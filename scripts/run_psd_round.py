"""Prepare, launch and close an attested PSD round without changing the Agent.

The normal Agent runner collects a fresh training-only rollout first. This
entry point then runs feedback repair, exact teacher scoring, target admission,
and resumable local assembly. Training is an explicit subcommand and requires
idle GPUs. It never stops inference services or the main SFT experiment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json
from ifv_training.psd_materialization import completed_package
from ifv_training.psd_candidates import build_psd_candidate_package, _load_train_case_allowlist
from ifv_training.psd_round import (
    PSD_ROLLOUT_GATE_SCHEMA_VERSION,
    complete_psd_round,
    validate_rollout_gate_for_candidates,
    verify_psd_round_rollout,
)
from ifv_training.psd_initialization import verify_initialization, frozen_base_binding
from ifv_training.psd_preflight import verify_psd_training_input
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.checkpoints import build_checkpoint_manifest
from scripts.continue_psd_canary import completed_stage
from scripts.postprocess_psd_training import postprocess
from scripts.run_psd_feedback_canary import run as run_feedback


def load_ready(path):
    saved = load_json(path)
    ready = load_bound(path, identity=saved["identity"])
    for name, digest in saved["identity"]["files"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("PSD ready-stage input changed")
    gate = verify_psd_training_input(datums_path=Path(ready["datums"]),
        manifest_path=Path(ready["datum_manifest"]), expected_topk=20, max_context=131072)
    if not gate["passed"]:
        raise ValueError("PSD prepared datums no longer pass verification")
    return ready


def attest(args):
    """Bind an already completed, verified PSD bank to its rollout policy.

    Repair search and exact top-k collection are expensive, crash-safe stages.
    This entry point lets an immutable completed bank enter the normal
    production launcher without resampling either stage.  It deliberately
    rechecks the rollout artifacts, datum package, target lineage and frozen
    student initialization before emitting ``ready.json``.
    """

    root = args.output.resolve()
    ready_path = root / "ready.json"
    if ready_path.exists():
        ready = load_ready(ready_path)
        expected = {
            "rollout_gate": str(args.rollout_gate.resolve()),
            "datums": str(args.datums.resolve()),
        }
        if any(ready.get(name) != value for name, value in expected.items()):
            raise ValueError("existing PSD ready stage is bound to different inputs")
        return {
            "status": "ready_for_training",
            "ready": str(ready_path),
            "training_started": False,
            "reused": True,
        }
    if root.exists() and any(root.iterdir()):
        raise ValueError("PSD attestation output must be new or already complete")
    root.mkdir(parents=True, exist_ok=True)

    gate_path = args.rollout_gate.resolve()
    gate = load_json(gate_path)
    gate_run = gate.get("run") if isinstance(gate.get("run"), dict) else {}
    gate_cases = (
        gate.get("train_cases")
        if isinstance(gate.get("train_cases"), dict)
        else {}
    )
    if (
        gate.get("schema_version") != PSD_ROLLOUT_GATE_SCHEMA_VERSION
        or gate.get("passed") is not True
    ):
        raise ValueError("PSD rollout gate is not a passing round gate")
    run_dir = Path(str(gate_run.get("directory") or "")).resolve()
    train_cases = Path(str(gate_cases.get("path") or "")).resolve()
    validate_rollout_gate_for_candidates(
        rollout_gate_path=gate_path,
        run_dir=run_dir,
        train_cases_path=train_cases,
    )

    datums = args.datums.resolve()
    datum_manifest = (
        args.datum_manifest.resolve()
        if args.datum_manifest
        else datums.parent / "manifest.json"
    )
    datum_gate = verify_psd_training_input(
        datums_path=datums,
        manifest_path=datum_manifest,
        expected_topk=20,
        max_context=131072,
    )
    if not datum_gate["passed"]:
        raise ValueError("completed PSD bank does not pass datum verification")

    datum_record = load_json(datum_manifest)
    targets = Path(datum_record["source"]["targets"]).resolve()
    target_rows = load_jsonl(targets)
    gate_sha256 = sha256_file(gate_path)
    round_index = gate.get("round_index")
    run_id = str(gate_run.get("run_id") or "")
    if not target_rows:
        raise ValueError("completed PSD bank has no source targets")
    lineage_errors = []
    for index, row in enumerate(target_rows):
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        if source.get("psd_rollout_gate_sha256") != gate_sha256:
            lineage_errors.append(f"target[{index}].rollout_gate")
        if source.get("psd_round_index") != round_index:
            lineage_errors.append(f"target[{index}].round_index")
        if source.get("source_run_id") != run_id:
            lineage_errors.append(f"target[{index}].run_id")
        if row.get("target_status") != "complete":
            lineage_errors.append(f"target[{index}].status")
    if lineage_errors:
        raise ValueError(
            "PSD target lineage differs from rollout gate: "
            + ",".join(lineage_errors[:20])
        )

    snapshot = args.snapshot.resolve()
    serving = snapshot / "serving-profile.json"
    checkpoint = snapshot / "checkpoint-manifest.json"
    profile = load_json(serving)
    base = profile.get("engine_model_path") or profile["model_path"]
    adapter = profile["model_path"] if base != profile["model_path"] else None
    initialization = verify_initialization(
        datum_manifest_path=datum_manifest,
        serving_profile_path=serving,
        checkpoint_manifest_path=checkpoint,
        model_path=base,
        adapter_path=adapter,
    )
    ready = {
        "schema_version": "ifv-psd-round-ready-v1",
        "round_index": round_index,
        "rollout_gate": str(gate_path),
        "datums": str(datums),
        "datum_manifest": str(datum_manifest),
        "serving_profile": str(serving),
        "checkpoint_manifest": str(checkpoint),
        "model": base,
        "adapter": adapter,
        "initialization": initialization,
        "preparation_mode": "attested_completed_bank_without_resampling",
    }
    identity_paths = (gate_path, datums, datum_manifest, targets, serving, checkpoint)
    ready_identity = {
        "files": {
            str(path.resolve()): sha256_file(path) for path in identity_paths
        }
    }
    save_bound(ready_path, identity=ready_identity, payload=ready)
    write_json(
        root / "attestation.json",
        {
            "schema_version": "ifv-psd-completed-bank-attestation-v1",
            "passed": True,
            "round_index": round_index,
            "source_run_id": run_id,
            "rollout_gate_sha256": gate_sha256,
            "target_count": len(target_rows),
            "datum_count": datum_gate["datums"]["rows"],
            "datum_sha256": datum_gate["datums"]["sha256"],
            "initialization": initialization,
        },
    )
    return {
        "status": "ready_for_training",
        "ready": str(ready_path),
        "training_started": False,
        "reused": False,
    }


async def prepare(args):
    root = args.output.resolve()
    snapshot = args.snapshot.resolve()
    serving, checkpoint = snapshot / "serving-profile.json", snapshot / "checkpoint-manifest.json"
    run_dir = args.run_dir.resolve()
    allowed = _load_train_case_allowlist(args.train_cases)
    public_rows = load_jsonl(args.benchmark)
    if not public_rows or len({row["case_id"] for row in public_rows}) != len(public_rows):
        raise ValueError("public training manifest must be nonempty and unique")
    if any(allowed.get(row["case_id"]) != "train" for row in public_rows):
        raise ValueError("PSD public manifest contains a non-training case")
    # Membership was selected before outcomes. Do not silently prepare only the
    # successful/fast cases from a partially completed collection.
    results = load_jsonl(run_dir / "run_results.jsonl")
    if {row["case_id"] for row in results} != {row["case_id"] for row in public_rows}:
        raise ValueError("PSD rollout does not cover the complete selected training manifest")
    paths = [args.benchmark, args.train_cases, args.private_gold, args.source_access_policy,
        serving, checkpoint, run_dir / "run_manifest.json", run_dir / "run_results.jsonl"]
    if args.previous_round_completion:
        paths.append(args.previous_round_completion)
    identity = {"round_index": args.round_index,
        "files": {str(path.resolve()): sha256_file(path) for path in paths}}
    root.mkdir(parents=True, exist_ok=True)
    completed_stage(root, "postprocess", identity, lambda: (
        postprocess(run_dir=run_dir, train_cases=args.train_cases, private_gold=args.private_gold,
            source_access_policy=args.source_access_policy),
        [run_dir / name for name in ("post_rollout_rewards.jsonl", "rollout_groups.jsonl", "psd-postprocess.json")]))
    gate = root / "rollout-gate.json"
    rollout = completed_stage(root, "rollout-gate", identity, lambda: (
        verify_psd_round_rollout(round_index=args.round_index, run_dir=run_dir,
            train_cases_path=args.train_cases, serving_profile_path=serving,
            round_start_checkpoint_manifest_path=checkpoint, output=gate,
            previous_round_completion_path=args.previous_round_completion), [gate]))
    if not rollout["passed"]:
        raise ValueError("PSD fresh-policy rollout gate failed")
    bank = root / "source-bank"
    candidates = bank / "candidates"
    completed_package(output_dir=candidates,
        input_files=[gate, args.train_cases, run_dir / "post_rollout_rewards.jsonl", run_dir / "rollout_groups.jsonl"],
        build=lambda destination: build_psd_candidate_package(run_dir=run_dir,
            train_cases_path=args.train_cases, rollout_gate_path=gate, output_dir=destination))
    preparation = {"schema_version": "ifv-psd-round-bank-v1", "round_index": args.round_index,
        "training_only": True, "preparation": {"benchmark": str(args.benchmark.resolve()),
            "case_split": str(args.train_cases.resolve()), "private_gold": str(args.private_gold.resolve()),
            "source_access_policy": str(args.source_access_policy.resolve()), "rollout_dir": str(run_dir)}}
    prepared = bank / "prepared.json"
    if prepared.exists() and load_json(prepared) != preparation:
        raise ValueError("PSD source preparation changed")
    write_json(prepared, preparation)
    result = await run_feedback(SimpleNamespace(source=bank, output=root / "search", snapshot=snapshot,
        attempts=args.attempts, case_concurrency=args.case_concurrency, judge_model=args.judge_model,
        score_missing_topk=True, teacher_device=args.teacher_device))
    if result["status"] != "search_complete_datums_materialized":
        return {"status": result["status"], "training_started": False, "search": str(root / "search/progress.json")}
    datums = root / "search/datums/datums.jsonl"
    manifest = datums.parent / "manifest.json"
    profile = load_json(serving)
    base = profile.get("engine_model_path") or profile["model_path"]
    adapter = profile["model_path"] if base != profile["model_path"] else None
    initialization = verify_initialization(datum_manifest_path=manifest, serving_profile_path=serving,
        checkpoint_manifest_path=checkpoint, model_path=base, adapter_path=adapter)
    ready = {"schema_version": "ifv-psd-round-ready-v1", "round_index": args.round_index,
        "rollout_gate": str(gate), "datums": str(datums), "datum_manifest": str(manifest),
        "serving_profile": str(serving), "checkpoint_manifest": str(checkpoint),
        "model": base, "adapter": adapter, "initialization": initialization}
    ready_identity = {"files": {str(path.resolve()): sha256_file(path)
        for path in (gate, datums, manifest, serving, checkpoint)}}
    save_bound(root / "ready.json", identity=ready_identity, payload=ready)
    return {"status": "ready_for_training", "ready": str(root / "ready.json"), "training_started": False}


def train(args):
    ready = load_ready(args.ready)
    training_data_root = args.training_data_root.resolve()
    if not training_data_root.is_dir():
        raise ValueError("PSD training data root must be an existing directory")
    env = os.environ.copy()
    env.update(IFV_MODEL_ID=ready["model"],
        IFV_TRAINING_DATA_ROOT=str(training_data_root),
        IFV_PSD_SERVING_PROFILE=ready["serving_profile"],
        IFV_PSD_ROUND_START_MANIFEST=ready["checkpoint_manifest"],
        IFV_PSD_INITIAL_ADAPTER=ready["adapter"] or "", IFV_PSD_ROUND_READY=str(args.ready.resolve()))
    command = ["bash", str(ROOT / "training/scripts/train/run_psd_topk.sh"),
        str(args.model_profile.resolve()), str(args.psd_profile.resolve()), ready["datums"], args.experiment_id]
    if args.resume_checkpoint:
        command.append(str(args.resume_checkpoint.resolve()))
    subprocess.run(command, env=env, check=True)
    return {"status": "training_command_completed", "round_index": ready["round_index"]}


def finalize(args):
    ready = load_ready(args.ready)
    initialization = load_json(args.initialization_gate)
    if initialization != ready["initialization"]:
        raise ValueError("optimizer initialization differs from the prepared round")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = output / "checkpoint-manifest.json"
    checkpoint_record = build_checkpoint_manifest(checkpoint_dir=args.checkpoint,
        dataset_manifest_path=Path(ready["datum_manifest"]), output_path=checkpoint_manifest,
        base_model_id=ready["model"], model_revision="", processor_revision="", method="lora")
    checkpoint_record["base_model_binding"] = frozen_base_binding(
        load_json(Path(ready["checkpoint_manifest"])), ready["model"])
    write_json(checkpoint_manifest, checkpoint_record)
    completion = complete_psd_round(rollout_gate_path=Path(ready["rollout_gate"]),
        training_profile_path=args.training_profile, output_checkpoint_manifest_path=checkpoint_manifest,
        initialization_gate_path=args.initialization_gate, output=output / "completion.json")
    if not completion["passed"]:
        raise ValueError("PSD round completion gate failed")
    next_round = {"round_index": ready["round_index"] + 1,
        "previous_round_completion": str(output / "completion.json"),
        "checkpoint_manifest": str(checkpoint_manifest), "model": ready["model"],
        "adapter": str(args.checkpoint.resolve()), "requires_fresh_training_rollout": True,
        "optimizer_state_for_new_round": "fresh", "heldout_improvement_measured": False}
    write_json(output / "next-round.json", next_round)
    return {"status": "round_completed", "completion": str(output / "completion.json"), "next_round": next_round}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("run-dir", "benchmark", "train-cases", "private-gold", "source-access-policy", "snapshot", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--round-index", type=int, required=True)
    prepare_parser.add_argument("--previous-round-completion", type=Path)
    prepare_parser.add_argument("--attempts", type=int, default=6)
    prepare_parser.add_argument("--case-concurrency", type=int, default=1)
    prepare_parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    prepare_parser.add_argument("--teacher-device", default="cpu")
    attest_parser = commands.add_parser("attest")
    for name in ("rollout-gate", "datums", "snapshot", "output"):
        attest_parser.add_argument("--" + name, type=Path, required=True)
    attest_parser.add_argument("--datum-manifest", type=Path)
    train_parser = commands.add_parser("train")
    for name in ("ready", "model-profile", "psd-profile"):
        train_parser.add_argument("--" + name, type=Path, required=True)
    train_parser.add_argument("--training-data-root", type=Path, required=True)
    train_parser.add_argument("--experiment-id", required=True)
    train_parser.add_argument("--resume-checkpoint", type=Path)
    finalize_parser = commands.add_parser("finalize")
    for name in ("ready", "checkpoint", "training-profile", "initialization-gate", "output"):
        finalize_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(prepare(args)) if args.command == "prepare" else globals()[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
