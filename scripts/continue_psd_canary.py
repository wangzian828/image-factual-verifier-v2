"""Resume a training-only canary through real repairs, admission and exact datums.

Does not launch training, change the Agent, or resample completed judgments.
Call after the on-policy run completes; successful stages are hash-checked.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_round import verify_psd_round_rollout
from ifv_training.psd_candidates import build_psd_candidate_package
from ifv_training.psd_gemini_judge import trace_steps
from ifv_training.psd_repairs import assemble_psd_repair_package
from ifv_training.psd import build_psd_target_package
from ifv_training.psd_datums import build_sparse_topk_package
from ifv_training.psd_repair_storage import load_bound, save_bound
from scripts.postprocess_psd_training import postprocess
from scripts.run_psd_repair_driver import _parser, _run


def completed_stage(root, name, identity, action):
    marker = root / "stage-cache" / (name + ".json")
    if marker.exists():
        saved = load_bound(marker, identity=identity)
        for path, digest in saved["files"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("completed PSD stage output changed: " + name)
        return saved["result"]
    result, files = action()
    save_bound(marker, identity=identity, payload={"result": result,
        "files": {str(path): sha256_file(path) for path in files}})
    return result


async def run(args):
    root = args.root.resolve()
    preparation_path = root / "prepared.json"
    preparation = load_json(preparation_path)["preparation"]
    run_dir = root / "on-policy-r1"
    manifest = load_json(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        return {"status": "waiting_for_on_policy_completion"}
    split, gold_path, policy = (Path(preparation[k]) for k in ("case_split", "private_gold", "source_access_policy"))
    benchmark = Path(preparation["benchmark"])
    serving = args.snapshot / "serving-profile.json"
    checkpoint = args.snapshot / "checkpoint-manifest.json"
    identity = {str(path): sha256_file(path) for path in (preparation_path, split, gold_path, policy,
        benchmark, serving, checkpoint, run_dir / "run_manifest.json", run_dir / "run_results.jsonl")}
    summary = {"status": "postprocessing", "canary_only": True, "training_started": False}
    def save():
        write_json(root / "acceptance-progress.json", summary)
    save()
    completed_stage(root, "postprocess", identity, lambda: (
        postprocess(run_dir=run_dir, train_cases=split, private_gold=gold_path, source_access_policy=policy),
        [run_dir / "post_rollout_rewards.jsonl", run_dir / "rollout_groups.jsonl", run_dir / "psd-postprocess.json"]))
    gate = root / "rollout-gate.json"
    gate_result = verify_psd_round_rollout(round_index=1, run_dir=run_dir, train_cases_path=split,
        serving_profile_path=serving, round_start_checkpoint_manifest_path=checkpoint, output=gate)
    if not gate_result["passed"]:
        raise ValueError("real PSD rollout provenance gate failed")
    # Gate includes created_at: freeze it once, rather than invalidating a cached bank on restart.
    gate_cache = root / "stage-cache/rollout-gate.json"
    if gate_cache.exists():
        saved_gate = load_bound(gate_cache, identity=identity)
        write_json(gate, saved_gate)
    else:
        save_bound(gate_cache, identity=identity, payload=gate_result)
    bank = root / "candidates"
    summary["candidate_bank"] = completed_stage(root, "candidates", identity, lambda: (
        build_psd_candidate_package(run_dir=run_dir, train_cases_path=split, rollout_gate_path=gate, output_dir=bank),
        [bank / name for name in ("manifest.json", "repair_candidates.jsonl", "preservation_candidates.jsonl")]))
    profile = load_json(serving)
    public = {row["case_id"]: row for row in load_jsonl(benchmark)}
    gold = {row["case_id"]: row for row in load_jsonl(gold_path)}
    summary.update(status="repairing", repairs=[])
    save()
    candidates, attempts = [], []
    for candidate in load_jsonl(bank / "repair_candidates.jsonl"):
        key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
        inputs = root / "repair-inputs" / key
        output = root / "repairs" / key
        trace_path = run_dir / candidate["source"]["source_trace_path"]
        trace = load_json(trace_path)
        case = candidate["case_id"]
        write_json(inputs / "candidate.json", candidate)
        write_json(inputs / "gold.json", gold[case])
        write_json(inputs / "public.json", {"case_id": case, "source_steps": trace_steps(trace)})
        cli = ["--trace", str(trace_path), "--candidate", str(inputs / "candidate.json"),
            "--audit", str(run_dir / "psd-audits" / (trace_path.stem + ".json")),
            "--image", str(benchmark.parent / public[case]["image_path"]), "--gold", str(inputs / "gold.json"),
            "--public-context", str(inputs / "public.json"), "--private-context", str(inputs / "gold.json"),
            "--train-cases", str(split), "--source-access-policy", str(policy), "--output-dir", str(output),
            "--policy-model", profile["profile_id"], "--policy-base-url", profile["base_url"],
            "--policy-serving-profile", str(serving), "--round-start-checkpoint", profile["model_path"],
            "--round-start-checkpoint-manifest", str(checkpoint), "--hint-constructor-provider", "gemini",
            "--hint-constructor-model", args.judge_model, "--hint-constructor-wire-api", "interactions",
            "--judge-model", args.judge_model, "--hint-count", "2", "--max-suffix-actions", "8"]
        if (output / "run-inputs.json").exists():
            cli.append("--resume")
        try:
            result = await _run(_parser().parse_args(cli))
            summary["repairs"].append({"case_id": case, "output": str(output), "result": result})
        except Exception as exc:
            # Keep a negative/no-site result; never manufacture a passing example.
            summary["repairs"].append({"case_id": case, "output": str(output), "error_type": type(exc).__name__})
        save()
        if (output / "repair_candidates.jsonl").exists() and (output / "repair_attempts.jsonl").exists():
            candidates.extend(load_jsonl(output / "repair_candidates.jsonl"))
            attempts.extend(load_jsonl(output / "repair_attempts.jsonl"))
    summary["accepted_attempts"] = sum(row.get("accepted") is True for row in attempts)
    if not summary["accepted_attempts"]:
        summary["status"] = "no_verified_repair_or_pending_verification"
        save()
        return summary
    merge = root / "merged"
    write_jsonl(merge / "repair_candidates.jsonl", candidates)
    write_jsonl(merge / "repair_attempts.jsonl", attempts)
    assembly_identity = {**identity, "candidates": sha256_file(merge / "repair_candidates.jsonl"),
                         "attempts": sha256_file(merge / "repair_attempts.jsonl")}
    assembly = root / "assembled"
    summary["assembly"] = completed_stage(root, "assembly", assembly_identity, lambda: (
        assemble_psd_repair_package(repair_candidates_path=merge / "repair_candidates.jsonl",
            repair_attempts_path=merge / "repair_attempts.jsonl",
            preservation_candidates_path=bank / "preservation_candidates.jsonl", output_dir=assembly),
        [assembly / name for name in ("manifest.json", "repairs.jsonl", "preservation.jsonl")]))
    targets = root / "targets"
    summary["targets"] = completed_stage(root, "targets", assembly_identity, lambda: (
        build_psd_target_package(repairs_path=assembly / "repairs.jsonl",
            preservation_path=assembly / "preservation.jsonl", output_dir=targets),
        [targets / "manifest.json", targets / "targets.jsonl"]))
    if summary["targets"]["status"] != "ready_for_training":
        summary["status"] = "requires_frozen_teacher_topk_or_missing_kind"
    else:
        datums = root / "datums"
        summary["datums"] = completed_stage(root, "datums", assembly_identity, lambda: (
            build_sparse_topk_package(targets_path=targets / "targets.jsonl", output_dir=datums,
                topk=20, max_sequence_length=131072, require_both_kinds=True), [datums / "manifest.json"]))
        summary["status"] = ("datums_materialized_pending_optimizer_acceptance"
            if summary["datums"]["status"] == "ready_for_trainer" else "blocked_invalid_datums")
    save()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))
