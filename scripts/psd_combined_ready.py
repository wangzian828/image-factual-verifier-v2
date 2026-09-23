"""Attest the selected two-source PSD bank without inventing a rollout gate.

This package binds the two historical collection lineages independently.  It
uses the same student-initialization and datum validators as a single round,
but deliberately does not claim that the targets came from one rollout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ifv_training.artifact_receipts import artifact_identity
from ifv_training.io import load_json, sha256_file, write_json
from ifv_training.psd_initialization import verify_initialization
from ifv_training.psd_preflight import verify_psd_training_input
from ifv_training.psd_repair_storage import save_bound


SCHEMA = "ifv-psd-combined-sft3-ready-v1"
MODEL = "ifv-qwen3.5-9b-sft-3084"
CHECKPOINT_SHA = "d5517f95d69ade6f72a4da5289254c0386603dc6c3550457b1e3ee16397cb98a"


def _path(root: Path, value: str) -> Path:
    path = Path(value).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or symlinked combined input: {path}")
    return path


def attest(*, datums: Path, scored_manifest: Path, small_manifest: Path,
           old_manifest: Path, selection_manifest: Path, freeze_manifest: Path,
           score_state: Path, snapshot: Path, output: Path, root: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    output.relative_to(root / "runs")
    if output.exists():
        raise FileExistsError(output)
    paths = [_path(root, str(p)) for p in (datums, scored_manifest, small_manifest,
        old_manifest, selection_manifest, freeze_manifest, score_state,
        snapshot / "serving-profile.json", snapshot / "checkpoint-manifest.json")]
    datums, scored_manifest, small_manifest, old_manifest, selection_manifest, \
        freeze_manifest, score_state, serving, checkpoint_manifest = paths
    datum_manifest = _path(root, str(datums.parent / "manifest.json"))
    scored_targets = _path(root, str(scored_manifest.parent / "targets.jsonl"))
    scores = load_json(score_state)
    combined = load_json(scored_manifest)
    small = load_json(small_manifest)
    old = load_json(old_manifest)
    freeze = load_json(freeze_manifest)
    selected = load_json(selection_manifest)
    if (scores.get("phase") != "raw_teacher_scoring_complete"
            or scores.get("services_restored") is not True
            or combined.get("schema_version") != "ifv-psd-combined-sft3-scored-v1"
            or combined.get("targets") != 3196
            or combined.get("repair_targets") != 1598
            or combined.get("preservation_targets") != 1598
            or combined.get("checkpoint_manifest_sha256") != CHECKPOINT_SHA
            or combined.get("teacher_logprob_semantics") != "pre_grammar_unprocessed_v1"
            or small.get("counts") != {"repair": 626, "preserve": 626}
            or old.get("counts") != {"repair": 972, "preserve": 972}
            or freeze.get("accepted_repair_targets") != 990
            or selected.get("metadata_coverage_gate_pass") is not True
            or selected.get("uses_gold_or_teacher_score") is not False
            or selected.get("status") != "metadata_only_not_authorized_to_score_or_train"):
        raise ValueError("two-source selection, freeze or raw-teacher gate failed")
    if len(combined.get("source_stat_bindings", {})) < 4:
        raise ValueError("combined bank lost source-stat bindings")
    for name, stat in combined["source_stat_bindings"].items():
        path = _path(root, name)
        s = path.stat()
        if stat != {"device": s.st_dev, "inode": s.st_ino,
                    "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}:
            raise ValueError("combined source stat binding changed")
    datum_record = load_json(datum_manifest)
    if (Path(datum_record.get("source", {}).get("targets", "")).resolve() != scored_targets
            or datum_record.get("targets_by_kind") != {"preserve": 1598, "repair": 1598}
            or datum_record.get("counts", {}).get("rejections") != 0):
        raise ValueError("combined datums differ from scored bank")
    gate = verify_psd_training_input(datums_path=datums,
        manifest_path=datum_manifest, expected_topk=20, max_context=131072)
    if (not gate["passed"] or gate["datums"]["rows"] != 3196
            or gate["datums"]["by_kind"] != {"preserve": 1598, "repair": 1598}):
        raise ValueError("combined datums failed exact balanced 3196-row gate")
    profile = load_json(serving)
    if (profile.get("profile_id") != MODEL
            or profile.get("base_url") != "http://127.0.0.1:19025/v1"
            or Path(profile.get("engine_model_path", "")).resolve() !=
                    (root / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model").resolve()):
        raise ValueError("combined serving snapshot is not frozen SFT3 at the owned gateway")
    initialization = verify_initialization(datum_manifest_path=datum_manifest,
        serving_profile_path=serving, checkpoint_manifest_path=checkpoint_manifest,
        model_path=Path(profile["engine_model_path"]))
    if (not initialization["passed"]
            or initialization["checkpoint_manifest_sha256"] != CHECKPOINT_SHA
            or initialization["targets"] != 3196):
        raise ValueError("combined student initialization differs from raw teacher")
    identity_paths = paths + [datum_manifest, scored_targets]
    identity = {"files": {str(p): artifact_identity(p) for p in identity_paths}}
    payload = {"schema_version": SCHEMA, "datums": str(datums),
        "datum_manifest": str(datum_manifest), "scored_targets": str(scored_targets),
        "scored_manifest": str(scored_manifest),
        "source_manifests": [str(small_manifest), str(old_manifest)],
        "selection_manifest": str(selection_manifest), "freeze_manifest": str(freeze_manifest),
        "score_state": str(score_state), "serving_profile": str(serving),
        "checkpoint_manifest": str(checkpoint_manifest),
        "native_resume_result": str(output / "dp4-resume-gate-combined-v1/result.json"),
        "model": profile["engine_model_path"], "adapter": None,
        "initialization": initialization, "target_counts": {"repair": 1598, "preserve": 1598},
        "preparation_mode": "attested_selected_two_source_bank_without_resampling"}
    output.mkdir(parents=True, exist_ok=False)
    save_bound(output / "ready.json", identity=identity, payload=payload)
    report = {"schema_version": "ifv-psd-combined-attestation-v1", "passed": True,
        "ready": str(output / "ready.json"), "targets": 3196,
        "repair_targets": 1598, "preservation_targets": 1598,
        "checkpoint_manifest_sha256": CHECKPOINT_SHA, "formal_training_started": False}
    write_json(output / "attestation.json", report)
    return report


def complete(*, ready: dict, profile_path: Path, checkpoint_manifest_path: Path,
             output: Path, initialization_gate_path: Path) -> dict:
    """Check a real optimizer run; never synthesize single-rollout completion."""
    if ready.get("schema_version") != SCHEMA:
        raise ValueError("not a combined bank")
    training = load_json(profile_path)
    checkpoint = load_json(checkpoint_manifest_path)
    initialization = load_json(initialization_gate_path)
    source_manifest = load_json(Path(ready["datum_manifest"]))
    record = checkpoint.get("checkpoint", {})
    raw_gate = training.get("raw_dataset_gate", {}).get("verification", {})
    profile_data = raw_gate.get("datums", {})
    expected_manifest_sha = sha256_file(Path(ready["datum_manifest"]))
    checks = {
        "training_production_gate": training.get("passed_production_gate") is True,
        "not_smoke": training.get("smoke_only") is False,
        "four_gpu_dp4_global32": training.get("parallelism", {}).get("world_size") == 4
            and training.get("parallelism", {}).get("data_parallel_size") == 4
            and training.get("parallelism", {}).get("unique_samples_per_optimizer_step") == 32,
        "optimizer_steps": isinstance(record.get("global_step"), int)
            and record["global_step"] > 1
            and training.get("train_step_metric_rows") == record["global_step"],
        "finite_loss": training.get("loss", {}).get("count", 0) > 1
            and isinstance(training.get("loss", {}).get("last"), (int, float))
            and math.isfinite(training["loss"]["last"]),
        "initialization": initialization == ready["initialization"]
            and initialization.get("checkpoint_manifest_sha256") == CHECKPOINT_SHA,
        "same_datums": raw_gate.get("passed") is True
            and profile_data.get("path") == ready["datums"]
            and profile_data.get("rows") == 3196,
        "dataset_manifest_bound": checkpoint.get("training_dataset", {}).get("manifest_sha256")
            == expected_manifest_sha,
        "balanced_training_bank": source_manifest.get("targets_by_kind")
            == {"preserve": 1598, "repair": 1598},
        "native_resumable_state": all(record.get(key) is True for key in
            ("optimizer_state_available", "scheduler_state_available", "rng_state_available"))
            and any(a.get("scope") == "training_state" for a in checkpoint.get("artifacts", [])),
        "adapter_export_present": any(a.get("scope") == "model"
            and a.get("path") == "adapter_model.safetensors"
            for a in checkpoint.get("artifacts", []))
            and (Path(record.get("path", "")) / "adapter_model.safetensors").is_file(),
        "native_resume_preflight": (Path(ready["native_resume_result"]).is_file()
            and load_json(Path(ready["native_resume_result"])).get("adapter_bitwise_equal") is True),
    }
    result = {"schema_version": "ifv-psd-combined-sft3-completion-v1",
        "passed": all(checks.values()), "checks": checks,
        "profile": str(profile_path), "output_checkpoint_manifest": str(checkpoint_manifest_path),
        "global_step": record.get("global_step"), "formal_training": True}
    write_json(output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("datums", "scored-manifest", "small-manifest", "old-manifest",
                 "selection-manifest", "freeze-manifest", "score-state", "snapshot",
                 "output", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(attest(**vars(args))))


if __name__ == "__main__":
    main()
