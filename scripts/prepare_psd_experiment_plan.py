"""Freeze PSD inputs, development observation IDs and launch gates; start no jobs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_storage import load_bound, save_bound


def select_balanced(rows, *, count, seed, label, case_id):
    """Deterministic selection made before observing any model outcome."""
    if type(count) is not int or count < 2 or count > len(rows):
        raise ValueError("invalid observation/canary count")
    groups = {name: [] for name in ("real", "fake")}
    if len({case_id(row) for row in rows}) != len(rows):
        raise ValueError("duplicate input case IDs")
    for row in rows:
        groups[label(row)].append(row)
    if any(not group for group in groups.values()):
        raise ValueError("both classes are required")
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256((seed + "\0" + case_id(row)).encode()).hexdigest())
    selected = []
    while len(selected) < count:
        for group in groups.values():
            if group and len(selected) < count:
                selected.append(group.pop(0))
    return selected


def prepare(*, pool, test_manifest, test_runtime, storage_root, output, dev_size=400, canary_size=32):
    storage_root = storage_root.resolve()
    for path in (pool, test_manifest, test_runtime, output):
        path.resolve().relative_to(storage_root)
    benchmark = pool / "train/runtime-release/runtime_input/cases.jsonl"
    gold_path = pool / "train/evaluator_private/private_gold.jsonl"
    split_path = pool / "train/evaluator_private/case_split.jsonl"
    policy = pool / "train/runtime-release/evaluator_private/source_access_policy.json"
    report_path, verification_path = pool / "selection-report.json", pool / "release-verification.json"
    report, verification = load_json(report_path), load_json(verification_path)
    if (report.get("status") != "ready_for_fresh_policy_rollouts" or verification.get("passed") is not True
            or verification.get("selection_report_sha256") != sha256_file(report_path)):
        raise ValueError("PSD pool verification binding is invalid")
    public, gold, splits = load_jsonl(benchmark), load_jsonl(gold_path), load_jsonl(split_path)
    case_ids = [r["case_id"] for r in public]
    if (len(case_ids) != 4000 or len(set(case_ids)) != 4000
            or case_ids != [r["case_id"] for r in gold] or case_ids != [r["case_id"] for r in splits]
            or any(r.get("split") != "train" for r in splits)
            or any(set(r) != {"case_id", "image_path", "image_sha256"} for r in public)
            or sha256_file(benchmark) != report["releases"]["train"]["sha256"]):
        raise ValueError("PSD training release differs from frozen 4000 pool")
    labels = {r["case_id"]: {"supported": "real", "refuted": "fake"}[r["factual_status"]] for r in gold}
    if Counter(labels.values()) != {"real": 1421, "fake": 2579}:
        raise ValueError("PSD frozen class counts changed")
    seed = "psd-observation-20260915-v1"
    canary = select_balanced(public, count=canary_size, seed=seed + "-training",
        label=lambda r: labels[r["case_id"]], case_id=lambda r: r["case_id"])
    test, runtime = load_jsonl(test_manifest), load_jsonl(test_runtime)
    test_by_id = {r["unified_case_id"]: r for r in test}
    runtime_by_id = {r["case_id"]: r for r in runtime}
    if (len(test) != 1527 or len(test_by_id) != 1527 or len(runtime) != 1526 or len(runtime_by_id) != 1526
            or not set(runtime_by_id).issubset(test_by_id)):
        raise ValueError("formal test denominator/membership changed")
    available = [r for r in test if r["unified_case_id"] in runtime_by_id]
    test_label = lambda r: {"supported": "real", "refuted": "fake"}[r["factual_status"]]
    dev = select_balanced(available, count=dev_size, seed=seed + "-evaluation",
        label=test_label, case_id=lambda r: r["unified_case_id"])
    # Observation list only: no copied images, test gold or trainable targets.
    dev_rows = [{"case_id": r["unified_case_id"], "split": "test", "training_prohibited": True} for r in dev]
    paths = (benchmark, gold_path, split_path, policy, report_path, verification_path, test_manifest, test_runtime)
    identity = {"schema_version": "ifv-psd-experiment-plan-v1", "seed": seed,
        "dev_size": dev_size, "canary_size": canary_size,
        "files": {str(p.resolve()): sha256_file(p) for p in paths}}
    marker = output / "prepared.json"
    if marker.exists():
        plan = load_bound(marker, identity=identity)
        for name, digest in plan["artifacts"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("prepared PSD plan artifact changed")
        return plan
    if output.exists() and any(output.iterdir()):
        raise ValueError("PSD plan output must be new or bound")
    write_jsonl(output / "training-canary-case-list.jsonl", [{"case_id": r["case_id"], "split": "train"} for r in canary])
    write_jsonl(output / "development-observation-case-list.jsonl", dev_rows)
    plan = {"schema_version": "ifv-psd-experiment-plan-v1", "training_started": False,
        "status": "prepared_waiting_for_main_experiments_and_new_sft_snapshot",
        "training": {"benchmark": str(benchmark), "private_gold": str(gold_path), "train_cases": str(split_path),
            "source_access_policy": str(policy), "samples": 4000, "class_counts": dict(Counter(labels.values())),
            "training_source_dev_holdout": 0, "canary_samples": canary_size,
            "canary_class_counts": dict(Counter(labels[r["case_id"]] for r in canary))},
        "development_observation": {"samples": dev_size, "class_counts": dict(Counter(test_label(r) for r in dev)),
            "source": "fixed formal-test subset; not an independent heldout evaluation", "training_prohibited": True,
            "selection_depends_on_model_results": False, "runtime": str(test_runtime)},
        "formal_evaluation": {"denominator": 1527, "available_images": 1526,
            "metrics": ["BAcc", "SESR", "completion", "tool_contract_errors", "thinking_length_stops"],
            "judge": "same frozen final-evaluation judge and full-material protocol across models"},
        "snapshot": {"path": None, "policy": "selected verified new SFT checkpoint; never silently substitute Base/old SFT"},
        "recipe": {"rounds": 1, "epochs_per_round": 5, "learning_rate": 4e-5, "lora_rank": 32,
            "topk": 20, "effective_unique_targets_per_update": 32, "max_context_tokens": 131072,
            "step_count": "unknown until admitted assistant-step targets are materialized",
            "step_formula": "5 * ceil(admitted_targets / 32), confirm with actual distributed dataloader",
            "repair_attempts_per_case": 6, "hint_proposal_rounds": 12, "source_review_concurrency": 4,
            "repair_case_concurrency_canary": 2, "repair_case_concurrency_candidates": [4, 8, 16, 32, 40],
            "scheduling": "asynchronous construction within one frozen policy; no stale-policy training",
            "initial_gpu_profile": "training/configs/psd/qwen3.5-lora-r32-h20-sp4-128k.env",
            "throughput_candidate": "training/configs/psd/qwen3.5-lora-r32-h20-dp4-128k.env",
            "profile_selection": "same canary targets and batch=32; fastest stable profile after worst-length and checkpoint tests"},
        "resource_policy": {"storage_root": str(storage_root), "preserve_old_weights": True,
            "stop_other_processes": False, "shared_free_space_is_user_quota": False,
            "checkpoint_estimate_bytes": 8000000000, "checkpoint_estimate_is_not_measured": True,
            "checkpoint_copies_peak": 2, "storage_reserve_factor": 1.25,
            "require_user_quota_or_explicit_budget_before_production": True},
        "gates": ["main experiment GPU work complete", "new SFT export/load and exact snapshot binding",
            "fresh 32-case multimodal/token/tool canary", "bound source task review and causal repair",
            "top20 target admission", "worst-length throughput/memory probe", "real GPU optimizer/full-state reload",
            "complete selected 4000-case fresh collection", "quota and round budget preflight",
            "explicit production start then fixed-observation and formal evaluation"],
        "artifacts": {name: sha256_file(output / name) for name in (
            "training-canary-case-list.jsonl", "development-observation-case-list.jsonl")}}
    write_json(output / "plan.json", plan)
    plan["artifacts"]["plan.json"] = sha256_file(output / "plan.json")
    save_bound(marker, identity=identity, payload=plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "test-manifest", "test-runtime", "storage-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--canary-size", type=int, default=32)
    plan = prepare(**vars(parser.parse_args()))
    print(json.dumps({"status": plan["status"], "training_samples": plan["training"]["samples"],
        "development_samples": plan["development_observation"]["samples"], "training_started": False}, ensure_ascii=False))
