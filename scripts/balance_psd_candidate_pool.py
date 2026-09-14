"""Downsample an accepted PSD input pool to a round, class-balanced size."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("IFV_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
if os.environ.get("IFV_PSD_SELECTOR_PATH"):
    spec = importlib.util.spec_from_file_location("psd_selector", os.environ["IFV_PSD_SELECTOR_PATH"])
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    describe, digest, release, reuse_verified_images, scoped = (
        getattr(selector, name) for name in ("describe", "digest", "release", "reuse_verified_images", "scoped"))
else:
    from scripts.prepare_psd_candidate_pool import describe, digest, release, reuse_verified_images, scoped
from src.orchestrator.source_access import SourceAccessPolicy


def balanced_sample(rows, target, seed):
    """Preserve scarce labels; capped equal allocation over routes per label.

    All multi-case groups are retained. Remaining slots are filled by singleton
    groups, preserving exact totals without splitting linked cases. Refuse an
    infeasible quota instead of quietly dropping part of a linked group.
    """
    counts = Counter(r["label"] for r in rows)
    if set(counts) != {"real", "fake"} or not 0 < target <= len(rows):
        raise ValueError("invalid target or unsupported label set")
    real = min(counts["real"], target // 2)
    fake = target - real
    if fake > counts["fake"]:
        fake, real = counts["fake"], target - counts["fake"]
    quotas = {"real": real, "fake": fake}
    groups = defaultdict(list)
    for row in rows:
        groups[row["selection_group_id"]].append(row)
    selected = [r for group in groups.values() if len(group) > 1 for r in group]
    used = Counter(r["label"] for r in selected)
    routes_used = Counter((r["label"], r["route"]) for r in selected)
    available = defaultdict(list)
    for group in groups.values():
        if len(group) == 1:
            row = group[0]
            available[(row["label"], row["route"])].append(row)
    for values in available.values():
        values.sort(key=lambda r: digest(seed + ":" + r["case_id"]), reverse=True)
    for label in ("real", "fake"):
        if used[label] > quotas[label]:
            raise ValueError("exact class quota incompatible with preserved multi-case groups")
        while used[label] < quotas[label]:
            candidates = [key for key, values in available.items() if key[0] == label and values]
            if not candidates:
                raise ValueError("insufficient singleton groups for exact class quota")
            key = min(candidates, key=lambda key: (routes_used[key], digest(seed + ":" + key[1])))
            selected.append(available[key].pop())
            used[label] += 1
            routes_used[key] += 1
    selected.sort(key=lambda r: r["case_id"])
    return selected


def prepare(source, output, storage_root, target, seed):
    source, output = scoped(source, storage_root), scoped(output, storage_root)
    if output.exists() or output == storage_root.resolve():
        raise ValueError("balanced output must be a new scoped directory")
    source_report = load_json(source / "selection-report.json")
    if source_report["status"] != "ready_for_fresh_policy_rollouts":
        raise ValueError("source candidate pool is not ready")
    rows = load_jsonl(source / "selection/train-final.jsonl")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source case")
    selected = balanced_sample(rows, target, seed)
    kept = {r["case_id"] for r in selected}
    prior_parts = {name: load_jsonl(source / "selection" / (name + "-final.jsonl"))
                   for name in ("hard_train", "sft_revisit", "development")}
    if prior_parts["development"]:
        raise ValueError("this workflow expects no training-source holdout")
    final = {name: [r for r in values if r["case_id"] in kept] for name, values in prior_parts.items()}
    final["train"] = selected
    output.mkdir(parents=True)
    images = reuse_verified_images(source, output, selected, storage_root)
    gold = {r["case_id"]: r for r in load_jsonl(source / "train/evaluator_private/private_gold.jsonl")}
    policy = SourceAccessPolicy.load(source / "train/runtime-release/evaluator_private/source_access_policy.json")
    identity = {"version": "ifv-psd-balanced-pool-v1", "source": str(source), "target": target, "seed": seed,
        "files": {str(source / "selection/train-final.jsonl"): sha256_file(source / "selection/train-final.jsonl"),
                  str(source / "selection-report.json"): sha256_file(source / "selection-report.json")}}
    report = {"schema_version": "ifv-psd-balanced-pool-v1", "source_identity": identity,
        "status": "selected_images_pending", "before": describe(rows), "after": describe(selected),
        "selection_policy": "keep scarce class; capped equal route allocation within label; preserve all multi-case groups; seeded singleton ordering",
        "reporting_name": "PSD training samples", "source_data_deleted": False,
        "fresh_current_policy_rollout_required": True, "psd_targets_created": False}
    write_json(output / "identity.json", identity)
    write_json(output / "selection-report.json", report)
    # Rehash the actual original bytes before publishing the revised inputs.
    checked = set()
    for row in selected:
        image = images[row["official_image_member"]]
        if image["path"] not in checked:
            if sha256_file(scoped(Path(image["path"]), storage_root)) != image["sha256"]:
                raise ValueError("cached original image bytes changed")
            checked.add(image["path"])
    report["releases"] = {name: release(output, name, values, images, gold, policy) for name, values in final.items()}
    for name, values in final.items():
        write_jsonl(output / "selection" / (name + "-final.jsonl"), values)
    write_jsonl(output / "selection/not-selected.jsonl", [r for r in rows if r["case_id"] not in kept])
    report.update(status="ready_for_fresh_policy_rollouts", final_pools={name: describe(values) for name, values in final.items()})
    write_json(output / "selection-report.json", report)
    return {"status": report["status"], "before": report["before"], "after": report["after"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "storage-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--seed", default="psd-balanced-20260914-v1")
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.storage_root, args.target, args.seed), flush=True)


if __name__ == "__main__":
    main()
