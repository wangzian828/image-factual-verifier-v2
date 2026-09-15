"""Prepare a 400-task PSD projection; copy no image bytes and start no jobs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_storage import load_bound, save_bound
from src.eval.public_release import load_public_release


def select_pilot(rows, *, labels, required_ids, count=400, seed="psd-pilot-20260915-v1"):
    """Balance labels without selecting on new policy outcomes; retain the canary."""
    by_id = {r["case_id"]: r for r in rows}
    if len(by_id) != len(rows) or len(set(required_ids)) != len(required_ids):
        raise ValueError("duplicate PSD pool/canary IDs")
    if type(count) is not int or count < len(required_ids) or count % 2 or count < 2:
        raise ValueError("PSD pilot must have an even, sufficient sample count")
    if not set(required_ids) <= set(by_id) or set(labels) != set(by_id):
        raise ValueError("PSD pilot source membership mismatch")
    selected = set(required_ids)
    for label in ("real", "fake"):
        remaining = count // 2 - sum(labels[c] == label for c in selected)
        available = [r for r in rows if r["case_id"] not in selected and labels[r["case_id"]] == label]
        available.sort(key=lambda r: hashlib.sha256((seed + "\0" + r["case_id"]).encode()).hexdigest())
        if remaining < 0 or len(available) < remaining:
            raise ValueError("PSD canary or pool cannot meet balanced pilot quota")
        selected.update(r["case_id"] for r in available[:remaining])
    if len(selected) != count:
        raise ValueError("PSD pilot class labels/count invalid")
    return [by_id[c] for c in sorted(selected)]


def prepare(*, previous_plan, output, storage_root, count=400):
    storage_root = storage_root.resolve()
    previous_plan, output = previous_plan.resolve(), output.resolve()
    for path in (previous_plan, output):
        path.relative_to(storage_root)
    if output == previous_plan or previous_plan in output.parents or output in previous_plan.parents:
        raise ValueError("PSD pilot must not overwrite/nest its frozen predecessor")
    saved = load_json(previous_plan / "prepared.json")
    old = load_bound(previous_plan / "prepared.json", identity=saved["identity"])
    for path, digest in saved["identity"]["files"].items():
        file = Path(path).resolve()
        file.relative_to(storage_root)
        if sha256_file(file) != digest:
            raise ValueError("frozen PSD preparation input changed")
    for path, digest in old["artifacts"].items():
        file = (previous_plan / path).resolve()
        file.relative_to(previous_plan)
        if sha256_file(file) != digest:
            raise ValueError("frozen PSD plan artifact changed")
    source = old["training"]
    benchmark = Path(source["benchmark"])
    release = load_public_release(benchmark)
    rows = load_jsonl(benchmark)
    gold_rows = load_jsonl(Path(source["private_gold"]))
    split_rows = load_jsonl(Path(source["train_cases"]))
    if ([r["case_id"] for r in rows] != [r["case_id"] for r in gold_rows]
            or [r["case_id"] for r in rows] != [r["case_id"] for r in split_rows]
            or any(r.get("split") != "train" for r in split_rows)):
        raise ValueError("PSD train/gold/split membership mismatch")
    gold, splits = ({r["case_id"]: r for r in rs} for rs in (gold_rows, split_rows))
    labels = {c: {"supported": "real", "refuted": "fake"}[r["factual_status"]] for c, r in gold.items()}
    canary = load_jsonl(previous_plan / "training-canary-case-list.jsonl")
    if any(r.get("split") != "train" for r in canary):
        raise ValueError("PSD canary is not training-only")
    selected = select_pilot(rows, labels=labels, required_ids=[r["case_id"] for r in canary], count=count)
    blocked = load_jsonl(Path(old["development_observation"]["runtime"]))
    blocked_ids = {r["case_id"] for r in blocked}
    blocked_images = {r["image_sha256"] for r in blocked}
    image_paths = {}
    for row in selected:
        if row["case_id"] in blocked_ids or row["image_sha256"] in blocked_images:
            raise ValueError("PSD pilot overlaps formal evaluation")
        image = (benchmark.parent / row["image_path"]).resolve()
        image.relative_to(storage_root)
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError("PSD pilot image hash mismatch")
        image_paths[row["case_id"]] = image
    identity = {"schema_version": "ifv-psd-pilot-v1", "count": count,
        "previous_plan_sha256": sha256_file(previous_plan / "prepared.json"),
        "release_manifest_sha256": sha256_file(release.manifest_path),
        "source_access_policy_sha256": sha256_file(Path(source["source_access_policy"])),
        "seed": "psd-pilot-20260915-v1", "group_size": 8, "temperature": 0.7}
    marker = output / "prepared.json"
    if marker.exists():
        result = load_bound(marker, identity=identity)
        for path, digest in result["artifacts"].items():
            file = (output / path).resolve()
            file.relative_to(output)
            if sha256_file(file) != digest:
                raise ValueError("PSD pilot artifact changed")
        return result
    if output.exists() and any(output.iterdir()):
        raise ValueError("PSD pilot output must be new or bound")
    runtime = output / "runtime-release/runtime_input"
    public = []
    for row in selected:
        image = image_paths[row["case_id"]]
        destination = runtime / "assets" / (row["image_sha256"] + image.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(image, destination)
        public.append({"case_id": row["case_id"], "image_path": "assets/" + destination.name,
            "image_sha256": row["image_sha256"]})
    write_jsonl(runtime / "cases.jsonl", public)
    write_jsonl(output / "evaluator_private/private_gold.jsonl", [gold[r["case_id"]] for r in selected])
    write_jsonl(output / "evaluator_private/case_split.jsonl", [splits[r["case_id"]] for r in selected])
    write_jsonl(output / "training-canary-case-list.jsonl", canary)
    write_jsonl(output / "development-observation-case-list.jsonl",
        load_jsonl(previous_plan / "development-observation-case-list.jsonl"))
    policy_path = runtime.parent / "evaluator_private/source_access_policy.json"
    write_json(policy_path, load_json(Path(source["source_access_policy"])))
    manifest = {**release.manifest, "release_id": "psd-pilot-400-v1", "release_stage": "psd_training_pilot",
        "source_access_policy": {"active": True, "path": "evaluator_private/source_access_policy.json",
            "sha256": sha256_file(policy_path)}}
    write_json(runtime.parent / "manifest.json", manifest)
    load_public_release(runtime / "cases.jsonl")
    result = {"schema_version": identity["schema_version"], "status": "inputs_ready_not_started",
        "training_started": False, "collection_started": False, "tasks": len(selected),
        "unique_images": len({r["image_sha256"] for r in selected}),
        "class_counts": dict(Counter(labels[r["case_id"]] for r in selected)),
        "canary_included": len(canary), "source_pool_retained": True, "new_image_bytes": 0,
        "benchmark": str(runtime / "cases.jsonl"),
        "private_gold": str(output / "evaluator_private/private_gold.jsonl"),
        "train_cases": str(output / "evaluator_private/case_split.jsonl"),
        "source_access_policy": str(policy_path),
        "sampling": {"rollouts_per_case": 8, "episodes": count * 8, "temperature": 0.7,
            "base_sampling_seed": 0, "collection_once_per_round_not_per_epoch": True},
        "repair_source_selection": "longest_failed_episode_per_task", "repair_attempts_per_task": 6,
        "recipe": {**old["recipe"], "repair_weight_per_target": 1, "preservation_weight_per_target": 1,
            "aggregate_source_rebalancing": False},
        "launch_gate": "main evaluation first; multi-position/real-canary validation and explicit launch required",
        "formal_evaluation": old["formal_evaluation"],
        "artifacts": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}}
    save_bound(marker, identity=identity, payload=result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-plan", "output", "storage-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k not in {"artifacts", "recipe"}}, ensure_ascii=False))
