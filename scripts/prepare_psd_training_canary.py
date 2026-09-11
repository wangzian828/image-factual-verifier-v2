"""Prepare a frozen PSD training-only canary using already delivered image bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from scripts.prepare_psd_training_metadata import SHA256, MEMBERS
from scripts.trajectory.run_teacher_rollout_autopilot import prepare_runtime_release, _case_id
from src.orchestrator.source_access import benchmark_source_access_policy


def prepare(args):
    metadata = load_json(args.metadata_root / "verified-metadata.json")
    if metadata.get("archive_sha256") != SHA256:
        raise ValueError("PSD official training metadata provenance mismatch")
    for name in MEMBERS - {"PACKAGE_MANIFEST.json"}:
        if sha256_file(args.metadata_root / name) != metadata["files"][name]:
            raise ValueError("PSD official metadata changed")
    split = {row["case_id"]: row for row in load_jsonl(args.case_split)}
    sft_ids = {row["case_id"] for row in load_jsonl(args.sft_index) if row.get("split") == "train"}
    prohibited = load_jsonl(args.prohibited_runtime)
    if not prohibited:
        raise ValueError("PSD canary requires the explicit held-out exclusion manifest")
    blocked_ids = {row["case_id"] for row in prohibited}
    blocked_images = {row["image_sha256"] for row in prohibited}
    rows = load_jsonl(args.metadata_root / "train-manifest.jsonl")
    eligible = [row for row in rows if _case_id(row) in sft_ids
                and split.get(_case_id(row), {}).get("split") == "train"]
    eligible.sort(key=lambda row: hashlib.sha256(("psd-canary-v1:" + _case_id(row)).encode()).hexdigest())
    selected = eligible[:args.limit]
    if args.limit < 2 or len(selected) != args.limit:
        raise ValueError("PSD canary scope is invalid or lacks enough training cases")
    for row in selected:
        case = _case_id(row)
        if case in blocked_ids or split[case]["image_sha256"] in blocked_images:
            raise ValueError("PSD canary overlaps a held-out case/image")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    staging = args.output_dir / "source"
    staging.mkdir()
    materialized = []
    for row in selected:
        case = _case_id(row)
        digest = split[case]["image_sha256"]
        matches = list(args.images_dir.glob(digest + ".*"))
        if len(matches) != 1 or sha256_file(matches[0]) != digest:
            raise ValueError("PSD canary delivered image hash mismatch")
        relative = Path("images") / matches[0].name
        target = staging / relative
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            os.link(matches[0], target)  # Same server filesystem; no second image copy.
        materialized.append({**row, "unified_image_path": relative.as_posix()})
    selected_ids = {_case_id(row) for row in selected}
    gold = [row for row in load_jsonl(args.metadata_root / "evaluator_private/private-gold-v1/train-private-gold.jsonl")
            if _case_id(row) in selected_ids]
    if len(gold) != len(selected):
        raise ValueError("PSD canary private-reference membership mismatch")
    write_jsonl(staging / "train-manifest.jsonl", materialized)
    write_jsonl(staging / "evaluator_private/private-gold-v1/train-private-gold.jsonl", gold)
    policy = benchmark_source_access_policy((str(row.get("source_url") or "") for row in rows),
                                            policy_id="psd-training-canary-v1")
    policy_path = staging / "source-access-policy.json"
    write_json(policy_path, policy.to_dict())
    prepared = prepare_runtime_release(dataset_root=staging, train_manifest=staging / "train-manifest.jsonl",
        output_dir=args.output_dir / "release", validation_count=0, source_access_policy=policy_path)
    report = {"schema_version": "ifv-psd-canary-preparation-v1", "training_only": True,
        "selection": "fixed hash order within current SFT train IDs; independent of model outcomes",
        "selected_case_ids": sorted(selected_ids), "new_image_bytes": 0,
        "official_archive_sha256": SHA256, "sft_index_sha256": sha256_file(args.sft_index),
        "case_split_sha256": sha256_file(args.case_split),
        "prohibited_runtime_sha256": sha256_file(args.prohibited_runtime), "preparation": prepared}
    write_json(args.output_dir / "prepared.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-root", "case-split", "sft-index", "images-dir", "prohibited-runtime", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
