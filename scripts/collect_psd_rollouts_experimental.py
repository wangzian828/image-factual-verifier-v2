"""Collect an isolated PSD source-sampling batch with a non-published group size.

This entry point deliberately keeps the native PSD capture path but does not
claim that a four-rollout group is the published eight-rollout PSD bank.
Artifacts from this command must remain in their own run directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from ifv_training.io import load_jsonl, load_json, sha256_file, write_json
from ifv_training.psd_capture_semantics import install_capture_semantics
from ifv_training.psd_collection import require_token_capture_environment
from scripts.collect_psd_rollouts import PSDWorkflow


def main() -> None:
    install_capture_semantics()
    require_token_capture_environment(os.environ)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-cases", type=Path, required=True)
    private_args, remaining = parser.parse_known_args()

    from src.eval import run_cases

    sys.argv = [sys.argv[0], *remaining]
    args = run_cases._parse_args()
    if args.rollouts_per_case < 1:
        raise ValueError("experimental PSD group size must be positive")
    if not args.benchmark or not args.output_dir:
        raise ValueError("experimental PSD collection requires benchmark and output directory")

    train_rows = load_jsonl(private_args.train_cases)
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("experimental PSD collection requires a training-only split")
    public_rows = load_jsonl(Path(args.benchmark))
    public_ids = [row["case_id"] for row in public_rows]
    allowed_ids = {row["case_id"] for row in train_rows}
    if len(public_ids) != len(set(public_ids)) or not set(public_ids) <= allowed_ids:
        raise ValueError("experimental PSD benchmark is not covered by the training split")

    run_cases.VerificationWorkflow = PSDWorkflow
    summary = asyncio.run(run_cases._run_cases(args))

    manifest_path = Path(args.output_dir) / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["agent"]["psd_sampling"] = {
        "temperature": 0.7,
        "rollouts_per_case": args.rollouts_per_case,
        "capture_policy_tokens": True,
        "policy_topk": 20,
        "collector_sha256": sha256_file(Path(__file__)),
        "stages": ["UNIFIED_REACT", "UNIFIED_JUDGMENT"],
        "formal_psd_bank": False,
        "experiment_label": "first_step_exploration_1000x4",
    }
    manifest["experimental_scope"] = {
        "published_group_size": 8,
        "formal_training_allowed": False,
        "selection_by_answer": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
