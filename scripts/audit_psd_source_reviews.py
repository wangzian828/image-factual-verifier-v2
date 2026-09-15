"""Audit saved real source reviews and deterministic/semantic routing, no API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json
from ifv_training.psd_source_review import source_review_reference, validate_source_review
from ifv_training.psd_repair_verifier import verify_source_rollout_failure
from ifv_training.psd_repair_storage import load_bound
from src.eval.postprocess_run import _trace_path
from src.trajectory.scoring import score_process_trace


def audit(*, run_dir, private_gold, reviews, output):
    marker = load_json(reviews / "inputs.json")
    load_bound(reviews / "inputs.json", identity=marker["identity"])
    for name, digest in marker["identity"]["inputs"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("review inputs changed")
    if str(private_gold.resolve()) not in marker["identity"]["inputs"]:
        raise ValueError("gold file differs from source review")
    results = {str(r.get("episode_id") or r["case_id"]): r for r in load_jsonl(run_dir / "run_results.jsonl")}
    gold = {r["case_id"]: r for r in load_jsonl(private_gold)}
    summary = load_json(reviews / "summary.json")
    rows = []
    for review in summary["reviews"]:
        if review["status"] == "pending_error":
            rows.append({"case_id": review["case_id"], "status": "pending_error"})
            continue
        trace = load_json(_trace_path(run_dir, results[review["episode_id"]]))
        artifact = source_review_reference({"source_task_review": review})
        status = validate_source_review(artifact, trace=trace, gold=gold[review["case_id"]])
        if status != review["status"]:
            raise ValueError("review status changed")
        metrics, _ = score_process_trace(trace, gold[review["case_id"]])
        repair = verify_source_rollout_failure(trace, gold=gold[review["case_id"]], source_task_review=artifact)
        if status == "fail" and not repair["passed"]:
            raise ValueError("semantic failure did not reach causal repair gate")
        rows.append({"case_id": review["case_id"], "status": status,
            "classification_correct": metrics.get("result_correct"), "repair_source_verified": repair["passed"],
            "source_semantic_failure": repair["semantic_failure"],
            "images_in_review": len({digest for row in artifact["media"]["policy_images"] for digest in row["image_sha256"]}),
            "explanation": artifact["decision"]["explanation"]})
    report = {"passed": not summary["pending"], "cases": rows,
        "classification_correct": sum(r.get("classification_correct") is True for r in rows),
        "source_status_counts": dict(Counter(r["status"] for r in rows)),
        "correct_label_semantic_failures": sum(r.get("classification_correct") is True and r["status"] == "fail" for r in rows),
        "source_run_unchanged": True, "new_provider_calls": 0, "training_started": False,
        "interpretation": "Small diagnostic of routing and judge behavior, not measured judge accuracy or PSD capability gain",
        "source_summary_sha256": sha256_file(reviews / "summary.json")}
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "private-gold", "reviews", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False, indent=2))
