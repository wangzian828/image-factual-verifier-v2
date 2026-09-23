"""Freeze the old-1000 preservation source without copying its large payload.

This is a CPU-only, one-pass index of the already reviewed candidate bank.  It
does not select a smaller training subset, call a model, or score teacher top-k.
The source JSONL remains the sole canonical payload; offsets are bound to its
inode and timestamps so later stages can seek only the chosen full episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics


def identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def freeze_preservation(
    *, candidates: Path, reviews: Path, candidate_manifest: Path, output: Path,
) -> dict:
    candidates, reviews, candidate_manifest = (
        candidates.resolve(), reviews.resolve(), candidate_manifest.resolve())
    if output.exists():
        raise FileExistsError(output)
    source_ids = {str(p): identity(p) for p in (candidates, reviews, candidate_manifest)}
    review_doc = json.loads(reviews.read_text(encoding="utf-8"))
    candidate_doc = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if review_doc.get("status") != "source_reviews_complete_with_unadopted":
        raise ValueError("source reviews are not complete")
    if review_doc.get("case_errors") != 0 or review_doc.get("completed_cases") not in (None, review_doc.get("selected_cases")):
        raise ValueError("source reviews have incomplete/error cases")
    if candidate_doc.get("status") != "ready_for_privileged_localization":
        raise ValueError("candidate bank is not frozen")
    expected: dict[str, tuple[str, str]] = {}
    seen_reviews: set[str] = set()
    review_count = Counter()
    cases = review_doc.get("cases")
    if not isinstance(cases, list) or len(cases) != review_doc.get("selected_cases"):
        raise ValueError("source review case count changed")
    for case in cases:
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid or cid in seen_reviews:
            raise ValueError("source review case ID missing or duplicated")
        seen_reviews.add(cid)
        passes = []
        for review in case.get("reviews", []):
            status = review.get("status")
            review_count[status] += 1
            if status == "pass":
                passes.append(review)
        if len(passes) > 1:
            raise ValueError(f"multiple verified successes for {cid}")
        if passes:
            first = passes[0]
            expected[cid] = (first["episode_id"], first["path"])
    if dict(review_count) != review_doc.get("review_counts"):
        raise ValueError("source review totals changed")
    if len(expected) != candidate_doc.get("counts", {}).get("preservation_candidates"):
        raise ValueError("review passes and preservation candidates differ")

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    seen: set[str] = set()
    seen_candidates: set[str] = set()
    stage_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    lengths: list[int] = []
    prompt_tokens = completion_tokens = 0
    with candidates.open("rb") as source, (output / "episodes.jsonl").open("x", encoding="utf-8") as index:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            row = json.loads(line)
            cid = row.get("case_id")
            if cid not in expected or cid in seen:
                raise ValueError(f"unexpected or duplicate preservation case: {cid}")
            seen.add(cid)
            candidate_id = row.get("candidate_id")
            if (not isinstance(candidate_id, str) or not candidate_id
                    or candidate_id in seen_candidates):
                raise ValueError(f"missing or duplicate candidate ID for {cid}")
            seen_candidates.add(candidate_id)
            episode_id, review_path = expected[cid]
            if row.get("episode_id") != episode_id:
                raise ValueError(f"first verified episode differs for {cid}")
            if row.get("source", {}).get("source_task_review", {}).get("path") != review_path:
                raise ValueError(f"review binding differs for {cid}")
            if (row.get("class") != "base_pass_preserve"
                    or row.get("candidate_status") != "ready_for_teacher_collection"
                    or row.get("verified_full_task") is not True
                    or row.get("strict_trace_audit_pass") is not True):
                raise ValueError(f"preservation admission failed for {cid}")
            steps = row.get("preservation_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"empty preservation episode for {cid}")
            step_ids: list[str] = []
            tools: list[str] = []
            max_context = 0
            episode_completion = 0
            for step in steps:
                sid = step.get("step_id")
                if not isinstance(sid, str) or not sid or sid in step_ids:
                    raise ValueError(f"missing or duplicate preservation step for {cid}")
                step_ids.append(sid)
                if step.get("protocol_rejected"):
                    raise ValueError(f"protocol-rejected preservation step for {cid}")
                capture = step.get("rollout_token_capture") or {}
                prompt = capture.get("prompt_token_ids")
                completion = capture.get("completion_token_ids")
                if (capture.get("status") != "complete" or not isinstance(prompt, list)
                        or not prompt or not isinstance(completion, list) or not completion):
                    raise ValueError(f"incomplete token capture for {cid}")
                prompt_tokens += len(prompt)
                completion_tokens += len(completion)
                episode_completion += len(completion)
                max_context = max(max_context, len(prompt))
                stage_counts[str(step.get("stage") or "unknown")] += 1
                action_counts[str(step.get("action_type") or "unknown")] += 1
                action = step.get("observed_policy_action") or {}
                tool = action.get("name") if isinstance(action, dict) else None
                if isinstance(tool, str) and tool:
                    tools.append(tool)
                    tool_counts[tool] += 1
            if sum(":judgment:" in sid for sid in step_ids) != 1:
                raise ValueError(f"preservation episode needs exactly one judgment for {cid}")
            lengths.append(len(steps))
            record = {
                "case_id": cid, "episode_id": episode_id,
                "candidate_id": candidate_id,
                "source_offset": offset, "source_length": len(line),
                "source_review_path": review_path,
                "source_trace_path": row.get("source", {}).get("source_trace_path"),
                "step_ids": step_ids, "step_count": len(steps),
                "tool_sequence": tools, "max_prompt_tokens": max_context,
                "completion_tokens": episode_completion,
            }
            index.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if seen != set(expected):
        raise ValueError(f"preservation cases missing from source: {len(set(expected) - seen)}")
    if any(identity(p) != source_ids[str(p)] for p in (candidates, reviews, candidate_manifest)):
        raise ValueError("a preservation input changed during indexing")
    if sum(lengths) != sum(stage_counts.values()):
        raise AssertionError("step accounting mismatch")
    manifest = {
        "schema_version": "ifv-old1000-fixed-preservation-index-v1",
        "status": "preservation_source_frozen_repair_pending",
        "selection_policy": "first_verified_success_per_case_complete_episode",
        "source_identities": source_ids,
        "index_identity": identity(output / "episodes.jsonl"),
        "counts": {
            "reviewed_cases": len(cases), "review_passes": review_count["pass"],
            "review_failures": review_count["fail"],
            "preservation_cases": len(seen), "preservation_episodes": len(lengths),
            "preservation_steps": sum(lengths),
            "prompt_token_ids": prompt_tokens, "completion_token_ids": completion_tokens,
            "min_steps_per_episode": min(lengths),
            "median_steps_per_episode": statistics.median(lengths),
            "max_steps_per_episode": max(lengths),
        },
        "stage_counts": dict(stage_counts), "action_counts": dict(action_counts),
        "tool_counts": dict(tool_counts),
        "raw_payload_copies": 0, "raw_file_hashes": 0, "provider_calls": 0,
        "teacher_scoring_complete": False, "train_ready": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    result = freeze_preservation(candidates=args.candidates, reviews=args.reviews,
                                 candidate_manifest=args.candidate_manifest, output=args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
