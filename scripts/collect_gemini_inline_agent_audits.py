"""Collect Gemini inline Batch results for Agent private-gold audits.

The submitter journal is the source of truth.  This collector is restartable:
each completed Batch job is checkpointed independently, then all checkpoints
are merged into one audit ledger and summary per candidate source.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from google import genai


TERMINAL_FAILURE_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A concurrently appended final line can be observed briefly.
                if line_number == sum(1 for _ in path.open("r", encoding="utf-8")):
                    break
                raise
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    return rows


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = json.loads(raw_line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _write_gzip_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(pending, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    pending.replace(path)


def _state_name(value: Any) -> str:
    return str(getattr(value, "name", None) or value or "JOB_STATE_UNSPECIFIED")


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, Mapping):
        return None, "judge response is not a JSON object"
    return dict(parsed), None


def _response_text(response: Any) -> str:
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def _checkpoint_path(root: Path, journal_row: Mapping[str, Any]) -> Path:
    display_name = str(journal_row.get("display_name") or "batch")
    digest = hashlib.sha256(str(journal_row["batch_name"]).encode()).hexdigest()[:12]
    return root / "batch-checkpoints" / f"{display_name}-{digest}.jsonl.gz"


def _audit_record(
    *,
    source_name: str,
    candidate: Mapping[str, Any],
    item: Any,
    judge_model: str,
    thinking_level: str,
) -> dict[str, Any]:
    metadata = dict(getattr(item, "metadata", None) or {})
    case_id = str(metadata.get("case_id") or candidate.get("case_id") or "")
    gold = dict(candidate.get("gold") or {})
    answer = dict(candidate.get("candidate_answer") or {})
    record: dict[str, Any] = {
        "case_id": case_id,
        "request_key": metadata.get("key"),
        "source_name": source_name,
        "source_model": candidate.get("source_model"),
        "source_status": candidate.get("source_status"),
        "source_trace_path": candidate.get("source_trace_path"),
        "judge_model": judge_model,
        "thinking_level": thinking_level,
        "audit_mode": "agent_trace_evidence",
        "private_gold_auditable": gold.get("auditable") is True,
        "gold_verdict": gold.get("expected_verdict"),
        "candidate_verdict": answer.get("verdict"),
        "verdict_matches_gold": bool(
            gold.get("expected_verdict")
            and answer.get("verdict") == gold.get("expected_verdict")
        ),
    }
    item_error = getattr(item, "error", None)
    response = getattr(item, "response", None)
    if item_error is not None or response is None:
        record.update(
            {
                "status": "error",
                "error_type": "BatchItemError",
                "error": str(item_error or "missing response"),
            }
        )
        return record
    output_text = _response_text(response)
    parsed, parse_error = _parse_json_object(output_text)
    record.update(
        {
            "status": "completed",
            "judge_output": parsed,
            "judge_json_parse_error": parse_error,
            "judge_output_text": output_text if parse_error else None,
            "usage": (
                response.usage_metadata.model_dump(exclude_none=True)
                if getattr(response, "usage_metadata", None) is not None
                else None
            ),
        }
    )
    if parsed is not None:
        record.update(
            {
                "quality_bucket": parsed.get("quality_bucket"),
                "fact_alignment": parsed.get("fact_alignment"),
                "private_gold_fact_match": parsed.get("fact_alignment"),
                "reason_quality": parsed.get("reason_quality"),
                "failure_modes": parsed.get("failure_modes"),
                "explanation": parsed.get("explanation"),
            }
        )
    return record


def _collect_one(
    journal_row: Mapping[str, Any],
    *,
    api_key: str,
    checkpoint_root: Path,
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    judge_model: str,
    thinking_level: str,
) -> tuple[str, str, int]:
    source_name = str(journal_row["source_name"])
    checkpoint = _checkpoint_path(checkpoint_root, journal_row)
    if checkpoint.is_file():
        return source_name, "CHECKPOINTED", int(journal_row["sample_count"])
    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=str(journal_row["batch_name"]))
    state = _state_name(batch.state)
    if state != "JOB_STATE_SUCCEEDED":
        return source_name, state, 0
    destination = getattr(batch, "dest", None)
    items = list(getattr(destination, "inlined_responses", None) or [])
    if len(items) != int(journal_row["sample_count"]):
        raise ValueError(
            f"{journal_row['display_name']} returned {len(items)} items; "
            f"expected {journal_row['sample_count']}"
        )
    source_candidates = candidates[source_name]
    records: list[dict[str, Any]] = []
    for item in items:
        metadata = dict(getattr(item, "metadata", None) or {})
        case_id = str(metadata.get("case_id") or "")
        if case_id not in source_candidates:
            raise KeyError(f"unknown case_id in Batch response: {case_id}")
        records.append(
            _audit_record(
                source_name=source_name,
                candidate=source_candidates[case_id],
                item=item,
                judge_model=judge_model,
                thinking_level=thinking_level,
            )
        )
    _write_gzip_jsonl(checkpoint, records)
    return source_name, state, len(records)


def _category(row: Mapping[str, Any]) -> str | None:
    if row.get("private_gold_auditable") is not True:
        return None
    if row.get("status") != "completed" or not row.get("quality_bucket"):
        return None
    if row.get("verdict_matches_gold") is False:
        return "wrong_verdict"
    if row.get("verdict_matches_gold") is True:
        if row.get("reason_quality") == "decisive_and_grounded":
            return "correct_point_with_strong_evidence"
        return "correct_verdict_insufficient_evidence"
    return None


def _merge_source(
    source_name: str,
    journal_rows: list[Mapping[str, Any]],
    *,
    checkpoint_root: Path,
    output_root: Path,
    expected_cases: int,
    reported_denominator: int,
    judge_model: str,
    thinking_level: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for journal_row in journal_rows:
        checkpoint = _checkpoint_path(checkpoint_root, journal_row)
        records.extend(_read_gzip_jsonl(checkpoint))
    by_case = {str(row["case_id"]): row for row in records}
    if len(by_case) != expected_cases:
        raise ValueError(
            f"{source_name} has {len(by_case)} unique results; expected {expected_cases}"
        )
    ordered = sorted(by_case.values(), key=lambda row: str(row["request_key"]))
    source_dir = output_root / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    output_path = source_dir / "audit-results.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            category = _category(row)
            if category is not None:
                row["private_gold_category"] = category
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    categories = Counter(_category(row) or "uncategorized" for row in ordered)
    status_counts = Counter(str(row.get("status") or "unknown") for row in ordered)
    quality_counts = Counter(str(row.get("quality_bucket") or "unknown") for row in ordered)
    strict_count = categories["correct_point_with_strong_evidence"]
    summary = {
        "schema_version": "ifv-gemini-inline-agent-audit-collector-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "judge_model": judge_model,
        "thinking_level": thinking_level,
        "available_cases": expected_cases,
        "reported_denominator": reported_denominator,
        "counts": {
            "rows": len(ordered),
            "unique_cases": len(by_case),
            "status": dict(status_counts),
        },
        "quality_buckets": dict(quality_counts),
        "private_gold_categories": dict(categories),
        "strict_evidence_sufficient_count": strict_count,
        "sesr_available_percent": 100.0 * strict_count / expected_cases,
        "sesr_reported_percent": 100.0 * strict_count / reported_denominator,
        "audit_results": str(output_path),
    }
    _write_json(source_dir / "summary.json", summary)
    return summary


def _source_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use NAME=/path/to/candidates.jsonl.gz")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source name and path must be non-empty")
    return name.strip(), Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", action="append", type=Path, required=True)
    parser.add_argument("--source", action="append", type=_source_argument, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gemini-3.7-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--expected-cases", type=int, default=1526)
    parser.add_argument("--reported-denominator", type=int, default=1527)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    sources = dict(args.source)
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name, path in sources.items():
        rows = _read_gzip_jsonl(path)
        by_case = {str(row["case_id"]): row for row in rows}
        if len(by_case) != args.expected_cases:
            parser.error(
                f"{source_name} has {len(by_case)} unique candidates; "
                f"expected {args.expected_cases}"
            )
        candidates[source_name] = by_case
    output_root = args.output_dir.expanduser().resolve()
    checkpoint_root = output_root / "collector-state"
    output_root.mkdir(parents=True, exist_ok=True)

    while True:
        journal_rows: list[dict[str, Any]] = []
        for raw_path in args.journal:
            path = raw_path.expanduser().resolve()
            if path.is_file():
                journal_rows.extend(_read_jsonl(path))
        unique_jobs = {str(row["batch_name"]): row for row in journal_rows}
        unknown_sources = {
            str(row["source_name"]) for row in unique_jobs.values()
        } - set(sources)
        if unknown_sources:
            raise ValueError(f"journals contain unknown sources: {sorted(unknown_sources)}")

        unresolved = [
            row
            for row in unique_jobs.values()
            if not _checkpoint_path(checkpoint_root, row).is_file()
        ]
        state_counts: Counter[str] = Counter()
        errors: list[str] = []
        if unresolved:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = [
                    executor.submit(
                        _collect_one,
                        row,
                        api_key=api_key,
                        checkpoint_root=checkpoint_root,
                        candidates=candidates,
                        judge_model=args.judge_model,
                        thinking_level=args.thinking_level,
                    )
                    for row in unresolved
                ]
                for future in as_completed(futures):
                    try:
                        _, state, count = future.result()
                        state_counts[state] += count or 1
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
        journal_cases: dict[str, set[str]] = {name: set() for name in sources}
        checkpoint_jobs = 0
        for row in unique_jobs.values():
            source_name = str(row["source_name"])
            journal_cases[source_name].update(str(x) for x in row.get("case_ids") or [])
            if _checkpoint_path(checkpoint_root, row).is_file():
                checkpoint_jobs += 1
        progress = {
            "time": datetime.now(timezone.utc).isoformat(),
            "journal_jobs": len(unique_jobs),
            "checkpoint_jobs": checkpoint_jobs,
            "journal_case_counts": {
                name: len(case_ids) for name, case_ids in journal_cases.items()
            },
            "observed_states": dict(state_counts),
            "errors": errors[:20],
        }
        _write_json(output_root / "collector-progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        complete = (
            not errors
            and checkpoint_jobs == len(unique_jobs)
            and all(len(case_ids) == args.expected_cases for case_ids in journal_cases.values())
        )
        if complete:
            summaries = {}
            for source_name in sources:
                source_rows = [
                    row
                    for row in unique_jobs.values()
                    if row["source_name"] == source_name
                ]
                summaries[source_name] = _merge_source(
                    source_name,
                    source_rows,
                    checkpoint_root=checkpoint_root,
                    output_root=output_root,
                    expected_cases=args.expected_cases,
                    reported_denominator=args.reported_denominator,
                    judge_model=args.judge_model,
                    thinking_level=args.thinking_level,
                )
            _write_json(output_root / "summary.json", {"sources": summaries})
            return 0
        if not args.wait:
            return 2 if errors else 1
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
