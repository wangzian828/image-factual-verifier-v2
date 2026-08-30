"""Audit completed Agent traces against private construction gold.

Unlike direct QA, this audit sees the persisted final report plus the actual
Evidence/basis selected by the Agent.  It never feeds private gold back into a
rollout and it never treats an Agent prose claim as evidence by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from src.eval.agent_private_gold import build_agent_private_gold_candidate
from src.eval.private_gold_metrics import (
    annotate_private_gold_category,
    private_gold_audit_summary,
    private_gold_category_counts,
)
from src.integrations.gemini import GeminiInteractionsClient, extract_text
from scripts.audit_direct_qa_baseline import (
    _build_private_gold_rows,
    _private_gold,
    _read_jsonl,
    _write_json,
)
from scripts.run_direct_qa_baseline import (
    _image_input,
    _parse_json_object,
    _resolve_image_path,
    extract_native_thought,
)


AGENT_PRIVATE_GOLD_JUDGE_PROMPT = """You are a frozen private-gold auditor for an image factual-verification benchmark.

You receive an image, evaluator-private construction gold, and a completed Agent trace packet. The private gold is authoritative for the intended image-level fact and expected binary verdict. The Agent report is untrusted prose. Only the supplied successful Evidence records and verdict basis are actual trace evidence; do not credit a report sentence merely because it says a source proved something.

First assess whether the Agent reached the same complete image-level fact. A compatible sub-fact is acceptable only when it decisively settles the same real/fake verdict. Do not accept an easier, broader fact that drops a decisive identity, event, relationship, time, place, number, or other condition.

Then assess the evidence chain actually selected by the trace:
- decisive: selected successful Evidence directly establishes the necessary relation and supports the recorded verdict;
- partial: it is relevant and useful but leaves a material part of the target open;
- not_grounded: no selected successful Evidence supports the factual conclusion;
- contradictory: selected Evidence conflicts with the recorded conclusion.

Assess the reader-facing report separately. It is faithful only when it accurately summarizes the supplied target, basis, and Evidence without inventing sources, observations, or stronger claims. A missing report is allowed only for historical traces and must be marked missing; it cannot by itself make an otherwise grounded historical trace incorrect.

Set quality_bucket:
- strong: correct verdict, same_fact or compatible_subfact, decisive trace evidence, and a faithful report;
- usable: correct verdict and aligned fact, with partial evidence or a report that is incomplete but not invented;
- rejected: wrong/different fact, no factual grounding, contradiction, artifact-only reasoning, invented evidence/source, or major overclaiming;
- not_auditable: private gold or source trace is unavailable.

Do not treat apparent AI generation, editing artifacts, image quality, distorted anatomy, search-result titles, generic lack of results, or an unsupported URL as factual proof. Return only JSON."""


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or "").strip()


def _trace_path(run_dir: Path, row: Mapping[str, Any]) -> Path:
    raw = str(row.get("trace_path") or "").strip()
    if not raw:
        raise ValueError("run result does not name a trace_path")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (run_dir / path).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError:
        raise ValueError(f"trace_path escapes run directory: {raw}") from None
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return dict(value)


async def _audit_one(
    result: Mapping[str, Any],
    gold_row: Mapping[str, Any] | None,
    *,
    run_dir: Path,
    manifest_root: Path,
    client: GeminiInteractionsClient,
    judge_model: str,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = _case_id(result)
    started = time.monotonic()
    audit: dict[str, Any] = {
        "case_id": case_id,
        "source_status": result.get("status"),
        "source_model": result.get("model"),
        "judge_model": judge_model,
        "audit_mode": "agent_trace_evidence",
        "status": "error",
    }
    try:
        if gold_row is None:
            raise ValueError("private gold row is missing")
        if str(result.get("status") or "") != "success":
            raise ValueError("Agent source result is not successful")
        trace_path = _trace_path(run_dir, result)
        trace = _read_json(trace_path)
        gold = _private_gold(gold_row)
        candidate = build_agent_private_gold_candidate(trace)
        audit.update(
            {
                "source_trace_path": str(trace_path),
                "gold_verdict": gold.get("expected_verdict"),
                "candidate_verdict": candidate.get("recorded_verdict"),
                "verdict_matches_gold": bool(
                    gold.get("expected_verdict")
                    and candidate.get("recorded_verdict")
                    == gold.get("expected_verdict")
                ),
                "candidate_output": candidate,
                "private_gold_auditable": gold.get("auditable") is True,
                "private_gold": gold,
            }
        )
        if not gold.get("auditable"):
            audit.update(
                {
                    "status": "not_auditable",
                    "quality_bucket": "not_auditable",
                    "private_gold_fact_match": "not_auditable",
                    "trace_evidence_grounded": "not_auditable",
                    "report_grounding": "not_auditable",
                    "explanation": "private gold lacks a usable factual target",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            return audit

        image_path = _resolve_image_path(
            {"image_path": trace.get("image_path")},
            manifest_root,
        )
        payload_text = json.dumps(
            {
                "private_gold": gold,
                "agent_trace_packet": candidate,
            },
            ensure_ascii=False,
            indent=2,
        )
        request_input: list[dict[str, Any]] = [
            _image_input(image_path),
            {
                "type": "text",
                "text": AGENT_PRIVATE_GOLD_JUDGE_PROMPT
                + "\n\nAUDIT INPUT:\n"
                + payload_text,
            },
        ]
        async with semaphore:
            payload = await client.create(
                model=judge_model,
                input=request_input,
                response_format=response_format,
                generation_config=generation_config,
                store=True,
            )
        output_text = extract_text(payload)
        parsed, parse_error = _parse_json_object(output_text)
        native_thought = extract_native_thought(payload)
        if parsed is not None:
            audit.update(
                {
                    "quality_bucket": parsed.get("quality_bucket"),
                    "private_gold_fact_match": parsed.get(
                        "private_gold_fact_match"
                    ),
                    # Keep the old display key available to generic summaries.
                    "fact_alignment": parsed.get("private_gold_fact_match"),
                    "trace_evidence_grounded": parsed.get(
                        "trace_evidence_grounded"
                    ),
                    "report_grounding": parsed.get("report_grounding"),
                    "failure_modes": parsed.get("failure_modes"),
                    "explanation": parsed.get("explanation"),
                }
            )
        audit.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "interaction_id": payload.get("id"),
                "interaction_status": payload.get("status"),
                "judge_output_text": output_text,
                "judge_output": parsed,
                "judge_json_parse_error": parse_error,
                "judge_native_thought": native_thought,
                "judge_native_thought_chars": len(native_thought),
                "judge_native_thought_tokens": int(
                    (payload.get("usage") or {}).get(
                        "total_thought_tokens",
                        (payload.get("usage") or {}).get("thought_tokens", 0),
                    )
                    or 0
                ),
                "usage": payload.get("usage"),
            }
        )
    except Exception as exc:
        audit.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return audit


async def _run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    archive_root = (
        Path(args.archive_root).expanduser().resolve()
        if args.archive_root
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "run_results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    manifest_rows = _read_jsonl(manifest)
    private_gold_rows = _build_private_gold_rows(
        manifest_rows,
        archive_root=archive_root,
        manifest_root=manifest.parent,
    )
    gold_by_case = {_case_id(row): row for row in private_gold_rows}
    results = _read_jsonl(results_path)
    if args.limit > 0:
        results = results[: args.limit]
    if args.case_id:
        selected = set(args.case_id)
        results = [row for row in results if _case_id(row) in selected]

    schema = {
        "type": "object",
        "properties": {
            "quality_bucket": {
                "type": "string",
                "enum": ["strong", "usable", "rejected", "not_auditable"],
            },
            "private_gold_fact_match": {
                "type": "string",
                "enum": [
                    "same_fact",
                    "compatible_subfact",
                    "overgeneralized_subfact",
                    "different_fact",
                    "unclear",
                    "not_auditable",
                ],
            },
            "trace_evidence_grounded": {
                "type": "string",
                "enum": [
                    "decisive",
                    "partial",
                    "not_grounded",
                    "contradictory",
                    "not_auditable",
                ],
            },
            "report_grounding": {
                "type": "string",
                "enum": [
                    "faithful",
                    "partially_faithful",
                    "overclaimed_or_invented",
                    "missing",
                    "not_auditable",
                ],
            },
            "failure_modes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
            },
            "explanation": {"type": "string"},
        },
        "required": [
            "quality_bucket",
            "private_gold_fact_match",
            "trace_evidence_grounded",
            "report_grounding",
            "failure_modes",
            "explanation",
        ],
        "additionalProperties": False,
    }
    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }
    generation_config = {
        "max_output_tokens": args.max_output_tokens,
        "thinking_level": args.thinking_level,
        "thinking_summaries": "auto",
    }
    (output_dir / "prompt.txt").write_text(
        AGENT_PRIVATE_GOLD_JUDGE_PROMPT + "\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "run-config.json",
        {
            "schema_version": "ifv-agent-private-gold-audit-config-v1",
            "run_dir": str(run_dir),
            "results_path": str(results_path),
            "manifest": str(manifest),
            "archive_root": str(archive_root) if archive_root else None,
            "judge_model": args.judge_model,
            "thinking_level": args.thinking_level,
            "max_output_tokens": args.max_output_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "concurrency": args.concurrency,
            "selected_results": len(results),
            "judge_input": [
                "image",
                "private_construction_gold",
                "agent_fact_check_report",
                "actual_trace_verdict_basis",
                "actual_successful_evidence",
                "actual_selected_evidence",
            ],
        },
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    output_path = output_dir / "audit-results.jsonl"
    records: list[dict[str, Any]] = []
    async with GeminiInteractionsClient(
        timeout=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        tasks = [
            asyncio.create_task(
                _audit_one(
                    result,
                    gold_by_case.get(_case_id(result)),
                    run_dir=run_dir,
                    manifest_root=manifest.parent,
                    client=client,
                    judge_model=args.judge_model,
                    response_format=response_format,
                    generation_config=generation_config,
                    semaphore=semaphore,
                )
            )
            for result in results
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                record = annotate_private_gold_category(await task)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "case_id": record.get("case_id"),
                            "status": record.get("status"),
                            "quality_bucket": record.get("quality_bucket"),
                            "private_gold_fact_match": record.get(
                                "private_gold_fact_match"
                            ),
                            "trace_evidence_grounded": record.get(
                                "trace_evidence_grounded"
                            ),
                            "report_grounding": record.get("report_grounding"),
                            "verdict_matches_gold": record.get(
                                "verdict_matches_gold"
                            ),
                            "error_type": record.get("error_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    completed = [row for row in records if row.get("status") == "completed"]
    summary = {
        "schema_version": "ifv-agent-private-gold-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "archive_root": str(archive_root) if archive_root else None,
        "judge_model": args.judge_model,
        "thinking_level": args.thinking_level,
        "concurrency": args.concurrency,
        "counts": {
            "selected": len(results),
            "completed": len(completed),
            "errors": len(records) - len(completed),
            "private_gold_auditable": sum(
                row.get("private_gold_auditable") is True for row in records
            ),
            "verdict_matches_gold": sum(
                row.get("verdict_matches_gold") is True for row in records
            ),
            "judge_thought_present": sum(
                row.get("judge_native_thought_chars", 0) > 0
                for row in completed
            ),
        },
        "quality_buckets": dict(
            Counter(str(row.get("quality_bucket") or "unknown") for row in completed)
        ),
        "fact_match": dict(
            Counter(
                str(row.get("private_gold_fact_match") or "unknown")
                for row in completed
            )
        ),
        "trace_evidence_grounded": dict(
            Counter(
                str(row.get("trace_evidence_grounded") or "unknown")
                for row in completed
            )
        ),
        "report_grounding": dict(
            Counter(
                str(row.get("report_grounding") or "unknown")
                for row in completed
            )
        ),
        "private_gold_categories": private_gold_category_counts(records),
        "private_gold_audit": private_gold_audit_summary(records),
        "private_gold_category_definition": {
            "correct_point_with_strong_evidence": (
                "verdict_matches_gold=true and quality_bucket=strong; for Agent "
                "this requires decisive selected trace Evidence and a faithful "
                "fact-check report"
            ),
            "correct_verdict_insufficient_evidence": (
                "verdict_matches_gold=true and quality_bucket!=strong"
            ),
            "wrong_verdict": "verdict_matches_gold=false",
        },
        "audit_results": str(output_path),
        "prompt_file": str(output_dir / "prompt.txt"),
        "config_file": str(output_dir / "run-config.json"),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit completed Agent traces with private construction gold."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
