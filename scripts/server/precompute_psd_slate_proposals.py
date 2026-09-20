"""Precompute first-round Gemini slate proposals without starting Qwen.

The later live repair search consumes the exact same per-case ``judge-cache``.
Only compact receipts and an offset index are added; source traces are read once
for cases that have not already committed a receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, path)


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "inode": value.st_ino,
        "device": value.st_dev,
    }


def build_offset_index(selected: Path, index_path: Path) -> list[dict[str, Any]]:
    """Index a JSONL once; later resumes never linearly scan it again."""
    selected = selected.resolve()
    identity = _stat_identity(selected)
    if index_path.exists():
        saved = _load_json(index_path)
        if (saved.get("schema_version") != "ifv-psd-slate-precompute-index-v1"
                or saved.get("selected_path") != str(selected)
                or saved.get("selected_stat") != identity):
            raise ValueError("selected candidate index binding changed")
        return list(saved["entries"])
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    with selected.open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            case_id = str(row.get("case_id") or "").strip()
            if not case_id or case_id in seen:
                raise ValueError("selected candidates require unique case IDs")
            seen.add(case_id)
            entries.append({"case_id": case_id, "offset": offset, "length": len(raw)})
            offset += len(raw)
    if offset != identity["size"]:
        raise RuntimeError("selected candidate file changed while indexing")
    _atomic_json(index_path, {
        "schema_version": "ifv-psd-slate-precompute-index-v1",
        "selected_path": str(selected),
        "selected_stat": identity,
        "entries": entries,
    })
    return entries


def read_indexed_row(selected: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    with selected.open("rb") as stream:
        stream.seek(int(entry["offset"]))
        raw = stream.read(int(entry["length"]))
    row = json.loads(raw)
    if row.get("case_id") != entry.get("case_id"):
        raise ValueError("indexed selected candidate changed")
    return row


def _jsonl_by_case(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case_id = str(row.get("case_id") or "").strip()
            if not case_id or case_id in result:
                raise ValueError(f"invalid or duplicate case in {path}")
            result[case_id] = row
    return result


def pending_entries(
    entries: list[dict[str, Any]], output: Path, max_new_cases: int | None
) -> tuple[list[dict[str, Any]], int, int]:
    pending = [entry for entry in entries
               if not (output / "cases" / entry["case_id"] / "proposal.json").exists()]
    pending_before_limit = len(pending)
    completed_before_run = len(entries) - pending_before_limit
    if max_new_cases is not None:
        pending = pending[:max_new_cases]
    return pending, pending_before_limit, completed_before_run


async def _prepare_case(
    *,
    entry: Mapping[str, Any],
    selected: Path,
    traces_root: Path,
    benchmark_root: Path,
    benchmark: Mapping[str, Mapping[str, Any]],
    private_gold: Mapping[str, Mapping[str, Any]],
    output: Path,
    model: str,
    proposal_budget: int,
    client: Any,
) -> dict[str, Any]:
    from ifv_training.psd_gemini_judge import review_images, trace_steps
    from ifv_training.psd_slate import decision_map, propose_slate, SlateProposalRejected
    from ifv_training.psd_slate_feedback import checker_feedback, diagnostic_position
    from ifv_training.psd_source_review import source_review_reference

    case_id = str(entry["case_id"])
    case_root = output / "cases" / case_id
    receipt = case_root / "proposal.json"
    if receipt.exists():
        saved = _load_json(receipt)
        if saved.get("case_id") != case_id or saved.get("model") != model:
            raise ValueError("saved proposal receipt binding changed")
        return saved

    candidate = read_indexed_row(selected, entry)
    source = candidate.get("source") or {}
    trace_path = Path(str(source.get("source_trace_path") or ""))
    if not trace_path.is_absolute():
        trace_path = traces_root / trace_path
    trace = await asyncio.to_thread(_load_json, trace_path)
    review = await asyncio.to_thread(source_review_reference, source)
    if review is None:
        raise ValueError("selected repair candidate lacks source review")
    case = benchmark.get(case_id)
    gold = private_gold.get(case_id)
    if case is None or gold is None:
        raise ValueError("selected case is missing benchmark or private reference")
    image = Path(str(case.get("image_path") or ""))
    if not image.is_absolute():
        image = benchmark_root / image

    feedback = checker_feedback(review, trace, withhold_invalid_citations=True)
    failed_position = diagnostic_position(feedback, trace)
    images, _ = await asyncio.to_thread(
        review_images, {"source": trace}, image_path=image
    )
    public = {
        "source_steps": trace_steps(trace),
        "observed_steps": trace_steps(trace),
        "decision_map": decision_map(trace),
        "checker_feedback": feedback,
    }
    rejected = 0
    while rejected < proposal_budget:
        try:
            hints, provenance = await propose_slate(
                client,
                public_context=public,
                previous={},
                passing_positions=[],
                failed_position=failed_position,
                model=model,
                cache_dir=case_root / "judge-cache",
                private_context=gold,
                images=images,
                proposal_feedback=(
                    {"rejected_proposals": rejected,
                     "reason": "invalid_or_nonprocedural_slate"}
                    if rejected else None
                ),
            )
            result = {
                "schema_version": "ifv-psd-slate-first-proposal-v1",
                "case_id": case_id,
                "candidate_id": candidate["candidate_id"],
                "episode_id": candidate["episode_id"],
                "model": model,
                "failed_position": failed_position,
                "hints": {str(position): hint.text for position, hint in hints.items()},
                "rejected_proposals": rejected,
                "request_binding": provenance.get("request_binding", {}),
                "response_sha256": provenance.get("response_sha256"),
                "status": "ready_for_qwen",
            }
            _atomic_json(receipt, result)
            (output / "errors" / f"{case_id}.json").unlink(missing_ok=True)
            return result
        except SlateProposalRejected:
            rejected += 1
    result = {
        "schema_version": "ifv-psd-slate-first-proposal-v1",
        "case_id": case_id,
        "candidate_id": candidate["candidate_id"],
        "episode_id": candidate["episode_id"],
        "model": model,
        "failed_position": failed_position,
        "hints": {},
        "rejected_proposals": rejected,
        "status": "proposal_budget_exhausted",
    }
    _atomic_json(receipt, result)
    (output / "errors" / f"{case_id}.json").unlink(missing_ok=True)
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.integrations.gemini import GeminiInteractionsClient

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = build_offset_index(args.selected, output / "selected-offset-index.json")
    benchmark = _jsonl_by_case(args.benchmark)
    private_gold = _jsonl_by_case(args.private_gold)
    missing, pending_before_limit, completed_before_run = pending_entries(
        entries, output, args.max_new_cases
    )
    state = {
        "schema_version": "ifv-psd-slate-precompute-state-v1",
        "phase": "running",
        "selected_cases": len(entries),
        "completed_cases": completed_before_run,
        "model": args.model,
        "gpu_required": False,
        "selected_file_rescans_after_index": 0,
    }
    _atomic_json(output / "state.json", state)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = state["completed_cases"]
    failed = 0

    async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
        async def one(entry: Mapping[str, Any]) -> None:
            nonlocal completed, failed
            async with semaphore:
                try:
                    await _prepare_case(
                        entry=entry,
                        selected=args.selected,
                        traces_root=args.traces_root,
                        benchmark_root=args.benchmark_root,
                        benchmark=benchmark,
                        private_gold=private_gold,
                        output=output,
                        model=args.model,
                        proposal_budget=args.proposal_budget,
                        client=client,
                    )
                    completed += 1
                except Exception as error:
                    failed += 1
                    case_id = str(entry["case_id"])
                    _atomic_json(output / "errors" / f"{case_id}.json", {
                        "case_id": case_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                _atomic_json(output / "state.json", {
                    **state,
                    "completed_cases": completed,
                    "failed_cases": failed,
                })

        await asyncio.gather(*(one(entry) for entry in missing))

    remaining = pending_before_limit - (completed - state["completed_cases"])
    final = {
        **state,
        "phase": (
            "partial_smoke"
            if remaining > 0 and failed == 0
            else "complete" if failed == 0
            else "complete_with_errors"
        ),
        "completed_cases": completed,
        "failed_cases": failed,
        "remaining_cases": remaining,
    }
    _atomic_json(output / "state.json", final)
    return final


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--selected", type=Path, required=True)
    result.add_argument("--traces-root", type=Path, required=True)
    result.add_argument("--benchmark", type=Path, required=True)
    result.add_argument("--benchmark-root", type=Path, required=True)
    result.add_argument("--private-gold", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", default="gemini-3.6-flash")
    result.add_argument("--concurrency", type=int, default=8)
    result.add_argument("--proposal-budget", type=int, default=12)
    result.add_argument("--max-new-cases", type=int)
    return result


def main() -> None:
    args = parser().parse_args()
    if not 1 <= args.concurrency <= 32:
        raise ValueError("concurrency must be between 1 and 32")
    if not 1 <= args.proposal_budget <= 64:
        raise ValueError("proposal budget must be between 1 and 64")
    if args.max_new_cases is not None and args.max_new_cases < 1:
        raise ValueError("max new cases must be positive")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
