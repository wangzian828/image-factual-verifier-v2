"""Audit an initial streaming judge plus explicitly independent supplements.

This never changes the original case receipts. An ambiguous paid request cannot
be called a provider failure or silently replayed. Its separately submitted
judgment is labelled as a supplement in every consolidated row and summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.server.stream_agent_judges import case_token, compact_record, terminal_judge
from scripts.summarize_agent_test_evaluation import _metric_payload
from src.eval.private_gold_metrics import agent_private_gold_category_counts


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_once(path: Path, data: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError(f"frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def _same_stat(path: Path, frozen: dict[str, Any]) -> bool:
    stat = path.stat()
    return (
        str(path.resolve()) == frozen.get("path")
        and stat.st_size == frozen.get("size")
        and stat.st_mtime_ns == frozen.get("mtime_ns")
    )


def consolidate(
    *,
    rollout_root: Path,
    benchmark: Path,
    original_judge: Path,
    supplement: Path,
    output: Path,
    expected_count: int,
    formal_denominator: int,
) -> dict[str, Any]:
    expected = [str(row["case_id"]) for row in _jsonl(benchmark)]
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise ValueError("frozen benchmark count/identity mismatch")
    inference = _json(rollout_root / "inference-summary.json")
    if inference.get("phase") != "inference_complete" or inference.get("success") != expected_count:
        raise ValueError("Agent inference has not fully succeeded")
    original_binding = _json(original_judge / "binding.json")
    if original_binding.get("rollout_root") != str(rollout_root.resolve()):
        raise ValueError("original judge source changed")
    if original_binding.get("benchmark", {}).get("path") != str(benchmark.resolve()):
        raise ValueError("original judge benchmark changed")
    if original_binding.get("formal_denominator") != formal_denominator:
        raise ValueError("original judge denominator changed")
    plan = _json(supplement / "plan.json")
    bindings = plan.get("bindings")
    if (
        plan.get("protocol_relation") not in (None, "independent_supplement_not_idempotent_replay")
        or plan.get("original_judge") != str(original_judge.resolve())
        or plan.get("source") != str(rollout_root.resolve())
        or not isinstance(bindings, list)
    ):
        raise ValueError("supplement is not bound to this original judge")
    supplements: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(bindings, 1):
        case_id = str(binding.get("case_id") or "")
        if case_id not in expected or case_id in supplements:
            raise ValueError("duplicate or unknown supplement case")
        if (
            binding.get("protocol_relation") != "independent_supplement_not_idempotent_replay"
            or binding.get("judge_model") != original_binding.get("judge_model")
            or binding.get("thinking_level") != original_binding.get("thinking_level")
            or binding.get("max_output_tokens") != original_binding.get("max_output_tokens")
        ):
            raise ValueError("supplement judge protocol changed")
        original_case = original_judge / "cases" / case_token(expected.index(case_id) + 1, case_id)
        ambiguous = original_case / "ambiguous.json"
        if (
            not ambiguous.is_file()
            or not (original_case / "attempt-01.intent.json").is_file()
            or (original_case / "attempt-01.result.json").exists()
            or (original_case / "result.json").exists()
            or not _same_stat(ambiguous, binding["original_ambiguous"])
            or _json(ambiguous).get("case_id") != case_id
        ):
            raise ValueError("original ambiguity was resolved or changed")
        trace = Path(binding["source_trace"]["path"])
        if not trace.is_file() or not _same_stat(trace, binding["source_trace"]):
            raise ValueError("supplement source trace changed")
        directory = supplement / f"case-{index:03d}"
        if _json(directory / "binding.json") != binding:
            raise ValueError("supplement case binding changed")
        config = _json(directory / "audit" / "run-config.json")
        if (
            config.get("judge_model") != original_binding.get("judge_model")
            or config.get("thinking_level") != original_binding.get("thinking_level")
            or config.get("max_output_tokens") != original_binding.get("max_output_tokens")
            or config.get("selected_results") != 1
            or config.get("manifest") != original_binding["manifest"]["path"]
            or config.get("private_gold_sidecar") != original_binding["gold"]["path"]
            or (directory / "audit" / "prompt.txt").read_text(encoding="utf-8")
            != (original_judge / "prompt.txt").read_text(encoding="utf-8")
        ):
            raise ValueError("supplement prompt, gold, or generation settings changed")
        rows = _jsonl(directory / "audit" / "audit-results.jsonl")
        if len(rows) != 1 or rows[0].get("case_id") != case_id or not terminal_judge(rows[0]):
            raise ValueError("supplement case has no valid terminal judgment")
        if rows[0].get("judge_model") != original_binding.get("judge_model"):
            raise ValueError("supplement response model changed")
        if Path(rows[0]["source_trace_path"]).resolve() != trace.resolve():
            raise ValueError("supplement judged a different trace")
        compact = compact_record(rows[0], source_trace=trace)
        compact["audit_source"] = "independent_ambiguous_supplement"
        compact["original_ambiguous_receipt"] = str(ambiguous)
        supplements[case_id] = compact

    original: dict[str, dict[str, Any]] = {}
    for index, case_id in enumerate(expected, 1):
        path = original_judge / "cases" / case_token(index, case_id) / "result.json"
        if path.is_file():
            record = _json(path)
            if record.get("case_id") != case_id or not terminal_judge(record):
                raise ValueError("original judge receipt is invalid")
            record["audit_source"] = "streaming_initial_or_explicit_503_retry"
            original[case_id] = record
    if set(original) & set(supplements):
        raise ValueError("same case judged in both sources")
    if set(original) | set(supplements) != set(expected):
        raise ValueError("judge coverage is incomplete")
    ordered = [original.get(case_id) or supplements[case_id] for case_id in expected]
    if any(not terminal_judge(row) for row in ordered):
        raise ValueError("nonterminal judgment in consolidated output")
    categories = agent_private_gold_category_counts(ordered)
    strict = categories.get("correct_point_with_strong_evidence", 0)
    summary = {
        "schema_version": "ifv-streaming-agent-judge-explicit-supplement-v1",
        "phase": "audited_complete_with_independent_supplements",
        "source_rollout": str(rollout_root.resolve()),
        "original_judge": str(original_judge.resolve()),
        "supplement": str(supplement.resolve()),
        "expected_runnable": expected_count,
        "formal_denominator": formal_denominator,
        "completed": len(ordered),
        "original_completed": len(original),
        "independent_supplements": len(supplements),
        "original_ambiguities_preserved": sorted(supplements),
        "judge_model": original_binding["judge_model"],
        "binary_metrics": _metric_payload(ordered),
        "private_gold_categories": categories,
        "strict_evidence_sufficient_count": strict,
        "sesr_reported_percent": 100.0 * strict / formal_denominator,
        "large_payload_hashing": False,
    }
    _write_once(output / "consolidated-audit.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    _write_once(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout-root", "benchmark", "original-judge", "supplement", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1526)
    parser.add_argument("--formal-denominator", type=int, default=1527)
    args = parser.parse_args()
    summary = consolidate(
        rollout_root=args.rollout_root.resolve(),
        benchmark=args.benchmark.resolve(),
        original_judge=args.original_judge.resolve(),
        supplement=args.supplement.resolve(),
        output=args.output.resolve(),
        expected_count=args.expected_count,
        formal_denominator=args.formal_denominator,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
