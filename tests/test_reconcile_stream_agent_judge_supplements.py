import json
from pathlib import Path

import pytest

from scripts.server.reconcile_stream_agent_judge_supplements import consolidate
from scripts.server.stream_agent_judges import case_token


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def stat(path: Path) -> dict[str, object]:
    value = path.stat()
    return {"path": str(path.resolve()), "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def terminal(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "status": "completed",
        "judge_model": "gemini-3.7-flash",
        "gold_verdict": "real",
        "candidate_verdict": "real",
        "verdict_matches_gold": True,
        "private_gold_auditable": True,
        "quality_bucket": "strong",
        "fact_alignment": "same_fact",
        "reason_quality": "decisive_and_grounded",
    }


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "rollout"
    original = root / "judge-stream"
    supplement = root / "supplement"
    benchmark = tmp_path / "benchmark.jsonl"
    output = root / "consolidated"
    manifest = tmp_path / "manifest.jsonl"
    gold = tmp_path / "gold.jsonl"
    for path in (manifest, gold):
        path.write_text("", encoding="utf-8")
    benchmark.write_text('{"case_id":"case-1"}\n{"case_id":"case-2"}\n', encoding="utf-8")
    write(root / "inference-summary.json", {"phase": "inference_complete", "success": 2})
    write(
        original / "binding.json",
        {
            "rollout_root": str(root.resolve()),
            "benchmark": {"path": str(benchmark.resolve())},
            "manifest": {"path": str(manifest.resolve())},
            "gold": {"path": str(gold.resolve())},
            "formal_denominator": 3,
            "judge_model": "gemini-3.7-flash",
            "thinking_level": "low",
            "max_output_tokens": 32768,
        },
    )
    (original / "prompt.txt").write_text("frozen prompt\n", encoding="utf-8")
    write(original / "cases" / case_token(1, "case-1") / "result.json", terminal("case-1"))
    original_case = original / "cases" / case_token(2, "case-2")
    write(original_case / "ambiguous.json", {"case_id": "case-2"})
    write(original_case / "attempt-01.intent.json", {"case_id": "case-2"})
    trace = root / "attempt-0" / "traces" / "case-2.json"
    write(trace, {"image_path": "image.jpg"})
    binding = {
        "case_id": "case-2",
        "source_trace": stat(trace),
        "original_ambiguous": stat(original_case / "ambiguous.json"),
        "protocol_relation": "independent_supplement_not_idempotent_replay",
        "judge_model": "gemini-3.7-flash",
        "thinking_level": "low",
        "max_output_tokens": 32768,
    }
    write(
        supplement / "plan.json",
        {"original_judge": str(original.resolve()), "source": str(root.resolve()), "bindings": [binding]},
    )
    case = supplement / "case-001"
    write(case / "binding.json", binding)
    write(
        case / "audit/run-config.json",
        {
            "judge_model": "gemini-3.7-flash",
            "thinking_level": "low",
            "max_output_tokens": 32768,
            "selected_results": 1,
            "manifest": str(manifest.resolve()),
            "private_gold_sidecar": str(gold.resolve()),
        },
    )
    (case / "audit/prompt.txt").write_text("frozen prompt\n", encoding="utf-8")
    write(case / "audit/audit-results.jsonl", {**terminal("case-2"), "source_trace_path": str(trace)})
    return {"root": root, "original": original, "supplement": supplement, "benchmark": benchmark, "output": output, "trace": trace}


def test_consolidate_keeps_ambiguous_original_separate(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    summary = consolidate(
        rollout_root=paths["root"],
        benchmark=paths["benchmark"],
        original_judge=paths["original"],
        supplement=paths["supplement"],
        output=paths["output"],
        expected_count=2,
        formal_denominator=3,
    )
    assert summary["completed"] == 2
    assert summary["original_completed"] == 1
    assert summary["independent_supplements"] == 1
    assert summary["strict_evidence_sufficient_count"] == 2
    assert summary["sesr_reported_percent"] == pytest.approx(200 / 3)
    assert (paths["original"] / "cases" / case_token(2, "case-2") / "ambiguous.json").is_file()
    assert not (paths["original"] / "cases" / case_token(2, "case-2") / "result.json").exists()
    assert consolidate(
        rollout_root=paths["root"], benchmark=paths["benchmark"],
        original_judge=paths["original"], supplement=paths["supplement"],
        output=paths["output"], expected_count=2, formal_denominator=3,
    ) == summary


def test_consolidate_rejects_changed_source_trace(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    paths["trace"].write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="source trace changed"):
        consolidate(
            rollout_root=paths["root"], benchmark=paths["benchmark"],
            original_judge=paths["original"], supplement=paths["supplement"],
            output=paths["output"], expected_count=2, formal_denominator=3,
        )
