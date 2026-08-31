import json
from pathlib import Path

import pytest

from scripts.summarize_agent_test_evaluation import summarize_agent_test_evaluation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest(case_id: str, route: str) -> dict:
    return {
        "unified_case_id": case_id,
        "split": "test",
        "construction_subroute": route,
        "production_class": route,
        "target_route": route,
        "capability_cell": "AF1",
        "factual_status": "supported",
    }


def _audit(
    case_id: str,
    gold: str,
    prediction: str,
    reason_quality: str,
) -> dict:
    return {
        "case_id": case_id,
        "status": "completed",
        "private_gold_auditable": True,
        "gold_verdict": gold,
        "candidate_verdict": prediction,
        "verdict_matches_gold": gold == prediction,
        "reason_quality": reason_quality,
        "fact_alignment": "same_fact",
        "failure_modes": [],
    }


def test_summarize_agent_test_evaluation_calculates_binary_and_three_way(
    tmp_path: Path,
):
    manifest = tmp_path / "test-manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest("a", "route-a"),
            _manifest("b", "route-a"),
            _manifest("c", "route-b"),
            _manifest("d", "route-b"),
        ],
    )
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    _write_jsonl(
        old,
        [
            _audit("a", "real", "real", "decisive_and_grounded"),
            _audit("b", "fake", "fake", "unsupported"),
        ],
    )
    _write_jsonl(
        new,
        [
            _audit("c", "real", "fake", "contradictory"),
            _audit("d", "fake", "real", "unsupported"),
        ],
    )

    summary = summarize_agent_test_evaluation(
        inputs=[("old", old), ("new", new)],
        test_manifest=manifest,
        output_dir=tmp_path / "summary",
        expected_case_count=4,
    )

    assert summary["binary_metrics"]["accuracy"] == 0.5
    assert summary["binary_metrics"]["balanced_accuracy"] == 0.5
    assert summary["three_way_private_gold_categories"] == {
        "correct_point_with_strong_evidence": 1,
        "correct_verdict_insufficient_evidence": 1,
        "wrong_verdict": 2,
    }
    assert summary["by_construction_subroute"]["route-a"]["selected"] == 2
    assert summary["failure_analysis"]["wrong_verdict_count"] == 2


def test_summarize_agent_test_evaluation_rejects_duplicate_case(tmp_path: Path):
    manifest = tmp_path / "test-manifest.jsonl"
    _write_jsonl(manifest, [_manifest("a", "route-a")])
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    _write_jsonl(source_a, [_audit("a", "real", "real", "unsupported")])
    _write_jsonl(source_b, [_audit("a", "real", "real", "unsupported")])

    with pytest.raises(ValueError, match="multiple audit inputs"):
        summarize_agent_test_evaluation(
            inputs=[("a", source_a), ("b", source_b)],
            test_manifest=manifest,
            output_dir=tmp_path / "summary",
        )
