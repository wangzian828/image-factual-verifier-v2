from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.archive_baseline import (
    score_archive_baseline,
    write_archive_baseline_metrics,
)


def _build_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "archive-summary.json").write_text(
        json.dumps({"archive_id": "archive-fixture"}),
        encoding="utf-8",
    )
    rows = [
        {
            "candidate_id": "candidate:1",
            "archive_source_version_id": "archive:1",
            "archive_image_path": "artifacts/images/1.jpg",
            "factual_status": "supported",
            "production_class": "pipeline_generated_supported",
            "target_subtype": "AF1",
            "target_capability_cell": "AF1",
        },
        {
            "candidate_id": "candidate:2",
            "archive_source_version_id": "archive:2",
            "archive_image_path": "artifacts/images/2.jpg",
            "factual_status": "refuted",
            "production_class": "pipeline_generated_refuted",
            "target_subtype": "EF3",
            "target_capability_cell": "EF3",
        },
        {
            "candidate_id": "candidate:3",
            "archive_source_version_id": "archive:3",
            "archive_image_path": "artifacts/images/3.jpg",
            "factual_status": "supported",
            "production_class": "web_crawled_supported",
            "target_subtype": "CD1",
            "target_capability_cell": "CD1",
        },
    ]
    candidate_path = archive / "human-review-candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return candidate_path


def test_score_archive_baseline_counts_raw_and_valid_only_accuracy(
    tmp_path: Path,
) -> None:
    candidate_path = _build_archive(tmp_path)
    summary, cases = score_archive_baseline(
        [
            {
                "case_id": "archive:1",
                "status": "success",
                "verdict": "real",
                "termination": "success",
            },
            {
                "case_id": "archive:2",
                "status": "success",
                "verdict": "real",
                "termination": "success",
            },
            {
                "case_id": "archive:3",
                "status": "error",
                "verdict": "error",
                "termination": "error",
                "error": "provider failed",
            },
        ],
        candidate_path,
        run_id="fixture-run",
        git_commit="a" * 40,
    )

    assert summary["total_cases"] == 3
    assert summary["correct"] == 1
    assert summary["valid_predictions"] == 2
    assert summary["raw_accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["valid_only_accuracy"] == 0.5
    assert summary["engineering_errors"] == 1
    assert summary["engineering_error_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert [row["correct"] for row in cases] == [True, False, False]
    assert cases[2]["prediction"] is None


def test_write_archive_baseline_metrics_writes_separate_artifacts(
    tmp_path: Path,
) -> None:
    candidate_path = _build_archive(tmp_path)
    run_dir = tmp_path / "run"
    summary = write_archive_baseline_metrics(
        run_dir,
        candidate_path,
        [
            {
                "case_id": "archive:1",
                "status": "success",
                "verdict": "real",
                "termination": "success",
            }
        ],
        run_id="fixture-run",
    )

    assert summary["raw_accuracy"] == 1.0
    assert (run_dir / "baseline_metrics.json").is_file()
    assert (run_dir / "baseline_case_metrics.jsonl").is_file()
    saved = json.loads(
        (run_dir / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    assert saved["archive_id"] == "archive-fixture"
