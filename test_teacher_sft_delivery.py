from __future__ import annotations

import json
from pathlib import Path

from scripts.trajectory.verify_teacher_sft_delivery import (
    verify_teacher_sft_delivery,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_text(path: Path, value: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _complete_pipeline(root: Path) -> None:
    _write_json(root / "preparation.json", {"case_count": 3, "limit": None})
    _write_json(root / "pipeline-state.json", {"status": "completed"})
    _write_json(
        root / "classification/final/classification.json",
        {"case_count": 3, "selected_case_count": 2, "hard_case_count": 1},
    )
    _write_json(
        root / "accepted-release/accepted_release_manifest.json",
        {"accepted_case_count": 2},
    )
    _write_text(root / "accepted-release/selected_episodes.jsonl")
    _write_text(root / "accepted-release/trajectory_sft.jsonl")
    _write_text(root / "accepted-release/perception_trajectories.jsonl", "")
    _write_text(root / "sft-training-package/ms-swift-policy/train.jsonl")
    _write_text(root / "sft-training-package/ms-swift-policy/validation.jsonl")
    _write_json(root / "sft-training-package/ms-swift-policy/manifest.json", {})
    _write_json(root / "sft-training-package/ms-swift-perception/manifest.json", {})
    _write_json(root / "sft-training-package/audits/policy.json", {"passed": True})
    _write_json(
        root / "sft-training-package/audits/perception.json",
        {"passed": True},
    )
    _write_json(root / "sft-training-package/training_plan.json", {})
    _write_json(
        root / "sft-training-package/MANIFEST.json",
        {
            "counts": {
                "selected_release_cases": 2,
                "policy_rows": 2,
                "action_only_rows": 0,
                "long_holdout_rows": 0,
            },
            "audits": {
                "policy": {"passed": True},
                "perception": {"passed": True},
            },
            "training": {
                "requested": False,
                "started": False,
            },
        },
    )
    _write_json(
        root / "audits/pipeline-summary.json",
        {"all_trace_audits_passed": True},
    )


def test_full_teacher_sft_delivery_passes(tmp_path: Path) -> None:
    _complete_pipeline(tmp_path)

    result = verify_teacher_sft_delivery(tmp_path, expected_case_count=3)

    assert result["passed"] is True
    assert result["final_delivery"] is True
    assert result["failed_check_count"] == 0


def test_limited_smoke_cannot_be_final_delivery(tmp_path: Path) -> None:
    _complete_pipeline(tmp_path)
    _write_json(tmp_path / "preparation.json", {"case_count": 3, "limit": 3})

    result = verify_teacher_sft_delivery(tmp_path, expected_case_count=3)

    assert result["passed"] is False
    assert result["final_delivery"] is False
    assert any(
        check["name"] == "unbounded_full_run" and not check["passed"]
        for check in result["checks"]
    )


def test_raw_trace_preview_cannot_be_final_delivery(tmp_path: Path) -> None:
    _write_text(tmp_path / "rollouts/initial/attempt-02/traces/example.json")

    result = verify_teacher_sft_delivery(tmp_path, expected_case_count=8490)

    assert result["passed"] is False
    assert result["final_delivery"] is False
    assert result["failed_check_count"] > 0
