from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.release_adapter import (
    release_companions,
    require_uniform_runtime_release,
    resolve_runtime_image_path,
    verification_case_from_runtime_row,
)


def _runtime_row(**updates):
    row = {
        "case_id": "case_0123456789abcdef",
        "image_path": "assets/sha256/00/image.jpg",
        "image_sha256": "0" * 64,
        "claim_mode": "external_claim",
        "user_claim": "A public factual claim.",
        "claim_surface": None,
        "claim_source_region": None,
        "claim_observed_at": "2024-01-02",
        "decision_policy_version": "reinspect-v1",
    }
    row.update(updates)
    return row


def test_v02_runtime_row_projects_to_runtime_owned_case(tmp_path: Path) -> None:
    benchmark = tmp_path / "release" / "runtime_input" / "cases.jsonl"
    row = resolve_runtime_image_path(_runtime_row(), benchmark)
    case = verification_case_from_runtime_row(row)

    assert case is not None
    assert case.case_id == "case_0123456789abcdef"
    assert case.image_path == str(
        (benchmark.parent / "assets/sha256/00/image.jpg").resolve()
    )
    assert case.decision_policy_version == "reinspect-v1"


def test_standard_release_companions_are_inferred(tmp_path: Path) -> None:
    benchmark = tmp_path / "release" / "runtime_input" / "cases.jsonl"
    companions = release_companions(benchmark)

    assert companions is not None
    assert companions.evaluator_private == (
        tmp_path / "release" / "evaluator_private" / "run_eval.jsonl"
    )
    assert companions.evaluation_gold == (
        tmp_path / "release" / "evaluation_gold" / "gold.jsonl"
    )


def test_image_only_is_fail_closed_until_v03_is_active() -> None:
    with pytest.raises(ValueError, match="v0.3/reinspect-v2"):
        verification_case_from_runtime_row(
            _runtime_row(
                claim_mode="image_only",
                user_claim=None,
                decision_policy_version="reinspect-v2",
            )
        )


def test_unactivated_decision_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="accepts decision_policy_version=reinspect-v1"):
        verification_case_from_runtime_row(
            _runtime_row(decision_policy_version="reinspect-v2")
        )


def test_release_row_requires_explicit_policy_version() -> None:
    row = _runtime_row()
    del row["decision_policy_version"]

    assert verification_case_from_runtime_row(row) is None


def test_mixed_release_and_legacy_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="mixes runtime release rows"):
        require_uniform_runtime_release(
            [_runtime_row(), {"sample_id": "legacy", "image_path": "image.jpg"}]
        )
