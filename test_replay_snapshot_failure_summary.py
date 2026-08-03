from __future__ import annotations

from scripts.replay_snapshot_manifest import _summary


def test_replay_summary_exposes_fail_closed_visual_stage() -> None:
    result = {
        "case_id": "case-a",
        "engineering_error": (
            "RuntimeError: Focused visual reinspection produced no pixel Evidence"
        ),
        "engineering_failure_codes": ["rate_limited"],
        "source_only_follow_up_blocked": True,
        "decision_2_skipped_reason": "focused_visual_failure_guard",
        "visual_inspection_update": None,
        "decision_2_update": None,
        "records": {
            "visual_reinspections": [
                {
                    "status": "failed",
                    "evidence_ids": [],
                }
            ],
            "evidence": [],
            "claim_assessments": [],
            "material_discrepancies": [],
            "discrepancy_decisions": [],
        },
        "compiled_basis": None,
        "composite_success": False,
        "compiled_verdict": "",
        "coverage_audit": {"stop_reason": "continue"},
    }

    summary = _summary(result)

    assert summary["engineering_failure_codes"] == ["rate_limited"]
    assert summary["source_only_follow_up_blocked"] is True
    assert (
        summary["decision_2_skipped_reason"]
        == "focused_visual_failure_guard"
    )
    assert summary["decision_2_ran"] is False
    assert summary["resolved_visual_evidence_ids"] == []
