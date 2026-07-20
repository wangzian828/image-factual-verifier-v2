from __future__ import annotations

from src.orchestrator.investigation_models import (
    ImageOnlyInvestigationState,
    InvestigationBrief,
)
from src.orchestrator.progress_control import (
    record_action_progress,
    record_decision_progress,
)


def _state(action_count: int = 1) -> ImageOnlyInvestigationState:
    return ImageOnlyInvestigationState(
        brief=InvestigationBrief(brief_id="brief", case_id="case"),
        action_count=action_count,
    )


def test_leads_remain_diagnostic_no_substantive_gain() -> None:
    state = _state()

    for index in range(3):
        state.action_count = index + 1
        event = record_action_progress(
            state,
            {"created_discovery_ids": [f"discovery-{index}"]},
        )

    assert event.gain == "lead_gain"
    assert state.no_substantive_gain_streak == 3
    assert event.no_substantive_gain_streak == 3


def test_evidence_and_decision_gain_reset_streak() -> None:
    state = _state()
    record_action_progress(state, {"created_failure_ids": ["failure-1"]})
    state.action_count = 2
    record_action_progress(state, {"created_failure_ids": ["failure-2"]})
    assert state.no_substantive_gain_streak == 2

    state.action_count = 3
    evidence = record_action_progress(
        state,
        {"created_evidence_ids": ["evidence-1"]},
    )
    assert evidence.gain == "evidence_gain"
    assert state.no_substantive_gain_streak == 0

    state.no_substantive_gain_streak = 2
    decision = record_decision_progress(
        state,
        {"accepted_discrepancy_id": "discrepancy-1"},
    )
    assert decision is not None and decision.gain == "decision_gain"
    assert state.no_substantive_gain_streak == 0


def test_no_gain_streak_never_sets_a_terminal_state() -> None:
    state = _state()
    for index in range(24):
        state.action_count = index + 1
        record_action_progress(
            state,
            {"created_failure_ids": [f"failure-{index}"]},
        )

    assert state.no_substantive_gain_streak == 24
    assert state.stop_reason == ""
    assert all(
        event.no_substantive_gain_streak == index + 1
        for index, event in enumerate(state.progress_events)
    )
