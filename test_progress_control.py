from __future__ import annotations

from src.orchestrator.investigation_models import (
    ImageOnlyInvestigationState,
    InvestigationBrief,
)
from src.orchestrator.progress_control import (
    grace_exhausted_without_gain,
    open_saturation_grace,
    record_action_progress,
    record_decision_progress,
    should_run_saturation_checkpoint,
)


def _state(action_count: int = 1) -> ImageOnlyInvestigationState:
    return ImageOnlyInvestigationState(
        brief=InvestigationBrief(brief_id="brief", case_id="case"),
        action_count=action_count,
    )


def test_leads_do_not_reset_no_substantive_gain(monkeypatch) -> None:
    monkeypatch.setenv("IFV_SOFT_NO_GAIN_ACTIONS", "3")
    state = _state()

    for index in range(3):
        state.action_count = index + 1
        event = record_action_progress(
            state,
            {"created_discovery_ids": [f"discovery-{index}"]},
        )

    assert event.gain == "lead_gain"
    assert state.no_substantive_gain_streak == 3
    assert should_run_saturation_checkpoint(state) is True


def test_evidence_and_decision_gain_reset_streak(monkeypatch) -> None:
    monkeypatch.setenv("IFV_SOFT_NO_GAIN_ACTIONS", "2")
    state = _state()
    record_action_progress(state, {"created_failure_ids": ["failure-1"]})
    state.action_count = 2
    record_action_progress(state, {"created_failure_ids": ["failure-2"]})
    assert should_run_saturation_checkpoint(state) is True

    state.action_count = 3
    evidence = record_action_progress(
        state,
        {"created_evidence_ids": ["evidence-1"]},
    )
    assert evidence.gain == "evidence_gain"
    assert state.no_substantive_gain_streak == 0
    assert state.saturation_checkpoint_action is None

    state.no_substantive_gain_streak = 2
    decision = record_decision_progress(
        state,
        {"accepted_discrepancy_id": "discrepancy-1"},
    )
    assert decision is not None and decision.gain == "decision_gain"
    assert state.no_substantive_gain_streak == 0


def test_grace_actions_are_bounded_and_then_saturate(monkeypatch) -> None:
    monkeypatch.setenv("IFV_SOFT_NO_GAIN_ACTIONS", "2")
    monkeypatch.setenv("IFV_SATURATION_GRACE_ACTIONS", "2")
    state = _state()
    record_action_progress(state, {"created_failure_ids": ["failure-1"]})
    state.action_count = 2
    record_action_progress(state, {"created_failure_ids": ["failure-2"]})
    assert open_saturation_grace(state) == 2

    state.action_count = 3
    record_action_progress(state, {"created_discovery_ids": ["lead-1"]})
    assert grace_exhausted_without_gain(state) is False
    state.action_count = 4
    record_action_progress(state, {"created_failure_ids": ["failure-3"]})
    assert grace_exhausted_without_gain(state) is True
