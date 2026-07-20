"""Auditable information-gain accounting and two-stage saturation control."""
from __future__ import annotations

import os
from typing import Any, Mapping

from src.orchestrator.investigation_models import (
    ImageOnlyInvestigationState,
    ProgressEvent,
)
from src.orchestrator.task_store import stable_id


SUBSTANTIVE_GAINS = {
    "evidence_gain",
    "decision_gain",
    "visual_understanding_gain",
}


def record_action_progress(
    state: ImageOnlyInvestigationState,
    update: Mapping[str, Any],
    *,
    visual_reinspection: bool = False,
) -> ProgressEvent:
    """Classify progress from reducer-created IDs, never from result wording."""

    evidence_ids = list(update.get("created_evidence_ids", []) or [])
    finding_ids = list(update.get("created_finding_ids", []) or [])
    discovery_ids = list(update.get("created_discovery_ids", []) or [])
    failure_ids = list(update.get("created_failure_ids", []) or [])
    recalled_candidate_ids = list(update.get("recalled_candidate_ids", []) or [])
    read_memory_ids = list(update.get("read_memory_ids", []) or [])
    if visual_reinspection and evidence_ids:
        gain = "visual_understanding_gain"
        source_ids = [*evidence_ids, *finding_ids]
        rationale = "Evidence-motivated image reinspection changed the recorded visual account."
    elif evidence_ids:
        gain = "evidence_gain"
        source_ids = [*evidence_ids, *finding_ids]
        rationale = "The accepted action created provenance-complete Evidence."
    elif discovery_ids or recalled_candidate_ids or read_memory_ids:
        gain = "lead_gain"
        source_ids = [*discovery_ids, *recalled_candidate_ids, *read_memory_ids]
        rationale = (
            "The accepted action exposed a candidate or exact archived span, "
            "but did not itself create qualified Evidence."
        )
    else:
        gain = "no_gain"
        source_ids = failure_ids
        rationale = "The accepted action created no new decision-capable material."

    if gain in SUBSTANTIVE_GAINS:
        state.no_substantive_gain_streak = 0
        state.saturation_checkpoint_action = None
        state.saturation_grace_remaining = 0
    else:
        state.no_substantive_gain_streak = min(
            24,
            state.no_substantive_gain_streak + 1,
        )
        if state.saturation_grace_remaining > 0:
            state.saturation_grace_remaining -= 1

    soft_limit = max(
        2,
        int(os.getenv("IFV_SOFT_NO_GAIN_ACTIONS", "4")),
    )
    checkpoint = bool(
        state.no_substantive_gain_streak >= soft_limit
        and state.saturation_checkpoint_action is None
    )
    if checkpoint:
        state.saturation_checkpoint_action = state.action_count

    event = ProgressEvent(
        progress_id=stable_id(
            "progress",
            state.brief.case_id,
            state.action_count,
            gain,
            source_ids,
        ),
        action_count=state.action_count,
        gain=gain,
        source_ids=list(dict.fromkeys(source_ids))[:40],
        no_substantive_gain_streak=state.no_substantive_gain_streak,
        soft_checkpoint_triggered=checkpoint,
        grace_remaining=state.saturation_grace_remaining,
        rationale=rationale,
    )
    state.progress_events.append(event)
    return event


def record_decision_progress(
    state: ImageOnlyInvestigationState,
    update: Mapping[str, Any],
) -> ProgressEvent | None:
    changed_ids = list(
        dict.fromkeys(
            [
                *(update.get("accepted_assessment_ids", []) or []),
                *(update.get("accepted_hypothesis_ids", []) or []),
                *(update.get("retired_hypothesis_ids", []) or []),
                *(
                    [update.get("accepted_discrepancy_id")]
                    if update.get("accepted_discrepancy_id")
                    else []
                ),
                *(
                    [update.get("accepted_visual_question_id")]
                    if update.get("accepted_visual_question_id")
                    else []
                ),
            ]
        )
    )
    if not changed_ids:
        return None
    state.no_substantive_gain_streak = 0
    state.saturation_checkpoint_action = None
    state.saturation_grace_remaining = 0
    event = ProgressEvent(
        progress_id=stable_id(
            "progress-decision",
            state.brief.case_id,
            state.action_count,
            changed_ids,
        ),
        action_count=max(1, state.action_count),
        gain="decision_gain",
        source_ids=changed_ids[:40],
        no_substantive_gain_streak=0,
        rationale="The semantic checkpoint changed claim, discrepancy, route, or visual state.",
    )
    state.progress_events.append(event)
    return event


def open_saturation_grace(state: ImageOnlyInvestigationState) -> int:
    grace = max(1, min(8, int(os.getenv("IFV_SATURATION_GRACE_ACTIONS", "2"))))
    state.saturation_grace_remaining = grace
    return grace


def should_run_saturation_checkpoint(state: ImageOnlyInvestigationState) -> bool:
    return bool(
        state.saturation_checkpoint_action == state.action_count
        and state.saturation_grace_remaining == 0
    )


def grace_exhausted_without_gain(state: ImageOnlyInvestigationState) -> bool:
    return bool(
        state.saturation_checkpoint_action is not None
        and state.action_count > state.saturation_checkpoint_action
        and state.saturation_grace_remaining == 0
        and state.no_substantive_gain_streak > 0
    )
