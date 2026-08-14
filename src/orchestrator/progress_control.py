"""Auditable information-gain accounting for investigation diagnostics."""
from __future__ import annotations

from typing import Any, Mapping

from src.orchestrator.evidence_semantics import evidence_is_qualified
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
    evidence_by_id = {
        item.evidence_id: item
        for item in state.evidence
    }
    decision_capable_evidence_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if (
            evidence_id in evidence_by_id
            and evidence_is_qualified(evidence_by_id[evidence_id])
            and (
                evidence_by_id[evidence_id].stance in {"support", "refute"}
                or evidence_by_id[evidence_id].evidence_kind
                in {"image_region", "reference_comparison"}
            )
        )
    ]
    if visual_reinspection and evidence_ids:
        gain = (
            "visual_understanding_gain"
            if decision_capable_evidence_ids
            else "lead_gain"
        )
        source_ids = [*evidence_ids, *finding_ids]
        rationale = (
            "Evidence-motivated image reinspection changed the recorded visual "
            "account."
            if decision_capable_evidence_ids
            else (
                "Visual reinspection returned observations, but none were "
                "qualified for a decision-capable update."
            )
        )
    elif decision_capable_evidence_ids:
        gain = "evidence_gain"
        source_ids = [*decision_capable_evidence_ids, *finding_ids]
        rationale = "The accepted action created decision-capable Evidence."
    elif evidence_ids:
        gain = "lead_gain"
        source_ids = [*evidence_ids, *finding_ids]
        rationale = (
            "The accepted action created Evidence, but none was qualified "
            "for a decision-capable update."
        )
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
    else:
        state.no_substantive_gain_streak = min(
            24,
            state.no_substantive_gain_streak + 1,
        )

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
