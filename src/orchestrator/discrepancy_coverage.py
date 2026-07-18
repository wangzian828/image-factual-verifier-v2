"""Deterministic v4 claim/discrepancy Coverage and verdict-basis compilation."""

from __future__ import annotations

from typing import List

from src.orchestrator.investigation_models import (
    DiscrepancyCoverageAudit,
    DiscrepancyVerdictBasis,
    ImageClaimCoverage,
    ImageOnlyInvestigationState,
)
from src.orchestrator.task_store import (
    MAX_TOOL_ACTIONS,
    remaining_claim_hypothesis_routes,
    stable_id,
)


def audit_discrepancy_coverage(
    state: ImageOnlyInvestigationState,
    *,
    decision_checkpoint: bool = False,
) -> DiscrepancyCoverageAudit:
    """Audit v4 terminal conditions without consulting legacy core-fact state."""

    latest_assessment = {}
    for item in state.claim_assessments:
        latest_assessment[item.claim_id] = item
    remaining_routes = remaining_claim_hypothesis_routes(state)
    open_task_ids = {
        route.split(":", 2)[2]
        if route.startswith("reverse_image_search:")
        and len(route.split(":", 2)) == 3
        else route.split(":", 2)[1]
        for route in remaining_routes
        if len(route.split(":", 2)) >= 2
    }
    task_by_id = {item.task_id: item for item in state.tasks}
    coverage_rows: List[ImageClaimCoverage] = []
    for claim in state.image_claims:
        assessment = latest_assessment.get(claim.claim_id)
        claim_route_open = any(
            claim.claim_id in task_by_id[task_id].claim_ids
            for task_id in open_task_ids
            if task_id in task_by_id
        )
        coverage_rows.append(
            ImageClaimCoverage(
                claim_id=claim.claim_id,
                salience=claim.salience,
                assessment=(
                    assessment.assessment if assessment is not None else "open"
                ),
                evidence_ids=(
                    list(assessment.evidence_ids) if assessment is not None else []
                ),
                remaining_gap=(
                    assessment.remaining_gap if assessment is not None else ""
                ),
                route_status="open" if claim_route_open else "closed",
            )
        )

    decisive = [
        item
        for item in state.material_discrepancies
        if item.materiality == "decisive" and item.status == "established"
    ]
    high_rows = [item for item in coverage_rows if item.salience == "high"]
    fake_complete = bool(decisive and state.proposed_verdict == "fake")
    real_complete = bool(
        high_rows
        and all(item.assessment == "supported" for item in high_rows)
        and all(item.route_status == "closed" for item in high_rows)
        and not decisive
        and state.proposed_verdict == "real"
    )
    unverifiable_complete = bool(
        any(
            item.assessment in {"insufficient", "conflicted"}
            and item.route_status == "closed"
            for item in high_rows
        )
        and not decisive
        and state.proposed_verdict == "unverifiable"
    )
    complete = fake_complete or real_complete or unverifiable_complete
    if complete:
        stop_reason = "verdict_determined"
        reason = {
            "fake": "An established decisive discrepancy closes the case.",
            "real": (
                "Every high-salience ImageClaim is supported and its routes close."
            ),
            "unverifiable": (
                "A high-salience ImageClaim remains unresolved after its routes close."
            ),
        }[state.proposed_verdict]
    elif state.action_count >= MAX_TOOL_ACTIONS:
        stop_reason = "hard_budget_exhausted"
        reason = "The action budget ended before v4 verdict preconditions closed."
    elif not remaining_routes and decision_checkpoint:
        stop_reason = "information_saturated"
        reason = (
            "No claim/hypothesis route remains, but the checkpoint did not propose "
            "an admissible terminal verdict."
        )
    else:
        stop_reason = "continue"
        reason = (
            "Claim/discrepancy coverage remains open."
            if remaining_routes
            else "A final Discrepancy Decision checkpoint is required."
        )

    previous = (
        state.discrepancy_coverage_audits[-1]
        if state.discrepancy_coverage_audits
        else None
    )
    signature = _coverage_signature(coverage_rows, decisive, state.proposed_verdict)
    substantive_gain = previous is None or signature != _audit_signature(previous)
    audit = DiscrepancyCoverageAudit(
        audit_id=stable_id(
            "discrepancy-coverage",
            state.brief.case_id,
            state.action_count,
            len(state.discrepancy_coverage_audits),
        ),
        action_count=state.action_count,
        claims=coverage_rows,
        decisive_discrepancy_ids=[item.discrepancy_id for item in decisive],
        proposed_verdict=state.proposed_verdict,
        complete=complete,
        stop_reason=stop_reason,
        decision_checkpoint=decision_checkpoint,
        substantive_gain=substantive_gain,
        reason=reason,
    )
    state.discrepancy_coverage_audits.append(audit)
    if stop_reason != "continue":
        state.stop_reason = stop_reason
    return audit


def compile_discrepancy_verdict_basis(
    state: ImageOnlyInvestigationState,
) -> tuple[str, DiscrepancyVerdictBasis]:
    """Compile the smallest admissible v4 claim/discrepancy/Evidence basis."""

    audit = (
        state.discrepancy_coverage_audits[-1]
        if state.discrepancy_coverage_audits
        else audit_discrepancy_coverage(state)
    )
    if not audit.complete or state.proposed_verdict not in {
        "fake",
        "real",
        "unverifiable",
    }:
        raise RuntimeError("v4 verdict basis requires complete discrepancy coverage")

    evidence_ids: List[str] = []
    claim_ids: List[str] = []
    discrepancy_ids: List[str] = []
    anchor_ids: List[str] = []
    unresolved_gaps: List[str] = []
    if state.proposed_verdict == "fake":
        selected = next(
            item
            for item in state.material_discrepancies
            if item.materiality == "decisive" and item.status == "established"
        )
        discrepancy_ids = [selected.discrepancy_id]
        claim_ids = list(selected.affected_claim_ids)
        anchor_ids = list(selected.visual_anchor_fact_ids)
        evidence_ids = list(selected.evidence_ids)
        verdict_target = selected.statement
    elif state.proposed_verdict == "real":
        high_claims = [
            item for item in state.image_claims if item.salience == "high"
        ]
        claim_ids = [item.claim_id for item in high_claims]
        anchor_ids = list(
            dict.fromkeys(
                fact_id
                for claim in high_claims
                for fact_id in claim.anchor_fact_ids
            )
        )
        latest = {}
        for item in state.claim_assessments:
            latest[item.claim_id] = item
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim_id in claim_ids
                for evidence_id in latest[claim_id].evidence_ids
            )
        )
        verdict_target = state.image_account_summary
    else:
        unresolved_rows = [
            item
            for item in audit.claims
            if item.salience == "high"
            and item.assessment in {"insufficient", "conflicted"}
        ]
        claim_ids = [item.claim_id for item in unresolved_rows]
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for item in unresolved_rows
                for evidence_id in item.evidence_ids
            )
        )
        unresolved_gaps = [
            item.remaining_gap or f"ImageClaim {item.claim_id} remains unresolved."
            for item in unresolved_rows
        ]
        verdict_target = state.image_account_summary

    finding_ids = [
        item.finding_id
        for item in state.findings
        if set(item.evidence_ids) & set(evidence_ids)
        and (not claim_ids or _finding_serves_claims(state, item.task_id, claim_ids))
    ]
    basis = DiscrepancyVerdictBasis(
        verdict_target=verdict_target,
        claim_ids=claim_ids,
        discrepancy_ids=discrepancy_ids,
        visual_anchor_fact_ids=anchor_ids,
        finding_ids=list(dict.fromkeys(finding_ids)),
        evidence_ids=evidence_ids,
        unresolved_gaps=unresolved_gaps[:12],
    )
    state.discrepancy_verdict_basis = basis
    return state.proposed_verdict, basis


def _finding_serves_claims(
    state: ImageOnlyInvestigationState,
    task_id: str,
    claim_ids: List[str],
) -> bool:
    return any(
        item.task_id == task_id and bool(set(item.claim_ids) & set(claim_ids))
        for item in state.tasks
    )


def _coverage_signature(
    claims: List[ImageClaimCoverage],
    decisive: List[object],
    proposed_verdict: str,
) -> tuple[object, ...]:
    return (
        tuple(
            (
                item.claim_id,
                item.assessment,
                tuple(item.evidence_ids),
                item.route_status,
            )
            for item in claims
        ),
        tuple(getattr(item, "discrepancy_id", "") for item in decisive),
        proposed_verdict,
    )


def _audit_signature(audit: DiscrepancyCoverageAudit) -> tuple[object, ...]:
    return (
        tuple(
            (
                item.claim_id,
                item.assessment,
                tuple(item.evidence_ids),
                item.route_status,
            )
            for item in audit.claims
        ),
        tuple(audit.decisive_discrepancy_ids),
        audit.proposed_verdict,
    )
