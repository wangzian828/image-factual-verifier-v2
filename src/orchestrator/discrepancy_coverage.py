"""Deterministic v4 claim/discrepancy Coverage and verdict-basis compilation."""

from __future__ import annotations

from typing import List

from src.orchestrator.evidence_semantics import evidence_is_qualified_for_stance
from src.orchestrator.investigation_models import (
    DiscrepancyCoverageAudit,
    DiscrepancyVerdictBasis,
    ImageClaimCoverage,
    ImageOnlyInvestigationState,
)
from src.orchestrator.task_store import (
    COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY,
    MAX_TOOL_ACTIONS,
    pending_discrepancy_evidence_ids,
    pending_visual_reinspection,
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
    pending_archive_ids = list(state.pending_archive_read_ids)
    pending_evidence_ids = pending_discrepancy_evidence_ids(state)
    pending_visual = pending_visual_reinspection(state)
    has_pending_terminal_work = bool(
        pending_archive_ids or pending_evidence_ids or pending_visual is not None
    )
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
    complete = fake_complete or real_complete
    if complete:
        stop_reason = "verdict_determined"
        reason = {
            "fake": "An established decisive discrepancy closes the case.",
            "real": (
                "Every high-salience ImageClaim is supported and its routes close."
            ),
        }[state.proposed_verdict]
    elif state.action_count >= MAX_TOOL_ACTIONS:
        stop_reason = "hard_budget_exhausted"
        reason = "The action budget ended before v4 verdict preconditions closed."
    elif (
        not remaining_routes
        and not has_pending_terminal_work
        and decision_checkpoint
    ):
        stop_reason = "meaningful_routes_exhausted"
        reason = (
            "No claim/hypothesis route, pending archive read, or visual reinspection "
            "remains after the final route audit."
        )
    else:
        stop_reason = "continue"
        reason = (
            "Claim/discrepancy coverage or pending terminal work remains open."
            if remaining_routes or has_pending_terminal_work
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
    if state.stop_reason not in {
        "verdict_determined",
        "meaningful_routes_exhausted",
        "hard_budget_exhausted",
    }:
        raise RuntimeError("v4 verdict basis requires a terminal investigation state")

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
        evidence_ids, finding_ids = _directional_verdict_chain(
            state,
            claim_ids=claim_ids,
            candidate_evidence_ids=list(selected.evidence_ids),
            stance="refute",
        )
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
        candidate_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim_id in claim_ids
                for evidence_id in latest[claim_id].evidence_ids
            )
        )
        evidence_ids, finding_ids = _directional_verdict_chain(
            state,
            claim_ids=claim_ids,
            candidate_evidence_ids=candidate_evidence_ids,
            stance="support",
        )
        verdict_target = state.image_account_summary
    else:
        high_rows = [
            item for item in audit.claims if item.salience == "high"
        ]
        unresolved_rows = [
            item
            for item in audit.claims
            if item.assessment not in {"supported", "refuted"}
        ]
        basis_rows = list(
            {
                item.claim_id: item
                for item in [*high_rows, *unresolved_rows]
            }.values()
        )
        claim_ids = [item.claim_id for item in basis_rows]
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for item in basis_rows
                for evidence_id in item.evidence_ids
            )
        )
        claim_by_id = {item.claim_id: item for item in state.image_claims}
        anchor_ids = list(
            dict.fromkeys(
                anchor_id
                for claim_id in claim_ids
                for anchor_id in claim_by_id[claim_id].anchor_fact_ids
            )
        )
        unresolved_gaps = [
            item.remaining_gap or f"ImageClaim {item.claim_id} remains unresolved."
            for item in unresolved_rows
        ]
        if not unresolved_gaps:
            unresolved_gaps = [
                "No evidence-determined verdict was accepted before "
                f"{state.stop_reason}."
            ]
        verdict_target = state.image_account_summary

    if state.proposed_verdict not in {"fake", "real"}:
        finding_ids = [
            item.finding_id
            for item in state.findings
            if set(item.evidence_ids) & set(evidence_ids)
            and (
                not claim_ids
                or _finding_serves_claims(state, item.task_id, claim_ids)
            )
        ]
    basis = DiscrepancyVerdictBasis(
        decision_mode=(
            "evidence_determined"
            if state.proposed_verdict in {"fake", "real"}
            else "bounded_binary_judgment"
        ),
        verdict_target=verdict_target,
        claim_ids=claim_ids,
        discrepancy_ids=discrepancy_ids,
        visual_anchor_fact_ids=anchor_ids,
        finding_ids=list(dict.fromkeys(finding_ids)),
        evidence_ids=evidence_ids,
        unresolved_gaps=unresolved_gaps[:12],
    )
    state.discrepancy_verdict_basis = basis
    return state.proposed_verdict if state.proposed_verdict in {"fake", "real"} else "", basis


def _directional_verdict_chain(
    state: ImageOnlyInvestigationState,
    *,
    claim_ids: List[str],
    candidate_evidence_ids: List[str],
    stance: str,
) -> tuple[List[str], List[str]]:
    claim_by_id = {item.claim_id: item for item in state.image_claims}
    task_by_id = {item.task_id: item for item in state.tasks}
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    selected_evidence_ids: List[str] = []
    selected_finding_ids: List[str] = []
    for claim_id in claim_ids:
        claim = claim_by_id[claim_id]
        matched = None
        if stance == "refute":
            composite = next(
                (
                    item
                    for item in state.findings
                    if item.stance == "refute"
                    and COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY
                    in item.source_family_ids
                    and claim.fact_id in item.fact_ids
                    and set(item.evidence_ids) <= set(candidate_evidence_ids)
                    and item.task_id in task_by_id
                    and claim_id in task_by_id[item.task_id].claim_ids
                ),
                None,
            )
            if composite is not None:
                selected_evidence_ids.extend(composite.evidence_ids)
                selected_finding_ids.append(composite.finding_id)
                continue
        for evidence_id in candidate_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or not evidence_is_qualified_for_stance(
                evidence,
                stance,
            ):
                continue
            task = task_by_id.get(evidence.task_id)
            if task is None or claim_id not in task.claim_ids:
                continue
            finding = next(
                (
                    item
                    for item in state.findings
                    if item.stance == stance
                    and item.task_id == task.task_id
                    and claim.fact_id in item.fact_ids
                    and evidence_id in item.evidence_ids
                ),
                None,
            )
            if finding is not None:
                matched = (evidence_id, finding.finding_id)
                break
        if matched is None:
            raise RuntimeError(
                f"{stance} verdict basis lacks a qualified Finding -> Evidence "
                f"chain for ImageClaim {claim_id!r}"
            )
        selected_evidence_ids.append(matched[0])
        selected_finding_ids.append(matched[1])
    return (
        list(dict.fromkeys(selected_evidence_ids)),
        list(dict.fromkeys(selected_finding_ids)),
    )


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
