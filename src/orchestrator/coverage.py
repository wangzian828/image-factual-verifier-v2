"""Deterministic v3 Coverage and reinspect-v2 verdict-basis compilation."""

from __future__ import annotations

from typing import Dict, List

from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.investigation_models import (
    FactCoverage,
    Finding,
    ImageOnlyCoverage,
    ImageOnlyInvestigationState,
    VerdictBasis,
)
from src.orchestrator.task_store import (
    DECISIVE_FACTS_MAX,
    MAX_TOOL_ACTIONS,
    stable_id,
)


def activate_initial_decisive_facts(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Activate up to three central, routed facts before the first Reflection."""

    if state.decisive_fact_ids and not all(
        fact.predicate == "visual_integrity"
        for fact in state.facts
        if fact.fact_id in state.decisive_fact_ids
    ):
        return list(state.decisive_fact_ids)
    task_priority = {
        fact_id: min(
            task.priority
            for task in state.tasks
            if fact_id in task.fact_ids
        )
        for fact_id in {
            fact_id
            for task in state.tasks
            for fact_id in task.fact_ids
        }
    }
    candidates = [
        fact
        for fact in state.facts
        if fact.fact_id in task_priority
        and fact.kind in {"relation", "attribute", "internal_consistency"}
        and fact.predicate != "context_suggested_by_text"
    ]
    candidates.sort(
        key=lambda fact: (
            task_priority[fact.fact_id],
            fact.predicate != "appears_to_depict",
            fact.kind == "relation",
            fact.fact_id,
        )
    )
    scene_candidates = [
        fact for fact in candidates if fact.predicate == "appears_to_depict"
    ]
    selected = scene_candidates[:1] or candidates[:3]
    selected = [
        fact
        for fact in selected
        if fact.fact_id not in state.decisive_fact_ids
    ]
    for fact in selected:
        fact.decision_relevance = "decisive"
        if fact.status == "candidate":
            fact.status = "active"
        state.decisive_fact_ids.append(fact.fact_id)
    return list(state.decisive_fact_ids)


def audit_coverage(
    state: ImageOnlyInvestigationState,
) -> ImageOnlyCoverage:
    decisive_ids = list(state.decisive_fact_ids[:DECISIVE_FACTS_MAX])
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    findings_by_fact: Dict[str, List[Finding]] = {}
    for finding in state.findings:
        for fact_id in finding.fact_ids:
            findings_by_fact.setdefault(fact_id, []).append(finding)
    evidence_ids = {item.evidence_id for item in state.evidence}
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    coverages: List[FactCoverage] = []
    for fact_id in decisive_ids:
        fact = fact_by_id[fact_id]
        findings = findings_by_fact.get(fact_id, [])
        assessment = assess_fact(
            fact,
            findings,
            evidence_by_id,
            all_fact_evidence=[
                item for item in state.evidence if fact_id in item.fact_ids
            ],
        )
        if assessment.status in {"supported", "refuted", "conflicted"}:
            status = assessment.status
        elif fact.status in {"blocked", "exhausted"}:
            status = fact.status
        else:
            status = "unresolved"
        finding_ids = [item.finding_id for item in findings]
        owned_evidence = list(
            dict.fromkeys(
                evidence_id
                for item in findings
                for evidence_id in item.evidence_ids
                if evidence_id in evidence_ids
            )
        )
        coverages.append(
            FactCoverage(
                fact_id=fact_id,
                status=status,
                finding_ids=finding_ids,
                evidence_ids=owned_evidence,
                winning_finding_ids=list(assessment.winning_finding_ids),
                winning_evidence_ids=list(assessment.winning_evidence_ids),
                support_score=assessment.support.score,
                refute_score=assessment.refute.score,
                conflict_resolution=assessment.conflict_resolution,
                reason=(
                    assessment.reason
                    if status in {"supported", "refuted", "conflicted"}
                    else _fact_reason(status, fact.statement)
                ),
            )
        )

    complete = bool(coverages) and all(
        item.status in {"supported", "refuted"}
        for item in coverages
    )
    previous = state.coverage_audits[-1] if state.coverage_audits else None
    previous_signature = (
        {
            (item.fact_id, item.status)
            for item in previous.facts
        }
        if previous
        else set()
    )
    current_signature = {
        (item.fact_id, item.status)
        for item in coverages
    }
    previous_evidence = (
        {
            evidence_id
            for item in previous.facts
            for evidence_id in item.evidence_ids
        }
        if previous
        else set()
    )
    current_evidence = {
        evidence_id
        for item in coverages
        for evidence_id in item.evidence_ids
    }
    substantive_gain = (
        current_signature != previous_signature
        or not current_evidence <= previous_evidence
    )
    low_gain = 0 if substantive_gain else (
        (previous.low_gain_intervals if previous else 0) + 1
    )
    open_high_priority = any(
        task.status in {"active", "pending"}
        and task.priority == 1
        and task.attempt_count == 0
        for task in state.tasks
    )
    determined_verdict = _coverage_verdict(coverages)
    unresolved_non_integrity = any(
        item.status not in {"supported", "refuted"}
        and fact_by_id[item.fact_id].predicate != "visual_integrity"
        for item in coverages
    )
    refuted_non_integrity = any(
        item.status == "refuted"
        and fact_by_id[item.fact_id].predicate != "visual_integrity"
        for item in coverages
    )
    if (
        determined_verdict == "fake"
        and not complete
        and (refuted_non_integrity or not unresolved_non_integrity)
    ):
        stop_reason = "verdict_determined"
        reason = (
            "At least one decisive proposition is refuted after evidence-conflict "
            "adjudication; remaining supporting facts cannot change the fake verdict."
        )
    elif complete:
        stop_reason = "coverage_complete"
        reason = "Every active decisive fact is supported or refuted."
    elif state.action_count >= MAX_TOOL_ACTIONS:
        stop_reason = "hard_budget_exhausted"
        reason = "The image-only tool action budget is exhausted."
    elif low_gain >= 2 and not open_high_priority:
        stop_reason = "information_saturated"
        reason = (
            "Two consecutive Reflection intervals produced no evidence or "
            "decision gain and no unattempted priority-1 task remains."
        )
    else:
        stop_reason = "continue"
        reason = "Decisive-fact coverage is incomplete."
    audit = ImageOnlyCoverage(
        audit_id=stable_id(
            "coverage",
            state.brief.case_id,
            state.action_count,
            len(state.coverage_audits) + 1,
        ),
        action_count=state.action_count,
        decisive_fact_ids=decisive_ids,
        facts=coverages,
        complete=complete,
        stop_reason=stop_reason,
        substantive_gain=substantive_gain,
        low_gain_intervals=low_gain,
        reason=reason,
    )
    state.coverage_audits.append(audit)
    if stop_reason != "continue":
        state.stop_reason = stop_reason
    return audit


def compile_verdict_basis(
    state: ImageOnlyInvestigationState,
) -> tuple[str, VerdictBasis]:
    audit = (
        state.coverage_audits[-1]
        if state.coverage_audits
        else audit_coverage(state)
    )
    refuted = [item for item in audit.facts if item.status == "refuted"]
    supported = [item for item in audit.facts if item.status == "supported"]
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    if refuted:
        verdict = "fake"
        selected = [
            sorted(
                refuted,
                key=lambda item: (
                    -item.refute_score,
                    len(item.winning_evidence_ids),
                    item.fact_id,
                ),
            )[0]
        ]
        mechanism = _infer_mechanism(
            [fact_by_id[item.fact_id].statement for item in selected]
        )
        unresolved: List[str] = []
    elif audit.facts and len(supported) == len(audit.facts):
        verdict = "real"
        selected = supported
        mechanism = None
        unresolved = []
    else:
        verdict = "unverifiable"
        selected = [
            item
            for item in audit.facts
            if item.status
            in {"conflicted", "blocked", "exhausted", "unresolved"}
        ]
        mechanism = None
        unresolved = [item.reason for item in selected]
        if not unresolved:
            unresolved = ["No decisive fact reached a qualified resolution."]
    basis = VerdictBasis(
        verdict_target=_verdict_target(state),
        fact_ids=[item.fact_id for item in selected],
        finding_ids=list(
            dict.fromkeys(
                finding_id
                for item in selected
                for finding_id in (
                    item.winning_finding_ids
                    if item.winning_finding_ids
                    else item.finding_ids
                )
            )
        ),
        evidence_ids=list(
            dict.fromkeys(
                evidence_id
                for item in selected
                for evidence_id in (
                    item.winning_evidence_ids
                    if item.winning_evidence_ids
                    else item.evidence_ids
                )
            )
        ),
        mechanism=mechanism,
        unresolved_gaps=unresolved,
    )
    state.verdict_basis = basis
    return verdict, basis


def verdict_is_determined(state: ImageOnlyInvestigationState) -> bool:
    """Return whether current adjudicated fact state fixes real/fake already."""

    decisive = [
        fact
        for fact in state.facts
        if fact.fact_id in state.decisive_fact_ids
    ]
    if not decisive:
        return False
    if any(fact.status == "refuted" for fact in decisive):
        return True
    return all(fact.status == "supported" for fact in decisive)


def _coverage_verdict(facts: List[FactCoverage]) -> str:
    if any(item.status == "refuted" for item in facts):
        return "fake"
    if facts and all(item.status == "supported" for item in facts):
        return "real"
    return ""


def _fact_reason(status: str, statement: str) -> str:
    if status == "supported":
        return f"Qualified evidence supports: {statement}"
    if status == "refuted":
        return f"Qualified evidence refutes: {statement}"
    if status == "conflicted":
        return f"Qualified evidence conflicts about: {statement}"
    if status == "blocked":
        return f"Required evidence access is blocked for: {statement}"
    if status == "exhausted":
        return f"Available evidence routes are exhausted for: {statement}"
    return f"Decisive evidence is absent for: {statement}"


def _verdict_target(state: ImageOnlyInvestigationState) -> str:
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    statements = [
        fact_by_id[fact_id].statement
        for fact_id in state.decisive_fact_ids
        if fact_id in fact_by_id
    ]
    if not statements:
        return (
            "Whether the image's strongest recoverable factual interpretation "
            "is supported by qualified evidence."
        )
    return " | ".join(statements)[:1200]


def _infer_mechanism(statements: List[str]) -> str:
    text = " ".join(statements).lower()
    if any(token in text for token in ("place", "location", "center", "building")):
        return "wrong_place"
    if any(token in text for token in ("person", "ship", "identity", "logo")):
        return "wrong_identity"
    if any(token in text for token in ("event", "ceremony", "participate")):
        return "wrong_event"
    return "decisive_fact_refuted"
