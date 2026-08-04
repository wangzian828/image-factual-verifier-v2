"""Deterministic Coverage and verdict-basis compilation for image-only runs."""

from __future__ import annotations

from typing import Dict, List

from src.orchestrator.evidence_semantics import (
    same_capture_can_support_visual_claim,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.investigation_models import (
    FactCoverage,
    Finding,
    ImageOnlyCoverage,
    ImageOnlyInvestigationState,
    VerdictBasis,
    VisualFact,
)
from src.orchestrator.task_store import (
    MAX_TOOL_ACTIONS,
    latest_evidence_decision,
    remaining_material_routes,
    reconcile_core_verdict_fact,
    refresh_core_evidence_gaps,
    stable_id,
)


def activate_initial_decisive_facts(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Establish a single initial core fact, with a scene fallback.

    Initial target planning can already have established a core.  Otherwise the
    most salient scene proposition becomes the stable fallback; no list of
    parallel decisive facts is ever activated.
    """

    if state.core_verdict_fact_id:
        return [state.core_verdict_fact_id]
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
        and fact.predicate not in {"context_suggested_by_text", "visual_integrity"}
    ]
    candidates.sort(
        key=lambda fact: (
            not same_capture_can_support_visual_claim(fact),
            task_priority[fact.fact_id],
            fact.kind == "relation",
            fact.fact_id,
        )
    )
    for fact in candidates:
        accepted, _ = reconcile_core_verdict_fact(
            state,
            fact,
            allow_initial=True,
        )
        if accepted:
            return [fact.fact_id]
    return []


def audit_coverage(
    state: ImageOnlyInvestigationState,
    *,
    reflection_checkpoint: bool = False,
    decision_checkpoint: bool = False,
) -> ImageOnlyCoverage:
    """Audit only the core fact and its bounded evidence gaps.

    Every accepted tool action calls this function with
    ``decision_checkpoint=True``.  Search leads, task churn, and optional
    attribution metadata do not constitute progress; the state can continue
    only when a core evidence gap changes.
    """

    core_id = state.core_verdict_fact_id
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    core = fact_by_id.get(core_id or "")
    coverage = _build_core_coverage(state, core)
    gaps = refresh_core_evidence_gaps(state)
    required_gaps = [
        gap
        for gap in gaps
        if gap.status != "not_required"
    ]
    complete = bool(
        coverage
        and coverage.status in {"supported", "refuted"}
        and all(gap.status == "resolved" for gap in required_gaps)
    )
    previous = state.coverage_audits[-1] if state.coverage_audits else None
    substantive_gain = _coverage_signature(coverage, gaps) != _audit_signature(
        previous
    )
    prior_low_gain = previous.low_gain_intervals if previous else 0
    low_gain = (
        0 if substantive_gain else prior_low_gain + 1
    ) if decision_checkpoint else prior_low_gain

    open_core_route = _has_executable_core_route(state)
    remaining_routes = (
        remaining_material_routes(state, fact_id=core_id)
        if core_id
        else []
    )
    if complete and coverage and coverage.status == "refuted":
        stop_reason = "verdict_determined"
        reason = "The core factual proposition is directly refuted."
    elif complete and coverage and coverage.status == "supported":
        stop_reason = "verdict_determined"
        reason = "The core factual proposition is supported and all required gaps close."
    elif state.action_count >= MAX_TOOL_ACTIONS:
        stop_reason = "hard_budget_exhausted"
        reason = "The image-only action budget ended before core coverage closed."
    elif core is None or not open_core_route:
        stop_reason = "information_saturated"
        reason = (
            "No executable task remains for an unresolved core evidence gap."
        )
    elif decision_checkpoint and low_gain >= 2:
        if remaining_routes:
            stop_reason = "continue"
            reason = (
                "Recent actions produced no qualified core gain, but distinct "
                "material routes remain: "
                + ", ".join(route.rsplit(":", 1)[0] for route in remaining_routes[:4])
                + "."
            )
        else:
            stop_reason = "information_saturated"
            reason = (
                "Two consecutive action checkpoints produced no qualified "
                "change to the core fact, its winning evidence, or its "
                "evidence gaps, and no distinct material route remains."
            )
    else:
        stop_reason = "continue"
        reason = (
            "Core-fact coverage remains open."
            if core is not None
            else "A core factual proposition has not been established."
        )

    audit = ImageOnlyCoverage(
        audit_id=stable_id(
            "coverage",
            state.brief.case_id,
            state.action_count,
            len(state.coverage_audits) + 1,
        ),
        action_count=state.action_count,
        decisive_fact_ids=[core_id] if core_id else [],
        facts=[coverage] if coverage else [],
        evidence_gaps=list(gaps),
        complete=complete,
        stop_reason=stop_reason,
        reflection_checkpoint=reflection_checkpoint,
        decision_checkpoint=decision_checkpoint,
        substantive_gain=substantive_gain,
        low_gain_intervals=low_gain,
        reason=reason,
    )
    state.coverage_audits.append(audit)
    if stop_reason != "continue":
        state.stop_reason = stop_reason
    return audit


def _build_core_coverage(
    state: ImageOnlyInvestigationState,
    core: VisualFact | None,
) -> FactCoverage | None:
    if core is None:
        return None
    decision = latest_evidence_decision(
        state,
        fact_id=core.fact_id,
    )
    if decision is not None:
        assessment = decision.output.assessment
        status = {
            "supported": "supported",
            "refuted": "refuted",
            "conflicted": "conflicted",
            "insufficient": "unresolved",
        }[assessment]
        support_score = 100.0 if assessment == "supported" else 0.0
        refute_score = 100.0 if assessment == "refuted" else 0.0
        return FactCoverage(
            fact_id=core.fact_id,
            status=status,
            finding_ids=list(decision.finding_ids),
            evidence_ids=list(decision.output.selected_evidence_ids),
            winning_finding_ids=(
                list(decision.finding_ids)
                if assessment in {"supported", "refuted"}
                else []
            ),
            winning_evidence_ids=(
                list(decision.output.selected_evidence_ids)
                if assessment in {"supported", "refuted"}
                else []
            ),
            support_score=support_score,
            refute_score=refute_score,
            conflict_resolution={
                "supported": "support_wins",
                "refuted": "refute_wins",
                "conflicted": "needs_discriminating_evidence",
                "insufficient": "not_applicable",
            }[assessment],
            reason=(
                decision.output.rationale
                if assessment != "insufficient"
                else (
                    decision.output.remaining_gap
                    or decision.output.rationale
                )
            ),
        )

    if core.status in {"blocked", "exhausted"}:
        status = core.status
    else:
        status = "unresolved"
    evidence_ids = [
        item.evidence_id
        for item in state.evidence
        if core.fact_id in item.fact_ids
    ]
    return FactCoverage(
        fact_id=core.fact_id,
        status=status,
        finding_ids=[],
        evidence_ids=evidence_ids,
        winning_finding_ids=[],
        winning_evidence_ids=[],
        support_score=0.0,
        refute_score=0.0,
        conflict_resolution="not_applicable",
        reason=(
            _fact_reason(status, core.statement)
            if status != "unresolved"
            else (
                "Evidence is recorded but awaits a semantic decision checkpoint."
            )
        ),
    )


def _coverage_signature(
    coverage: FactCoverage | None,
    gaps: List[object],
) -> tuple[object, ...]:
    if coverage is None:
        coverage_signature: tuple[object, ...] = ("no_core",)
    else:
        coverage_signature = (
            coverage.fact_id,
            coverage.status,
            coverage.support_score,
            coverage.refute_score,
            tuple(coverage.winning_evidence_ids),
            coverage.conflict_resolution,
        )
    gap_signature = tuple(
        (
            getattr(gap, "kind", ""),
            getattr(gap, "status", ""),
            tuple(getattr(gap, "evidence_ids", []) or []),
        )
        for gap in gaps
    )
    return coverage_signature + (gap_signature,)


def _audit_signature(
    audit: ImageOnlyCoverage | None,
) -> tuple[object, ...] | None:
    if audit is None:
        return None
    coverage = audit.facts[0] if audit.facts else None
    gap_signature = tuple(
        (
            gap.kind,
            gap.status,
            tuple(gap.evidence_ids),
        )
        for gap in audit.evidence_gaps
    )
    if coverage is None:
        return ("no_core", gap_signature)
    return (
        coverage.fact_id,
        coverage.status,
        coverage.support_score,
        coverage.refute_score,
        tuple(coverage.winning_evidence_ids),
        coverage.conflict_resolution,
        gap_signature,
    )


def _has_executable_core_route(
    state: ImageOnlyInvestigationState,
) -> bool:
    core_id = state.core_verdict_fact_id
    return bool(
        core_id
        and remaining_material_routes(
            state,
            fact_id=core_id,
        )
    )


def compile_verdict_basis(
    state: ImageOnlyInvestigationState,
) -> tuple[str, VerdictBasis]:
    audit = (
        state.coverage_audits[-1]
        if state.coverage_audits
        else audit_coverage(state)
    )
    core = audit.facts[0] if audit.facts else None
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    if audit.complete and core and core.status == "refuted":
        verdict = "fake"
        selected = [core]
        mechanism = _infer_mechanism(
            [fact_by_id[core.fact_id].statement]
            if core.fact_id in fact_by_id
            else []
        )
        unresolved: List[str] = []
    elif audit.complete and core and core.status == "supported":
        verdict = "real"
        selected = [core]
        mechanism = None
        unresolved = []
    else:
        verdict = "unverifiable"
        selected = [core] if core else []
        mechanism = None
        unresolved = [
            gap.reason or f"{gap.kind} is {gap.status}"
            for gap in state.evidence_gaps
            if gap.status != "resolved" and gap.status != "not_required"
        ] or [audit.reason]
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
        unresolved_gaps=unresolved[:12],
    )
    state.verdict_basis = basis
    return verdict, basis


def verdict_is_determined(state: ImageOnlyInvestigationState) -> bool:
    """Return whether the one core fact is resolved with all required gaps."""

    core_id = state.core_verdict_fact_id
    core = next(
        (fact for fact in state.facts if fact.fact_id == core_id),
        None,
    )
    if core is None or core.status not in {"supported", "refuted"}:
        return False
    gaps = refresh_core_evidence_gaps(state)
    return all(
        gap.status in {"resolved", "not_required"}
        for gap in gaps
    )


def _fact_reason(status: str, statement: str) -> str:
    if status == "blocked":
        return f"Required evidence access is blocked for: {statement}"
    if status == "exhausted":
        return f"Available evidence routes are exhausted for: {statement}"
    return f"Qualified evidence is still absent for: {statement}"


def _verdict_target(state: ImageOnlyInvestigationState) -> str:
    core_id = state.core_verdict_fact_id
    fact = next(
        (item for item in state.facts if item.fact_id == core_id),
        None,
    )
    if fact is None:
        return (
            "Whether the image's strongest recoverable factual interpretation "
            "is supported by qualified evidence."
        )
    return fact.statement[:1200]


def _infer_mechanism(statements: List[str]) -> str:
    text = " ".join(statements).lower()
    if any(token in text for token in ("place", "location", "center", "building")):
        return "wrong_place"
    if any(token in text for token in ("person", "ship", "identity", "logo")):
        return "wrong_identity"
    if any(token in text for token in ("event", "ceremony", "participate")):
        return "wrong_event"
    return "decisive_fact_refuted"
