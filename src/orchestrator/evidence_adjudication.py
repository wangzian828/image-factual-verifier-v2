"""Deterministic evidence qualification and support/refute adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.orchestrator.investigation_models import (
    Finding,
    InvestigationEvidence,
    VisualFact,
)


DECISIVE_SINGLE_EVIDENCE_SCORE = 70.0
CORROBORATING_EVIDENCE_SCORE = 55.0
CONFLICT_WIN_MARGIN = 12.0


@dataclass(frozen=True)
class DirectionAssessment:
    stance: str
    decisive: bool
    score: float
    evidence_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    strongest_binding: str
    strongest_source_class: str


@dataclass(frozen=True)
class FactAssessment:
    status: str
    support: DirectionAssessment
    refute: DirectionAssessment
    conflict_resolution: str
    winning_evidence_ids: tuple[str, ...]
    winning_finding_ids: tuple[str, ...]
    reason: str


_QUALITY_SCORE = {
    "strong": 30.0,
    "moderate": 20.0,
    "weak": 5.0,
}
_SOURCE_SCORE = {
    "official": 35.0,
    "visual": 30.0,
    "news": 25.0,
    "unknown": 10.0,
    "ugc": 0.0,
}
_BINDING_SCORE = {
    "same_capture": 40.0,
    "pixel_observation": 25.0,
    "source_assertion": 15.0,
    "same_subject": 5.0,
    "none": 0.0,
}
_BINDING_RANK = {
    "same_capture": 5,
    "pixel_observation": 4,
    "source_assertion": 3,
    "same_subject": 2,
    "none": 1,
}
_SOURCE_RANK = {
    "official": 5,
    "visual": 5,
    "news": 4,
    "unknown": 2,
    "ugc": 1,
}


def assess_fact(
    fact: VisualFact,
    findings: Sequence[Finding],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    *,
    all_fact_evidence: Iterable[InvestigationEvidence] = (),
) -> FactAssessment:
    """Resolve a fact only after claim binding and source conflict are evaluated."""

    fact_evidence = tuple(all_fact_evidence)
    support = _assess_direction(
        fact,
        "support",
        findings,
        evidence_by_id,
        fact_evidence=fact_evidence,
    )
    refute = _assess_direction(
        fact,
        "refute",
        findings,
        evidence_by_id,
        fact_evidence=fact_evidence,
    )
    if support.decisive and refute.decisive:
        winner = _resolve_conflict(support, refute)
        if winner == "support":
            return FactAssessment(
                status="supported",
                support=support,
                refute=refute,
                conflict_resolution="support_wins",
                winning_evidence_ids=support.evidence_ids,
                winning_finding_ids=support.finding_ids,
                reason=(
                    "Qualified support and refute evidence were both present; "
                    "support won on claim binding, source originality/directness, "
                    "independence, and evidence strength."
                ),
            )
        if winner == "refute":
            return FactAssessment(
                status="refuted",
                support=support,
                refute=refute,
                conflict_resolution="refute_wins",
                winning_evidence_ids=refute.evidence_ids,
                winning_finding_ids=refute.finding_ids,
                reason=(
                    "Qualified support and refute evidence were both present; "
                    "refutation won on claim binding, source originality/directness, "
                    "independence, and evidence strength."
                ),
            )
        return FactAssessment(
            status="conflicted",
            support=support,
            refute=refute,
            conflict_resolution="needs_discriminating_evidence",
            winning_evidence_ids=(),
            winning_finding_ids=(),
            reason=(
                "Qualified support and refute evidence remain materially tied. "
                "A more direct, original, independent, or better scene-bound source "
                "is required before this fact can be resolved."
            ),
        )
    if refute.decisive:
        return FactAssessment(
            status="refuted",
            support=support,
            refute=refute,
            conflict_resolution="not_applicable",
            winning_evidence_ids=refute.evidence_ids,
            winning_finding_ids=refute.finding_ids,
            reason="Qualified evidence directly refutes the decisive proposition.",
        )
    if support.decisive:
        return FactAssessment(
            status="supported",
            support=support,
            refute=refute,
            conflict_resolution="not_applicable",
            winning_evidence_ids=support.evidence_ids,
            winning_finding_ids=support.finding_ids,
            reason="Qualified evidence directly supports the decisive proposition.",
        )
    return FactAssessment(
        status="active",
        support=support,
        refute=refute,
        conflict_resolution="not_applicable",
        winning_evidence_ids=(),
        winning_finding_ids=(),
        reason=(
            "Available evidence is not sufficiently direct, independent, or bound "
            "to the exact visual proposition."
        ),
    )


def _assess_direction(
    fact: VisualFact,
    stance: str,
    findings: Sequence[Finding],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    *,
    fact_evidence: Sequence[InvestigationEvidence],
) -> DirectionAssessment:
    rows: list[tuple[float, InvestigationEvidence, str]] = []
    for finding in findings:
        if finding.stance != stance:
            continue
        for evidence_id in finding.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.stance != stance:
                continue
            score = _evidence_score(
                fact,
                evidence,
                stance=stance,
                fact_evidence=fact_evidence,
            )
            if score > 0:
                rows.append((score, evidence, finding.finding_id))
    rows.sort(
        key=lambda item: (
            -item[0],
            -_BINDING_RANK.get(item[1].claim_binding, 0),
            -_SOURCE_RANK.get(item[1].source_class, 0),
            item[1].evidence_id,
        )
    )
    if not rows:
        return DirectionAssessment(
            stance=stance,
            decisive=False,
            score=0.0,
            evidence_ids=(),
            finding_ids=(),
            source_families=(),
            strongest_binding="none",
            strongest_source_class="unknown",
        )

    selected: list[tuple[float, InvestigationEvidence, str]] = []
    scene_support_pair = (
        _scene_support_pair(rows)
        if stance == "support" and fact.predicate == "appears_to_depict"
        else []
    )
    official_exact_capture = (
        rows[0]
        if (
            stance == "support"
            and fact.predicate == "appears_to_depict"
            and rows[0][1].claim_binding == "same_capture"
            and rows[0][1].source_class == "official"
            and rows[0][0] >= 90.0
        )
        else None
    )
    scene_support_requires_pair = (
        stance == "support" and fact.predicate == "appears_to_depict"
    )
    if official_exact_capture is not None:
        selected = [official_exact_capture]
    elif scene_support_pair:
        selected = scene_support_pair
    elif (
        not scene_support_requires_pair
        and rows[0][0] >= DECISIVE_SINGLE_EVIDENCE_SCORE
    ):
        selected = [rows[0]]
    elif not scene_support_requires_pair:
        families: set[str] = set()
        for row in rows:
            score, evidence, _ = row
            if score < CORROBORATING_EVIDENCE_SCORE:
                continue
            if evidence.source_family in families:
                continue
            families.add(evidence.source_family)
            selected.append(row)
            if len(selected) == 2:
                break
    decisive = bool(selected) and (
        official_exact_capture is not None
        or bool(scene_support_pair)
        or selected[0][0] >= DECISIVE_SINGLE_EVIDENCE_SCORE
        or len(selected) >= 2
    )
    selected_rows = selected if decisive else rows[:1]
    selected_evidence = tuple(
        dict.fromkeys(row[1].evidence_id for row in selected_rows)
    )
    selected_findings = tuple(
        dict.fromkeys(row[2] for row in selected_rows)
    )
    families = tuple(
        dict.fromkeys(row[1].source_family for row in selected_rows)
    )
    score = rows[0][0]
    if scene_support_pair:
        score = scene_support_pair[0][0] + (
            0.25 * scene_support_pair[1][0]
        )
    if decisive and len(families) > 1:
        score += min(10.0, 5.0 * (len(families) - 1))
    return DirectionAssessment(
        stance=stance,
        decisive=decisive,
        score=round(score, 3),
        evidence_ids=selected_evidence,
        finding_ids=selected_findings,
        source_families=families,
        strongest_binding=rows[0][1].claim_binding,
        strongest_source_class=rows[0][1].source_class,
    )


def _scene_support_pair(
    rows: Sequence[tuple[float, InvestigationEvidence, str]],
) -> list[tuple[float, InvestigationEvidence, str]]:
    captures = [
        row
        for row in rows
        if row[1].claim_binding == "same_capture"
        and row[0] >= 65.0
    ]
    assertions = [
        row
        for row in rows
        if row[1].claim_binding == "source_assertion"
        and row[0] >= 45.0
    ]
    if not captures or not assertions:
        return []
    return [captures[0], assertions[0]]


def _evidence_score(
    fact: VisualFact,
    evidence: InvestigationEvidence,
    *,
    stance: str,
    fact_evidence: Sequence[InvestigationEvidence],
) -> float:
    if evidence.risk_flags:
        return 0.0
    score = (
        _QUALITY_SCORE.get(evidence.quality, 0.0)
        + _SOURCE_SCORE.get(evidence.source_class, 0.0)
        + _BINDING_SCORE.get(evidence.claim_binding, 0.0)
    )
    if evidence.directness != "direct":
        score -= 25.0
    if evidence.temporal_alignment.casefold() in {
        "after_cutoff",
        "outside_target_window",
        "misaligned",
    }:
        score -= 35.0
    if evidence.confidence is not None and evidence.confidence < 0.6:
        score -= 15.0

    if stance == "support" and fact.predicate == "appears_to_depict":
        if evidence.claim_binding != "same_capture":
            score = min(score, CORROBORATING_EVIDENCE_SCORE - 1.0)
    elif stance == "support" and fact.predicate in {"visible_in", "reads"}:
        has_pixel_bridge = any(
            item.claim_binding in {"pixel_observation", "same_capture"}
            and not item.risk_flags
            for item in fact_evidence
        )
        if evidence.claim_binding not in {
            "pixel_observation",
            "same_capture",
        }:
            if has_pixel_bridge:
                score += 15.0
            else:
                score = min(score, CORROBORATING_EVIDENCE_SCORE - 1.0)
    return max(0.0, round(score, 3))


def _resolve_conflict(
    support: DirectionAssessment,
    refute: DirectionAssessment,
) -> str:
    score_delta = support.score - refute.score
    if abs(score_delta) >= CONFLICT_WIN_MARGIN:
        return "support" if score_delta > 0 else "refute"

    binding_delta = (
        _BINDING_RANK.get(support.strongest_binding, 0)
        - _BINDING_RANK.get(refute.strongest_binding, 0)
    )
    if binding_delta >= 2 and score_delta >= -5.0:
        return "support"
    if binding_delta <= -2 and score_delta <= 5.0:
        return "refute"

    source_delta = (
        _SOURCE_RANK.get(support.strongest_source_class, 0)
        - _SOURCE_RANK.get(refute.strongest_source_class, 0)
    )
    if source_delta >= 2 and score_delta >= -5.0:
        return "support"
    if source_delta <= -2 and score_delta <= 5.0:
        return "refute"

    family_delta = len(support.source_families) - len(refute.source_families)
    if abs(family_delta) >= 2:
        return "support" if family_delta > 0 else "refute"
    return ""
