"""Shared qualification rules for Evidence used by semantic decisions."""

from __future__ import annotations

from typing import Any, Mapping


QUALIFIED_EVIDENCE_QUALITIES = frozenset({"strong", "moderate"})
_NON_BLOCKING_RISK_FLAGS = frozenset({"user_generated_content"})


def evidence_value(evidence: Any, field: str, default: Any = None) -> Any:
    if isinstance(evidence, Mapping):
        return evidence.get(field, default)
    return getattr(evidence, field, default)


def evidence_direction_is_coherent(evidence: Any, stance: str) -> bool:
    """Return whether recorded Evidence metadata can carry ``stance``."""

    if str(evidence_value(evidence, "stance", "")) != stance:
        return False
    if str(evidence_value(evidence, "evidence_kind", "")) != (
        "reference_comparison"
    ):
        return True

    same_capture = evidence_value(
        evidence,
        "same_capture_or_near_duplicate",
        False,
    ) is True
    different_capture = evidence_value(
        evidence,
        "likely_different_original_capture",
        False,
    ) is True
    edit_present = evidence_value(
        evidence,
        "edit_evidence_present",
        False,
    ) is True
    if stance == "refute":
        return same_capture and not different_capture and edit_present
    if stance == "support":
        return same_capture and not different_capture and not edit_present
    return False


def evidence_is_qualified_for_stance(evidence: Any, stance: str) -> bool:
    """Return whether Evidence may support a terminal semantic direction."""

    hard_risk_flags = {
        str(item)
        for item in evidence_value(evidence, "risk_flags", []) or []
    } - _NON_BLOCKING_RISK_FLAGS
    return bool(
        stance in {"support", "refute"}
        and str(evidence_value(evidence, "directness", "")) == "direct"
        and str(evidence_value(evidence, "quality", ""))
        in QUALIFIED_EVIDENCE_QUALITIES
        and not hard_risk_flags
        and evidence_direction_is_coherent(evidence, stance)
    )


def required_assessment_stances(assessment: str) -> frozenset[str]:
    return {
        "supported": frozenset({"support"}),
        "refuted": frozenset({"refute"}),
        "conflicted": frozenset({"support", "refute"}),
        "insufficient": frozenset(),
    }.get(str(assessment), frozenset())
