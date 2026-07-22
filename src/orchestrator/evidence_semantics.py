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
    evidence_kind = str(evidence_value(evidence, "evidence_kind", ""))
    if evidence_kind == "web_span":
        expected_relation_stance = {
            "support": "supports",
            "refute": "contradicts",
        }.get(stance)
        return bool(
            expected_relation_stance
            and str(evidence_value(evidence, "claim_binding", ""))
            == "source_assertion"
            and str(evidence_value(evidence, "relation_scope", ""))
            == "same_relation"
            and str(evidence_value(evidence, "relation_stance", ""))
            == expected_relation_stance
        )
    if evidence_kind != "reference_comparison":
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


def evidence_is_qualified(evidence: Any) -> bool:
    """Return whether Evidence may enter a material semantic decision."""

    hard_risk_flags = {
        str(item)
        for item in evidence_value(evidence, "risk_flags", []) or []
    } - _NON_BLOCKING_RISK_FLAGS
    return bool(
        str(evidence_value(evidence, "directness", "")) == "direct"
        and str(evidence_value(evidence, "quality", ""))
        in QUALIFIED_EVIDENCE_QUALITIES
        and not hard_risk_flags
    )


def evidence_is_qualified_for_stance(evidence: Any, stance: str) -> bool:
    """Return whether Evidence can support the requested directional chain.

    Qualification and direction are separate checks.  ``evidence_is_qualified``
    validates provenance/quality/risk, while ``evidence_direction_is_coherent``
    validates the recorded stance and any reference-comparison constraints.
    Keeping this small composition here lets reducers, audits, and exporters use
    one semantic gate without duplicating either rule.
    """

    return bool(
        evidence_is_qualified(evidence)
        and evidence_direction_is_coherent(evidence, stance)
    )


def required_assessment_stances(assessment: str) -> frozenset[str]:
    """Return directional Evidence requirements for an internal assessment."""

    return {
        "supported": frozenset({"support"}),
        "refuted": frozenset({"refute"}),
        "conflicted": frozenset({"support", "refute"}),
        "insufficient": frozenset(),
        "unclear": frozenset(),
    }.get(str(assessment).strip().lower(), frozenset())


def reference_only_assessment_is_admissible(
    assessment: str,
    evidence_rows: list[Any],
) -> bool:
    """Preserve v3's conservative standalone reference-comparison boundary."""

    if assessment not in {"supported", "refuted"} or not evidence_rows:
        return True
    qualified_rows = [item for item in evidence_rows if evidence_is_qualified(item)]
    if not qualified_rows or any(
        str(evidence_value(item, "evidence_kind", ""))
        != "reference_comparison"
        for item in qualified_rows
    ):
        return True
    if assessment == "supported":
        return not any(
            evidence_value(item, "edit_evidence_present", False) is True
            for item in qualified_rows
        )
    return any(
        evidence_value(item, "same_capture_or_near_duplicate", False) is True
        and evidence_value(item, "likely_different_original_capture", False)
        is not True
        and evidence_value(item, "edit_evidence_present", False) is True
        for item in qualified_rows
    )
