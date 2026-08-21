"""Deterministic compatibility rules between claim scopes and tool evidence."""
from __future__ import annotations

import re
from typing import Any, Final, Mapping


EXTERNAL_FACT: Final = "external_fact"
IMAGE_AUTHENTICITY: Final = "image_authenticity"
IMAGE_PROVENANCE: Final = "image_provenance"
VISIBLE_CONTENT: Final = "visible_content"

CLAIM_SCOPES: Final = {
    EXTERNAL_FACT,
    IMAGE_AUTHENTICITY,
    IMAGE_PROVENANCE,
    VISIBLE_CONTENT,
}

_ANOMALY_TOOLS: Final = {
    "analyze_visual_anomalies",
    "check_consistency",
}
_REFERENCE_TOOLS: Final = {"compare_with_reference"}
_REGION_TOOLS: Final = {
    "crop_and_inspect",
    "focused_visual_inspection",
    "count_objects",
    "ocr_with_position",
}

VISUAL_OBSERVATION_TOOLS: Final = (
    _ANOMALY_TOOLS | _REFERENCE_TOOLS | _REGION_TOOLS
)

AS_OF_GOAL_MARKER: Final = "As-of constraint: evaluate this claim as of "


def tool_can_decide_claim(tool_name: str, claim_scope: str) -> bool:
    """Return whether a tool observation may support/refute this claim slot.

    Web passages are governed by their own exact-span and stance contracts. This
    policy prevents an image-level observation from deciding an unrelated
    external event claim merely because the model attached it to that question.
    """

    tool = str(tool_name or "").strip()
    scope = str(claim_scope or EXTERNAL_FACT).strip()
    if tool in _ANOMALY_TOOLS:
        return scope == IMAGE_AUTHENTICITY
    if tool in _REFERENCE_TOOLS:
        return scope in {IMAGE_AUTHENTICITY, IMAGE_PROVENANCE, VISIBLE_CONTENT}
    if tool == "crop_and_inspect":
        return scope in {IMAGE_AUTHENTICITY, VISIBLE_CONTENT}
    if tool == "focused_visual_inspection":
        return scope in {IMAGE_AUTHENTICITY, VISIBLE_CONTENT}
    if tool == "ocr_with_position":
        return scope == VISIBLE_CONTENT
    if tool in {"crop_and_search", "count_objects"}:
        return scope in {IMAGE_AUTHENTICITY, VISIBLE_CONTENT}
    if tool in _REGION_TOOLS:
        return scope in {IMAGE_AUTHENTICITY, IMAGE_PROVENANCE, VISIBLE_CONTENT}
    return True


def query_targets_fact_check_answer(value: str) -> bool:
    """Detect queries aimed at a ready-made fact-check verdict rather than sources."""

    text = re.sub(r"[-_]+", " ", str(value or "").casefold())
    return bool(
        re.search(r"\bfact\s*check(?:er|ing|ed)?\b", text)
        or re.search(
            r"\b(?:fake|false|misleading|debunk(?:ed|ing)?|hoax)\s+(?:news|claim|report|image|photo|video|story)\b",
            text,
        )
        or re.search(
            r"\b(?:claim|report|image|photo|video|story)\s+(?:fake|false|misleading|debunked|hoax)\b",
            text,
        )
    )


def query_policy_violation(
    value: str,
    *,
    source_access_policy: Any = None,
) -> str:
    """Return the deterministic reason a search query cannot be accepted."""

    if query_targets_fact_check_answer(value):
        return "fact-check-oriented query"
    if source_access_policy is not None:
        blocked_reference = source_access_policy.blocked_query_reference(value)
        if blocked_reference:
            return f"excluded source reference {blocked_reference!r}"
    return ""


def text_targets_verdict_or_media_origin(value: str) -> bool:
    """Detect route text that presupposes verdict/media-origin classification."""

    text = re.sub(r"[-_]+", " ", str(value or "").casefold())
    return bool(
        query_targets_fact_check_answer(text)
        or re.search(r"\b(?:ai|a i|midjourney|dall\s*e|stable\s+diffusion)\b", text)
        or re.search(
            r"\b(?:ai\s+generated|generated\s+by\s+ai|synthetic|computer\s+generated|digitally\s+generated)\b",
            text,
        )
        or re.search(
            r"\b(?:real\s+or\s+fake|fake\s+or\s+real|authenticity|creation\s+method|generation\s+source)\b",
            text,
        )
        or re.search(
            r"\b(?:original\s+creator|publication\s+context|platform\s+of\s+publication|media\s+background|generation\s+context)\b",
            text,
        )
    )


def neutralize_planning_route_text(value: str) -> str:
    """Remove prohibited AI-generation wording from a factual route.

    ``AI`` can be part of a legitimate on-image title such as ``AI Governance
    Framework``.  Planning routes must not carry that token because it is also
    the marker used to reject generation-method/provenance searches.  Keep the
    remaining distinctive factual words so the route remains searchable.
    """

    text = str(value or "")
    text = re.sub(r"\b(?:ai|a\s+i)\s+for\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ai|a\s+i)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" \t\r\n-:;,")


def goal_has_as_of_constraint(goal: str) -> bool:
    """Return whether a browse goal carries an explicit historical cutoff."""

    return AS_OF_GOAL_MARKER in str(goal or "")


def web_record_is_temporally_eligible(
    record: Mapping[str, Any],
    goal: str,
) -> bool:
    """Require explicit pre-cutoff alignment for historical claim evidence."""

    if not goal_has_as_of_constraint(goal):
        return True
    return str(record.get("temporal_alignment", "")).strip().lower() == (
        "before_or_at_cutoff"
    )
