"""Deterministic compatibility rules between claim scopes and tool evidence."""
from __future__ import annotations

import re
from typing import Final


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
    "count_objects",
    "ocr_with_position",
}

VISUAL_OBSERVATION_TOOLS: Final = (
    _ANOMALY_TOOLS | _REFERENCE_TOOLS | _REGION_TOOLS
)


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
    if tool in _REGION_TOOLS:
        return scope in {IMAGE_AUTHENTICITY, IMAGE_PROVENANCE, VISIBLE_CONTENT}
    return True


def query_targets_fact_check_answer(value: str) -> bool:
    """Detect queries aimed at a ready-made fact-check verdict rather than sources."""

    text = re.sub(r"[-_]+", " ", str(value or "").casefold())
    return bool(re.search(r"\bfact\s*check(?:er|ing|ed)?\b", text))
