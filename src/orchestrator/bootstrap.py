"""Deterministic image-only bootstrap from validated perception and OCR."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, List

from src.orchestrator.investigation_models import (
    BootstrapInvestigation,
    FactOrigin,
    InvestigationBrief,
    ResearchTask,
    RetrievalAnchor,
    VisualEntity,
    VisualFact,
)
from src.orchestrator.state import ImageOnlyRuntimeCase, PerceptionReport, TextRegion


MIN_OCR_FACT_CONFIDENCE = 0.5


def _id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _clean_text(value: str, *, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _quad_region(region: TextRegion) -> List[float] | None:
    points = region.bbox_quad
    if not points or not all(
        isinstance(point, list) and len(point) == 2 for point in points
    ):
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError):
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def _text_score(anchor: RetrievalAnchor) -> tuple[float, int, str]:
    compact = "".join(char for char in anchor.value if char.isalnum())
    words = anchor.value.split()
    score = anchor.confidence
    score += min(len(compact) / 30.0, 1.0)
    score += 0.25 if len(words) >= 2 else 0.0
    score += 0.2 if any(char.isdigit() for char in anchor.value) else 0.0
    return score, len(compact), anchor.anchor_id


def _entity_score(
    anchor: RetrievalAnchor,
    entity_by_id: dict[str, VisualEntity],
) -> tuple[float, int, str]:
    compact = "".join(char for char in anchor.value if char.isalnum())
    words = anchor.value.split()
    entity = entity_by_id.get(anchor.entity_id or "")
    score = anchor.confidence
    score += min(len(compact) / 30.0, 1.0)
    score += 0.15 if len(words) >= 2 else 0.0
    score += 0.1 if anchor.kind == "logo" else 0.0
    if entity is not None and entity.entity_type == "text_region":
        score -= 1.0
    return score, len(compact), anchor.anchor_id


def _usable_ocr_region(region: TextRegion) -> bool:
    text = _clean_text(region.text)
    if not text or float(region.confidence) < MIN_OCR_FACT_CONFIDENCE:
        return False
    compact = "".join(char for char in text if char.isalnum())
    if len(compact) >= 3:
        return True
    return len(compact) >= 2 and any(ord(char) > 127 for char in compact)


def _unique_by_value(
    anchors: Iterable[RetrievalAnchor],
) -> List[RetrievalAnchor]:
    result: List[RetrievalAnchor] = []
    seen: set[str] = set()
    for anchor in anchors:
        key = anchor.value.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(anchor)
    return result


def build_bootstrap_investigation(
    case: ImageOnlyRuntimeCase,
    perception: PerceptionReport,
) -> BootstrapInvestigation:
    """Create low-commitment, image-grounded facts and initial tasks."""

    brief = InvestigationBrief(
        brief_id=_id("brief", case.case_id, case.image_sha256),
        case_id=case.case_id,
    )
    image_entity = VisualEntity(
        entity_id=_id("ve", case.case_id, "input-image"),
        name="input image",
        entity_type="image",
        origin="input_image",
        origin_ids=[case.case_id],
        region=[0.0, 0.0, 1.0, 1.0],
        confidence=1.0,
    )
    entities: List[VisualEntity] = [image_entity]
    entity_by_id: dict[str, VisualEntity] = {
        image_entity.entity_id: image_entity
    }
    facts: List[VisualFact] = []
    anchors: List[RetrievalAnchor] = []
    fact_by_anchor: dict[str, VisualFact] = {}
    relation_fact_by_anchor: dict[str, VisualFact] = {}

    scene = _clean_text(perception.scene_description, limit=500)
    if scene:
        scene_fact_id = _id("vf", case.case_id, "scene", scene)
        scene_anchor_id = _id("anchor", case.case_id, "scene", scene)
        anchors.append(
            RetrievalAnchor(
                anchor_id=scene_anchor_id,
                kind="scene_pattern",
                value=scene,
                entity_id=image_entity.entity_id,
                region=image_entity.region,
                confidence=0.5,
            )
        )
        scene_fact = VisualFact(
            fact_id=scene_fact_id,
            kind="attribute",
            statement=f"The image appears to depict: {scene}",
            subject_entity_id=image_entity.entity_id,
            predicate="appears_to_depict",
            basis_ids=[image_entity.entity_id, scene_anchor_id],
            origin=FactOrigin(
                type="input_image",
                origin_ids=[image_entity.entity_id, scene_anchor_id],
            ),
        )
        facts.append(scene_fact)
        fact_by_anchor[scene_anchor_id] = scene_fact

    for index, item in enumerate(perception.entities[:8]):
        name = _clean_text(item.name)
        if not name:
            continue
        region = [round(float(value), 6) for value in item.bbox] if item.bbox else None
        entity_id = _id(
            "ve",
            case.case_id,
            "perception",
            index,
            item.entity_type,
            name,
            region,
        )
        entity = VisualEntity(
            entity_id=entity_id,
            name=name,
            entity_type=_clean_text(item.entity_type, limit=100) or "object",
            origin="input_image",
            origin_ids=[case.case_id],
            region=region,
            confidence=float(item.confidence),
        )
        entities.append(entity)
        entity_by_id[entity.entity_id] = entity
        anchor_kind = "logo" if entity.entity_type == "logo" else "entity"
        anchor = RetrievalAnchor(
            anchor_id=_id("anchor", entity_id, name),
            kind=anchor_kind,
            value=name,
            entity_id=entity_id,
            region=region,
            confidence=entity.confidence,
        )
        anchors.append(anchor)
        visible_fact = VisualFact(
            fact_id=_id("vf", entity_id, "visible"),
            kind="attribute",
            statement=(
                f"The image visibly contains {name} "
                f"(visual type: {entity.entity_type})."
            ),
            subject_entity_id=entity_id,
            predicate="visible_in",
            object_entity_id=image_entity.entity_id,
            basis_ids=[entity_id, anchor.anchor_id],
            origin=FactOrigin(
                type="input_image",
                origin_ids=[entity_id, anchor.anchor_id],
            ),
        )
        facts.append(visible_fact)
        fact_by_anchor[anchor.anchor_id] = visible_fact

    text_anchors: List[RetrievalAnchor] = []
    for index, region in enumerate(perception.text_regions[:16]):
        if not _usable_ocr_region(region):
            continue
        text = _clean_text(region.text)
        bbox = _quad_region(region)
        entity_id = _id("ve", case.case_id, "ocr", index, text, bbox)
        entity = VisualEntity(
            entity_id=entity_id,
            name=text,
            entity_type="text_region",
            origin="ocr",
            origin_ids=[case.case_id],
            region=bbox,
            confidence=float(region.confidence),
        )
        entities.append(entity)
        entity_by_id[entity.entity_id] = entity
        anchor = RetrievalAnchor(
            anchor_id=_id("anchor", entity_id, "text", text),
            kind="text",
            value=text,
            entity_id=entity_id,
            region=bbox,
            confidence=float(region.confidence),
        )
        anchors.append(anchor)
        text_anchors.append(anchor)
        text_fact = VisualFact(
            fact_id=_id("vf", entity_id, "reads", text),
            kind="text_claim",
            statement=f'The image visibly contains the text "{text}".',
            subject_entity_id=entity_id,
            predicate="reads",
            basis_ids=[entity_id, anchor.anchor_id],
            origin=FactOrigin(
                type="ocr",
                origin_ids=[entity_id, anchor.anchor_id],
            ),
        )
        relation_fact = VisualFact(
            fact_id=_id("vf", image_entity.entity_id, "context-text", entity_id),
            kind="relation",
            statement=(
                f'The visible text "{text}" is a candidate anchor for identifying '
                "the depicted entity, place, or event."
            ),
            subject_entity_id=image_entity.entity_id,
            predicate="context_suggested_by_text",
            object_entity_id=entity_id,
            basis_ids=[image_entity.entity_id, entity_id, anchor.anchor_id],
            origin=FactOrigin(
                type="ocr",
                origin_ids=[entity_id, anchor.anchor_id],
            ),
        )
        facts.extend([text_fact, relation_fact])
        fact_by_anchor[anchor.anchor_id] = text_fact
        relation_fact_by_anchor[anchor.anchor_id] = relation_fact

    anchors = _unique_by_value(anchors)
    tasks: List[ResearchTask] = []

    scene_fact = next(
        (fact for fact in facts if fact.predicate == "appears_to_depict"),
        None,
    )
    if scene_fact is not None:
        tasks.append(
            ResearchTask(
                task_id=_id("task", case.case_id, "provenance"),
                fact_ids=[scene_fact.fact_id],
                question=(
                    "What is the earliest verifiable public context or source for "
                    "this image or a close visual match?"
                ),
                purpose=(
                    "Establish image provenance before accepting any event, place, "
                    "identity, or authenticity attribution."
                ),
                priority=1,
                origin_ids=list(scene_fact.basis_ids),
                suggested_tools=["reverse_image_search", "text_search", "visit"],
                suggested_queries=[],
            )
        )

    ranked_text = sorted(
        _unique_by_value(text_anchors),
        key=_text_score,
        reverse=True,
    )
    selected_text = ranked_text[:3]
    if selected_text:
        text_facts: List[VisualFact] = []
        relation_facts: List[VisualFact] = []
        for anchor in selected_text:
            fact = fact_by_anchor.get(anchor.anchor_id)
            relation_fact = relation_fact_by_anchor.get(anchor.anchor_id)
            if fact is not None:
                text_facts.append(fact)
            if relation_fact is not None:
                relation_facts.append(relation_fact)
        quoted_values = [
            anchor.value.replace('"', "'")
            for anchor in selected_text
        ]
        fact_ids = [
            fact.fact_id
            for fact in [*text_facts, *relation_facts]
        ][:6]
        origin_ids = list(
            dict.fromkeys(
                origin_id
                for fact in [*text_facts, *relation_facts]
                for origin_id in fact.basis_ids
            )
        )[:12]
        combined_query = " ".join(quoted_values)
        suggested_queries = [combined_query]
        suggested_queries.extend(
            f'"{value}"' for value in quoted_values[:2]
        )
        tasks.append(
            ResearchTask(
                task_id=_id("task", case.case_id, "joint-text-context", fact_ids),
                fact_ids=fact_ids,
                question=(
                    "What real-world entity, place, or event is jointly indicated "
                    "by the visible text anchors "
                    + ", ".join(f'"{value}"' for value in quoted_values)
                    + ", and does reliable source context match the depicted scene?"
                ),
                purpose=(
                    "Investigate multiple pixel-grounded text anchors together so "
                    "partial OCR strings do not become isolated factual targets."
                ),
                priority=1,
                origin_ids=origin_ids,
                suggested_tools=["text_search", "reverse_image_search", "visit"],
                suggested_queries=suggested_queries[:3],
            )
        )

    entity_anchors = sorted(
        (
            anchor
            for anchor in anchors
            if anchor.kind in {"logo", "entity"}
            and len("".join(char for char in anchor.value if char.isalnum())) >= 3
        ),
        key=lambda item: _entity_score(item, entity_by_id),
        reverse=True,
    )
    for anchor in entity_anchors:
        if len(tasks) >= 4:
            break
        fact = fact_by_anchor.get(anchor.anchor_id)
        if fact is None:
            continue
        quoted = anchor.value.replace('"', "'")
        tasks.append(
            ResearchTask(
                task_id=_id("task", fact.fact_id, "entity-context"),
                fact_ids=[fact.fact_id],
                question=(
                    f'What verifiable real-world identity or context corresponds to '
                    f'the visible {anchor.kind} "{quoted}"?'
                ),
                purpose=(
                    "Resolve a salient visual identity without inferring facts that "
                    "are not present in the pixels."
                ),
                priority=2,
                origin_ids=list(fact.basis_ids),
                suggested_tools=["reverse_image_search", "text_search", "visit"],
                suggested_queries=[quoted],
            )
        )

    if not tasks and facts:
        fact = facts[0]
        tasks.append(
            ResearchTask(
                task_id=_id("task", fact.fact_id, "context"),
                fact_ids=[fact.fact_id],
                question=(
                    "What reliable external context can identify the visible scene "
                    "without assuming an event, place, or person?"
                ),
                purpose="Turn an image-grounded observation into a checkable target.",
                priority=1,
                origin_ids=list(fact.basis_ids),
                suggested_tools=["reverse_image_search", "text_search", "visit"],
            )
        )

    return BootstrapInvestigation(
        brief=brief,
        entities=entities,
        facts=facts,
        tasks=tasks[:4],
        retrieval_anchors=anchors,
        findings=[],
    )
