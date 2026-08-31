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
    RetrievalAnchor,
    ResearchTask,
    VisualEntity,
    VisualFact,
    VisualBootstrap,
)
from src.orchestrator.state import ImageOnlyRuntimeCase, PerceptionReport, TextRegion


MIN_OCR_FACT_CONFIDENCE = 0.5


def _id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _clean_text(value: str, *, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _relation_entity_id(
    label: str,
    entities: Iterable[VisualEntity],
    *,
    image_entity_id: str,
) -> str | None:
    """Resolve a perception relation label to one visible entity deterministically."""

    normalized = _clean_text(label, limit=160).casefold()
    if not normalized:
        return None
    if normalized in {"image", "the image", "scene", "the scene", "background"}:
        return image_entity_id

    rows = list(entities)
    exact = [
        item
        for item in rows
        if item.name.casefold() == normalized
    ]
    if exact:
        return exact[0].entity_id

    candidates: list[tuple[int, int, str, VisualEntity]] = []
    for item in rows:
        name = item.name.casefold()
        if normalized in name or name in normalized:
            candidates.append(
                (min(len(name), len(normalized)), len(name), item.entity_id, item)
            )
    if candidates:
        candidates.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
        return candidates[0][3].entity_id

    label_tokens = {
        token
        for token in re.findall(r"[\w\u3400-\u9fff]+", normalized)
        if len(token) >= 3
    }
    if not label_tokens:
        return None
    overlaps: list[tuple[int, int, str, VisualEntity]] = []
    for item in rows:
        entity_tokens = {
            token
            for token in re.findall(r"[\w\u3400-\u9fff]+", item.name.casefold())
            if len(token) >= 3
        }
        overlap = len(label_tokens & entity_tokens)
        if overlap:
            overlaps.append((overlap, len(entity_tokens), item.entity_id, item))
    if not overlaps:
        return None
    overlaps.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
    return overlaps[0][3].entity_id


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


def build_visual_bootstrap(
    case: ImageOnlyRuntimeCase,
    perception: PerceptionReport,
) -> VisualBootstrap:
    """Materialize only image/OCR-grounded state for unified ReAct.

    This deliberately creates no ResearchTask. The unified ReAct loop creates
    its first target, route, and task atomically with the first real
    investigation action.
    """

    media_type = str(perception.image_type or "photo").strip().lower()
    if media_type not in {
        "photo",
        "screenshot",
        "document",
        "illustration",
        "meme",
        "unknown",
    }:
        media_type = "unknown"
    brief = InvestigationBrief(
        brief_id=_id("brief", case.case_id, case.image_sha256),
        case_id=case.case_id,
        media_type=media_type,
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

    scene = _clean_text(perception.scene_description, limit=1000)
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
            statement=f"The visible scene is: {scene}",
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

    for index, item in enumerate(perception.entities[:16]):
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
        attributes = {
            _clean_text(key, limit=80): _clean_text(value, limit=260)
            for key, value in item.attributes.items()
            if _clean_text(key, limit=80) and _clean_text(value, limit=260)
        }
        entity = VisualEntity(
            entity_id=entity_id,
            name=name,
            entity_type=_clean_text(item.entity_type, limit=100) or "object",
            origin="input_image",
            origin_ids=[case.case_id],
            region=region,
            confidence=float(item.confidence),
            attributes=dict(list(attributes.items())[:8]),
            text_role=item.text_role,
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
        attribute_text = "; ".join(
            f"{key}: {value}"
            for key, value in list(attributes.items())[:8]
        )
        statement = (
            f"The image visibly contains {name} "
            f"(visual type: {entity.entity_type})."
        )
        if attribute_text:
            statement += f" Visible attributes: {attribute_text}."
        visible_fact = VisualFact(
            fact_id=_id("vf", entity_id, "visible"),
            kind="attribute",
            statement=_clean_text(statement, limit=1150),
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
            text_role=region.text_role,
            attributes=(
                {"text_role": region.text_role}
                if region.text_role != "unknown"
                else {}
            ),
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
            statement=(
                f'The image visibly contains the text "{text}".'
                + (
                    f" Its visible layout role is {region.text_role}."
                    if region.text_role != "unknown"
                    else ""
                )
            ),
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

    # Relation rows are kept as image-grounded facts only after all visible
    # entities and OCR text regions have been materialized.  This lets a
    # perception label such as "woman" resolve to a richer entity name such as
    # "woman at podium" without inventing a new object.
    for index, relation in enumerate(perception.relations[:16]):
        subject_entity_id = _relation_entity_id(
            relation.subject,
            entities,
            image_entity_id=image_entity.entity_id,
        )
        if not subject_entity_id:
            continue
        object_entity_id = _relation_entity_id(
            relation.object,
            entities,
            image_entity_id=image_entity.entity_id,
        )
        predicate = _clean_text(relation.predicate, limit=100)
        description = _clean_text(relation.description, limit=1050)
        if not predicate or not description:
            continue
        basis_ids = [subject_entity_id]
        if object_entity_id and object_entity_id != subject_entity_id:
            basis_ids.append(object_entity_id)
        facts.append(
            VisualFact(
                fact_id=_id(
                    "vf",
                    case.case_id,
                    "relation",
                    index,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    description,
                ),
                kind="relation",
                statement=f"Visible relation: {description}",
                subject_entity_id=subject_entity_id,
                predicate=predicate,
                object_entity_id=object_entity_id,
                basis_ids=basis_ids,
                origin=FactOrigin(
                    type="input_image",
                    origin_ids=basis_ids,
                ),
            )
        )

    return VisualBootstrap(
        brief=brief,
        entities=entities,
        facts=facts[:72],
        retrieval_anchors=anchors,
        notable_details=list(perception.notable_details[:16]),
        uncertainties=list(perception.uncertainties[:8]),
    )


def build_bootstrap_investigation(
    case: ImageOnlyRuntimeCase,
    perception: PerceptionReport,
) -> BootstrapInvestigation:
    """Build the archived bootstrap shape for legacy replay/tests only.

    The active runtime starts from :func:`build_visual_bootstrap` and lets the
    unified ReAct policy choose its first action.  This function deliberately
    stays outside that path; it only reconstructs the old speculative task
    container needed to inspect pre-refactor traces.
    """

    visual = build_visual_bootstrap(case, perception)
    facts = list(visual.facts)
    anchors = list(visual.retrieval_anchors)
    entities = list(visual.entities)
    entity_by_id = {item.entity_id: item for item in entities}

    def fact_for_anchor(anchor_id: str, predicate: str) -> VisualFact | None:
        return next(
            (
                fact
                for fact in facts
                if fact.predicate == predicate and anchor_id in fact.basis_ids
            ),
            None,
        )

    tasks: List[ResearchTask] = []
    selected_text = sorted(
        _unique_by_value(
            anchor for anchor in anchors if anchor.kind == "text"
        ),
        key=_text_score,
        reverse=True,
    )[:3]
    quoted_values = [anchor.value.replace('"', "'") for anchor in selected_text]
    suggested_text_queries: List[str] = []
    if quoted_values:
        suggested_text_queries.append(" ".join(quoted_values))
        suggested_text_queries.extend(
            f'"{value}"' for value in quoted_values[:2]
        )

    scene_fact = next(
        (fact for fact in facts if fact.predicate == "appears_to_depict"),
        None,
    )
    if scene_fact is not None:
        tasks.append(
            ResearchTask(
                task_id=_id("task", case.case_id, "scene-relation"),
                fact_ids=[scene_fact.fact_id],
                question=(
                    "Does reliable external evidence support or refute this "
                    f"image-grounded scene proposition: {scene_fact.statement}"
                ),
                purpose=(
                    "Investigate the central visible subject, place, event, or "
                    "scene relation. Source metadata may guide retrieval but must "
                    "not replace the depicted-world question."
                ),
                priority=1,
                origin_ids=list(scene_fact.basis_ids),
                suggested_tools=[
                    "reverse_image_search",
                    "compare_with_reference",
                    "text_search",
                    "visit",
                ],
                suggested_queries=suggested_text_queries[:3],
            )
        )

    if selected_text and scene_fact is None:
        text_facts = [
            fact
            for anchor in selected_text
            if (fact := fact_for_anchor(anchor.anchor_id, "reads")) is not None
        ]
        relation_facts = [
            fact
            for anchor in selected_text
            if (
                fact := fact_for_anchor(
                    anchor.anchor_id,
                    "context_suggested_by_text",
                )
            )
            is not None
        ]
        task_facts = [*text_facts, *relation_facts]
        fact_ids = [fact.fact_id for fact in task_facts][:6]
        origin_ids = list(
            dict.fromkeys(
                origin_id
                for fact in task_facts
                for origin_id in fact.basis_ids
            )
        )[:12]
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
                suggested_tools=[
                    "text_search",
                    "reverse_image_search",
                    "compare_with_reference",
                    "visit",
                ],
                suggested_queries=suggested_text_queries[:3],
            )
        )

    entity_anchors = sorted(
        (
            anchor
            for anchor in _unique_by_value(anchors)
            if anchor.kind in {"logo", "entity"}
            and len(
                "".join(char for char in anchor.value if char.isalnum())
            )
            >= 3
        ),
        key=lambda item: _entity_score(item, entity_by_id),
        reverse=True,
    )
    for anchor in entity_anchors:
        if len(tasks) >= 4:
            break
        fact = fact_for_anchor(anchor.anchor_id, "visible_in")
        if fact is None:
            continue
        quoted = anchor.value.replace('"', "'")
        tasks.append(
            ResearchTask(
                task_id=_id("task", fact.fact_id, "entity-context"),
                fact_ids=[fact.fact_id],
                question=(
                    "What verifiable real-world identity or context corresponds "
                    f'to the visible {anchor.kind} "{quoted}"?'
                ),
                purpose=(
                    "Resolve a salient visual identity without inferring facts "
                    "that are not present in the pixels."
                ),
                priority=2,
                origin_ids=list(fact.basis_ids),
                suggested_tools=[
                    "reverse_image_search",
                    "compare_with_reference",
                    "text_search",
                    "visit",
                ],
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
                    "What reliable external context can identify the visible "
                    "scene without assuming an event, place, or person?"
                ),
                purpose=(
                    "Turn an image-grounded observation into a checkable target."
                ),
                priority=1,
                origin_ids=list(fact.basis_ids),
                suggested_tools=[
                    "reverse_image_search",
                    "compare_with_reference",
                    "text_search",
                    "visit",
                ],
            )
        )

    return BootstrapInvestigation(
        brief=visual.brief,
        entities=entities[:32],
        facts=facts[:48],
        tasks=tasks[:4],
        retrieval_anchors=anchors[:32],
        findings=[],
    )
