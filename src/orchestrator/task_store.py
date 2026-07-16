"""Deterministic reducer for the image-only VisualFact investigation state."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from src.orchestrator.investigation_models import (
    AttributionOutput,
    BootstrapInvestigation,
    FactOrigin,
    Finding,
    ImageOnlyInvestigationState,
    InvestigationDiscovery,
    InvestigationEvidence,
    InvestigationFailure,
    ReflectionOutput,
    ReflectionRecord,
    ResearchTask,
    TargetPlanningOutput,
    VisualFact,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.route_policy import route_signature
from src.orchestrator.source_provenance import classify_source
from src.orchestrator.tool_result import parse_tool_result


MAX_TOOL_ACTIONS = 24
REFLECTION_INTERVAL = 4
MAX_REFLECTIONS = 6
INITIAL_TASKS_MAX = 4
TOTAL_TASKS_MAX = 12
NEW_TASKS_PER_REFLECTION_MAX = 3
DECISIVE_FACTS_MAX = 6
NEW_DECISIVE_FACTS_PER_REFLECTION_MAX = 2
ATTRIBUTION_FACTS_PER_PASS_MAX = 2
MAX_ATTEMPTS_PER_TASK = 5


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def next_action_boundary(action_count: int) -> int:
    """Return the next Reflection/action-cap boundary after current progress."""

    current = max(0, int(action_count))
    if current >= MAX_TOOL_ACTIONS:
        return MAX_TOOL_ACTIONS
    return min(
        MAX_TOOL_ACTIONS,
        ((current // REFLECTION_INTERVAL) + 1) * REFLECTION_INTERVAL,
    )


def state_from_bootstrap(
    bootstrap: BootstrapInvestigation,
) -> ImageOnlyInvestigationState:
    return ImageOnlyInvestigationState(
        brief=bootstrap.brief,
        entities=list(bootstrap.entities),
        facts=list(bootstrap.facts),
        tasks=list(bootstrap.tasks),
        retrieval_anchors=list(bootstrap.retrieval_anchors),
        findings=list(bootstrap.findings),
    )


def record_tool_observation(
    state: ImageOnlyInvestigationState,
    step: Any,
    *,
    image_sha256: str,
) -> Dict[str, Any]:
    """Reduce one real tool call into immutable records and state transitions."""

    if getattr(step, "action_type", "") != "tool_call":
        return {}
    if state.action_count >= MAX_TOOL_ACTIONS:
        raise RuntimeError("image-only tool action budget is exhausted")

    tool_name = str(getattr(step, "tool_name", "")).strip()
    tool_args = dict(getattr(step, "tool_args", {}) or {})
    metadata = dict(getattr(step, "metadata", {}) or {})
    call_id = str(metadata.get("function_call_id", "")).strip()
    if not call_id:
        call_id = stable_id(
            "call",
            state.brief.case_id,
            state.action_count + 1,
            tool_name,
            tool_args,
        )
        metadata["function_call_id"] = call_id
        step.metadata = metadata

    task_id = str(
        tool_args.get("task_id", "") or tool_args.get("__question_id", "")
    ).strip()
    task = _task_by_id(state, task_id)
    if task is None:
        raise RuntimeError(
            f"image-only tool call {call_id} references unknown task_id={task_id!r}"
        )

    state.action_count += 1
    task.attempt_count += 1
    task.status = "active"
    route = json.dumps(
        route_signature(tool_name, tool_args),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if route not in state.attempted_routes:
        state.attempted_routes.append(route)

    serialized = str(getattr(step, "tool_result", "") or "")
    try:
        data, succeeded = parse_tool_result(serialized)
    except Exception as exc:
        data = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        succeeded = False

    discovery_ids = _record_discoveries(
        state,
        task,
        function_call_id=call_id,
        tool_name=tool_name,
        data=data,
    )
    evidence_ids: List[str] = []
    finding_ids: List[str] = []
    failure_ids: List[str] = []

    if succeeded:
        evidence_ids, finding_ids = _record_evidence_and_findings(
            state,
            task,
            function_call_id=call_id,
            tool_name=tool_name,
            data=data,
            metadata=metadata,
            image_sha256=image_sha256,
        )
        if not discovery_ids and not evidence_ids and _is_empty_result(tool_name, data):
            failure_ids.append(
                _append_failure(
                    state,
                    task,
                    call_id=call_id,
                    tool_name=tool_name,
                    code="no_results",
                    message="Tool completed successfully but returned no usable result.",
                )
            )
    else:
        failure_ids.append(
            _append_failure(
                state,
                task,
                call_id=call_id,
                tool_name=tool_name,
                code=_failure_code(str(data.get("error", ""))),
                message=str(data.get("error", "tool call failed")),
            )
        )

    _refresh_fact_states(state)
    if finding_ids and not _task_has_unresolved_decisive_fact(state, task):
        task.finding_ids = list(dict.fromkeys([*task.finding_ids, *finding_ids]))
        task.status = "resolved"
    elif finding_ids:
        task.finding_ids = list(dict.fromkeys([*task.finding_ids, *finding_ids]))
        task.status = "active"
    elif (
        failure_ids
        and task.attempt_count >= 3
        and not _task_owns_scene_fact(state, task)
    ):
        task.status = "exhausted"
    if (
        task.status == "active"
        and task.attempt_count >= MAX_ATTEMPTS_PER_TASK
    ):
        task.status = "exhausted"

    return {
        "action_count": state.action_count,
        "task_id": task.task_id,
        "task_status": task.status,
        "created_discovery_ids": discovery_ids,
        "created_evidence_ids": evidence_ids,
        "created_finding_ids": finding_ids,
        "created_failure_ids": failure_ids,
        "fact_statuses": {
            fact.fact_id: fact.status
            for fact in state.facts
            if fact.fact_id in task.fact_ids
        },
    }


def apply_target_planning(
    state: ImageOnlyInvestigationState,
    output: TargetPlanningOutput,
) -> Dict[str, Any]:
    """Apply image-grounded investigation targets without a media-type pipeline."""

    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    anchor_text = " ".join(
        anchor.value for anchor in state.retrieval_anchors
    )
    accepted_fact_ids: List[str] = []
    accepted_task_ids: List[str] = []
    rejected_reasons: List[str] = []
    has_external_decisive_target = any(
        proposal.decision_relevance == "decisive"
        and proposal.predicate != "visual_integrity"
        for proposal in output.proposals
    )
    for proposal in output.proposals[:3]:
        parent_ids = list(dict.fromkeys(proposal.parent_fact_ids))
        parents = [
            fact_by_id[fact_id]
            for fact_id in parent_ids
            if fact_id in fact_by_id
        ]
        if len(parents) != len(parent_ids):
            rejected_reasons.append(
                "target proposal cites unknown parent facts"
            )
            continue
        if any(
            parent.origin.type not in {"input_image", "ocr"}
            for parent in parents
        ):
            rejected_reasons.append(
                "initial targets must be grounded in image or OCR facts"
            )
            continue
        if (
            proposal.predicate == "visual_integrity"
            and _visual_integrity_target_contains_world_relation(proposal)
        ):
            rejected_reasons.append(
                "visual_integrity describes a subject-to-place, event, date, "
                "or coexistence relation; propose that depicted-world relation "
                "as a separate decisive fact"
            )
            continue
        if (
            proposal.predicate == "located_at"
            and _located_at_target_is_overbroad(proposal, fact_by_id)
        ):
            rejected_reasons.append(
                "located_at target must bind one salient visible subject to one "
                "concrete visible place or habitat; generic coexistence and "
                "multi-entity location conjunctions are not atomic"
            )
            continue
        if (
            proposal.predicate == "source_record_matches"
            and _contains_visual_integrity_scope(
                " ".join(
                    (
                        proposal.statement,
                        proposal.question,
                        proposal.purpose,
                    )
                )
            )
        ):
            rejected_reasons.append(
                "source_record_matches must test source existence/content only; "
                "propose visual_integrity separately for pixel alteration"
            )
            continue
        if (
            proposal.predicate == "source_record_matches"
            and not _source_record_target_preserves_visible_text(
                proposal.statement,
                parents,
            )
        ):
            rejected_reasons.append(
                "source_record_matches must preserve at least one distinctive "
                "visible text anchor in the fact statement"
            )
            continue
        if (
            proposal.predicate == "visual_integrity"
            and _visual_integrity_target_is_negative(proposal.statement)
        ):
            rejected_reasons.append(
                "visual_integrity must be a positive authenticity proposition "
                "that anomaly evidence can refute"
            )
            continue
        if (
            proposal.predicate == "provenance_matches"
            and not _target_preserves_question_slots(
                proposal.statement,
                " ".join((proposal.question, proposal.purpose)),
            )
        ):
            rejected_reasons.append(
                "provenance target statement omits the identity, creator, "
                "place, event, date, or source slot asked by its task"
            )
            continue
        grounding_text = " ".join(
            [
                *(parent.statement for parent in parents),
                anchor_text,
            ]
        )
        if not _target_text_is_grounded(
            proposal.statement,
            grounding_text,
            predicate=proposal.predicate,
        ):
            rejected_reasons.append(
                "target statement is not grounded in visible facts or anchors"
            )
            continue
        if not _target_queries_are_grounded(
            proposal.suggested_queries,
            grounding_text,
        ):
            rejected_reasons.append(
                "target search query introduces terms absent from visible anchors"
            )
            continue
        existing = _matching_planned_target(
            state,
            proposal.statement,
            proposal.predicate,
        )
        decision_relevance = proposal.decision_relevance
        if (
            proposal.predicate == "visual_integrity"
            and has_external_decisive_target
        ):
            decision_relevance = "supporting"
        if existing is None:
            if len(state.facts) >= 72:
                rejected_reasons.append("VisualFact budget exhausted")
                break
            parent = parents[0]
            basis_ids = list(
                dict.fromkeys(
                    [
                        *parent_ids,
                        *[
                            basis_id
                            for item in parents
                            for basis_id in item.basis_ids
                        ],
                    ]
                )
            )[:12]
            origin_type = (
                "ocr"
                if any(item.origin.type == "ocr" for item in parents)
                else "input_image"
            )
            fact = VisualFact(
                fact_id=stable_id(
                    "vf",
                    state.brief.case_id,
                    "target",
                    proposal.predicate,
                    proposal.statement.casefold(),
                ),
                kind=proposal.kind,
                statement=re.sub(
                    r"\s+",
                    " ",
                    proposal.statement,
                ).strip(),
                subject_entity_id=parent.subject_entity_id,
                predicate=proposal.predicate,
                object_entity_id=parent.object_entity_id,
                status="active",
                basis_ids=basis_ids,
                decision_relevance=decision_relevance,
                origin=FactOrigin(
                    type=origin_type,
                    origin_ids=parent_ids[:8],
                ),
            )
            state.facts.append(fact)
            fact_by_id[fact.fact_id] = fact
        else:
            fact = existing
            if decision_relevance == "decisive":
                fact.decision_relevance = "decisive"

        if decision_relevance == "decisive":
            if proposal.predicate != "visual_integrity":
                _replace_generic_decisive_parents(
                    state,
                    parents,
                    fact,
                )
            else:
                state.decisive_fact_ids = list(
                    dict.fromkeys(
                        [*state.decisive_fact_ids, fact.fact_id]
                    )
                )[:DECISIVE_FACTS_MAX]
        task_id = stable_id(
            "task",
            fact.fact_id,
            "initial-target",
        )
        task = next(
            (item for item in state.tasks if item.task_id == task_id),
            None,
        )
        if task is None:
            if len(state.tasks) >= TOTAL_TASKS_MAX:
                rejected_reasons.append("total task budget exhausted")
                continue
            task = ResearchTask(
                task_id=task_id,
                fact_ids=[fact.fact_id],
                question=proposal.question,
                purpose=proposal.purpose,
                priority=1,
                status="active",
                parent_task_id=None,
                origin_ids=list(
                    dict.fromkeys([fact.fact_id, *parent_ids])
                )[:12],
                suggested_tools=list(
                    dict.fromkeys(proposal.suggested_tools)
                )[:4],
                suggested_queries=list(
                    dict.fromkeys(
                        query.strip()
                        for query in proposal.suggested_queries
                        if query.strip()
                    )
                )[:3],
            )
            state.tasks.append(task)
        accepted_fact_ids.append(fact.fact_id)
        accepted_task_ids.append(task.task_id)

    state.recommended_next_task_ids = list(
        dict.fromkeys(
            [
                *accepted_task_ids,
                *state.recommended_next_task_ids,
            ]
        )
    )[:4]
    return {
        "accepted_fact_ids": list(dict.fromkeys(accepted_fact_ids)),
        "accepted_task_ids": list(dict.fromkeys(accepted_task_ids)),
        "rejected_reasons": rejected_reasons,
        "remaining_target_gaps": output.remaining_target_gaps[:4],
    }


def _target_text_is_grounded(
    statement: str,
    grounding_text: str,
    *,
    predicate: str,
) -> bool:
    target = _attribution_tokens(statement)
    grounding = _attribution_tokens(grounding_text)
    if not target or not grounding:
        return False
    overlap = target & grounding
    minimum = 1 if predicate == "visual_integrity" else 2
    minimum_ratio = 0.05 if predicate == "visual_integrity" else 0.12
    return len(overlap) >= minimum and (
        len(overlap) / len(target) >= minimum_ratio
    )


def _located_at_target_is_overbroad(
    proposal: Any,
    fact_by_id: Mapping[str, VisualFact],
) -> bool:
    parents = [
        fact_by_id[fact_id]
        for fact_id in proposal.parent_fact_ids
        if fact_id in fact_by_id
    ]
    visible_subject_count = sum(
        parent.predicate == "visible_in" for parent in parents
    )
    text = " ".join(
        (
            proposal.statement,
            proposal.question,
            proposal.purpose,
        )
    ).casefold()
    generic_location = any(
        phrase in text
        for phrase in (
            "coexist",
            "same real-world geographic location",
            "same geographic location",
            "same real-world location",
            "same location",
            "any real-world habitat",
            "any geographic location",
        )
    )
    return visible_subject_count > 1 or generic_location


def _contains_visual_integrity_scope(value: str) -> bool:
    lowered = str(value or "").casefold()
    return any(
        token in lowered
        for token in (
            "unmodified",
            "not modified",
            "unaltered",
            "not altered",
            "fabricated",
            "edited image",
            "image is edited",
            "image was edited",
            "digitally fabricated",
            "digitally manipulated",
            "pixel manipulation",
            "edit seam",
            "compositing",
        )
    )


def _visual_integrity_target_is_negative(value: str) -> bool:
    lowered = " ".join(str(value or "").casefold().split())
    return any(
        phrase in lowered
        for phrase in (
            "is fake",
            "is fabricated",
            "is synthetic",
            "is ai-generated",
            "is ai generated",
            "is digitally manipulated",
            "is digitally altered",
            "is a composite",
            "was fabricated",
            "was generated by ai",
            "was digitally manipulated",
            "was digitally altered",
            "not authentic",
            "not an authentic",
            "physically impossible",
        )
    )


_QUESTION_SLOT_TERMS = {
    "identity": {
        "identity",
        "identify",
        "identified",
        "name",
        "named",
        "title",
        "titled",
        "creator",
        "created",
        "artist",
        "author",
        "weaver",
        "attribution",
        "attributed",
    },
    "place": {
        "where",
        "place",
        "location",
        "located",
        "site",
        "venue",
        "city",
        "country",
        "region",
        "museum",
        "collection",
        "repository",
    },
    "event": {
        "event",
        "ceremony",
        "incident",
        "occasion",
        "meeting",
        "protest",
        "festival",
    },
    "date": {
        "when",
        "date",
        "dated",
        "year",
        "time",
        "period",
    },
    "source": {
        "source",
        "origin",
        "provenance",
        "publication",
        "published",
        "record",
        "post",
        "page",
        "museum",
        "collection",
        "repository",
    },
}


def _target_preserves_question_slots(
    statement: str,
    task_text: str,
) -> bool:
    statement_tokens = _attribution_tokens(statement)
    task_tokens = _attribution_tokens(task_text)
    requested = {
        slot
        for slot, terms in _QUESTION_SLOT_TERMS.items()
        if task_tokens & terms
    }
    if not requested:
        return True
    preserved = {
        slot
        for slot, terms in _QUESTION_SLOT_TERMS.items()
        if statement_tokens & terms
    }
    return bool(requested & preserved)


def _visual_integrity_target_contains_world_relation(proposal: Any) -> bool:
    if proposal.predicate != "visual_integrity":
        return False
    relation_markers = (
        "real-world scene where",
        "coexist",
        "alongside",
        "located ",
        "located at",
        "takes place",
        "took place",
        "occurred ",
        "during ",
        " in a landscape",
        " in an landscape",
        " in the landscape",
        " in a polar",
        " in an antarctic",
        " in an arctic",
    )
    rendered = " ".join(
        (
            proposal.statement,
            proposal.question,
            proposal.purpose,
        )
    ).casefold()
    return any(marker in rendered for marker in relation_markers)


def _source_record_target_preserves_visible_text(
    statement: str,
    parents: Sequence[VisualFact],
) -> bool:
    visible_anchors: list[str] = []
    for parent in parents:
        for value in re.findall(r'"([^"]+)"', parent.statement):
            compact = _compact_visible_text(value)
            if _is_distinctive_visible_text(compact):
                visible_anchors.append(compact)
    if not visible_anchors:
        return True
    rendered = _compact_visible_text(statement)
    return any(anchor in rendered for anchor in visible_anchors)


def _compact_visible_text(value: str) -> str:
    return "".join(
        char.casefold()
        for char in str(value or "")
        if char.isalnum() or char in {"@", "_"}
    )


def _is_distinctive_visible_text(value: str) -> bool:
    if len(value) < 8:
        return False
    if value.startswith("@"):
        return False
    if re.fullmatch(r"\d+", value):
        return False
    return value not in {
        "showtranslation",
        "viewquotes",
        "relevant",
        "pinned",
    }


def _target_queries_are_grounded(
    queries: Sequence[str],
    grounding_text: str,
) -> bool:
    grounding = _attribution_tokens(grounding_text)
    for query in queries:
        query_tokens = _attribution_tokens(query)
        if query_tokens and not query_tokens & grounding:
            return False
    return True


def _matching_planned_target(
    state: ImageOnlyInvestigationState,
    statement: str,
    predicate: str,
) -> VisualFact | None:
    target = _attribution_tokens(statement)
    for fact in state.facts:
        if fact.predicate != predicate:
            continue
        tokens = _attribution_tokens(fact.statement)
        if not target or not tokens:
            continue
        overlap = len(target & tokens)
        union = len(target | tokens)
        if union and overlap / union >= 0.7:
            return fact
    return None


def attribution_planning_needed(
    state: ImageOnlyInvestigationState,
    update: Mapping[str, Any],
) -> bool:
    """Return whether new public evidence may warrant a specific attribution fact."""

    created_ids = {
        str(item)
        for key in (
            "created_discovery_ids",
            "created_evidence_ids",
            "created_finding_ids",
        )
        for item in update.get(key, []) or []
    }
    if not created_ids:
        return False
    affected_fact_ids = {
        fact_id
        for item in [
            *state.discoveries,
            *state.evidence,
            *state.findings,
        ]
        if (
            getattr(item, "discovery_id", "")
            or getattr(item, "evidence_id", "")
            or getattr(item, "finding_id", "")
        )
        in created_ids
        for fact_id in item.fact_ids
    }
    facts = {fact.fact_id: fact for fact in state.facts}
    if affected_fact_ids and all(
        facts.get(fact_id) is not None
        and facts[fact_id].predicate == "visual_integrity"
        for fact_id in affected_fact_ids
    ):
        return False
    decisive = [
        fact
        for fact in state.facts
        if fact.fact_id in state.decisive_fact_ids
    ]
    if any(
        fact.fact_id in affected_fact_ids
        and fact.origin.type == "web_discovery"
        and fact.decision_relevance != "decisive"
        for fact in state.facts
    ) and any(
        (
            getattr(item, "evidence_id", "")
            or getattr(item, "finding_id", "")
        )
        in created_ids
        for item in [*state.evidence, *state.findings]
    ):
        return True
    if any(fact.status == "refuted" for fact in decisive):
        return False
    promotable_predicates = {
        "appears_to_depict",
        "visible_in",
        "reads",
        "context_suggested_by_text",
        "provenance_matches",
    }
    if any(
        fact.fact_id in affected_fact_ids
        and fact.predicate in promotable_predicates
        and fact.origin.type != "web_discovery"
        for fact in decisive
    ):
        return True
    decisive_subject_ids = {
        fact.subject_entity_id
        for fact in decisive
        if fact.subject_entity_id
    }
    if any(
        fact.fact_id in affected_fact_ids
        and fact.predicate in promotable_predicates
        and fact.origin.type != "web_discovery"
        and fact.subject_entity_id in decisive_subject_ids
        for fact in state.facts
    ):
        return True
    return False


def apply_attribution(
    state: ImageOnlyInvestigationState,
    output: AttributionOutput,
) -> Dict[str, Any]:
    """Validate and apply evidence-grounded web attribution fact proposals."""

    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    task_by_id = {task.task_id: task for task in state.tasks}
    discovery_by_id = {
        item.discovery_id: item for item in state.discoveries
    }
    evidence_by_id = {
        item.evidence_id: item for item in state.evidence
    }
    finding_by_id = {
        item.finding_id: item for item in state.findings
    }
    accepted_fact_ids: List[str] = []
    created_task_ids: List[str] = []
    linked_evidence_ids: List[str] = []
    linked_finding_ids: List[str] = []
    rejected_reasons: List[str] = []

    for proposal in output.proposals[:ATTRIBUTION_FACTS_PER_PASS_MAX]:
        parent_ids = list(dict.fromkeys(proposal.parent_fact_ids))
        parent_facts = [
            fact_by_id[fact_id]
            for fact_id in parent_ids
            if fact_id in fact_by_id
        ]
        if len(parent_facts) != len(parent_ids):
            rejected_reasons.append(
                "attribution proposal cites unknown parent facts"
            )
            continue
        discoveries = [
            discovery_by_id[item]
            for item in dict.fromkeys(proposal.discovery_ids)
            if item in discovery_by_id
        ]
        evidence_rows = [
            evidence_by_id[item]
            for item in dict.fromkeys(proposal.evidence_ids)
            if item in evidence_by_id
        ]
        findings = [
            finding_by_id[item]
            for item in dict.fromkeys(proposal.finding_ids)
            if item in finding_by_id
        ]
        if (
            len(discoveries) != len(set(proposal.discovery_ids))
            or len(evidence_rows) != len(set(proposal.evidence_ids))
            or len(findings) != len(set(proposal.finding_ids))
        ):
            rejected_reasons.append(
                "attribution proposal cites unknown discovery/evidence/finding"
            )
            continue
        if not discoveries and not evidence_rows and not findings:
            rejected_reasons.append(
                "attribution proposal has no public grounding records"
            )
            continue
        central_lineage = {
            *state.decisive_fact_ids,
            *[
                origin_id
                for fact in state.facts
                if fact.fact_id in state.decisive_fact_ids
                and fact.origin.type == "web_discovery"
                for origin_id in fact.origin.origin_ids
            ],
            *[
                fact.fact_id
                for fact in state.facts
                if fact.origin.type == "web_discovery"
                and set(fact.origin.origin_ids) & set(state.decisive_fact_ids)
            ],
        }
        if (
            proposal.decision_relevance == "decisive"
            and not set(parent_ids) & central_lineage
        ):
            rejected_reasons.append(
                "decisive attribution must descend from the current central "
                "fact lineage"
            )
            continue
        record_parent_ids = list(
            dict.fromkeys(
                [
                    *parent_ids,
                    *[
                        origin_id
                        for fact in parent_facts
                        for origin_id in fact.origin.origin_ids
                    ],
                ]
            )
        )
        if not _attribution_records_own_parents(
            record_parent_ids,
            discoveries,
            evidence_rows,
            findings,
        ):
            rejected_reasons.append(
                "attribution records do not belong to the proposed parent facts"
            )
            continue
        statement = re.sub(r"\s+", " ", proposal.statement).strip()
        if (
            proposal.decision_relevance == "decisive"
            and proposal.predicate
            in {
                "identified_as",
                "attributed_as",
                "created_by",
                "dated_as",
                "located_at",
                "occurred_at",
                "depicts_event",
            }
            and _attribution_statement_is_negative(statement)
        ):
            rejected_reasons.append(
                "decisive attribution must preserve the positive image claim; "
                "attach refuting evidence to that claim instead of promoting "
                "a negated world fact"
            )
            continue
        decision_relevance = proposal.decision_relevance
        visual_bridge_present = _attribution_has_visual_bridge(
            evidence_rows,
            findings,
            evidence_by_id,
        )
        if (
            _contains_fabrication_attribution(statement)
            and not visual_bridge_present
            and not _records_explicitly_assert_fabrication(
                discoveries,
                evidence_rows,
                findings,
            )
        ):
            rejected_reasons.append(
                "fabrication/composite attribution requires a same-capture "
                "visual bridge or a cited record that explicitly asserts the "
                "fabrication mechanism"
            )
            continue
        if (
            decision_relevance == "decisive"
            and proposal.predicate
            in {
                "identified_as",
                "attributed_as",
                "created_by",
                "dated_as",
                "located_at",
                "occurred_at",
                "depicts_event",
            }
            and not visual_bridge_present
        ):
            decision_relevance = "supporting"
        if not _attribution_statement_is_grounded(
            statement,
            discoveries,
            evidence_rows,
            findings,
        ):
            rejected_reasons.append(
                "attribution statement is not grounded in cited public records"
            )
            continue
        if (
            _contains_fabrication_attribution(statement)
            and not evidence_rows
            and not findings
            and not (
                len(discoveries) == 1
                and bool(discoveries[0].reference_image_url)
            )
        ):
            rejected_reasons.append(
                "discovery-only attribution cannot synthesize a fabrication "
                "claim from multiple unverified search leads"
            )
            continue

        existing = _matching_attribution_fact(state, statement)
        target_fact_id = existing.fact_id if existing is not None else ""
        if not _attribution_link_capacity_available(
            target_fact_id,
            task_by_id,
            evidence_by_id,
            evidence_rows,
            findings,
        ):
            rejected_reasons.append(
                "attribution records have no remaining fact-link capacity"
            )
            continue
        origin_ids = list(
            dict.fromkeys(
                [
                    *parent_ids,
                    *proposal.discovery_ids,
                    *proposal.evidence_ids,
                    *proposal.finding_ids,
                ]
            )
        )
        if existing is None:
            if len(state.facts) >= 72:
                rejected_reasons.append("VisualFact budget exhausted")
                break
            parent = parent_facts[0]
            fact_id = stable_id(
                "vf",
                state.brief.case_id,
                "attribution",
                statement.casefold(),
            )
            fact = VisualFact(
                fact_id=fact_id,
                kind=proposal.kind,
                statement=statement,
                subject_entity_id=parent.subject_entity_id,
                predicate=proposal.predicate,
                object_entity_id=parent.object_entity_id,
                status="active",
                basis_ids=origin_ids[:12],
                decision_relevance=decision_relevance,
                origin=FactOrigin(
                    type="web_discovery",
                    origin_ids=origin_ids[:8],
                ),
            )
            state.facts.append(fact)
            fact_by_id[fact_id] = fact
        else:
            fact = existing
            fact_id = fact.fact_id
            fact.basis_ids = list(
                dict.fromkeys([*fact.basis_ids, *origin_ids])
            )[:12]
            fact.origin.origin_ids = list(
                dict.fromkeys([*fact.origin.origin_ids, *origin_ids])
            )[:8]
            if decision_relevance == "decisive":
                fact.decision_relevance = "decisive"

        owned_task_ids: set[str] = set()
        for evidence in evidence_rows:
            if fact_id not in evidence.fact_ids:
                evidence.fact_ids.append(fact_id)
                evidence.fact_ids = evidence.fact_ids[:6]
            owned_task_ids.add(evidence.task_id)
            linked_evidence_ids.append(evidence.evidence_id)
        for finding in findings:
            if fact_id not in finding.fact_ids:
                finding.fact_ids.append(fact_id)
                finding.fact_ids = finding.fact_ids[:6]
            owned_task_ids.add(finding.task_id)
            linked_finding_ids.append(finding.finding_id)
            for evidence_id in finding.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is not None and fact_id not in evidence.fact_ids:
                    evidence.fact_ids.append(fact_id)
                    evidence.fact_ids = evidence.fact_ids[:6]
                    linked_evidence_ids.append(evidence.evidence_id)
        for task_id in owned_task_ids:
            task = task_by_id.get(task_id)
            if task is not None and fact_id not in task.fact_ids:
                task.fact_ids.append(fact_id)
                task.fact_ids = task.fact_ids[:6]

        if decision_relevance == "decisive":
            _replace_generic_decisive_parents(
                state,
                parent_facts,
                fact,
            )
        _refresh_fact_states(state)
        if fact.status not in {"supported", "refuted"}:
            task_id = stable_id("task", fact_id, "verify-attribution")
            task = task_by_id.get(task_id)
            candidate_is_actionable = bool(
                visual_bridge_present
                or evidence_rows
                or findings
                or any(
                    discovery.reference_image_url
                    for discovery in discoveries
                )
            )
            if (
                task is None
                and candidate_is_actionable
                and len(state.tasks) < TOTAL_TASKS_MAX
            ):
                task = ResearchTask(
                    task_id=task_id,
                    fact_ids=[fact_id],
                    question=(
                        "What reliable original or direct source verifies or "
                        f"refutes this image attribution: {statement}"
                    ),
                    purpose=(
                        "Resolve the specific identity, event, place, creator, "
                        "or date discovered from public source context."
                    ),
                    priority=1,
                    status="active",
                    parent_task_id=None,
                    origin_ids=list(
                        dict.fromkeys([fact_id, *origin_ids])
                    )[:12],
                    suggested_tools=[
                        "visit",
                        "text_search",
                        "compare_with_reference",
                        "reverse_image_search",
                    ],
                    suggested_queries=(
                        list(dict.fromkeys(proposal.suggested_queries))[:3]
                        or [statement[:500]]
                    ),
                )
                state.tasks.append(task)
                task_by_id[task_id] = task
                created_task_ids.append(task_id)
            if task is not None:
                if task.status in {"resolved", "superseded"}:
                    task.status = "active"
                state.recommended_next_task_ids = list(
                    dict.fromkeys(
                        [task.task_id, *state.recommended_next_task_ids]
                    )
                )[:4]
        else:
            for task_id in owned_task_ids:
                task = task_by_id.get(task_id)
                if task is None:
                    continue
                task.finding_ids = list(
                    dict.fromkeys(
                        [
                            *task.finding_ids,
                            *[
                                finding.finding_id
                                for finding in findings
                                if finding.task_id == task_id
                            ],
                        ]
                    )
                )[:20]
        accepted_fact_ids.append(fact_id)

    return {
        "accepted_fact_ids": list(dict.fromkeys(accepted_fact_ids)),
        "created_task_ids": list(dict.fromkeys(created_task_ids)),
        "linked_evidence_ids": list(dict.fromkeys(linked_evidence_ids)),
        "linked_finding_ids": list(dict.fromkeys(linked_finding_ids)),
        "rejected_reasons": rejected_reasons,
        "remaining_attribution_gaps": output.remaining_attribution_gaps[:4],
    }


def _attribution_records_own_parents(
    parent_ids: Sequence[str],
    discoveries: Sequence[InvestigationDiscovery],
    evidence_rows: Sequence[InvestigationEvidence],
    findings: Sequence[Finding],
) -> bool:
    parent_set = set(parent_ids)
    records = [*discoveries, *evidence_rows, *findings]
    return all(parent_set & set(item.fact_ids) for item in records)


def _attribution_tokens(value: str) -> set[str]:
    text = str(value or "").casefold()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_@.-]+", text)
        if token
        not in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "at",
            "in",
            "of",
            "to",
            "this",
            "image",
            "photo",
            "photograph",
            "shows",
            "depicts",
        }
    }
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.add(sequence)
            continue
        tokens.update(
            sequence[index : index + 2]
            for index in range(len(sequence) - 1)
        )
    return tokens


def _attribution_statement_is_grounded(
    statement: str,
    discoveries: Sequence[InvestigationDiscovery],
    evidence_rows: Sequence[InvestigationEvidence],
    findings: Sequence[Finding],
) -> bool:
    source_text = " ".join(
        [
            *[
                " ".join((item.title, item.snippet))
                for item in discoveries
            ],
            *[
                item.exact_text
                for item in evidence_rows
            ],
            *[item.statement for item in findings],
        ]
    )
    statement_tokens = _attribution_tokens(statement)
    source_tokens = _attribution_tokens(source_text)
    if not statement_tokens or not source_tokens:
        return False
    overlap = statement_tokens & source_tokens
    return len(overlap) >= 2 and (
        len(overlap) / len(statement_tokens) >= 0.25
    )


def _attribution_has_visual_bridge(
    evidence_rows: Sequence[InvestigationEvidence],
    findings: Sequence[Finding],
    evidence_by_id: Mapping[str, InvestigationEvidence],
) -> bool:
    linked = [
        evidence_by_id[evidence_id]
        for finding in findings
        for evidence_id in finding.evidence_ids
        if evidence_id in evidence_by_id
    ]
    return any(
        item.claim_binding == "same_capture"
        and bool(item.same_capture_or_near_duplicate)
        and not bool(item.likely_different_original_capture)
        and not item.risk_flags
        for item in [*evidence_rows, *linked]
    )


def _contains_fabrication_attribution(value: str) -> bool:
    lowered = " ".join(str(value or "").casefold().split())
    return any(
        phrase in lowered
        for phrase in (
            "digital creation",
            "ai-generated",
            "ai generated",
            "generated by ai",
            "digitally manipulated",
            "digitally altered",
            "digital composite",
            "composite image",
            "synthetic image",
            "fabricated image",
            "physically impossible",
        )
    )


def _records_explicitly_assert_fabrication(
    discoveries: Sequence[InvestigationDiscovery],
    evidence_rows: Sequence[InvestigationEvidence],
    findings: Sequence[Finding],
) -> bool:
    source_text = " ".join(
        [
            *[
                " ".join((item.title, item.snippet))
                for item in discoveries
            ],
            *[item.exact_text for item in evidence_rows],
            *[item.statement for item in findings],
        ]
    ).casefold()
    return any(
        phrase in source_text
        for phrase in (
            "digital composite",
            "composite image",
            "photomontage",
            "photo montage",
            "photoshop",
            "digitally manipulated",
            "digitally altered",
            "ai-generated",
            "ai generated",
            "synthetic image",
            "digital artwork",
            "digital art",
        )
    )


def _attribution_statement_is_negative(value: str) -> bool:
    lowered = f" {' '.join(str(value or '').casefold().split())} "
    return any(
        phrase in lowered
        for phrase in (
            " do not ",
            " does not ",
            " did not ",
            " are not ",
            " is not ",
            " was not ",
            " were not ",
            " cannot ",
            " can't ",
            " never ",
            " no real-world ",
            " except antarctica ",
            " absent from ",
            " not found ",
            " not naturally ",
        )
    )


def _attribution_link_capacity_available(
    target_fact_id: str,
    task_by_id: Mapping[str, ResearchTask],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    evidence_rows: Sequence[InvestigationEvidence],
    findings: Sequence[Finding],
) -> bool:
    linked_evidence = [
        evidence_by_id[evidence_id]
        for finding in findings
        for evidence_id in finding.evidence_ids
        if evidence_id in evidence_by_id
    ]
    records = [
        *evidence_rows,
        *linked_evidence,
        *findings,
    ]
    for item in records:
        if target_fact_id and target_fact_id in item.fact_ids:
            continue
        if len(item.fact_ids) >= 6:
            return False
        task = task_by_id.get(item.task_id)
        if task is None:
            return False
        if target_fact_id and target_fact_id in task.fact_ids:
            continue
        if len(task.fact_ids) >= 6:
            return False
    return True


def _matching_attribution_fact(
    state: ImageOnlyInvestigationState,
    statement: str,
) -> VisualFact | None:
    target = _attribution_tokens(statement)
    for fact in state.facts:
        if fact.origin.type != "web_discovery":
            continue
        tokens = _attribution_tokens(fact.statement)
        if not target or not tokens:
            continue
        overlap = len(target & tokens)
        union = len(target | tokens)
        if union and overlap / union >= 0.7:
            return fact
    return None


def _replace_generic_decisive_parents(
    state: ImageOnlyInvestigationState,
    parent_facts: Sequence[VisualFact],
    attribution_fact: VisualFact,
) -> None:
    attribution_subject_id = attribution_fact.subject_entity_id
    ancestor_ids = {
        *attribution_fact.origin.origin_ids,
        *[
            origin_id
            for fact in parent_facts
            for origin_id in fact.origin.origin_ids
        ],
    }
    parent_ids = {
        fact.fact_id
        for fact in state.facts
        if (
            fact.fact_id in ancestor_ids
            or fact in parent_facts
            or (
                fact.subject_entity_id == attribution_subject_id
                and fact.origin.type != "web_discovery"
            )
        )
        and fact.predicate
        in {
            "appears_to_depict",
            "visible_in",
            "reads",
            "context_suggested_by_text",
            "provenance_matches",
            "identified_as",
            "located_at",
            "dated_as",
            "depicts_event",
        }
        and fact.fact_id != attribution_fact.fact_id
    }
    retained = [
        fact_id
        for fact_id in state.decisive_fact_ids
        if fact_id not in parent_ids
    ]
    for fact in state.facts:
        if fact.fact_id in parent_ids:
            fact.decision_relevance = "supporting"
    attribution_fact.decision_relevance = "decisive"
    state.decisive_fact_ids = list(
        dict.fromkeys([*retained, attribution_fact.fact_id])
    )[:DECISIVE_FACTS_MAX]
    _supersede_same_subject_tasks(state, attribution_fact)


def _supersede_same_subject_tasks(
    state: ImageOnlyInvestigationState,
    attribution_fact: VisualFact,
) -> None:
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    for task in state.tasks:
        if attribution_fact.fact_id in task.fact_ids:
            continue
        related = [
            fact_by_id[fact_id]
            for fact_id in task.fact_ids
            if fact_id in fact_by_id
        ]
        if not related or not any(
            fact.subject_entity_id == attribution_fact.subject_entity_id
            for fact in related
        ):
            continue
        if any(
            fact.fact_id in state.decisive_fact_ids
            for fact in related
        ):
            continue
        if task.status in {"active", "pending"}:
            task.status = "superseded"


def apply_reflection(
    state: ImageOnlyInvestigationState,
    output: ReflectionOutput,
    *,
    evidence_gain: bool,
    decision_gain: bool,
) -> ReflectionRecord:
    """Validate and apply a bounded Reflection delta."""

    if len(state.reflections) >= MAX_REFLECTIONS:
        raise RuntimeError("maximum image-only Reflection count is exhausted")
    task_by_id = {task.task_id: task for task in state.tasks}
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    finding_ids = {item.finding_id for item in state.findings}
    failure_ids = {item.failure_id for item in state.failures}
    origin_ids = {
        state.brief.brief_id,
        *[item.entity_id for item in state.entities],
        *[item.fact_id for item in state.facts],
        *[item.anchor_id for item in state.retrieval_anchors],
        *[item.discovery_id for item in state.discoveries],
        *[item.evidence_id for item in state.evidence],
        *finding_ids,
        *failure_ids,
    }
    accepted_updates: List[str] = []
    accepted_new: List[str] = []
    accepted_decisive: List[str] = []
    rejected: List[str] = []

    if len(output.task_updates) > 12:
        rejected.append("task update budget truncated to 12")
    if len(output.new_tasks) > NEW_TASKS_PER_REFLECTION_MAX:
        rejected.append(
            f"new task budget truncated to {NEW_TASKS_PER_REFLECTION_MAX}"
        )
    if (
        len(output.proposed_decisive_fact_ids)
        > NEW_DECISIVE_FACTS_PER_REFLECTION_MAX
    ):
        rejected.append(
            "decisive fact proposal budget truncated to "
            f"{NEW_DECISIVE_FACTS_PER_REFLECTION_MAX}"
        )
    if len(output.recommended_next_task_ids) > 4:
        rejected.append("recommended task budget truncated to 4")
    if len(output.remaining_gaps) > 8:
        rejected.append("remaining gap budget truncated to 8")
    bounded_output = output.model_copy(
        update={
            "task_updates": output.task_updates[:12],
            "new_tasks": output.new_tasks[:NEW_TASKS_PER_REFLECTION_MAX],
            "proposed_decisive_fact_ids": output.proposed_decisive_fact_ids[
                :NEW_DECISIVE_FACTS_PER_REFLECTION_MAX
            ],
            "recommended_next_task_ids": output.recommended_next_task_ids[:4],
            "remaining_gaps": output.remaining_gaps[:8],
        }
    )

    for update in bounded_output.task_updates:
        task = task_by_id.get(update.task_id)
        if task is None:
            rejected.append(f"unknown task update {update.task_id}")
            continue
        if update.priority is None:
            rejected.append(f"{update.task_id} proposed no priority change")
            continue
        task.priority = update.priority
        accepted_updates.append(update.task_id)

    semantic_keys = {
        _semantic_task_key(task.question): task.task_id
        for task in state.tasks
    }
    unresolved_decisive = {
        fact_id
        for fact_id in state.decisive_fact_ids
        if fact_by_id.get(fact_id) is not None
        and fact_by_id[fact_id].status not in {"supported", "refuted"}
    }
    for task in bounded_output.new_tasks:
        if len(state.tasks) >= TOTAL_TASKS_MAX:
            rejected.append("total task budget exhausted")
            break
        if task.task_id in task_by_id:
            rejected.append(f"duplicate task id {task.task_id}")
            continue
        if not set(task.fact_ids) <= set(fact_by_id):
            rejected.append(f"{task.task_id} cites unknown facts")
            continue
        if unresolved_decisive and not (
            set(task.fact_ids) & unresolved_decisive
        ):
            rejected.append(
                f"{task.task_id} does not own an unresolved decisive fact"
            )
            continue
        if not set(task.origin_ids) <= origin_ids:
            rejected.append(f"{task.task_id} cites unknown origins")
            continue
        key = _semantic_task_key(task.question)
        if key in semantic_keys:
            rejected.append(
                f"{task.task_id} duplicates {semantic_keys[key]}"
            )
            continue
        task.status = "active"
        state.tasks.append(task)
        task_by_id[task.task_id] = task
        semantic_keys[key] = task.task_id
        accepted_new.append(task.task_id)

    for fact_id in bounded_output.proposed_decisive_fact_ids:
        if len(state.decisive_fact_ids) >= DECISIVE_FACTS_MAX:
            rejected.append("decisive fact budget exhausted")
            break
        fact = fact_by_id.get(fact_id)
        if fact is None:
            rejected.append(f"unknown decisive fact {fact_id}")
            continue
        if fact_id in state.decisive_fact_ids:
            continue
        if (
            fact.predicate in {"visible_in", "reads"}
            and any(
                item.fact_id in state.decisive_fact_ids
                and item.predicate == "appears_to_depict"
                for item in state.facts
            )
        ):
            rejected.append(
                f"{fact_id} is incidental and already subsumed by the "
                "central scene proposition"
            )
            continue
        if not _fact_has_executable_route(state, fact_id):
            rejected.append(f"{fact_id} has no executable route")
            continue
        fact.decision_relevance = "decisive"
        if fact.status == "candidate":
            fact.status = "active"
        state.decisive_fact_ids.append(fact_id)
        accepted_decisive.append(fact_id)

    scene_task_ids = [
        task.task_id
        for task in state.tasks
        if task.status in {"active", "pending"}
        and _task_owns_scene_fact(state, task)
    ]
    model_recommendations = [
        task_id
        for task_id in bounded_output.recommended_next_task_ids
        if task_id in task_by_id
        and task_by_id[task_id].status in {"active", "pending"}
    ]
    state.recommended_next_task_ids = list(
        dict.fromkeys([*model_recommendations, *scene_task_ids])
    )[:4]
    record = ReflectionRecord(
        reflection_id=stable_id(
            "reflection",
            state.brief.case_id,
            state.action_count,
            len(state.reflections) + 1,
        ),
        action_count=state.action_count,
        output=bounded_output,
        accepted_task_update_ids=accepted_updates,
        accepted_new_task_ids=accepted_new,
        accepted_decisive_fact_ids=accepted_decisive,
        rejected_reasons=rejected,
        evidence_gain=evidence_gain,
        decision_gain=decision_gain,
    )
    state.reflections.append(record)
    state.reflection_failure_streak = 0
    return record


def _record_discoveries(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    function_call_id: str,
    tool_name: str,
    data: Mapping[str, Any],
) -> List[str]:
    rows: List[tuple[str, str, str, str, str]] = []
    if tool_name == "reverse_image_search":
        valid_references = {
            str(value).strip()
            for value in data.get("reference_image_candidates", []) or []
            if str(value).strip()
        }
        for item in data.get("lens_results", []) or []:
            if isinstance(item, Mapping):
                reference = str(item.get("image_url", "")).strip()
                rows.append(
                    (
                        str(item.get("url", "")),
                        str(item.get("title", "")),
                        str(item.get("snippet", "")),
                        "reverse_image",
                        reference if reference in valid_references else "",
                    )
                )
        for item in data.get("semantic_results", []) or []:
            if isinstance(item, Mapping):
                reference = str(item.get("image_url", "")).strip()
                rows.append(
                    (
                        str(item.get("url", "")),
                        str(item.get("title", "")),
                        str(item.get("snippet", "")),
                        "serp",
                        reference if reference in valid_references else "",
                    )
                )
    elif tool_name == "text_search":
        for query in data.get("queries", []) or []:
            if not isinstance(query, Mapping):
                continue
            for item in query.get("results", []) or []:
                if isinstance(item, Mapping):
                    rows.append(
                        (
                            str(item.get("url", "")),
                            str(item.get("title", "")),
                            str(item.get("snippet", "")),
                            "serp",
                            "",
                        )
                    )
    elif tool_name == "crop_and_search":
        valid_references = {
            str(value).strip()
            for value in data.get("reference_image_candidates", []) or []
            if str(value).strip()
        }
        for region in data.get("regions", []) or []:
            if not isinstance(region, Mapping):
                continue
            for item in region.get("lens_results", []) or []:
                if isinstance(item, Mapping):
                    reference = str(item.get("image_url", "")).strip()
                    rows.append(
                        (
                            str(item.get("url", "")),
                            str(item.get("title", "")),
                            str(item.get("snippet", "")),
                            "visual_reference",
                            reference if reference in valid_references else "",
                        )
                    )
            for item in region.get("semantic_results", []) or []:
                if isinstance(item, Mapping):
                    reference = str(item.get("image_url", "")).strip()
                    rows.append(
                        (
                            str(item.get("url", "")),
                            str(item.get("title", "")),
                            str(item.get("snippet", "")),
                            "serp",
                            reference if reference in valid_references else "",
                        )
                    )

    created: List[str] = []
    seen: set[tuple[str, str, str]] = set()
    existing = {item.discovery_id for item in state.discoveries}
    for raw_url, raw_title, raw_snippet, candidate_type, reference in rows:
        url = str(raw_url or "").strip()
        title = " ".join(str(raw_title or "").split())
        snippet = " ".join(str(raw_snippet or "").split())
        if not url:
            continue
        key = (url, title, candidate_type)
        if key in seen:
            continue
        seen.add(key)
        discovery_id = stable_id(
            "discovery",
            function_call_id,
            task.task_id,
            candidate_type,
            url,
            title,
        )
        if discovery_id in existing:
            continue
        state.discoveries.append(
            InvestigationDiscovery(
                discovery_id=discovery_id,
                task_id=task.task_id,
                fact_ids=list(task.fact_ids),
                function_call_id=function_call_id,
                tool_name=tool_name,
                candidate_url=url,
                reference_image_url=reference,
                title=title,
                snippet=snippet,
                candidate_type=candidate_type,
            )
        )
        created.append(discovery_id)
        existing.add(discovery_id)
    return created


def _record_evidence_and_findings(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    function_call_id: str,
    tool_name: str,
    data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    image_sha256: str,
) -> tuple[List[str], List[str]]:
    evidence_ids: List[str] = []
    finding_ids: List[str] = []
    eligible_fact_ids = _eligible_fact_ids(state, task, tool_name)

    for record in _web_evidence_records(data):
        evidence = str(record.get("evidence", "")).strip()
        source_url = str(
            record.get("selected_url", "") or record.get("url", "")
        ).strip()
        artifact = str(record.get("artifact_sha256", "")).strip()
        span = record.get("evidence_span", {})
        if (
            not evidence
            or not source_url
            or len(artifact) != 64
            or not isinstance(span, Mapping)
            or not isinstance(span.get("start"), int)
            or not isinstance(span.get("end"), int)
            or int(span["end"]) <= int(span["start"])
            or not str(record.get("retrieved_at", "")).strip()
            or record.get("injection_flags")
            or not bool(record.get("evidence_eligible", False))
        ):
            continue
        identity = classify_source(
            source_url,
            injection_flags=record.get("injection_flags", []),
        )
        stance = {
            "support": "support",
            "refute": "refute",
        }.get(str(record.get("stance", "")).strip().lower(), "neutral")
        relevance = str(record.get("relevance", "")).strip().lower()
        quality = (
            "strong"
            if identity.source_class == "official"
            and relevance == "high"
            and str(record.get("directness", "")).strip().lower() == "direct"
            else "moderate"
            if relevance in {"high", "medium"}
            else "weak"
        )
        evidence_id = stable_id(
            "evidence",
            function_call_id,
            task.task_id,
            source_url,
            artifact,
            int(span["start"]),
            int(span["end"]),
        )
        if evidence_id not in {item.evidence_id for item in state.evidence}:
            state.evidence.append(
                InvestigationEvidence(
                    evidence_id=evidence_id,
                    task_id=task.task_id,
                    fact_ids=eligible_fact_ids,
                    function_call_id=function_call_id,
                    tool_name=tool_name,
                    evidence_kind="web_span",
                    source_url=source_url,
                    source_family=identity.source_family,
                    source_class=identity.source_class,
                    exact_text=evidence,
                    span_start=int(span["start"]),
                    span_end=int(span["end"]),
                    artifact_sha256=artifact,
                    retrieved_at=str(record.get("retrieved_at", "")),
                    stance=stance,
                    quality=quality,
                    directness=(
                        "direct"
                        if str(record.get("directness", "")).strip().lower()
                        == "direct"
                        else "indirect"
                    ),
                    claim_binding="source_assertion",
                    temporal_alignment=str(
                        record.get("temporal_alignment", "")
                    ).strip(),
                    risk_flags=list(identity.risk_flags),
                )
            )
            evidence_ids.append(evidence_id)
        if stance in {"support", "refute"} and eligible_fact_ids:
            finding_id = stable_id(
                "finding",
                task.task_id,
                stance,
                evidence_id,
                eligible_fact_ids,
            )
            if finding_id not in {item.finding_id for item in state.findings}:
                state.findings.append(
                    Finding(
                        finding_id=finding_id,
                        task_id=task.task_id,
                        fact_ids=eligible_fact_ids,
                        statement=evidence[:1200],
                        stance=stance,
                        evidence_ids=[evidence_id],
                        source_family_ids=[identity.source_family],
                        quality=(
                            "decisive"
                            if quality == "strong"
                            else "supporting"
                        ),
                    )
                )
                finding_ids.append(finding_id)

    visual = _visual_evidence_record(
        state=state,
        task=task,
        function_call_id=function_call_id,
        tool_name=tool_name,
        data=data,
        metadata=metadata,
        image_sha256=image_sha256,
        fact_ids=eligible_fact_ids,
    )
    if visual is not None:
        evidence, finding = visual
        if evidence.evidence_id not in {
            item.evidence_id for item in state.evidence
        }:
            state.evidence.append(evidence)
            evidence_ids.append(evidence.evidence_id)
        if finding is not None and finding.finding_id not in {
            item.finding_id for item in state.findings
        }:
            state.findings.append(finding)
            finding_ids.append(finding.finding_id)

    return evidence_ids, finding_ids


def _visual_evidence_record(
    *,
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    function_call_id: str,
    tool_name: str,
    data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    image_sha256: str,
    fact_ids: List[str],
) -> tuple[InvestigationEvidence, Finding | None] | None:
    statement = ""
    stance = "neutral"
    kind = "image_region"
    claim_binding = "pixel_observation"
    source_url = ""
    region = data.get("crop_bbox") or data.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    if tool_name == "compare_with_reference":
        statement = str(data.get("overall_observation", "")).strip()
        source_url = str(
            data.get("resolved_reference_url")
            or data.get("reference_url", "")
        ).strip()
        kind = "reference_comparison"
        if bool(data.get("edit_evidence_present", False)):
            stance = "refute"
            claim_binding = "pixel_observation"
        elif _task_owns_scene_fact(state, task):
            if bool(data.get("same_capture_or_near_duplicate", False)):
                stance = "support"
                claim_binding = "same_capture"
            elif bool(data.get("same_subject_or_scene", False)):
                claim_binding = "same_subject"
        elif bool(data.get("same_subject_or_scene", False)):
            stance = "support"
            claim_binding = (
                "same_capture"
                if bool(data.get("same_capture_or_near_duplicate", False))
                else "same_subject"
            )
    elif tool_name == "crop_and_inspect":
        statement = str(
            data.get("answer", "") or data.get("description", "")
        ).strip()
    elif tool_name == "check_consistency":
        statement = str(data.get("details", "")).strip()
        if data.get("consistent") is False:
            stance = "refute"
        elif data.get("consistent") is True:
            stance = "support"
    elif tool_name == "analyze_visual_anomalies":
        statement = str(data.get("notes", "")).strip()
        if not statement:
            statement = "; ".join(
                str(
                    item.get("phenomenon")
                    or item.get("reasoning")
                    or item.get("name", "")
                ).strip()
                for item in data.get("anomalies", []) or []
                if isinstance(item, Mapping)
            ).strip()
        if not statement and str(data.get("overall_authenticity", "")).strip():
            statement = (
                "The targeted visual scan reported overall_authenticity="
                + str(data.get("overall_authenticity", "")).strip()
                + "."
            )
        if data.get("anomalies") and str(
            data.get("overall_authenticity", "")
        ) in {"likely_ai", "likely_manipulated"}:
            stance = "refute"
        elif (
            not data.get("anomalies")
            and str(data.get("overall_authenticity", "")) == "authentic"
        ):
            stance = "support"
    elif tool_name == "ocr_with_position":
        statement = str(data.get("full_text", "")).strip()
    else:
        return None
    if not statement or not fact_ids:
        return None

    observed_at = str(
        metadata.get("observed_at")
        or datetime.now(timezone.utc).isoformat()
    )
    artifact = (
        hashlib.sha256(
            (source_url + "\n" + statement).encode("utf-8")
        ).hexdigest()
        if kind == "reference_comparison"
        else image_sha256
    )
    identity = classify_source(source_url) if source_url else None
    evidence_id = stable_id(
        "evidence",
        function_call_id,
        task.task_id,
        tool_name,
        statement,
    )
    evidence = InvestigationEvidence(
        evidence_id=evidence_id,
        task_id=task.task_id,
        fact_ids=fact_ids,
        function_call_id=function_call_id,
        tool_name=tool_name,
        evidence_kind=kind,
        source_url=source_url,
        source_family=(
            identity.source_family
            if identity is not None
            else f"image:{image_sha256}"
        ),
        source_class=(
            identity.source_class
            if identity is not None
            else "visual"
        ),
        exact_text=statement,
        image_region=(
            list(region)
            if kind == "image_region"
            and isinstance(region, Sequence)
            and len(region) == 4
            else None
        ),
        artifact_sha256=artifact,
        retrieved_at=observed_at,
        stance=stance,
        quality="moderate",
        directness="direct",
        claim_binding=claim_binding,
        same_subject_or_scene=(
            bool(data.get("same_subject_or_scene", False))
            if tool_name == "compare_with_reference"
            else None
        ),
        same_capture_or_near_duplicate=(
            bool(data.get("same_capture_or_near_duplicate", False))
            if tool_name == "compare_with_reference"
            else None
        ),
        likely_different_original_capture=(
            bool(data.get("likely_different_original_capture", False))
            if tool_name == "compare_with_reference"
            else None
        ),
        edit_evidence_present=(
            bool(data.get("edit_evidence_present", False))
            if tool_name == "compare_with_reference"
            else None
        ),
        confidence=(
            float(data["confidence"])
            if tool_name == "compare_with_reference"
            and isinstance(data.get("confidence"), (int, float))
            else None
        ),
        risk_flags=list(identity.risk_flags) if identity is not None else [],
    )
    finding = None
    if stance in {"support", "refute"}:
        finding = Finding(
            finding_id=stable_id(
                "finding",
                task.task_id,
                stance,
                evidence_id,
                fact_ids,
            ),
            task_id=task.task_id,
            fact_ids=fact_ids,
            statement=statement[:1200],
            stance=stance,
            evidence_ids=[evidence_id],
            source_family_ids=[evidence.source_family],
            quality="supporting",
        )
    return evidence, finding


def _web_evidence_records(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if (
            "evidence" in value
            and "evidence_eligible" in value
            and ("url" in value or "selected_url" in value)
        ):
            yield value
        for child in value.values():
            yield from _web_evidence_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _web_evidence_records(child)


def _eligible_fact_ids(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    tool_name: str,
) -> List[str]:
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    if tool_name in {
        "text_search",
        "visit",
        "reverse_image_search",
        "crop_and_search",
    }:
        filtered = [
            fact_id
            for fact_id in task.fact_ids
            if fact_by_id.get(fact_id) is not None
            and fact_by_id[fact_id].kind != "text_claim"
        ]
        return filtered or list(task.fact_ids)
    return list(task.fact_ids)


def _refresh_fact_states(state: ImageOnlyInvestigationState) -> None:
    findings_by_fact: Dict[str, List[Finding]] = {}
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    for finding in state.findings:
        for fact_id in finding.fact_ids:
            findings_by_fact.setdefault(fact_id, []).append(finding)
    for fact in state.facts:
        fact_findings = findings_by_fact.get(fact.fact_id, [])
        fact_evidence = [
            item
            for item in state.evidence
            if fact.fact_id in item.fact_ids
        ]
        assessment = assess_fact(
            fact,
            fact_findings,
            evidence_by_id,
            all_fact_evidence=fact_evidence,
        )
        if assessment.status in {"supported", "refuted", "conflicted"}:
            fact.status = assessment.status
        elif fact.fact_id in state.decisive_fact_ids:
            fact.status = "active"


def _task_by_id(
    state: ImageOnlyInvestigationState,
    task_id: str,
) -> ResearchTask | None:
    return next(
        (item for item in state.tasks if item.task_id == task_id),
        None,
    )


def _append_failure(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    call_id: str,
    tool_name: str,
    code: str,
    message: str,
) -> str:
    failure_id = stable_id(
        "failure",
        call_id,
        task.task_id,
        tool_name,
        code,
        message,
    )
    if failure_id not in {item.failure_id for item in state.failures}:
        state.failures.append(
            InvestigationFailure(
                failure_id=failure_id,
                task_id=task.task_id,
                fact_ids=list(task.fact_ids),
                function_call_id=call_id,
                tool_name=tool_name,
                code=code,
                message=message[:4000],
            )
        )
    return failure_id


def _failure_code(message: str) -> str:
    lowered = str(message or "").lower()
    if "timeout" in lowered:
        return "timeout"
    if any(
        token in lowered
        for token in ("blocked", "access", "captcha", "download")
    ):
        return "access_limited"
    if any(token in lowered for token in ("provider", "unavailable")):
        return "provider_unavailable"
    if any(token in lowered for token in ("argument", "unknown", "schema")):
        return "protocol_error"
    if any(token in lowered for token in ("json", "malformed")):
        return "malformed_result"
    return "tool_error"


def _is_empty_result(tool_name: str, data: Mapping[str, Any]) -> bool:
    if tool_name == "reverse_image_search":
        return not (
            data.get("candidate_page_urls")
            or data.get("reference_image_candidates")
            or data.get("lens_results")
            or data.get("semantic_results")
        )
    if tool_name == "text_search":
        return not any(
            isinstance(query, Mapping) and query.get("results")
            for query in data.get("queries", []) or []
        )
    if tool_name == "visit":
        return not str(data.get("evidence", "")).strip()
    return False


def _semantic_task_key(question: str) -> str:
    return " ".join(
        "".join(
            char.lower() if char.isalnum() else " "
            for char in str(question or "")
        ).split()
    )


def _fact_has_executable_route(
    state: ImageOnlyInvestigationState,
    fact_id: str,
) -> bool:
    return any(
        fact_id in task.fact_ids
        and task.status in {"active", "pending", "resolved"}
        and bool(task.suggested_tools)
        for task in state.tasks
    )


def _task_has_unresolved_decisive_fact(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    facts = {fact.fact_id: fact for fact in state.facts}
    return any(
        fact_id in state.decisive_fact_ids
        and facts.get(fact_id) is not None
        and facts[fact_id].status not in {"supported", "refuted"}
        for fact_id in task.fact_ids
    )


def _task_owns_scene_fact(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    facts = {fact.fact_id: fact for fact in state.facts}
    return any(
        facts.get(fact_id) is not None
        and facts[fact_id].predicate == "appears_to_depict"
        for fact_id in task.fact_ids
    )
