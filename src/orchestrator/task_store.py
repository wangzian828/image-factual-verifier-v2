"""Deterministic reducer for the image-only VisualFact investigation state."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from src.orchestrator.investigation_models import (
    BootstrapInvestigation,
    EvidenceDecisionOutput,
    EvidenceDecisionRecord,
    FactOrigin,
    Finding,
    EvidenceGap,
    ImageOnlyInvestigationState,
    InvestigationDiscovery,
    InvestigationEvidence,
    InvestigationFailure,
    ReflectionOutput,
    ReflectionRecord,
    ResearchTask,
    TargetFactProposal,
    TargetPlanningOutput,
    VisualFact,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.route_policy import route_signature
from src.orchestrator.source_provenance import (
    canonicalize_url,
    classify_source,
)
from src.orchestrator.tool_result import parse_tool_result


MAX_TOOL_ACTIONS = 24
REFLECTION_INTERVAL = 4
MAX_REFLECTIONS = 6
INITIAL_TASKS_MAX = 4
TOTAL_TASKS_MAX = 12
NEW_TASKS_PER_REFLECTION_MAX = 3
MAX_TEXT_SEARCH_ROUTES_PER_TASK = 2
MAX_CORE_FACT_REFINEMENTS = 1
MAX_INSPECTION_CANDIDATES_PER_BATCH = 4
MAX_INSPECTION_ATTEMPTS_PER_BATCH = 2


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


_CORE_BINDING_PREDICATES = {
    "appears_to_depict",
    "source_record_matches",
    "provenance_matches",
    "identified_as",
}
_CORE_REFINEMENT_PREDICATES = {
    "appears_to_depict",
    "source_record_matches",
    "provenance_matches",
    "identified_as",
    "located_at",
    "occurred_at",
    "depicts_event",
}
_CORE_METADATA_PREDICATES = {
    "attributed_as",
    "created_by",
    "dated_as",
}


def reconcile_core_verdict_fact(
    state: ImageOnlyInvestigationState,
    candidate_fact: VisualFact,
    *,
    parent_facts: Sequence[VisualFact] = (),
    allow_initial: bool = False,
    allow_refinement: bool = False,
    allow_active_refinement: bool = False,
) -> tuple[bool, str]:
    """Apply the only allowed core-verdict ownership transition.

    One image investigation owns exactly one factual proposition. Planning
    establishes it once. A later semantic Evidence Decision may either promote an
    already adjudicated descendant or perform the one allowed active visual-slot
    refinement: the same visible subject/place/event relation with a previously
    unknown slot made more specific from pixel/OCR anchors and newly reviewed
    Evidence. Title, creator, date, platform, and other optional metadata never
    become new stopping conditions.
    """

    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    candidate = fact_by_id.get(candidate_fact.fact_id)
    if candidate is None:
        return False, "core candidate does not exist in the investigation state"
    if candidate.predicate == "visual_integrity":
        return False, "pixel-integrity checks are diagnostic and cannot own verdicts"
    if candidate.predicate in _CORE_METADATA_PREDICATES:
        return False, "creator, title, or date metadata cannot own the verdict"

    current_id = state.core_verdict_fact_id
    if current_id is None:
        if not allow_initial:
            return False, "initial core fact was not authorized"
        _set_core_verdict_fact(state, candidate)
        return True, "initial core fact established"

    if current_id == candidate.fact_id:
        _set_core_verdict_fact(state, candidate)
        return True, "candidate already owns the core verdict"

    if not allow_refinement:
        return False, "core fact is stable; only a bounded visual refinement may replace it"
    if state.core_fact_refinement_count >= MAX_CORE_FACT_REFINEMENTS:
        return False, "the bounded core-fact refinement budget is exhausted"
    current = fact_by_id.get(current_id)
    if current is None:
        return False, "current core fact is missing"
    if (
        candidate.status not in {"supported", "refuted"}
        and not (
            allow_active_refinement
            and candidate.status == "active"
        )
    ):
        return False, (
            "a refinement must be supported/refuted, or be the authorized active "
            "visual-slot refinement"
        )
    if candidate.predicate not in _CORE_REFINEMENT_PREDICATES:
        return False, "candidate predicate is not a verdict-preserving refinement"
    if candidate.subject_entity_id != current.subject_entity_id:
        return False, "a refinement must preserve the core fact's salient subject"
    if not _fact_descends_from(candidate, current.fact_id, fact_by_id):
        return False, "a refinement must descend from the current core fact lineage"
    if not _core_refinement_is_atomic(candidate.statement):
        return False, "a refinement must contain one factual slot, not metadata bundle"

    _set_core_verdict_fact(state, candidate)
    state.core_fact_refinement_count += 1
    _supersede_same_subject_tasks(state, candidate)
    return True, "bounded atomic visual refinement replaced the core fact"


def refresh_core_evidence_gaps(
    state: ImageOnlyInvestigationState,
) -> List[EvidenceGap]:
    """Synchronize bounded gaps from the latest semantic decision checkpoint."""

    core_id = state.core_verdict_fact_id
    if not core_id:
        state.evidence_gaps = []
        return []
    facts = {fact.fact_id: fact for fact in state.facts}
    core = facts.get(core_id)
    if core is None:
        state.evidence_gaps = []
        return []

    owned_evidence = [
        evidence
        for evidence in state.evidence
        if core_id in evidence.fact_ids
    ]
    decision = latest_evidence_decision(state, fact_id=core_id)
    if decision is None:
        direct_status = (
            "blocked"
            if core.status == "blocked"
            else "exhausted"
            if core.status == "exhausted"
            else "open"
        )
        gaps = [
            EvidenceGap(
                gap_id=stable_id("gap", core_id, "image_source_binding"),
                fact_id=core_id,
                kind="image_source_binding",
                status="not_required",
                evidence_ids=[],
                reason=(
                    "A semantic evidence checkpoint decides whether this "
                    "proposition needs source-to-image binding."
                ),
            ),
            EvidenceGap(
                gap_id=stable_id("gap", core_id, "direct_support_or_refute"),
                fact_id=core_id,
                kind="direct_support_or_refute",
                status=direct_status,
                evidence_ids=[],
                reason=(
                    "No semantic Evidence decision has resolved the active "
                    "proposition."
                )
                if direct_status == "open"
                else "",
            ),
            EvidenceGap(
                gap_id=stable_id("gap", core_id, "conflict_resolution"),
                fact_id=core_id,
                kind="conflict_resolution",
                status="not_required",
                evidence_ids=[],
            ),
        ]
        state.evidence_gaps = gaps
        return gaps

    selected_ids = [
        item
        for item in decision.output.selected_evidence_ids
        if item in {evidence.evidence_id for evidence in owned_evidence}
    ]
    selected_evidence = [
        evidence
        for evidence in owned_evidence
        if evidence.evidence_id in selected_ids
    ]
    capture_ids = [
        item.evidence_id
        for item in selected_evidence
        if (
            item.claim_binding == "same_capture"
            or item.same_capture_or_near_duplicate is True
        )
    ]
    binding_required = (
        decision.output.binding_requirement == "same_capture_required"
    )
    binding_status = (
        "resolved"
        if capture_ids
        else "open"
    )
    assessment = decision.output.assessment
    direct_status = (
        "resolved"
        if assessment in {"supported", "refuted"}
        else "open"
    )
    conflict_status = (
        "open"
        if assessment == "conflicted"
        else "not_required"
    )
    gaps = [
        EvidenceGap(
            gap_id=stable_id("gap", core_id, "image_source_binding"),
            fact_id=core_id,
            kind="image_source_binding",
            status=binding_status if binding_required else "not_required",
            evidence_ids=capture_ids[:40],
            reason=(
                "The semantic decision requires a same-capture or near-duplicate "
                "bridge before this proposition can close."
                if binding_required and binding_status == "open"
                else ""
            ),
        ),
        EvidenceGap(
            gap_id=stable_id("gap", core_id, "direct_support_or_refute"),
            fact_id=core_id,
            kind="direct_support_or_refute",
            status=direct_status,
            evidence_ids=selected_ids[:40],
            reason=(
                decision.output.remaining_gap
                or decision.output.rationale
            )
            if direct_status != "resolved"
            else "",
        ),
        EvidenceGap(
            gap_id=stable_id("gap", core_id, "conflict_resolution"),
            fact_id=core_id,
            kind="conflict_resolution",
            status=conflict_status,
            evidence_ids=selected_ids[:40],
            reason=decision.output.rationale if conflict_status == "open" else "",
        ),
    ]
    state.evidence_gaps = gaps
    return gaps


def _set_core_verdict_fact(
    state: ImageOnlyInvestigationState,
    candidate: VisualFact,
) -> None:
    for fact in state.facts:
        if fact.fact_id != candidate.fact_id and fact.decision_relevance == "decisive":
            fact.decision_relevance = "supporting"
    candidate.decision_relevance = "decisive"
    if candidate.status == "candidate":
        candidate.status = "active"
    state.core_verdict_fact_id = candidate.fact_id
    state.decisive_fact_ids = [candidate.fact_id]
    refresh_core_evidence_gaps(state)


def _fact_descends_from(
    candidate: VisualFact,
    ancestor_id: str,
    fact_by_id: Mapping[str, VisualFact],
) -> bool:
    pending = [
        *candidate.origin.origin_ids,
        *candidate.basis_ids,
    ]
    seen: set[str] = set()
    while pending:
        current_id = pending.pop()
        if current_id == ancestor_id:
            return True
        if current_id in seen:
            continue
        seen.add(current_id)
        parent = fact_by_id.get(current_id)
        if parent is not None:
            pending.extend(parent.origin.origin_ids)
            pending.extend(parent.basis_ids)
    return False


def _core_refinement_is_atomic(statement: str) -> bool:
    text = " ".join(str(statement or "").casefold().split())
    if not text or ";" in text:
        return False
    metadata_slots = sum(
        token in text
        for token in (
            "title",
            "creator",
            "author",
            "photographer",
            "date",
            "asset id",
            "platform",
        )
    )
    return metadata_slots <= 1


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
    serialized = str(getattr(step, "tool_result", "") or "")
    try:
        data, succeeded = parse_tool_result(serialized)
    except Exception as exc:
        data = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        succeeded = False

    discovery_ids: List[str] = []
    evidence_ids: List[str] = []
    finding_ids: List[str] = []
    failure_ids: List[str] = []

    if succeeded:
        discovery_ids = _record_discoveries(
            state,
            task,
            function_call_id=call_id,
            tool_name=tool_name,
            data=data,
        )
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

    route_payload = route_signature(tool_name, tool_args)
    route_payload["function_call_id"] = call_id
    route_payload["outcome"] = (
        "evidence"
        if evidence_ids
        else "discovery"
        if discovery_ids
        else "empty"
        if succeeded
        else "failed"
    )
    route = json.dumps(
        route_payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if route not in state.attempted_routes:
        state.attempted_routes.append(route)

    _refresh_fact_states(state)
    if finding_ids and not _task_has_unresolved_decisive_fact(state, task):
        task.finding_ids = list(dict.fromkeys([*task.finding_ids, *finding_ids]))
        task.status = "resolved"
    elif finding_ids:
        task.finding_ids = list(dict.fromkeys([*task.finding_ids, *finding_ids]))
        task.status = "active"
    elif (
        failure_ids
        and not _task_has_remaining_material_route(state, task)
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
        grounding_text = " ".join(
            [
                *(parent.statement for parent in parents),
                anchor_text,
            ]
        )
        proposal = _normalize_target_parenthetical_alias(
            proposal,
            grounding_text=grounding_text,
        )
        proposal = _normalize_target_authenticity_wrapper(proposal)
        unobserved_named_values = _unobserved_named_values(
            proposal.statement,
            proposal.suggested_queries,
            grounding_text,
        )
        invalid_initial_scope = (
            proposal.predicate != "visual_integrity"
            and _target_describes_unobserved_media_state(
                proposal.statement
            )
        )
        if invalid_initial_scope:
            reasons = [
                "target describes an unseen original, unaltered, or "
                "counterfactual media state instead of the visible positive "
                "subject-object, person-product, place, or event relation"
            ]
            if unobserved_named_values:
                rendered_values = ", ".join(unobserved_named_values[:4])
                reasons.append(
                    "target introduces named value(s) absent from image/OCR "
                    f"grounding: {rendered_values}"
                )
            rejected_reasons.append(
                "; ".join(reasons)
                + ". Preserve the visible relation and remove only unsupported "
                "values; do not replace it with hidden source-image details or "
                "incidental OCR metadata"
            )
            continue
        if (
            proposal.predicate != "visual_integrity"
            and unobserved_named_values
        ):
            rendered_values = ", ".join(unobserved_named_values[:4])
            rejected_reasons.append(
                "target introduces named value(s) absent from image/OCR "
                f"grounding: {rendered_values}. Remove only the unsupported "
                "value(s) while preserving the visible subject, object, place, "
                "and event relation; do not replace the relation with incidental "
                "OCR metadata"
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
            proposal.predicate != "visual_integrity"
            and _target_mixes_visual_integrity_and_world_relation(
                proposal.statement
            )
        ):
            rejected_reasons.append(
                "target must not combine visual authenticity with an external "
                "event, identity, place, date, or source relation; plan the "
                "world relation and any integrity diagnostic separately"
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
        if _attribution_statement_is_negative(proposal.statement):
            rejected_reasons.append(
                "target statement must preserve the positive image claim; "
                "attach refuting evidence to that claim instead of planning "
                "a negated world fact"
            )
            continue
        if (
            proposal.predicate != "visual_integrity"
            and _contains_fabrication_attribution(proposal.statement)
        ):
            rejected_reasons.append(
                "initial target cannot assert fabrication, AI generation, or "
                "compositing without source evidence; plan a positive "
                "depicted-world or source relation instead"
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
        core_candidate_requested = (
            state.core_verdict_fact_id is None
            and proposal.decision_relevance == "decisive"
            and proposal.predicate != "visual_integrity"
        )
        has_visible_source_anchor = any(
            parent.origin.type == "ocr" or parent.predicate == "reads"
            for parent in parents
        )
        if (
            proposal.predicate
            in {"source_record_matches", "provenance_matches"}
            and not has_visible_source_anchor
        ):
            core_candidate_requested = False
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
                decision_relevance="supporting",
                origin=FactOrigin(
                    type=origin_type,
                    origin_ids=parent_ids[:8],
                ),
            )
            state.facts.append(fact)
            fact_by_id[fact.fact_id] = fact
        else:
            fact = existing
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
        if core_candidate_requested:
            accepted_core, core_reason = reconcile_core_verdict_fact(
                state,
                fact,
                parent_facts=parents,
                allow_initial=True,
            )
            if not accepted_core:
                rejected_reasons.append(core_reason)
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


def _normalize_target_parenthetical_alias(
    proposal: TargetFactProposal,
    *,
    grounding_text: str,
) -> TargetFactProposal:
    """Remove only ungrounded scientific binomials in parenthetical apposition."""

    grounding = _attribution_tokens(grounding_text)

    def normalize(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            content = " ".join(match.group(1).split())
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", content)
            if (
                len(words) != 2
                or " ".join(words) != content
                or not words[0][0].isupper()
                or not words[0][1:].islower()
                or not words[1].islower()
                or any(word.casefold() in grounding for word in words)
            ):
                return match.group(0)
            return ""

        return re.sub(
            r"\s*\(([^()]*)\)",
            replace,
            str(value or ""),
        ).strip()

    statement = normalize(proposal.statement)
    question = normalize(proposal.question)
    purpose = normalize(proposal.purpose)
    if (
        statement == proposal.statement
        and question == proposal.question
        and purpose == proposal.purpose
    ):
        return proposal
    return proposal.model_copy(
        update={
            "statement": statement,
            "question": question,
            "purpose": purpose,
        }
    )


def _normalize_target_authenticity_wrapper(
    proposal: TargetFactProposal,
) -> TargetFactProposal:
    """Preserve a valid world relation while removing image-authenticity scope.

    Initial planning sometimes wraps an otherwise useful event proposition in
    wording such as "a real historical event" and then asks whether the pixels
    are a genuine photograph or an AI composite.  Regenerating the whole target
    after rejecting that wrapper can discard the salient subject-object-event
    relation and collapse onto incidental OCR metadata.  Normalize only the
    authenticity wrapper here; all ordinary grounding, atomicity, and named-value
    checks still run on the preserved proposition.
    """

    if proposal.predicate in {
        "visual_integrity",
        "source_record_matches",
        "provenance_matches",
    }:
        return proposal

    statement = str(proposal.statement or "")
    normalized_statement = re.sub(
        r"^\s*(the\s+(?:input\s+)?image)\s+is\s+(?:a|an)\s+"
        r"(?:real|genuine|authentic)\s+(?:photograph|photo|image)\s+"
        r"(?:that\s+)?(?:depicts|depicting|shows|showing)\s+",
        r"\1 depicts ",
        statement,
        flags=re.IGNORECASE,
    )
    normalized_statement = re.sub(
        r"^\s*(?:this|the\s+(?:input\s+)?image)\s+is\s+(?:a|an)\s+"
        r"(?:real|genuine|authentic)\s+(?:photograph|photo|image)\s+of\s+",
        "The image depicts ",
        normalized_statement,
        flags=re.IGNORECASE,
    )
    normalized_statement = re.sub(
        r"\b(?:a|an)\s+(?:real|genuine|authentic)\s+"
        r"(?:historical\s+)?event\b",
        "an event",
        normalized_statement,
        flags=re.IGNORECASE,
    )
    normalized_statement = re.sub(
        r"\b(?:a|an)\s+(?:real|genuine|authentic)\s+"
        r"(?:historical\s+)?occurrence\b",
        "an occurrence",
        normalized_statement,
        flags=re.IGNORECASE,
    )
    normalized_statement = re.sub(
        r"\s+",
        " ",
        normalized_statement,
    ).strip()

    context = " ".join(
        (
            proposal.statement,
            proposal.question,
            proposal.purpose,
        )
    )
    if (
        normalized_statement == proposal.statement
        and not _contains_initial_image_authenticity_scope(context)
    ):
        return proposal

    question = proposal.question
    purpose = proposal.purpose
    task_context = " ".join((question, purpose))
    if _contains_initial_image_authenticity_scope(task_context):
        question = (
            "Does this image-grounded proposition hold: "
            f"{normalized_statement.rstrip('.')}?"
        )
        purpose = (
            "Verify the image-grounded "
            f"{proposal.predicate.replace('_', ' ')} relation."
        )
    return proposal.model_copy(
        update={
            "statement": normalized_statement,
            "question": question,
            "purpose": purpose,
        }
    )


def _contains_initial_image_authenticity_scope(value: str) -> bool:
    lowered = " ".join(str(value or "").casefold().split())
    return (
        _contains_visual_integrity_scope(value)
        or _contains_fabrication_attribution(value)
        or any(
            phrase in lowered
            for phrase in (
                "genuine photograph",
                "genuine photo",
                "genuine image",
                "authentic photograph",
                "authentic photo",
                "authentic image",
                "real photograph",
                "real photo",
                "real image",
                "real historical event",
                "real historical occurrence",
                "image authenticity",
                "photo authenticity",
                "photograph authenticity",
            )
        )
    )


def _target_describes_unobserved_media_state(value: str) -> bool:
    """Reject initial cores about a hidden pre-edit or alternative source image."""

    lowered = " ".join(str(value or "").casefold().split())
    hidden_media = any(
        phrase in lowered
        for phrase in (
            "original photograph",
            "original photo",
            "original image",
            "unaltered photograph",
            "unaltered photo",
            "unaltered image",
            "unedited photograph",
            "unedited photo",
            "unedited image",
            "underlying photograph",
            "underlying photo",
            "underlying image",
            "background plate",
            "source photograph",
            "source photo",
            "source image",
        )
    )
    counterfactual = any(
        phrase in lowered
        for phrase in (
            "rather than",
            "actually holding",
            "actually wearing",
            "actually shows",
            "before editing",
            "before alteration",
            "before manipulation",
        )
    )
    return hidden_media and (
        counterfactual
        or _contains_visual_integrity_scope(lowered)
        or _contains_fabrication_attribution(lowered)
    )


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


_TARGET_GENERIC_CAPITALIZED_WORDS = {
    "A",
    "An",
    "And",
    "Antarctic",
    "Arctic",
    "Image",
    "Input",
    "Millions",
    "The",
    "This",
}


def _target_introduces_unobserved_named_values(
    statement: str,
    queries: Sequence[str],
    grounding_text: str,
) -> bool:
    """Reject model-memory proper names and dates from pixel-only planning."""

    return bool(
        _unobserved_named_values(
            statement,
            queries,
            grounding_text,
        )
    )


def _unobserved_named_values(
    statement: str,
    queries: Sequence[str],
    grounding_text: str,
) -> List[str]:
    """Return unsupported named/date tokens for actionable planning feedback."""

    grounding = _attribution_tokens(grounding_text)
    rendered = " ".join([statement, *queries])
    missing: List[str] = []
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", rendered)
    missing.extend(
        year
        for year in years
        if year.casefold() not in grounding
    )
    handles = re.findall(r"@[A-Za-z0-9_]+", rendered)
    missing.extend(
        handle
        for handle in handles
        if handle.casefold() not in grounding
    )
    capitalized = re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", rendered)
    missing.extend(
        token
        for token in capitalized
        if token not in _TARGET_GENERIC_CAPITALIZED_WORDS
        and not _named_token_is_grounded(token, grounding)
    )
    return list(dict.fromkeys(missing))


def _named_token_is_grounded(token: str, grounding: set[str]) -> bool:
    lowered = token.casefold()
    if lowered in grounding:
        return True
    # Permit ordinary inflectional/geographic variants such as
    # Antarctic/Antarctica without permitting unrelated remembered names.
    return len(lowered) >= 6 and any(
        len(candidate) >= 6
        and (
            lowered.startswith(candidate)
            or candidate.startswith(lowered)
        )
        for candidate in grounding
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


def latest_evidence_decision(
    state: ImageOnlyInvestigationState,
    *,
    fact_id: str,
) -> EvidenceDecisionRecord | None:
    """Return the latest semantic checkpoint that judged one active fact."""

    return next(
        (
            item
            for item in reversed(state.evidence_decisions)
            if item.output.active_fact_id == fact_id
        ),
        None,
    )


def pending_evidence_decision_ids(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return core Evidence not yet reviewed at a semantic checkpoint."""

    core_id = state.core_verdict_fact_id
    if not core_id:
        return []
    reviewed = {
        evidence_id
        for item in state.evidence_decisions
        if item.output.active_fact_id == core_id
        for evidence_id in item.reviewed_evidence_ids
    }
    return [
        item.evidence_id
        for item in state.evidence
        if core_id in item.fact_ids
        and item.evidence_id not in reviewed
    ]


def evidence_decision_checkpoint_reason(
    state: ImageOnlyInvestigationState,
    *,
    update: Mapping[str, Any] | None = None,
    before_reflection: bool = False,
    before_replan: bool = False,
    before_unverifiable: bool = False,
) -> str:
    """Choose sparse semantic checkpoints without deciding the verdict.

    The deterministic layer only decides whether a batch is worth reviewing.
    Evidence meaning, sufficiency, and image-binding requirements stay with the
    model.
    """

    pending_ids = pending_evidence_decision_ids(state)
    if not pending_ids:
        return ""
    if before_unverifiable:
        return "before_unverifiable"
    if before_replan:
        return "before_replan"
    if before_reflection:
        return "before_reflection"

    created_ids = {
        str(item)
        for item in (update or {}).get("created_evidence_ids", []) or []
    }
    if not created_ids:
        return ""
    evidence_by_id = {
        item.evidence_id: item for item in state.evidence
    }
    new_rows = [
        evidence_by_id[item]
        for item in created_ids
        if item in evidence_by_id
        and item in set(pending_ids)
    ]
    if not new_rows:
        return ""

    core_id = state.core_verdict_fact_id or ""
    prior = latest_evidence_decision(state, fact_id=core_id)
    directly_inspected = [
        item
        for item in new_rows
        if item.tool_name
        in {
            "visit",
            "compare_with_reference",
            "crop_and_inspect",
        }
        and item.directness == "direct"
        and item.quality in {"strong", "moderate"}
    ]
    if directly_inspected and prior is None:
        return "decisive_evidence"
    if (
        prior is not None
        and prior.output.assessment == "insufficient"
        and prior.output.binding_requirement
        in {"same_capture_helpful", "same_capture_required"}
        and any(
            item.evidence_kind == "web_span"
            and item.tool_name == "visit"
            and item.directness == "direct"
            for item in new_rows
        )
    ):
        return "decisive_evidence"
    if any(
        item.tool_name == "compare_with_reference"
        or item.stance == "refute"
        or (
            item.source_class in {"official", "news"}
            and item.directness == "direct"
            and item.quality in {"strong", "moderate"}
        )
        for item in directly_inspected
    ):
        return "decisive_evidence"

    pending_rows = [
        evidence_by_id[item]
        for item in pending_ids
        if item in evidence_by_id
    ]
    direct_families = {
        item.source_family
        for item in pending_rows
        if item.directness == "direct"
        and item.quality in {"strong", "moderate"}
    }
    if len(pending_rows) >= 2 and len(direct_families) >= 2:
        return "decisive_evidence"
    return ""


def apply_evidence_decision(
    state: ImageOnlyInvestigationState,
    output: EvidenceDecisionOutput,
    *,
    reviewed_evidence_ids: Sequence[str],
    trigger: str,
) -> Dict[str, Any]:
    """Apply one model-led semantic decision with auditable boundaries."""

    core_id = state.core_verdict_fact_id
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    core = fact_by_id.get(core_id or "")
    if core is None:
        return {
            "accepted": False,
            "rejected_reason": "no active core fact exists",
        }
    if output.active_fact_id != core.fact_id:
        return {
            "accepted": False,
            "rejected_reason": "decision does not target the active core fact",
        }

    reviewed_ids = list(dict.fromkeys(reviewed_evidence_ids))
    if not reviewed_ids:
        return {
            "accepted": False,
            "rejected_reason": "decision checkpoint has no new Evidence",
        }
    evidence_by_id = {
        item.evidence_id: item for item in state.evidence
    }
    if any(item not in evidence_by_id for item in reviewed_ids):
        return {
            "accepted": False,
            "rejected_reason": "decision checkpoint cites unknown reviewed Evidence",
        }
    selected_ids = list(dict.fromkeys(output.selected_evidence_ids))
    if any(item not in evidence_by_id for item in selected_ids):
        return {
            "accepted": False,
            "rejected_reason": "decision cites unknown Evidence",
        }
    if any(
        core.fact_id not in evidence_by_id[item].fact_ids
        for item in selected_ids
    ):
        return {
            "accepted": False,
            "rejected_reason": "selected Evidence is outside the active core fact",
        }
    task_by_id = {task.task_id: task for task in state.tasks}
    if any(
        evidence_by_id[item].task_id not in task_by_id
        or core.fact_id
        not in task_by_id[evidence_by_id[item].task_id].fact_ids
        for item in selected_ids
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "selected Evidence must belong to a task that owns the active fact"
            ),
        }
    if (
        output.assessment in {"supported", "refuted", "conflicted"}
        and not selected_ids
    ):
        return {
            "accepted": False,
            "rejected_reason": "a material assessment must select Evidence",
        }
    if selected_ids and not set(selected_ids) & set(reviewed_ids):
        return {
            "accepted": False,
            "rejected_reason": (
                "the decision must use at least one newly reviewed Evidence item"
            ),
        }
    if (
        output.assessment in {"supported", "refuted"}
        and output.binding_requirement == "same_capture_required"
        and not any(
            evidence_by_id[item].claim_binding == "same_capture"
            or evidence_by_id[item].same_capture_or_near_duplicate is True
            for item in selected_ids
        )
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "terminal decision requires same-capture binding but selected "
                "Evidence does not provide it"
            ),
        }
    selected_reference_evidence = [
        evidence_by_id[item]
        for item in selected_ids
        if evidence_by_id[item].evidence_kind == "reference_comparison"
    ]
    selected_non_reference_evidence = [
        evidence_by_id[item]
        for item in selected_ids
        if evidence_by_id[item].evidence_kind != "reference_comparison"
    ]
    if (
        output.assessment in {"supported", "refuted"}
        and not selected_non_reference_evidence
        and any(
            not _reference_comparison_is_same_capture(item)
            for item in selected_reference_evidence
        )
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "a different original capture of the same subject or event is "
                "a discovery/refinement bridge, not terminal Evidence; select "
                "fetched source Evidence, same-capture Evidence, or keep the "
                "fact insufficient and refine the discovered visual slot"
            ),
        }
    if (
        output.assessment == "supported"
        and not selected_non_reference_evidence
        and any(
            item.edit_evidence_present is True
            for item in selected_reference_evidence
        )
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "reference comparison with explicit edit evidence cannot "
                "terminally support the active proposition"
            ),
        }
    if (
        output.assessment == "refuted"
        and selected_reference_evidence
        and all(
            item.evidence_kind == "reference_comparison"
            for item in (evidence_by_id[item] for item in selected_ids)
        )
        and not any(
            item.edit_evidence_present is True
            for item in selected_reference_evidence
        )
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "same-capture comparison without explicit edit evidence is an "
                "image-binding bridge, not standalone terminal refutation; "
                "select factual source Evidence or keep the fact insufficient"
            ),
        }

    accepted_refinement_fact_id = ""
    accepted_refinement_task_id = ""
    refinement = output.refinement
    if refinement is not None:
        if output.assessment not in {"insufficient", "conflicted"}:
            return {
                "accepted": False,
                "rejected_reason": (
                    "a resolved fact must stop instead of opening a refinement"
                ),
            }
        anchor_ids = list(dict.fromkeys(refinement.anchor_fact_ids))
        anchors = [
            fact_by_id[item]
            for item in anchor_ids
            if item in fact_by_id
        ]
        if len(anchors) != len(anchor_ids) or not all(
            item.origin.type in {"input_image", "ocr"}
            for item in anchors
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "refinement anchors must be existing pixel/OCR facts"
                ),
            }
        grounding_ids = list(
            dict.fromkeys(refinement.grounding_evidence_ids)
        )
        if not set(grounding_ids) <= set(reviewed_ids):
            return {
                "accepted": False,
                "rejected_reason": (
                    "refinement grounding must use newly reviewed Evidence"
                ),
            }
        if not _refinement_slot_preserves_relation(
            refinement.slot,
            current_predicate=core.predicate,
            proposed_predicate=refinement.predicate,
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    f"{refinement.slot} refinement must preserve the active "
                    "image-world relation; refine the visible subject, place, "
                    "or event rather than switching to source record, platform, "
                    "provenance, or other metadata attribution"
                ),
            }
        if not _valid_visual_refinement_transition(
            core.predicate,
            refinement.predicate,
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "refinement changes the visual relation instead of narrowing it"
                ),
            }
        if (
            not _core_refinement_is_atomic(refinement.statement)
            or _refinement_is_peripheral_metadata(refinement.statement)
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "refinement must remain one image-visible factual relation"
                ),
            }
        if not _refinement_preserves_core_scope(
            core.statement,
            refinement.statement,
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "refinement changes non-target parts of the active visual "
                    "relation instead of filling one unknown slot"
                ),
            }
        if len(state.tasks) >= TOTAL_TASKS_MAX:
            return {
                "accepted": False,
                "rejected_reason": "task budget cannot hold the refinement",
            }

        refinement_fact = VisualFact(
            fact_id=stable_id(
                "vf",
                state.brief.case_id,
                "evidence-refinement",
                refinement.predicate,
                refinement.statement.casefold(),
            ),
            kind=core.kind,
            statement=re.sub(r"\s+", " ", refinement.statement).strip(),
            subject_entity_id=core.subject_entity_id,
            predicate=refinement.predicate,
            object_entity_id=core.object_entity_id,
            status="active",
            basis_ids=list(
                dict.fromkeys(
                    [
                        core.fact_id,
                        *anchor_ids,
                        *grounding_ids,
                    ]
                )
            )[:12],
            decision_relevance="supporting",
            origin=FactOrigin(
                type="web_discovery",
                origin_ids=list(
                    dict.fromkeys(
                        [
                            core.fact_id,
                            *anchor_ids,
                            *grounding_ids,
                        ]
                    )
                )[:8],
            ),
        )
        created_refinement_fact = False
        if refinement_fact.fact_id in fact_by_id:
            refinement_fact = fact_by_id[refinement_fact.fact_id]
        else:
            state.facts.append(refinement_fact)
            fact_by_id[refinement_fact.fact_id] = refinement_fact
            created_refinement_fact = True

        refinement_task = ResearchTask(
            task_id=stable_id(
                "task",
                refinement_fact.fact_id,
                "evidence-refinement",
            ),
            fact_ids=[refinement_fact.fact_id],
            question=refinement.question,
            purpose=refinement.purpose,
            priority=1,
            status="active",
            parent_task_id=None,
            origin_ids=list(
                dict.fromkeys(
                    [
                        refinement_fact.fact_id,
                        core.fact_id,
                        *grounding_ids,
                    ]
                )
            )[:12],
            suggested_tools=list(
                dict.fromkeys(refinement.suggested_tools)
            )[:4],
            suggested_queries=list(
                dict.fromkeys(
                    item.strip()
                    for item in refinement.suggested_queries
                    if item.strip()
                )
            )[:3],
        )
        existing_task = next(
            (
                item
                for item in state.tasks
                if item.task_id == refinement_task.task_id
            ),
            None,
        )
        if existing_task is None:
            state.tasks.append(refinement_task)
        else:
            refinement_task = existing_task
            refinement_task.status = "active"

        accepted_core, reason = reconcile_core_verdict_fact(
            state,
            refinement_fact,
            parent_facts=[core, *anchors],
            allow_refinement=True,
            allow_active_refinement=True,
        )
        if not accepted_core:
            if refinement_task in state.tasks and existing_task is None:
                state.tasks.remove(refinement_task)
            if created_refinement_fact and refinement_fact in state.facts:
                state.facts.remove(refinement_fact)
            return {
                "accepted": False,
                "rejected_reason": reason,
            }
        state.recommended_next_task_ids = list(
            dict.fromkeys(
                [
                    refinement_task.task_id,
                    *state.recommended_next_task_ids,
                ]
            )
        )[:4]
        accepted_refinement_fact_id = refinement_fact.fact_id
        accepted_refinement_task_id = refinement_task.task_id

    finding_ids: List[str] = []
    if (
        output.assessment in {"supported", "refuted"}
        and not accepted_refinement_fact_id
    ):
        stance = (
            "support"
            if output.assessment == "supported"
            else "refute"
        )
        # Evidence text, offsets, URL, artifact, and call provenance remain
        # unchanged. Stance is the audited interpretation relative to the
        # current active proposition, so the semantic checkpoint may correct
        # the extractor's query-relative label.
        for evidence_id in selected_ids:
            evidence_by_id[evidence_id].stance = stance
        evidence_ids_by_task: Dict[str, List[str]] = {}
        for evidence_id in selected_ids:
            evidence_ids_by_task.setdefault(
                evidence_by_id[evidence_id].task_id,
                [],
            ).append(evidence_id)
        for task_id, owned_ids in evidence_ids_by_task.items():
            finding_id = stable_id(
                "finding",
                "evidence-decision",
                core.fact_id,
                task_id,
                stance,
                owned_ids,
                output.rationale,
            )
            if finding_id not in {
                item.finding_id for item in state.findings
            }:
                state.findings.append(
                    Finding(
                        finding_id=finding_id,
                        task_id=task_id,
                        fact_ids=[core.fact_id],
                        statement=output.rationale,
                        stance=stance,
                        evidence_ids=owned_ids,
                        source_family_ids=list(
                            dict.fromkeys(
                                evidence_by_id[item].source_family
                                for item in owned_ids
                            )
                        )[:20],
                        quality="decisive",
                    )
                )
            finding_ids.append(finding_id)
            task = task_by_id[task_id]
            task.finding_ids = list(
                dict.fromkeys([*task.finding_ids, finding_id])
            )[:20]
        for task in state.tasks:
            if core.fact_id not in task.fact_ids:
                continue
            if task.status in {"active", "pending"}:
                task.status = "resolved"

    decision = EvidenceDecisionRecord(
        decision_id=stable_id(
            "evidence-decision",
            state.brief.case_id,
            state.action_count,
            len(state.evidence_decisions) + 1,
            output.model_dump(mode="json"),
            reviewed_ids,
        ),
        action_count=state.action_count,
        trigger=trigger,
        reviewed_evidence_ids=reviewed_ids[:40],
        output=output,
        finding_ids=finding_ids,
        accepted_refinement_fact_id=(
            accepted_refinement_fact_id or None
        ),
    )
    state.evidence_decisions.append(decision)
    if not accepted_refinement_fact_id:
        core.status = {
            "supported": "supported",
            "refuted": "refuted",
            "conflicted": "conflicted",
            "insufficient": "active",
        }[output.assessment]
        refresh_core_evidence_gaps(state)
    return {
        "accepted": True,
        "decision_id": decision.decision_id,
        "finding_ids": finding_ids,
        "accepted_refinement_fact_id": accepted_refinement_fact_id,
        "accepted_refinement_task_id": accepted_refinement_task_id,
        "reviewed_evidence_ids": reviewed_ids,
    }


def _valid_visual_refinement_transition(
    current_predicate: str,
    proposed_predicate: str,
) -> bool:
    if proposed_predicate == current_predicate:
        return True
    return (
        current_predicate == "appears_to_depict"
        and proposed_predicate
        in {
            "identified_as",
            "located_at",
            "depicts_event",
        }
    )


def _refinement_slot_preserves_relation(
    slot: str,
    *,
    current_predicate: str,
    proposed_predicate: str,
) -> bool:
    """Keep a semantic slot refinement inside the active image-world relation."""

    allowed = {
        "subject_identity": {
            current_predicate,
            "identified_as",
        },
        "scene_location": {
            current_predicate,
            "located_at",
            "occurred_at",
            "depicts_event",
        },
        "event_identity": {
            current_predicate,
            "occurred_at",
            "depicts_event",
        },
    }
    return proposed_predicate in allowed.get(slot, set())


def _reference_comparison_is_same_capture(
    evidence: InvestigationEvidence,
) -> bool:
    """Return whether structured comparison fields bind one original capture."""

    return (
        evidence.evidence_kind == "reference_comparison"
        and evidence.claim_binding == "same_capture"
        and evidence.same_subject_or_scene is True
        and evidence.same_capture_or_near_duplicate is True
        and evidence.likely_different_original_capture is not True
    )


_REFINEMENT_SCOPE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "depicted",
    "does",
    "image",
    "in",
    "input",
    "is",
    "of",
    "on",
    "shown",
    "the",
    "this",
    "to",
    "visible",
    "visibly",
}


def _refinement_preserves_core_scope(
    current_statement: str,
    proposed_statement: str,
) -> bool:
    """Reject obvious relation replacement while allowing one slot to narrow.

    Evidence Decision owns the semantic proposal. This guard only checks that
    most of the existing image-grounded relation remains and that named places,
    events, organizations, or dates already present in the active proposition
    are not silently exchanged for different values.
    """

    current_tokens = (
        _attribution_tokens(current_statement) - _REFINEMENT_SCOPE_STOPWORDS
    )
    proposed_tokens = (
        _attribution_tokens(proposed_statement) - _REFINEMENT_SCOPE_STOPWORDS
    )
    if not current_tokens or not proposed_tokens:
        return False
    shared_ratio = len(current_tokens & proposed_tokens) / len(current_tokens)
    if shared_ratio < 0.6:
        return False

    preserved_named_values = {
        item.casefold()
        for item in re.findall(
            r"\b(?:[A-Z][A-Za-z0-9'’-]{2,}|(?:18|19|20)\d{2})\b",
            current_statement,
        )
        if item.casefold() not in {"the", "this", "image", "input"}
    }
    return preserved_named_values <= proposed_tokens


def _refinement_is_peripheral_metadata(statement: str) -> bool:
    text = " ".join(str(statement or "").casefold().split())
    return any(
        token in text
        for token in (
            "photographer",
            "creator",
            "author",
            "asset id",
            "upload date",
            "platform account",
            "stock image title",
        )
    )


def _attribution_tokens(value: str) -> set[str]:
    text = str(value or "").casefold()
    tokens = {
        token
        for token in re.findall(
            r"@?[a-z0-9_]+(?:[.-][a-z0-9_]+)*",
            text,
        )
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


def _target_mixes_visual_integrity_and_world_relation(value: str) -> bool:
    """Keep an initial target to one adjudicable world or pixel proposition."""

    lowered = " ".join(str(value or "").casefold().split())
    has_visual_authenticity = any(
        phrase in lowered
        for phrase in (
            "genuine photograph",
            "genuine photo",
            "authentic photograph",
            "authentic photo",
            "unmodified image",
            "unaltered image",
        )
    )
    has_world_relation = any(
        phrase in lowered
        for phrase in (
            "depicts",
            "depicting",
            "located at",
            "occurred",
            "took place",
            "created by",
            "matches a public",
            "source record",
            "identified as",
        )
    )
    return has_visual_authenticity and has_world_relation


def _contains_fabrication_attribution(value: str) -> bool:
    lowered = " ".join(str(value or "").casefold().split())
    return any(
        phrase in lowered
        for phrase in (
            "digital creation",
            "digital artwork",
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
            "fictional scene",
            "fictional concept",
            "surrealist concept",
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


def _supersede_same_subject_tasks(
    state: ImageOnlyInvestigationState,
    refinement_fact: VisualFact,
) -> None:
    fact_by_id = {fact.fact_id: fact for fact in state.facts}
    for task in state.tasks:
        if refinement_fact.fact_id in task.fact_ids:
            continue
        related = [
            fact_by_id[fact_id]
            for fact_id in task.fact_ids
            if fact_id in fact_by_id
        ]
        if not related or not any(
            fact.subject_entity_id == refinement_fact.subject_entity_id
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
    follow_up_origin_ids = {
        *[item.discovery_id for item in state.discoveries],
        *[item.evidence_id for item in state.evidence],
        *finding_ids,
        *failure_ids,
    }
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
    rejected: List[str] = []

    if len(output.task_updates) > 12:
        rejected.append("task update budget truncated to 12")
    if len(output.new_tasks) > NEW_TASKS_PER_REFLECTION_MAX:
        rejected.append(
            f"new task budget truncated to {NEW_TASKS_PER_REFLECTION_MAX}"
        )
    if len(output.recommended_next_task_ids) > 4:
        rejected.append("recommended task budget truncated to 4")
    if len(output.remaining_gaps) > 8:
        rejected.append("remaining gap budget truncated to 8")
    bounded_output = output.model_copy(
        update={
            "task_updates": output.task_updates[:12],
            "new_tasks": output.new_tasks[:NEW_TASKS_PER_REFLECTION_MAX],
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
    unresolved_core_id = state.core_verdict_fact_id
    if (
        unresolved_core_id
        and fact_by_id.get(unresolved_core_id) is not None
        and fact_by_id[unresolved_core_id].status in {"supported", "refuted"}
    ):
        unresolved_core_id = None
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
        if unresolved_core_id and unresolved_core_id not in task.fact_ids:
            rejected.append(
                f"{task.task_id} does not own the unresolved core fact"
            )
            continue
        if not set(task.origin_ids) <= origin_ids:
            rejected.append(f"{task.task_id} cites unknown origins")
            continue
        if not set(task.origin_ids) & follow_up_origin_ids:
            rejected.append(
                f"{task.task_id} needs a new discovery, evidence, finding, "
                "or failure origin; a bare fact cannot justify another "
                "same-fact search task"
            )
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

    core_task_ids = [
        task.task_id
        for task in state.tasks
        if task.status in {"active", "pending"}
        and state.core_verdict_fact_id in task.fact_ids
    ]
    model_recommendations = [
        task_id
        for task_id in bounded_output.recommended_next_task_ids
        if task_id in task_by_id
        and task_by_id[task_id].status in {"active", "pending"}
    ]
    state.recommended_next_task_ids = list(
        dict.fromkeys([*model_recommendations, *core_task_ids])
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
    if tool_name in {"check_consistency", "analyze_visual_anomalies"}:
        # General VLM consistency/anomaly opinions are diagnostics. They are not
        # calibrated forensic Evidence and cannot support or refute authenticity.
        return None
    if tool_name == "compare_with_reference":
        statement = str(data.get("overall_observation", "")).strip()
        source_url = str(
            data.get("resolved_reference_url")
            or data.get("reference_url", "")
        ).strip()
        kind = "reference_comparison"
        same_subject = bool(data.get("same_subject_or_scene", False))
        same_capture = bool(data.get("same_capture_or_near_duplicate", False))
        different_capture = bool(
            data.get("likely_different_original_capture", False)
        )
        edit_present = bool(data.get("edit_evidence_present", False))
        if not same_subject:
            # An unrelated reference image supplies neither image binding nor a
            # valid alteration baseline. The attempted inspection remains in
            # route history, but the comparison is not promoted to Evidence.
            return None
        claim_binding = "same_capture" if same_capture else "same_subject"
        if edit_present and same_capture and not different_capture:
            stance = "refute"
        else:
            # A visual match binds source context to the input pixels. It does
            # not itself support a location, event, identity, or other world
            # assertion printed on the surrounding page.
            if (
                claim_binding == "same_capture"
                and _task_owns_scene_fact(state, task)
            ):
                stance = "support"
    elif tool_name == "crop_and_inspect":
        statement = str(
            data.get("answer", "") or data.get("description", "")
        ).strip()
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
        else str(data.get("artifact_sha256", "")).strip() or image_sha256
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
        if fact.fact_id == state.core_verdict_fact_id:
            decision = latest_evidence_decision(
                state,
                fact_id=fact.fact_id,
            )
            if decision is None:
                fact.status = "active"
            else:
                fact.status = {
                    "supported": "supported",
                    "refuted": "refuted",
                    "conflicted": "conflicted",
                    "insufficient": "active",
                }[decision.output.assessment]
            continue
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


def _task_has_uninspected_discovery(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    """Keep a task runnable while its own search leads offer a next action."""

    attempts = _attempted_routes_by_task(state).get(task.task_id, [])
    return bool(
        _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="visit",
        )
        or _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="compare_with_reference",
        )
    )


def _task_has_remaining_material_route(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    """Keep a task active while any bounded retrieval or inspection route remains."""

    attempts = _attempted_routes_by_task(state).get(task.task_id, [])
    return bool(
        _remaining_task_material_routes(
            state,
            task,
            attempts,
            global_reverse_branches=_attempted_reverse_branches(state),
        )
    )


def remaining_material_routes(
    state: ImageOnlyInvestigationState,
    *,
    fact_id: str,
) -> List[str]:
    """Return bounded, still-untried routes for one unresolved core fact.

    This is deliberately a route inventory rather than an information-gain
    score.  A failed Lens upload, an empty OCR pass, or a weak query cannot
    establish saturation while another material retrieval or inspection route
    remains.  Conversely, query reformulation is bounded so new wording cannot
    keep an investigation alive indefinitely.
    """

    tasks = [
        task
        for task in state.tasks
        if (
            task.status in {"active", "pending"}
            and fact_id in task.fact_ids
        )
    ]
    if not tasks:
        return []
    attempted = _attempted_routes_by_task(state)
    global_reverse_branches = _attempted_reverse_branches(state)
    routes: List[str] = []
    for task in tasks:
        routes.extend(
            _remaining_task_material_routes(
                state,
                task,
                attempted.get(task.task_id, []),
                global_reverse_branches=global_reverse_branches,
            )
        )
    return list(dict.fromkeys(routes))


def runtime_task_tool_names(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> set[str]:
    """Return Planning tools plus inspection tools implied by real Discoveries.

    Planning proposes useful first-hop tools. It cannot know which concrete page
    or reference image retrieval will discover, so task-owned Discoveries
    deterministically authorize their matching inspection operation.
    """

    allowed = set(task.suggested_tools)
    for discovery in state.discoveries:
        if discovery.task_id != task.task_id:
            continue
        if canonicalize_url(discovery.candidate_url):
            allowed.add("visit")
        if canonicalize_url(discovery.reference_image_url):
            allowed.add("compare_with_reference")
    return allowed


def _attempted_routes_by_task(
    state: ImageOnlyInvestigationState,
) -> Dict[str, List[Mapping[str, Any]]]:
    attempts: Dict[str, List[Mapping[str, Any]]] = {}
    for raw in state.attempted_routes:
        try:
            route = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(route, Mapping):
            continue
        task_id = str(route.get("task_id", "")).strip()
        if task_id:
            attempts.setdefault(task_id, []).append(route)
    return attempts


def _iter_attempted_routes(
    state: ImageOnlyInvestigationState,
) -> Iterator[Mapping[str, Any]]:
    for raw in state.attempted_routes:
        try:
            route = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(route, Mapping):
            yield route


def _attempted_reverse_branches(
    state: ImageOnlyInvestigationState,
) -> set[str]:
    return {
        str(route.get("branch", "lens")).strip().lower() or "lens"
        for route in _iter_attempted_routes(state)
        if str(route.get("tool", "")).strip() == "reverse_image_search"
    }


def _remaining_task_material_routes(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    attempts: Sequence[Mapping[str, Any]],
    *,
    global_reverse_branches: set[str],
) -> List[str]:
    allowed = runtime_task_tool_names(state, task)
    if not allowed:
        return []

    text_search_count = 0
    one_shot_tools: set[str] = set()
    for route in attempts:
        tool_name = str(route.get("tool", "")).strip()
        if tool_name == "text_search":
            text_search_count += 1
        elif tool_name not in {"visit", "compare_with_reference"}:
            one_shot_tools.add(tool_name)

    pending_page_batches = (
        _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="visit",
        )
        if "visit" in allowed
        else []
    )
    pending_reference_batches = (
        _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="compare_with_reference",
        )
        if "compare_with_reference" in allowed
        else []
    )

    pending_routes = [
        *(pending_page_batches[-1] if pending_page_batches else []),
        *(
            pending_reference_batches[-1]
            if pending_reference_batches
            else []
        ),
    ]
    if pending_routes:
        # Each retrieval batch receives bounded, model-selected inspections.
        # Useful Evidence consumes the batch immediately; empty or blocked
        # inspections may expose only the small sibling fallback budget.
        return pending_routes

    routes: List[str] = []
    if "reverse_image_search" in allowed:
        for branch in ("lens", "semantic"):
            if branch not in global_reverse_branches:
                routes.append(
                    f"reverse_image_search:{branch}:{task.task_id}"
                )
    if (
        "text_search" in allowed
        and text_search_count < MAX_TEXT_SEARCH_ROUTES_PER_TASK
    ):
        routes.append(f"text_search:{task.task_id}")
    for tool_name in (
        "ocr_with_position",
        "crop_and_inspect",
        "check_consistency",
        "analyze_visual_anomalies",
    ):
        if tool_name in allowed and tool_name not in one_shot_tools:
            routes.append(f"{tool_name}:{task.task_id}")
    return routes


def _pending_inspection_batches(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    attempts: Sequence[Mapping[str, Any]],
    *,
    tool_name: str,
) -> List[List[str]]:
    """Expose bounded sibling fallbacks after an empty or failed inspection.

    One useful Evidence item consumes the batch. An empty or blocked first page
    may be followed by one sibling candidate, but repeated inspection failure
    cannot turn a retrieval batch into an unbounded page sweep.
    """

    if tool_name not in {"visit", "compare_with_reference"}:
        return []
    attempted_outcomes: Dict[str, str] = {}
    for route in attempts:
        attempted_tool = str(route.get("tool", "")).strip()
        if attempted_tool not in {"visit", "compare_with_reference"}:
            continue
        if attempted_tool == "visit":
            urls = route.get("urls", []) or []
            if isinstance(urls, str):
                urls = [urls]
        else:
            urls = [route.get("reference_url", "")]
        outcome = str(route.get("outcome", "")).strip() or "evidence"
        for raw_url in urls:
            url = canonicalize_url(str(raw_url))
            if url:
                attempted_outcomes[url] = outcome

    discovery_batches: Dict[str, List[InvestigationDiscovery]] = {}
    for discovery in state.discoveries:
        if discovery.task_id == task.task_id:
            discovery_batches.setdefault(
                discovery.function_call_id,
                [],
            ).append(discovery)

    pending: List[List[str]] = []
    for discoveries in discovery_batches.values():
        paired_page_by_reference = {
            canonicalize_url(item.reference_image_url): canonicalize_url(
                item.candidate_url
            )
            for item in discoveries
            if canonicalize_url(item.reference_image_url)
            and canonicalize_url(item.candidate_url)
        }
        candidate_urls = list(
            dict.fromkeys(
                canonicalize_url(
                    item.candidate_url
                    if tool_name == "visit"
                    else item.reference_image_url
                )
                for item in discoveries
                if canonicalize_url(
                    item.candidate_url
                    if tool_name == "visit"
                    else item.reference_image_url
                )
            )
        )[:MAX_INSPECTION_CANDIDATES_PER_BATCH]
        if not candidate_urls:
            continue
        batch_page_urls = {
            canonicalize_url(item.candidate_url)
            for item in discoveries
            if canonicalize_url(item.candidate_url)
        }
        batch_reference_urls = {
            canonicalize_url(item.reference_image_url)
            for item in discoveries
            if canonicalize_url(item.reference_image_url)
        }
        attempted_pages = {
            url for url in batch_page_urls if url in attempted_outcomes
        }
        attempted_references = {
            url for url in batch_reference_urls if url in attempted_outcomes
        }
        if any(
            attempted_outcomes[url] == "evidence"
            for url in attempted_pages
        ):
            # Fetched source text resolves or materially advances the batch;
            # sibling pages and images are no longer mandatory.
            continue
        evidence_reference_urls = [
            url
            for url in attempted_references
            if attempted_outcomes[url] == "evidence"
        ]
        if evidence_reference_urls:
            # A visual match is an image/source bridge, not the surrounding
            # page's factual assertion. Permit only the paired source page as
            # the second inspection; do not sweep sibling references or pages.
            if tool_name != "visit":
                continue
            paired_pages = list(
                dict.fromkeys(
                    paired_page_by_reference.get(url, "")
                    for url in evidence_reference_urls
                    if paired_page_by_reference.get(url, "")
                    and paired_page_by_reference[url] not in attempted_outcomes
                )
            )
            if paired_pages:
                pending.append(
                    [
                        f"visit:{task.task_id}:{url}"
                        for url in paired_pages[:1]
                    ]
                )
            continue
        attempted_count = len(attempted_pages | attempted_references)
        if attempted_count >= MAX_INSPECTION_ATTEMPTS_PER_BATCH:
            continue
        remaining = [
            url for url in candidate_urls if url not in attempted_outcomes
        ]
        if not remaining:
            continue
        prefix = "visit" if tool_name == "visit" else "compare_with_reference"
        pending.append(
            [
                f"{prefix}:{task.task_id}:{url}"
                for url in remaining
            ]
        )
    return pending
