"""Deterministic reducer for the image-only VisualFact investigation state."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from src.orchestrator.investigation_models import (
    BootstrapInvestigation,
    Finding,
    ImageOnlyInvestigationState,
    InvestigationDiscovery,
    InvestigationEvidence,
    InvestigationFailure,
    ReflectionOutput,
    ReflectionRecord,
    ResearchTask,
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


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


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
        dict.fromkeys([*scene_task_ids, *model_recommendations])
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
    elif tool_name == "analyze_visual_anomalies":
        statement = str(data.get("notes", "")).strip()
        if data.get("anomalies") and str(
            data.get("overall_authenticity", "")
        ) in {"likely_ai", "likely_manipulated"}:
            stance = "refute"
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
