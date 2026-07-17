"""Prompts and compact context renderers for the v3 image-only runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from src.orchestrator.investigation_models import (
    ImageOnlyCoverage,
    ImageOnlyInvestigationState,
    QueryConceptExtractionOutput,
    VerdictBasis,
)
from src.orchestrator.task_store import remaining_material_routes
from src.orchestrator.source_provenance import classify_source
from src.orchestrator.source_provenance import canonicalize_url


REACT_SYSTEM_PROMPT = """\
Investigate one supplied active task with one tool call. It must serve the stated
core fact and an open evidence gap. Search results are leads, not evidence: inspect
a promising page or reference image before another retrieval for that task. Use
visual comparison for an image match and webpage text for factual claims.

Use only observed tool results. Return the segment summary and reflection boundary;
the runtime owns task state, evidence, duplicate control, source policy, and verdicts.
"""


REFLECTION_SYSTEM_PROMPT = """\
You are the structured Reflection step of an image-only factual investigation.
Review the global state after a scheduled interval.

You may reprioritize tasks, add up to three grounded tasks that serve the open core
evidence gaps, recommend next tasks, and identify remaining gaps.

You may not create Evidence or Findings, write a verdict, modify the immutable
brief, delete history, cite unknown ids, change task status, or change the core
fact. Task status, core ownership, coverage, and stopping are deterministic.
Return exactly one JSON object matching the schema.
"""


QUERY_CONCEPT_EXTRACTION_SYSTEM_PROMPT = """\
Extract searchable concepts newly introduced by the supplied exact Evidence, rather
than repeating subjects, products, places, or relations already explicit in the
active proposition or attempted queries. Include each materially distinct novel
concept that could open a different retrieval direction. Each concept must cite one
supplied Evidence id and copy a short exact evidence phrase. Do not translate or
normalize it into a query, choose the next query, judge the proposition, or add
knowledge absent from the Evidence.
"""


QUERY_REPLAN_SYSTEM_PROMPT = """\
Choose one supplied Evidence-derived concept that best closes the remaining gap in
the active proposition, then write one complete replacement web query.

The query must remain about the active proposition while changing the stalled
direction represented by the attempted queries. This is a retrieval hypothesis, not
a verdict. Select exactly one candidate concept, translate or condense it into one
concise concept_term, and include that exact concept_term in the replacement query.
"""


TARGET_PLANNING_SYSTEM_PROMPT = """\
You are the initial target-planning step of an open-domain image investigation.
Return exactly one decisive, pixel-grounded, externally checkable proposition.
Choose the smallest salient positive relation: one visible subject bound to one
visible object or activity, place, event, identity, date, or source record. When the
image visibly combines a subject with another entity or a concrete environment,
plan that visible relation directly (for example, whether a person endorses the
shown product or whether the visible species naturally occurs in the depicted place).

Do not plan image authenticity, manipulation, AI generation, compositing, creator,
title, platform, software, or earliest-source metadata. Those may be later retrieval
context, but they are not the initial factual target. Do not assume public-web facts
or add named values absent from pixel/OCR state. The target statement must be a
positive atomic proposition that later Evidence can support or refute. The runtime
validates grounding, atomicity, task state, and output structure.
"""


EVIDENCE_DECISION_SYSTEM_PROMPT = """\
You are a semantic decision checkpoint for an image-grounded investigation.
Evaluate the active proposition against the supplied eligible Evidence, not against
the wording of the search query that found it. Decide whether the proposition is
supported, refuted, materially conflicted, or still insufficient, and cite only
supplied Evidence ids.

Only the exact webpage span or visual observation contained in eligible_evidence is
factual input. Source class and source family are provenance metadata, not additional
page content. Do not fill gaps from source names, URL wording, search-result
titles/snippets, prior model explanations, or outside knowledge. If the supplied
Evidence text does not itself establish the claimed relation or a logically
incompatible fact, keep the proposition insufficient.

Decide flexibly whether the proposition needs source-to-image binding. Reliable text
can be sufficient for ecological, geographic, temporal, or other world relations.
A same-capture image is required only when the conclusion depends on proving that a
source assertion describes this exact input image; the absence of a reference image
is not itself a reason to keep searching.

Whether a reference is the same photographic capture is not a general quality gate.
A different capture may directly establish a stable visible identity, place, event,
or relation when its actual observation answers the active proposition. It cannot by
itself establish hidden authorization, creator, upload history, provenance, or the
existence of an exact source image. Failure to find an exact capture is never
refutation of a visible-world proposition.

Judge the supplied active proposition as written. Do not replace a subject-to-place,
event, identity, or date relation with the different question of whether the whole
image is authentic, manipulated, composite, or AI-generated. Reliable range,
habitat, chronology, or event evidence that is incompatible with the depicted
relation can be text-sufficient refutation.

If the evidence reveals a more specific visible subject, place, or event but does
not yet resolve the active relation, you may propose one narrower refinement grounded
in supplied pixel/OCR anchor facts and Evidence. You may instead use object_category
once to replace a visible product or object SKU with the smallest category needed by
the evidence, but only while preserving the same visible subject, object entity, and
relation. For example, a named packet may become "the shown health product" when
exact Evidence establishes that category-level impersonation relation. Do not
generalize to every product, replace the subject, or switch to authenticity,
provenance, creator, or platform questions. Creator, title, platform, upload date,
and asset metadata are retrieval context unless visibly part of the image. Do not
invent facts, ids, sources, or a requirement for a second source.

When newly reviewed Evidence introduces a concrete hypothesis about a property
that should be visible in the original pixels, request visual_reinspection before
refining or continuing broad search if a focused re-observation could materially
confirm, contradict, or disambiguate that hypothesis. This includes newly learned
subject identity, visible relation, scene/location cue, event cue, text, or image
integrity. Do not request it for creator, upload date, hidden provenance, or other
facts the image cannot show. Do not request it merely for reassurance after the
active proposition is already supported or refuted. A decision must choose either
visual_reinspection or refinement, never both.
"""


JUDGMENT_SYSTEM_PROMPT = """\
You are the constrained final synthesizer for reinspect-v2.
The deterministic runtime has already compiled the only allowed verdict and basis.
Return exactly that verdict, policy rule, and allowed ids. Explain the conclusion
concisely using only the supplied facts, Findings, and Evidence. Do not add facts,
citations, or ids. If the compiled verdict is unverifiable, preserve the supplied
fact-specific gaps.
"""


def select_react_tasks(
    state: ImageOnlyInvestigationState,
) -> List[Any]:
    """Expose only executable work that owns the unresolved core fact."""

    active = [
        task
        for task in state.tasks
        if task.status in {"active", "pending"}
    ]
    core_id = state.core_verdict_fact_id
    if not core_id:
        return []
    blocking = [
        task
        for task in active
        if core_id in task.fact_ids
    ]
    blocking.sort(
        key=lambda task: (
            task.priority,
            task.task_id not in state.recommended_next_task_ids,
            task.attempt_count,
            task.task_id,
        )
    )
    return blocking


def pending_discovery_routes(
    state: ImageOnlyInvestigationState,
    *,
    task_ids: set[str] | None = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Return task-linked candidate pages/images not yet inspected."""

    attempted_pages: set[str] = set()
    attempted_references: set[str] = set()
    for route in state.attempted_routes:
        try:
            parsed = json.loads(route)
        except (TypeError, ValueError):
            continue
        tool = str(parsed.get("tool", "")).strip()
        if tool == "visit":
            attempted_pages.update(
                canonicalize_url(str(url))
                for url in parsed.get("urls", []) or []
                if canonicalize_url(str(url))
            )
        elif tool == "compare_with_reference":
            reference_url = canonicalize_url(
                str(parsed.get("reference_url", ""))
            )
            if reference_url:
                attempted_references.add(reference_url)

    active_task_ids = task_ids or {
        task.task_id
        for task in state.tasks
        if task.status in {"active", "pending"}
    }
    pages: List[Dict[str, str]] = []
    references: List[Dict[str, str]] = []
    seen_pages: set[str] = set()
    seen_references: set[str] = set()
    for item in state.discoveries:
        if item.task_id not in active_task_ids or item.abandoned:
            continue
        page_url = canonicalize_url(item.candidate_url)
        if (
            page_url
            and page_url not in attempted_pages
            and page_url not in seen_pages
        ):
            seen_pages.add(page_url)
            source_class = classify_source(item.candidate_url).source_class
            pages.append(
                {
                    "discovery_id": item.discovery_id,
                    "function_call_id": item.function_call_id,
                    "task_id": item.task_id,
                    "url": item.candidate_url,
                    "title": item.title,
                    "snippet": item.snippet,
                    "source_class": source_class,
                }
            )
        reference_url = canonicalize_url(item.reference_image_url)
        if (
            reference_url
            and reference_url not in attempted_references
            and reference_url not in seen_references
        ):
            seen_references.add(reference_url)
            source_class = classify_source(
                item.reference_image_url
            ).source_class
            references.append(
                {
                    "discovery_id": item.discovery_id,
                    "function_call_id": item.function_call_id,
                    "task_id": item.task_id,
                    "reference_image_url": item.reference_image_url,
                    "page_url": item.candidate_url,
                    "title": item.title,
                    "snippet": item.snippet,
                    "source_class": source_class,
                }
            )
    source_rank = {
        "official": 0,
        "news": 1,
        "visual": 2,
        "unknown": 3,
        "ugc": 4,
    }
    pages.sort(
        key=lambda item: (
            source_rank.get(item["source_class"], 9),
            item["task_id"],
            item["discovery_id"],
        )
    )
    references.sort(
        key=lambda item: (
            source_rank.get(item["source_class"], 9),
            item["task_id"],
            item["discovery_id"],
        )
    )
    return {
        "pages": pages[:8],
        "references": references[:8],
    }


def render_react_context(state: ImageOnlyInvestigationState) -> str:
    active = select_react_tasks(state)
    facts = {fact.fact_id: fact for fact in state.facts}
    core = facts.get(state.core_verdict_fact_id or "")
    task_lines: List[str] = []
    for task in active[:8]:
        related = [
            facts[fact_id].statement
            for fact_id in task.fact_ids
            if fact_id in facts
        ]
        task_lines.append(
            f"- [{task.task_id}] priority={task.priority}; "
            f"status={task.status}; attempts={task.attempt_count}; "
            f"question={task.question}; facts={related}; "
            f"suggested_tools={task.suggested_tools}; "
            f"suggested_queries={task.suggested_queries}"
        )
    active_task_ids = {task.task_id for task in active}
    remaining_routes = (
        remaining_material_routes(
            state,
            fact_id=core.fact_id,
        )
        if core is not None
        else []
    )
    pending_routes = pending_discovery_routes(
        state,
        task_ids=active_task_ids,
    )
    pending_pages = pending_routes["pages"]
    pending_references = [
        {
            **item,
        }
        for item in pending_routes["references"]
    ]
    pending_references.sort(
        key=lambda item: (
            item["source_class"] != "official",
            item["task_id"],
            item["discovery_id"],
        )
    )
    executable_page_urls = {
        canonicalize_url(route.split(":", 2)[2])
        for route in remaining_routes
        if route.startswith("visit:") and len(route.split(":", 2)) == 3
    }
    executable_reference_urls = {
        canonicalize_url(route.split(":", 2)[2])
        for route in remaining_routes
        if route.startswith("compare_with_reference:")
        and len(route.split(":", 2)) == 3
    }
    pending_pages = [
        item
        for item in pending_pages
        if canonicalize_url(item["url"]) in executable_page_urls
    ]
    pending_references = [
        item
        for item in pending_references
        if canonicalize_url(item["reference_image_url"])
        in executable_reference_urls
    ]
    discoveries = [
        {
            "discovery_id": item.discovery_id,
            "task_id": item.task_id,
            "url": item.candidate_url,
            "reference_image_url": item.reference_image_url,
            "title": item.title,
            "type": item.candidate_type,
        }
        for item in state.discoveries[-16:]
        if not item.abandoned
    ]
    evidence = [
        {
            "evidence_id": item.evidence_id,
            "task_id": item.task_id,
            "fact_ids": item.fact_ids,
            "source_url": item.source_url,
            "stance": item.stance,
            "quality": item.quality,
            "directness": item.directness,
            "exact_text": item.exact_text[:500],
        }
        for item in state.evidence[-16:]
    ]
    findings = [
        item.model_dump(mode="json")
        for item in state.findings[-12:]
    ]
    failures = [
        item.model_dump(mode="json")
        for item in state.failures[-10:]
    ]
    attempted_routes = []
    for route in state.attempted_routes[-16:]:
        try:
            attempted_routes.append(json.loads(route))
        except (TypeError, ValueError):
            continue
    observed_image_context = [
        {
            "fact_id": fact.fact_id,
            "statement": fact.statement,
            "origin": fact.origin.type,
        }
        for fact in state.facts
        if fact.origin.type in {"input_image", "ocr"}
    ][:12]
    observed_retrieval_anchors = [
        item.model_dump(mode="json")
        for item in state.retrieval_anchors[:12]
    ]
    return (
        f"Investigation brief: {state.brief.objective}\n"
        f"Core fact: {core.statement if core else 'none'}\n"
        "Observed image/OCR context:\n"
        + json.dumps(
            observed_image_context,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nObserved retrieval anchors:\n"
        + json.dumps(
            observed_retrieval_anchors,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
        "Open evidence gaps: "
        + json.dumps(
            [
                gap.model_dump(mode="json")
                for gap in state.evidence_gaps
                if gap.status == "open"
            ],
            ensure_ascii=False,
        )
        + "\n"
        f"Real tool actions used: {state.action_count}/24\n"
        f"Next Reflection at action: "
        f"{min(24, ((state.action_count // 4) + 1) * 4)}\n"
        "Active tasks:\n"
        + ("\n".join(task_lines) or "- none")
        + "\n\nRecent Discoveries (not Evidence):\n"
        + json.dumps(discoveries, ensure_ascii=False, indent=2)
        + "\n\nUntested reference images (compare visually before relying on them):\n"
        + json.dumps(pending_references[:8], ensure_ascii=False, indent=2)
        + "\n\nUnvisited candidate pages (visit before another retrieval query):\n"
        + json.dumps(pending_pages[:8], ensure_ascii=False, indent=2)
        + "\n\nEligible Evidence:\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
        + "\n\nFindings:\n"
        + json.dumps(findings, ensure_ascii=False, indent=2)
        + "\n\nFailures:\n"
        + json.dumps(failures, ensure_ascii=False, indent=2)
        + "\n\nRemaining material routes (choose one; do not repeat a route):\n"
        + json.dumps(remaining_routes[:12], ensure_ascii=False, indent=2)
        + "\n\nAttempted semantic routes (do not repeat):\n"
        + json.dumps(attempted_routes, ensure_ascii=False, indent=2)
    )


def render_target_planning_context(
    state: ImageOnlyInvestigationState,
) -> str:
    return json.dumps(
        {
            "brief": state.brief.model_dump(mode="json"),
            "entities": [
                item.model_dump(mode="json")
                for item in state.entities[:24]
            ],
            "pixel_grounded_facts": [
                item.model_dump(mode="json")
                for item in state.facts
                if item.origin.type in {"input_image", "ocr"}
            ][:36],
            "retrieval_anchors": [
                item.model_dump(mode="json")
                for item in state.retrieval_anchors[:24]
            ],
            "bootstrap_tasks": [
                item.model_dump(mode="json")
                for item in state.tasks
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def render_reflection_context(
    state: ImageOnlyInvestigationState,
) -> str:
    attempted_routes = []
    for route in state.attempted_routes[-16:]:
        try:
            attempted_routes.append(json.loads(route))
        except (TypeError, ValueError):
            continue
    return json.dumps(
        {
            "brief": state.brief.model_dump(mode="json"),
            "action_count": state.action_count,
            "tasks": [
                task.model_dump(mode="json")
                for task in state.tasks
            ],
            "facts": [
                fact.model_dump(mode="json")
                for fact in state.facts
            ],
            "findings": [
                item.model_dump(mode="json")
                for item in state.findings
            ],
            "recent_discoveries": [
                item.model_dump(mode="json")
                for item in state.discoveries[-20:]
            ],
            "recent_evidence": [
                item.model_dump(mode="json")
                for item in state.evidence[-12:]
            ],
            "attempted_routes": attempted_routes,
            "source_families": sorted(
                {
                    item.source_family
                    for item in state.evidence
                }
            ),
            "failures": [
                item.model_dump(mode="json")
                for item in state.failures
            ],
            "core_verdict_fact_id": state.core_verdict_fact_id,
            "evidence_gaps": [
                gap.model_dump(mode="json")
                for gap in state.evidence_gaps
            ],
            "remaining_material_routes": (
                remaining_material_routes(
                    state,
                    fact_id=state.core_verdict_fact_id or "",
                )
                if state.core_verdict_fact_id
                else []
            ),
            "remaining_actions": max(0, 24 - state.action_count),
        },
        ensure_ascii=False,
        indent=2,
    )


def render_query_concept_extraction_context(
    state: ImageOnlyInvestigationState,
    *,
    task_id: str,
    new_evidence_ids: List[str],
) -> str:
    task = next(
        (item for item in state.tasks if item.task_id == task_id),
        None,
    )
    core = next(
        (
            item
            for item in state.facts
            if item.fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    evidence_ids = set(new_evidence_ids)
    attempted_queries: List[str] = []
    for raw in state.attempted_routes:
        try:
            route = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if (
            route.get("task_id") != task_id
            or route.get("tool") != "text_search"
        ):
            continue
        values = route.get("queries", []) or []
        if isinstance(values, str):
            values = [values]
        attempted_queries.extend(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    return json.dumps(
        {
            "task_id": task_id,
            "active_proposition": core.statement if core is not None else "",
            "task_question": task.question if task is not None else "",
            "attempted_queries": list(dict.fromkeys(attempted_queries)),
            "new_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "evidence_kind": item.evidence_kind,
                    "claim_binding": item.claim_binding,
                    "exact_text": item.exact_text,
                }
                for item in state.evidence
                if item.evidence_id in evidence_ids
                and state.core_verdict_fact_id in item.fact_ids
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def render_query_replan_context(
    state: ImageOnlyInvestigationState,
    *,
    task_id: str,
    concept_extraction: QueryConceptExtractionOutput,
) -> str:
    task = next(
        (item for item in state.tasks if item.task_id == task_id),
        None,
    )
    core = next(
        (
            item
            for item in state.facts
            if item.fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    attempted_queries: List[str] = []
    for raw in state.attempted_routes:
        try:
            route = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if (
            route.get("task_id") != task_id
            or route.get("tool") != "text_search"
        ):
            continue
        values = route.get("queries", []) or []
        if isinstance(values, str):
            values = [values]
        attempted_queries.extend(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    decision = next(
        (
            item
            for item in reversed(state.evidence_decisions)
            if item.output.active_fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    return json.dumps(
        {
            "task_id": task_id,
            "active_proposition": core.statement if core is not None else "",
            "task_question": task.question if task is not None else "",
            "attempted_queries": list(dict.fromkeys(attempted_queries)),
            "candidate_concepts": [
                item.model_dump(mode="json")
                for item in concept_extraction.concepts
            ],
            "remaining_gap": (
                decision.output.remaining_gap if decision is not None else ""
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def render_evidence_decision_context(
    state: ImageOnlyInvestigationState,
    *,
    reviewed_evidence_ids: List[str],
) -> str:
    core = next(
        (
            fact
            for fact in state.facts
            if fact.fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    reviewed = set(reviewed_evidence_ids)
    core_evidence = [
        item
        for item in state.evidence
        if (
            core is not None
            and core.fact_id in item.fact_ids
        )
    ]
    evidence_rows = [
        {
            **_render_semantic_evidence(item),
            "new_since_last_decision": item.evidence_id in reviewed,
        }
        for item in core_evidence[-24:]
    ]
    pixel_facts = [
        fact.model_dump(mode="json")
        for fact in state.facts
        if fact.origin.type in {"input_image", "ocr"}
    ]
    return json.dumps(
        {
            "brief": state.brief.model_dump(mode="json"),
            "active_fact": (
                core.model_dump(mode="json") if core is not None else None
            ),
            "pixel_or_ocr_anchor_facts": pixel_facts[:36],
            "retrieval_anchors": [
                item.model_dump(mode="json")
                for item in state.retrieval_anchors[:24]
            ],
            "evidence_under_review_ids": reviewed_evidence_ids,
            "eligible_evidence": evidence_rows,
            "prior_evidence_decisions": [
                {
                    "decision_id": item.decision_id,
                    "action_count": item.action_count,
                    "trigger": item.trigger,
                    "reviewed_evidence_ids": item.reviewed_evidence_ids,
                    "assessment": item.output.assessment,
                    "selected_evidence_ids": item.output.selected_evidence_ids,
                    "binding_requirement": item.output.binding_requirement,
                }
                for item in state.evidence_decisions[-4:]
            ],
            "current_evidence_gaps": [
                {
                    "gap_id": item.gap_id,
                    "fact_id": item.fact_id,
                    "kind": item.kind,
                    "status": item.status,
                    "evidence_ids": item.evidence_ids,
                }
                for item in state.evidence_gaps
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _render_semantic_evidence(item: Any) -> Dict[str, Any]:
    """Expose evidence content without retrieval-side semantic leakage."""

    return {
        "evidence_id": item.evidence_id,
        "task_id": item.task_id,
        "fact_ids": item.fact_ids,
        "tool_name": item.tool_name,
        "evidence_kind": item.evidence_kind,
        "source_family": item.source_family,
        "source_class": item.source_class,
        "exact_text": item.exact_text,
        "claim_binding": item.claim_binding,
        "same_subject_or_scene": item.same_subject_or_scene,
        "same_capture_or_near_duplicate": item.same_capture_or_near_duplicate,
        "likely_different_original_capture": (
            item.likely_different_original_capture
        ),
        "edit_evidence_present": item.edit_evidence_present,
        "confidence": item.confidence,
        "temporal_alignment": item.temporal_alignment,
        "risk_flags": item.risk_flags,
        "visual_question_id": item.visual_question_id,
        "visual_scope": item.visual_scope,
        "visual_answer_status": item.visual_answer_status,
        "visual_observations": [
            observation.model_dump(mode="json")
            for observation in item.visual_observations
        ],
    }


def render_judgment_context(
    state: ImageOnlyInvestigationState,
    coverage: ImageOnlyCoverage,
    verdict: str,
    basis: VerdictBasis,
) -> str:
    allowed_facts = {
        fact.fact_id: fact.model_dump(mode="json")
        for fact in state.facts
        if fact.fact_id in basis.fact_ids
    }
    allowed_findings = {
        item.finding_id: item.model_dump(mode="json")
        for item in state.findings
        if item.finding_id in basis.finding_ids
    }
    allowed_evidence = {
        item.evidence_id: _render_semantic_evidence(item)
        for item in state.evidence
        if item.evidence_id in basis.evidence_ids
    }
    return json.dumps(
        {
            "compiled_verdict": verdict,
            "compiled_basis": basis.model_dump(mode="json"),
            "coverage": coverage.model_dump(mode="json"),
            "allowed_facts": allowed_facts,
            "allowed_findings": allowed_findings,
            "allowed_evidence": allowed_evidence,
        },
        ensure_ascii=False,
        indent=2,
    )
