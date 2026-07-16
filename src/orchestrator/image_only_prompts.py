"""Prompts and compact context renderers for the v3 image-only runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from src.orchestrator.investigation_models import (
    ImageOnlyCoverage,
    ImageOnlyInvestigationState,
    VerdictBasis,
)
from src.orchestrator.source_provenance import classify_source
from src.orchestrator.source_provenance import canonicalize_url


REACT_SYSTEM_PROMPT = """\
Investigate one supplied active task with one tool call. Search results are leads,
not evidence: inspect a promising page or reference image before another retrieval
for that task. Use visual comparison for an image match and webpage text for factual
claims; keep source-record verification, depicted-world facts, and pixel integrity
separate. For screenshots, prefer an original or time-aligned source record.

Use only observed tool results. Return the segment summary and reflection boundary;
the runtime owns task state, evidence, duplicate control, source policy, and verdicts.
"""


REFLECTION_SYSTEM_PROMPT = """\
You are the structured Reflection step of an image-only factual investigation.
Review the global state after a four-action interval.

You may reprioritize tasks, add up to three grounded tasks, propose at most two
decisive facts, recommend next tasks, and identify remaining gaps.

You may not create Evidence or Findings, write a verdict, modify the immutable
brief, delete history, cite unknown ids, or change task status. Task status is
owned by the deterministic Finding/Failure reducer. Return exactly one JSON
object matching the schema.
"""


TARGET_PLANNING_SYSTEM_PROMPT = """\
You are the initial target-planning step of an open-domain image investigation.
Propose up to three pixel-grounded, checkable targets for an open-domain image
investigation. Favor the smallest salient real-world relation: one visible subject
bound to one visible place, event, identity, date, or source record. The target
statement must be the positive proposition that evidence can support or refute.

Choose independent alternatives when the image contains several factual relations.
Use visual integrity only as a separate supporting check when an external factual
relation is available. Do not assume public-web facts or add details absent from the
pixel/OCR state. The runtime validates grounding, atomicity, queries, task state,
and output structure.
"""


TARGET_REFRESH_SYSTEM_PROMPT = """\
A decisive image-grounded target has exhausted its current evidence routes without
resolving the image. Propose up to two new, independent, pixel-grounded factual
targets that could still verify the same scene. Prefer an atomic relation with a
visible subject and place, event, identity, date, or source record for which an
authoritative page could answer the question directly. Do not paraphrase the
exhausted target, use visual-integrity/anomaly claims as a substitute, or assume
outside facts. The runtime validates grounding, duplicate control, and output
structure.
"""


ATTRIBUTION_SYSTEM_PROMPT = """\
You are the attribution-planning step of an image-only factual investigation.
Turn newly retrieved public context into at most two specific, checkable facts
about the central image. Useful facts identify a title, creator, subject, place,
date, event, or an original public record matching a screenshot.

Rules:
1. Every proposal must cite one or more existing parent VisualFact ids and the
   supplied Discovery, Evidence, or Finding ids that ground its wording.
2. A Discovery is a search lead only. It may justify a candidate fact and a
   follow-up query, but it is never Evidence.
3. Do not merely restate a generic scene description or visible OCR text.
4. Do not use model memory or add details absent from the supplied records.
5. For screenshots, bind the visible author/account, distinctive text, displayed
   date, and reply/thread relation when those details are recoverable. Use the
   source_record_matches predicate for that proposition.
6. Prefer one central specific fact over several weak peripheral facts.
7. A decisive proposal must descend from a current decisive fact or its existing
   attribution lineage. Do not promote profile pictures, replies, side objects, or
   unrelated discoveries merely because they are recent.
8. Preserve the investigation question's factual slots. If a task asks for a title,
   creator, place, event, date, or source and a Discovery names that slot, promote
   the named candidate into the fact statement before allowing the broad parent fact
   to resolve.
9. When public context identifies the real-world relation behind a visibly anomalous
   scene, promote that relation itself (for example subject-to-place or
   subject-to-event). Do not replace it with the broader claim that the pixels are
   a digital creation, synthetic, impossible, edited, composite, or AI-generated.
10. Preserve positive image-claim polarity. If sources say the depicted relation is
    false, keep the fact as the positive relation that would make the image real and
    let Evidence refute it. Never promote "does not", "is not", "cannot", "never",
    or "except Antarctica" as the decisive image fact.
11. Do not write a verdict. Return exactly one JSON object matching the schema.
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
    """Expose blocking work plus a planned screenshot integrity companion."""

    active = [
        task
        for task in state.tasks
        if task.status in {"active", "pending"}
    ]
    facts = {fact.fact_id: fact for fact in state.facts}
    unresolved_decisive_ids = {
        fact_id
        for fact_id in state.decisive_fact_ids
        if fact_id in facts
        and facts[fact_id].status not in {"supported", "refuted"}
    }
    blocking = [
        task
        for task in active
        if set(task.fact_ids) & unresolved_decisive_ids
    ]
    supplemental_integrity = [
        task
        for task in active
        if state.brief.media_type == "screenshot"
        and any(
            facts.get(fact_id) is not None
            and facts[fact_id].predicate == "visual_integrity"
            and facts[fact_id].decision_relevance == "supporting"
            and facts[fact_id].status not in {"supported", "refuted"}
            for fact_id in task.fact_ids
        )
    ]
    blocking_task_ids = {task.task_id for task in blocking}
    if blocking:
        active = [
            *blocking,
            *[
                task
                for task in supplemental_integrity
                if task.task_id not in blocking_task_ids
            ],
        ]
    active.sort(
        key=lambda task: (
            task.task_id not in blocking_task_ids,
            task.priority,
            task.task_id not in state.recommended_next_task_ids,
            task.attempt_count,
            task.task_id,
        )
    )
    return active


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
        if item.task_id not in active_task_ids:
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
                    "task_id": item.task_id,
                    "url": item.candidate_url,
                    "title": item.title,
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
                    "task_id": item.task_id,
                    "reference_image_url": item.reference_image_url,
                    "page_url": item.candidate_url,
                    "title": item.title,
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
    return (
        f"Investigation brief: {state.brief.objective}\n"
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


def render_target_refresh_context(
    state: ImageOnlyInvestigationState,
) -> str:
    payload = json.loads(render_target_planning_context(state))
    facts = {fact.fact_id: fact for fact in state.facts}
    payload["exhausted_decisive_targets"] = [
        {
            "fact_id": fact_id,
            "statement": facts[fact_id].statement,
            "predicate": facts[fact_id].predicate,
        }
        for fact_id in state.decisive_fact_ids
        if fact_id in facts
        and facts[fact_id].status in {"exhausted", "blocked", "candidate", "active"}
        and any(
            task.status == "exhausted" and fact_id in task.fact_ids
            for task in state.tasks
        )
    ]
    payload["existing_target_statements"] = [
        {
            "fact_id": fact.fact_id,
            "statement": fact.statement,
            "predicate": fact.predicate,
            "status": fact.status,
        }
        for fact in state.facts
        if fact.predicate
        not in {"appears_to_depict", "visible_in", "reads", "context_suggested_by_text"}
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_reflection_context(state: ImageOnlyInvestigationState) -> str:
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
            "decisive_fact_ids": state.decisive_fact_ids,
            "remaining_actions": max(0, 24 - state.action_count),
        },
        ensure_ascii=False,
        indent=2,
    )


def render_attribution_context(state: ImageOnlyInvestigationState) -> str:
    recent_discoveries = state.discoveries[-12:]
    recent_evidence = state.evidence[-8:]
    recent_findings = state.findings[-8:]
    record_parent_ids = {
        fact_id
        for item in [
            *recent_discoveries,
            *recent_evidence,
            *recent_findings,
        ]
        for fact_id in item.fact_ids
    }
    decisive = [
        fact.model_dump(mode="json")
        for fact in state.facts
        if fact.fact_id in state.decisive_fact_ids
    ]
    existing_attributions = [
        fact.model_dump(mode="json")
        for fact in state.facts
        if fact.origin.type == "web_discovery"
    ]
    return json.dumps(
        {
            "brief": state.brief.model_dump(mode="json"),
            "current_decisive_facts": decisive,
            "record_parent_facts": [
                fact.model_dump(mode="json")
                for fact in state.facts
                if fact.fact_id in record_parent_ids
            ],
            "existing_attribution_facts": existing_attributions[-8:],
            "recent_discoveries": [
                item.model_dump(mode="json")
                for item in recent_discoveries
            ],
            "recent_evidence": [
                item.model_dump(mode="json")
                for item in recent_evidence
            ],
            "recent_findings": [
                item.model_dump(mode="json")
                for item in recent_findings
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


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
        item.evidence_id: item.model_dump(mode="json")
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
