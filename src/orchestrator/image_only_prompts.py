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
You are the single investigation agent for an image-only factual verification run.

Choose exactly one active ResearchTask and exactly one tool call per action turn.
Use the task id as question_id. Search for independent source context and inspect
visual details when needed. A search result title, snippet, or reverse-image match is
Discovery only. It is not Evidence until a page is fetched or a visual comparison
successfully observes the relevant property.

Rules:
1. Never use evaluator gold, benchmark labels, or model memory as Evidence.
2. Never invent Evidence, Finding ids, source text, URLs, or tool results.
3. Avoid exact duplicate routes.
4. Prefer priority-1 unresolved tasks, then recommended tasks.
5. For text_search and visit, the immutable verification goal is supplied by runtime.
6. If a Discovery has a non-empty reference_image_url, use
   compare_with_reference to test the visual match and pass its page_url as
   source_page_url so the tool can recover an expired or hotlink-blocked image.
   Visiting only the surrounding page text does not validate that the current image
   matches the reference.
7. Prefer an untested official reference image when one is available. Use visit
   separately to fetch the surrounding caption or event/place context. Supporting a
   full scene proposition requires both a same-capture comparison and a direct source
   assertion; either one alone is incomplete.
8. Discovery, Evidence, Finding, task, and fact state are reduced by the runtime.
   Do not propose or invent state transitions in the segment output.
9. Do not write a verdict. Return only the segment summary and
   ready_for_reflection flag.
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


JUDGMENT_SYSTEM_PROMPT = """\
You are the constrained final synthesizer for reinspect-v2.
The deterministic runtime has already compiled the only allowed verdict and basis.
Return exactly that verdict, policy rule, and allowed ids. Explain the conclusion
concisely using only the supplied facts, Findings, and Evidence. Do not add facts,
citations, or ids. If the compiled verdict is unverifiable, preserve the supplied
fact-specific gaps.
"""


def render_react_context(state: ImageOnlyInvestigationState) -> str:
    active = [
        task
        for task in state.tasks
        if task.status in {"active", "pending"}
    ]
    active.sort(
        key=lambda task: (
            task.task_id not in state.recommended_next_task_ids,
            task.priority,
            task.attempt_count,
            task.task_id,
        )
    )
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
    attempted_reference_urls: set[str] = set()
    for route in state.attempted_routes:
        try:
            parsed_route = json.loads(route)
        except (TypeError, ValueError):
            continue
        if str(parsed_route.get("tool", "")).strip() != "compare_with_reference":
            continue
        args = parsed_route.get("args")
        reference_url = str(parsed_route.get("reference_url", "")).strip()
        if not reference_url and isinstance(args, dict):
            reference_url = str(args.get("reference_url", "")).strip()
        if reference_url:
            attempted_reference_urls.add(canonicalize_url(reference_url))
    active_task_ids = {task.task_id for task in active}
    pending_references = [
        {
            "discovery_id": item.discovery_id,
            "task_id": item.task_id,
            "source_class": classify_source(
                item.reference_image_url
            ).source_class,
            "page_url": item.candidate_url,
            "reference_image_url": item.reference_image_url,
            "title": item.title,
        }
        for item in state.discoveries
        if item.task_id in active_task_ids
        and item.reference_image_url
        and canonicalize_url(item.reference_image_url)
        not in attempted_reference_urls
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
        + "\n\nEligible Evidence:\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
        + "\n\nFindings:\n"
        + json.dumps(findings, ensure_ascii=False, indent=2)
        + "\n\nFailures:\n"
        + json.dumps(failures, ensure_ascii=False, indent=2)
        + "\n\nAttempted semantic routes (do not repeat):\n"
        + json.dumps(attempted_routes, ensure_ascii=False, indent=2)
    )


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
