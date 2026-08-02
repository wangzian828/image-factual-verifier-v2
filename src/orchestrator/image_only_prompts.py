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
from src.orchestrator.task_store import (
    discrepancy_visual_reinspection_binding,
    remaining_claim_hypothesis_routes,
    remaining_material_routes,
)
from src.orchestrator.evidence_semantics import (
    evidence_is_qualified_for_stance,
)
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
Review the global state at a scheduled interval or before an unresolved stop.

You may reprioritize tasks, add up to three grounded tasks that serve the open core
evidence gaps, recommend next tasks, and identify remaining gaps. Also choose exactly
one investigation strategy:

- continue: a still-open route has a concrete chance to add decisive information;
- replan: the current direction is stalled, but one genuinely different web query
  could retrieve a named kind of decisive information;
- stop_unresolved: the bounded investigation has no worthwhile new direction.

For replan, name the existing task, provide one replacement query, and state what
decisive information it is expected to recover. Change the investigation angle, not
merely the wording. For continue, explain the concrete remaining route. Do not keep
searching merely because a formal tool route remains.

You may not create Evidence or Findings, write a verdict, modify the immutable
brief, delete history, cite unknown ids, change task status, or change the core
fact. Task status, core ownership, factual coverage, duplicate rejection, immutable
budgets, and final termination enforcement remain deterministic. Return exactly one
JSON object matching the schema.
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
Return exactly one small, decisive, pixel-grounded proposition: subject, event or
context, relation slot, and the value shown by the image. The proposition is the
real-world fact the image asks the viewer to accept, not a list of visible details.

Prefer a specific, discriminative relation over a generic scene description:
an action, identity, number, text, date, object, color, physical position, unusual
anatomy, or similar slot. State the purported world event or property, even when the
pixels may be synthetic or composited. Image integrity is an investigation route,
not a self-confirming claim that the pixels are an illustration. Never replace the
world relation with creator, title, upload history, or other provenance metadata.
When a person, place, event, or artifact is identifiable, a defining factual
relation about that identity may be more decisive than transient scene appearance.
Keep the target positive and atomic. The runtime validates grounding, ownership,
state, and output structure.
"""


DISCREPANCY_REACT_SYSTEM_PROMPT = """\
Choose exactly one runtime-authorized tool action that most reduces uncertainty
about an unresolved ImageClaim. Its attached SearchHypothesis supplies context and
ownership, not a boundary on the investigation. Frame retrieval around what actually
happened, not merely whether the image's proposed value or an identical image can be
found. When direct queries repeat the proposed value without useful evidence, omit
that value and retrieve the actual value of the same relation slot. Prior knowledge
may supply leads, but only tool Evidence establishes a fact.
Do not change the ImageClaim.

Inspect a promising page or reference image before repeating retrieval for that
route. Search titles, snippets, and reverse-image matches are Discovery only.
For page inspection, select one owned ImageClaim and state the passage sought.
Qualified Evidence requires a fetched exact span or a successful visual
observation with recorded provenance. Use only supplied observations, do not decide
a verdict, and do not introduce external identities or metadata as new
ImageClaims. The runtime owns IDs, claim/hypothesis ownership, route duplication,
budgets, Evidence eligibility, state transitions, and stopping.
"""


IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT = """\
You are the Image Account Planning root. Plan an open fact-check of the account
communicated by the image. Return one high-salience ImageClaim containing the
complete subject-event relation, relation slot, and depicted value; add a second
claim only when independently verdict-changing. For an identified person, place,
event, or artifact, a defining factual relation may be more useful than a transient
scene detail. Do not inventory details.

Write each claim as the positive world proposition the image asks the viewer to
accept. Prefer unusual or discriminative visible relations over generic presence or
scene descriptions. Image integrity may be a hypothesis that refutes the purported
world relation, but do not turn an obvious visual style into a claim that proves
itself. SearchHypotheses ask what actually happened and what the slot's verified
value is. Image clues guide retrieval but do not restrict it.
Prior knowledge is a lead; only tool Evidence establishes a fact. Hypotheses do not
own the verdict.

Output: account_summary; image_claims[{claim_key, statement, kind, predicate,
anchor_fact_ids, salience}]; search_hypotheses[{hypothesis_key, statement, queries,
expected_information, suggested_tools, priority}].
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


DISCREPANCY_DECISION_SYSTEM_PROMPT = """\
You are the sparse multimodal Discrepancy Decision checkpoint. Compare the reviewed Evidence with the current image account. Update
only affected Claim assessments; establish a MaterialDiscrepancy only when cited
Evidence and visible anchors support it. You may retire or add a bounded,
non-duplicate hypothesis or request one Evidence-motivated image reinspection.
Omit Claims that have no reviewed owned Evidence.
Use Evidence only in its recorded admissible_stances; neutral Evidence cannot
support or refute a Claim.
Assessment labels describe the exact ImageClaim: support means it is true and
refute means it is false.
Direct Evidence stating a competing value for the same subject-event relation
refutes the depicted value even when the exact depicted wording is absent.
Before supporting a Claim or proposing real, align every newly reviewed source fact
with the pixels. A source may verify that the depicted subject or event exists while
still contradicting a visible attribute or relation in this image. Event-level or
identity-level agreement therefore does not prove that the complete visible account
matches the source.
When reviewed Evidence introduces a concrete value for a property that is visible
but absent, coarse, ambiguous, or potentially incompatible in the current visual
account, request visual_reinspection while budget remains. Phrase the question as a
direct pixel discriminator for that property (for example glove versus bare hand,
on versus beside, or one instrument versus another), and name the competing
alternatives in the question. Set expected_property to one concrete visible
property to verify in the image, not an A-vs-B label; prefer the image-side
candidate when the current image account and source fact already name opposing
values. Use scope=relation for properties of an interaction or spatial relation
(for example a glove on the hand used in a handshake, or on versus beside),
scope=subject for an attribute of one object/person, and scope=text only for
legible text. Do not redirect such a check to generic AI artifact, anatomy,
realism, or provenance inspection unless the Evidence itself introduces an
integrity question.
If the current pixel/OCR anchors do not state either visual alternative, the new
source detail is an unverified visible hypothesis, not yet support or refutation of
the complete image account. In that situation, keep the affected Claim insufficient
and request visual_reinspection; do not create a MaterialDiscrepancy until a pixel
observation establishes which alternative the original image shows.
After focused visual Evidence is available, explicitly reconcile its observations
with the source fact. If the pixels and source disagree on the visible property, do
not ignore the visual observation or treat the source's event match as support for
the complete ImageClaim. In a source-pixel conflict, cite both the source Evidence
and the focused visual Evidence in claim_assessments[].selected_evidence_ids and
material_discrepancy.evidence_ids; keep their recorded stances unchanged while the
Decision records the composite discrepancy.
When resolved_focused_visual_evidence_requirements is non-empty, this is the second
Decision after a source-grounded pixel check. Consume its listed pixel Evidence in a
Claim assessment or MaterialDiscrepancy. Only when the pixel observation genuinely
does not answer the current Claim/discrepancy may you set
visual_evidence_disposition.disposition to
irrelevant_to_current_claim_or_discrepancy and give a concrete rationale. Never
silently revert to source-only support or a source-only verdict after that check.
For a visual_reinspection transition, emit only reason, scope, question, and
expected_property inside visual_reinspection. Leave claim_assessments,
material_discrepancy, retire_hypothesis_ids, new_hypotheses, and
visual_evidence_disposition empty, and keep
verdict_proposal=continue. The runtime binds the unique canonical Claim, pixel/OCR
anchors, and reviewed grounding Evidence from runtime_visual_reinspection_binding;
never copy those IDs into the visual proposal. Request reinspection only when that
binding reports status=available.
When proposing material_discrepancy, do not copy visual_anchor_fact_ids. The
runtime derives the canonical pixel/OCR anchors from affected_claim_ids. Select
only the exact affected Claim(s) and reviewed Evidence chain; never broaden a
discrepancy to another task-owned Claim without its own directional chain.
Task ownership permits review but does not establish semantic coverage; update only
the Claims the Evidence actually addresses and use their allowed visual anchors.
Treat qualified refutation of a high-salience Claim as decisive; unresolved other
Claims do not weaken it. Propose fake for a decisive high-salience discrepancy,
real when all high-salience claims are supported and meaningful routes are closed,
otherwise continue. Use only supplied IDs and return the required JSON schema.
"""


JUDGMENT_SYSTEM_PROMPT = """\
You are the constrained final synthesizer for reinspect-v2.
The deterministic runtime has already compiled the only allowed verdict and basis.
Return exactly that verdict, policy rule, and allowed ids. Explain the conclusion
concisely using only the supplied facts, Findings, and Evidence. Do not add facts,
citations, or ids. If the compiled verdict is unverifiable, preserve the supplied
fact-specific gaps.
"""


DISCREPANCY_JUDGMENT_SYSTEM_PROMPT = """\
You are the constrained final synthesizer for discrepancy-first-v4. Return only
the binary verdict, confidence, and a concise assessment of the supplied compiled
basis. When compiled_verdict is non-empty, reproduce it. Do not emit claim,
discrepancy, finding, Evidence, or gap IDs; the runtime injects those from the
accepted investigation state. Do not add historical facts or reopen search.
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


def select_discrepancy_react_tasks(
    state: ImageOnlyInvestigationState,
) -> List[Any]:
    """Expose active tasks that still own an executable investigation route.

    A task can remain ``active`` after its bounded routes have all been tried.
    It must not win scheduling merely because it has a lower priority or fewer
    attempts: doing so leaves the caller with an empty tool schema even when a
    sibling task still has a visit or retrieval route.  Keep archive reads as
    a separate executable phase; while one is pending, any valid owner remains
    selectable so the caller can expose only ``read_evidence``.
    """

    claims = {claim.claim_id: claim for claim in state.image_claims}
    hypotheses = {
        item.hypothesis_id: item
        for item in state.search_hypotheses
        if item.status in {"open", "active"}
    }
    active = [
        task
        for task in state.tasks
        if task.status in {"active", "pending"}
        and task.hypothesis_id in hypotheses
        and task.claim_ids
        and set(task.claim_ids) <= set(claims)
        and set(task.claim_ids) <= set(hypotheses[task.hypothesis_id].claim_ids)
        and any(
            claims[claim_id].status in {"open", "conflicted", "unresolved"}
            for claim_id in task.claim_ids
        )
        and (
            bool(state.pending_archive_read_ids)
            or bool(
                remaining_claim_hypothesis_routes(
                    state,
                    task_ids={task.task_id},
                )
            )
        )
    ]
    active.sort(
        key=lambda task: (
            task.priority,
            task.task_id not in state.recommended_next_task_ids,
            min(
                0 if claims[claim_id].salience == "high" else 1
                for claim_id in task.claim_ids
            ),
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


def render_discrepancy_react_context(
    state: ImageOnlyInvestigationState,
    *,
    task_ids: set[str] | None = None,
) -> str:
    """Render the v4 claim/hypothesis loop without any core-fact fallback."""

    active = select_discrepancy_react_tasks(state)
    if task_ids is not None:
        active = [task for task in active if task.task_id in task_ids]
    active_task_ids = {task.task_id for task in active}
    claims = {claim.claim_id: claim for claim in state.image_claims}
    hypotheses = {
        item.hypothesis_id: item for item in state.search_hypotheses
    }
    pending_routes = pending_discovery_routes(
        state,
        task_ids=active_task_ids,
    )
    remaining_routes = remaining_claim_hypothesis_routes(
        state,
        task_ids=active_task_ids,
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
    attempted_routes: List[Dict[str, Any]] = []
    for route in state.attempted_routes[-24:]:
        try:
            attempted_routes.append(json.loads(route))
        except (TypeError, ValueError):
            continue
    return json.dumps(
        {
            "image_account_summary": state.image_account_summary,
            "image_claims": [
                claim.model_dump(mode="json")
                for claim in state.image_claims
            ],
            "active_search_hypotheses": [
                hypothesis.model_dump(mode="json")
                for hypothesis in state.search_hypotheses
                if hypothesis.status in {"open", "active"}
            ],
            "active_tasks": [
                {
                    **task.model_dump(mode="json"),
                    "owned_claims": [
                        claims[claim_id].model_dump(mode="json")
                        for claim_id in task.claim_ids
                    ],
                    "owned_hypothesis": hypotheses[
                        task.hypothesis_id
                    ].model_dump(mode="json"),
                }
                for task in active
            ],
            "recent_discoveries": [
                item.model_dump(mode="json")
                for item in state.discoveries[-16:]
                if not item.abandoned
            ],
            "eligible_evidence": [
                item.model_dump(mode="json")
                for item in state.evidence[-20:]
            ],
            "claim_assessments": [
                item.model_dump(mode="json")
                for item in state.claim_assessments[-12:]
            ],
            "material_discrepancies": [
                item.model_dump(mode="json")
                for item in state.material_discrepancies
            ],
            "pending_pages": [
                item
                for item in pending_routes["pages"]
                if canonicalize_url(item["url"]) in executable_page_urls
            ],
            "pending_reference_images": [
                item
                for item in pending_routes["references"]
                if canonicalize_url(item["reference_image_url"])
                in executable_reference_urls
            ],
            "failures": [
                item.model_dump(mode="json")
                for item in state.failures[-12:]
            ],
            "remaining_routes": remaining_routes[:16],
            "attempted_routes": attempted_routes,
            "action_count": state.action_count,
            "remaining_action_budget": max(0, 24 - state.action_count),
        },
        ensure_ascii=False,
        indent=2,
    )


def render_image_account_planning_context(
    state: ImageOnlyInvestigationState,
    *,
    perception: Any = None,
) -> str:
    """Render a loss-aware observation projection for initial v4 Planning."""

    perception_payload: Dict[str, Any] = {}
    if perception is not None:
        if hasattr(perception, "model_dump"):
            perception_payload = perception.model_dump(mode="json")
        elif isinstance(perception, dict):
            perception_payload = dict(perception)
    entities: List[Dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    for item in perception_payload.get("entities", []):
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("entity_type", "")).strip()
        key = (name.casefold(), entity_type.casefold())
        if not name or key in seen_entities:
            continue
        seen_entities.add(key)
        entities.append(
            {
                "name": name,
                "entity_type": entity_type,
                "confidence": item.get("confidence", 0.0),
            }
        )

    positioned_ocr: List[Dict[str, Any]] = []
    seen_ocr: set[tuple[str, str]] = set()
    for item in perception_payload.get("text_regions", []):
        text = str(item.get("text", "")).strip()
        quad = item.get("bbox_quad") or []
        region: List[float] = []
        if len(quad) == 4 and all(len(point) == 2 for point in quad):
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            region = [min(xs), min(ys), max(xs), max(ys)]
        key = (text.casefold(), json.dumps(region))
        if not text or key in seen_ocr:
            continue
        seen_ocr.add(key)
        positioned_ocr.append(
            {
                "text": text,
                "region": region,
                "confidence": item.get("confidence", 0.0),
            }
        )

    visual_fact_anchors: List[Dict[str, Any]] = []
    seen_fact_statements: set[str] = set()
    for item in state.facts:
        if item.origin.type not in {"input_image", "ocr"}:
            continue
        # OCR bootstrap creates a second, mechanically derived
        # context_suggested_by_text relation for every literal token. Planning
        # retains the literal text_claim and image fact, so this duplicate adds
        # no observation and previously doubled the prompt.
        if item.predicate == "context_suggested_by_text":
            continue
        statement_key = " ".join(item.statement.casefold().split())
        if statement_key in seen_fact_statements:
            continue
        seen_fact_statements.add(statement_key)
        visual_fact_anchors.append(
            {
                "fact_id": item.fact_id,
                "kind": item.kind,
                "statement": item.statement,
                "predicate": item.predicate,
            }
        )

    retrieval_clues: List[Dict[str, Any]] = []
    seen_clues: set[tuple[str, str]] = set()
    for item in state.retrieval_anchors:
        key = (item.kind, " ".join(item.value.casefold().split()))
        if key in seen_clues:
            continue
        seen_clues.add(key)
        retrieval_clues.append(
            {
                "kind": item.kind,
                "value": item.value,
                "confidence": item.confidence,
            }
        )

    # Do not include bootstrap ResearchTasks here. They are deterministic
    # retrieval scaffolding created before semantic Planning and are replaced
    # by the accepted SearchHypotheses. Presenting them as input made smaller
    # models mistake internal state for the requested output schema.
    return json.dumps(
        {
            "context_role": "observations_only",
            "case": {
                "case_id": state.brief.case_id,
                "input_mode": state.brief.input_mode,
                "media_type": state.brief.media_type,
            },
            "scene": {
                "description": perception_payload.get("scene_description", ""),
                "image_type": perception_payload.get(
                    "image_type", state.brief.media_type
                ),
            },
            "salient_entities": entities,
            "positioned_ocr": positioned_ocr,
            "visual_fact_anchors": visual_fact_anchors,
            "retrieval_clues": retrieval_clues,
            "planning_limits": {
                "image_claims": 3,
                "high_salience_image_claims": 1,
                "search_hypotheses": 6,
                "queries_per_hypothesis": 3,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def render_discrepancy_decision_context(
    state: ImageOnlyInvestigationState,
    *,
    reviewed_evidence_ids: Iterable[str],
    trigger: str,
) -> str:
    """Render one sparse v4 decision checkpoint from canonical state only."""

    reviewed = list(dict.fromkeys(str(item) for item in reviewed_evidence_ids))
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    task_by_id = {item.task_id: item for item in state.tasks}
    reviewed_set = set(reviewed)
    reviewed_evidence_ownership = []
    reviewable_claim_ids: List[str] = []
    for evidence_id in reviewed:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        task = task_by_id.get(evidence.task_id)
        claim_ids = list(task.claim_ids) if task is not None else []
        reviewable_claim_ids.extend(claim_ids)
        reviewed_evidence_ownership.append(
            {
                "evidence_id": evidence_id,
                "task_id": evidence.task_id,
                "claim_ids": claim_ids,
                "hypothesis_id": (
                    task.hypothesis_id if task is not None else None
                ),
            }
        )
    claim_update_space = [
        {
            "claim_id": claim.claim_id,
            "allowed_visual_anchor_fact_ids": list(claim.anchor_fact_ids),
        }
        for claim in state.image_claims
    ]
    reviewed_directional_chains = []
    for finding in state.findings:
        task = task_by_id.get(finding.task_id)
        if task is None or finding.stance not in {"support", "refute"}:
            continue
        chain_evidence_ids = [
            evidence_id
            for evidence_id in finding.evidence_ids
            if evidence_id in reviewed_set
            and evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].task_id == finding.task_id
            and evidence_is_qualified_for_stance(
                evidence_by_id[evidence_id],
                finding.stance,
            )
        ]
        if not chain_evidence_ids:
            continue
        reviewed_directional_chains.append(
            {
                "stance": finding.stance,
                "finding_id": finding.finding_id,
                "evidence_ids": chain_evidence_ids,
                "task_id": finding.task_id,
                "task_owned_claim_ids": list(task.claim_ids),
            }
        )
    runtime_visual_binding = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=reviewed,
    )
    binding_hint_by_claim_id = {
        str(item.get("claim_id")): str(
            item.get("source_visible_property_hint", "")
        )
        for item in runtime_visual_binding.get("candidates", [])
        if isinstance(item, dict)
    }
    visual_alignment_candidates = []
    if len(state.visual_reinspections) < 1:
        facts_by_id = {item.fact_id: item for item in state.facts}
        claims_by_id = {item.claim_id: item for item in state.image_claims}
        for ownership in reviewed_evidence_ownership:
            evidence_id = ownership["evidence_id"]
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.evidence_kind == "image_region":
                continue
            for claim_id in ownership["claim_ids"]:
                claim = claims_by_id.get(claim_id)
                if claim is None or claim.status not in {
                    "open",
                    "unresolved",
                    "conflicted",
                }:
                    continue
                if claim.fact_id not in evidence.fact_ids:
                    continue
                visual_alignment_candidates.append(
                    # This is advisory context, not a deterministic verdict rule.
                    # The Decision stage still owns the semantic judgment, but it
                    # now sees the exact source text beside the current pixel
                    # anchors so evidence-introduced visible attributes are less
                    # likely to be mistaken for complete image-account support.
                    {
                        "evidence_id": evidence_id,
                        "evidence_text": evidence.exact_text[:1200],
                        "claim_id": claim_id,
                        "claim_statement": claim.statement,
                        "claim_status": claim.status,
                        "source_visible_property_hint": (
                            binding_hint_by_claim_id.get(claim_id, "")
                        ),
                        "current_image_account": state.image_account_summary,
                        "allowed_visual_anchors": [
                            {
                                "fact_id": fact_id,
                                "statement": facts_by_id[fact_id].statement,
                            }
                            for fact_id in claim.anchor_fact_ids
                            if fact_id in facts_by_id
                        ],
                        "required_review": (
                            "Check whether this Evidence introduces a concrete "
                            "visible attribute or relation that is absent, coarse, "
                            "ambiguous, or potentially incompatible in the current "
                            "visual account. If so, request one targeted "
                            "visual_reinspection before supporting the Claim or "
                            "proposing real."
                        ),
                    }
                )
    resolved_focused_visual_evidence_requirements = []
    for record in state.visual_reinspections:
        if (
            record.status != "resolved"
            or not (
                set(record.request.grounding_evidence_ids)
                & reviewed_set
            )
        ):
            continue
        visual_evidence_ids = [
            evidence_id
            for evidence_id in record.evidence_ids
            if evidence_id in reviewed_set
            and evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].evidence_kind == "image_region"
            and evidence_by_id[evidence_id].tool_name
            == "focused_visual_inspection"
            and evidence_by_id[evidence_id].visual_question_id
            == record.visual_question_id
        ]
        if visual_evidence_ids:
            resolved_focused_visual_evidence_requirements.append(
                {
                    "visual_question_id": record.visual_question_id,
                    "question": record.request.question,
                    "expected_property": record.request.expected_property,
                    "grounding_evidence_ids": list(
                        record.request.grounding_evidence_ids
                    ),
                    "visual_evidence_ids": visual_evidence_ids,
                    "required_review": (
                        "Consume the pixel Evidence in this Decision or explicitly "
                        "record why it is irrelevant to the current "
                        "Claim/discrepancy."
                    ),
                }
            )
    return json.dumps(
        {
            "trigger": trigger,
            "image_account_summary": state.image_account_summary,
            "image_claims": [
                item.model_dump(mode="json") for item in state.image_claims
            ],
            "search_hypotheses": [
                item.model_dump(mode="json")
                for item in state.search_hypotheses
            ],
            "reviewed_evidence": [
                {
                    **evidence_by_id[evidence_id].model_dump(mode="json"),
                    "admissible_stances": [
                        stance
                        for stance in ("support", "refute")
                        if evidence_is_qualified_for_stance(
                            evidence_by_id[evidence_id],
                            stance,
                        )
                    ],
                }
                for evidence_id in reviewed
                if evidence_id in evidence_by_id
            ],
            "reviewed_evidence_ownership": reviewed_evidence_ownership,
            "reviewable_claim_ids": list(dict.fromkeys(reviewable_claim_ids)),
            "claim_update_space": claim_update_space,
            "reviewed_directional_chains": reviewed_directional_chains,
            "evidence_to_visual_alignment_candidates": (
                visual_alignment_candidates[:12]
            ),
            "runtime_visual_reinspection_binding": runtime_visual_binding,
            "resolved_focused_visual_evidence_requirements": (
                resolved_focused_visual_evidence_requirements[:4]
            ),
            "ownership_note": (
                "Ownership permits review; it does not prove that Evidence "
                "semantically addresses every owned Claim."
            ),
            "prior_claim_assessments": [
                item.model_dump(mode="json")
                for item in state.claim_assessments[-12:]
            ],
            "prior_material_discrepancies": [
                item.model_dump(mode="json")
                for item in state.material_discrepancies
            ],
            "pixel_ocr_anchor_facts": [
                item.model_dump(mode="json")
                for item in state.facts
                if item.origin.type in {"input_image", "ocr"}
            ][:48],
            "attempted_routes": [
                json.loads(route)
                for route in state.attempted_routes[-24:]
                if _is_json_object(route)
            ],
            "remaining_routes": remaining_claim_hypothesis_routes(state)[:16],
            "action_count": state.action_count,
            "remaining_action_budget": max(0, 24 - state.action_count),
            "remaining_hypothesis_budget": max(
                0,
                12 - len(state.search_hypotheses),
            ),
            "remaining_visual_reinspection_budget": max(
                0,
                1 - len(state.visual_reinspections),
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _is_json_object(value: Any) -> bool:
    try:
        return isinstance(json.loads(str(value)), dict)
    except (TypeError, ValueError):
        return False


def render_discrepancy_judgment_context(
    state: ImageOnlyInvestigationState,
    compiled_verdict: str,
    basis: Any,
) -> str:
    claims = {item.claim_id: item for item in state.image_claims}
    discrepancies = {
        item.discrepancy_id: item for item in state.material_discrepancies
    }
    findings = {item.finding_id: item for item in state.findings}
    evidence = {item.evidence_id: item for item in state.evidence}
    facts = {item.fact_id: item for item in state.facts}
    return json.dumps(
        {
            "compiled_verdict": compiled_verdict,
            "compiled_basis": basis.model_dump(mode="json"),
            "selected_claims": [
                claims[item].model_dump(mode="json")
                for item in basis.claim_ids
                if item in claims
            ],
            "selected_discrepancies": [
                discrepancies[item].model_dump(mode="json")
                for item in basis.discrepancy_ids
                if item in discrepancies
            ],
            "selected_visual_anchors": [
                facts[item].model_dump(mode="json")
                for item in basis.visual_anchor_fact_ids
                if item in facts
            ],
            "selected_findings": [
                findings[item].model_dump(mode="json")
                for item in basis.finding_ids
                if item in findings
            ],
            "selected_evidence": [
                evidence[item].model_dump(mode="json")
                for item in basis.evidence_ids
                if item in evidence
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def render_reflection_context(
    state: ImageOnlyInvestigationState,
    *,
    trigger: str = "interval",
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
            "strategy_trigger": trigger,
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
            "latest_coverage": (
                state.coverage_audits[-1].model_dump(mode="json")
                if state.coverage_audits
                else None
            ),
            "strategy_replan_budget": [
                {
                    "task_id": task.task_id,
                    "remaining": max(0, 1 - task.query_replan_count),
                    "attempt_count": task.attempt_count,
                    "suggested_queries": task.suggested_queries,
                }
                for task in state.tasks
                if state.core_verdict_fact_id in task.fact_ids
                and task.status in {"active", "pending", "exhausted"}
                and "text_search" in task.suggested_tools
            ],
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
                    "relation_scope": item.relation_scope,
                    "relation_stance": item.relation_stance,
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
        "relation_scope": item.relation_scope,
        "relation_stance": item.relation_stance,
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
