"""Deterministic reducer for the image-only VisualFact investigation state."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from src.orchestrator.investigation_models import (
    BootstrapInvestigation,
    ClaimAssessment,
    DiscrepancyDecisionProposalOutput,
    DiscrepancyDecisionOutput,
    DiscrepancyDecisionRecord,
    EvidenceDecisionOutput,
    EvidenceDecisionRecord,
    FactOrigin,
    Finding,
    EvidenceGap,
    ImageAccountPlanningOutput,
    ImageClaim,
    ImageOnlyInvestigationState,
    InvestigationDiscovery,
    InvestigationEvidence,
    InvestigationFailure,
    MaterialDiscrepancy,
    NewSearchHypothesis,
    QueryConceptExtractionOutput,
    QueryReplanOutput,
    QueryReplanRecord,
    ReflectionOutput,
    ReflectionRecord,
    ResearchTask,
    RouteLocalReplanOutput,
    RouteLocalReplanRecord,
    SearchHypothesis,
    TargetFactProposal,
    TargetPlanningOutput,
    VisualFact,
    VisualObservation,
    VisualReinspectionRecord,
    VisualReinspectionRequest,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.evidence_policy import query_policy_violation
from src.orchestrator.evidence_semantics import (
    evidence_is_qualified,
    evidence_is_qualified_for_stance,
    required_assessment_stances,
    same_capture_can_support_visual_claim,
)
from src.orchestrator.route_policy import (
    route_signature,
    routes_semantically_equivalent,
)
from src.orchestrator.source_provenance import (
    canonicalize_url,
    classify_source,
)
from src.orchestrator.tool_result import parse_tool_result


MAX_TOOL_ACTIONS = 24
REFLECTION_INTERVAL = 4
MAX_REFLECTIONS = 7
INITIAL_TASKS_MAX = 4
TOTAL_TASKS_MAX = 12
NEW_TASKS_PER_REFLECTION_MAX = 3
MAX_TEXT_SEARCH_ROUTES_PER_TASK = 2
MAX_ARCHIVE_RECALL_ROUTES_PER_TASK = 2
MAX_CORE_FACT_REFINEMENTS = 1
MAX_TEXT_SEARCH_CANDIDATES_PER_BATCH = 10
MAX_REVERSE_SEARCH_CANDIDATES_PER_BATCH = 3
MAX_VISUAL_REINSPECTIONS = 2
MAX_IMAGE_CLAIMS = 3
MAX_SEARCH_HYPOTHESES = 3
MAX_NEW_HYPOTHESES_PER_DECISION = 3
MAX_V4_VISUAL_REINSPECTIONS = 1
ROOT_IMAGE_TARGET = "root_image"
REVERSE_IMAGE_SEARCH_BRANCHES = ("lens", "semantic")


def _unique_visual_view_artifacts(
    items: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep stable visual artifact records without hashing mapping objects."""

    unique: List[Dict[str, Any]] = []
    fingerprints: set[str] = set()
    for item in items:
        normalized = dict(item)
        fingerprint = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(normalized)
    return unique


COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY = (
    "composite:source_visual_discrepancy"
)
_GENERIC_VISUAL_REQUEST_TOKENS = {
    "check",
    "confirm",
    "determine",
    "evidence",
    "focus",
    "focused",
    "image",
    "inspect",
    "observation",
    "original",
    "photo",
    "photograph",
    "picture",
    "pixel",
    "pixels",
    "property",
    "question",
    "review",
    "see",
    "show",
    "shows",
    "text",
    "visible",
    "visual",
}
_INTEGRITY_REQUEST_TOKENS = {
    "ai",
    "anatomy",
    "artifact",
    "artifacts",
    "digital",
    "fake",
    "generated",
    "manipulated",
    "photoshop",
    "provenance",
    "realism",
}
_INTEGRITY_EVIDENCE_TOKENS = {
    "ai",
    "artificial",
    "digitally",
    "fake",
    "generated",
    "manipulated",
    "photoshop",
    "synthetic",
}
_VISUAL_DISCRIMINATOR_STOP_TOKENS = {
    "about",
    "after",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "has",
    "have",
    "image",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "photo",
    "photograph",
    "pictured",
    "says",
    "shows",
    "source",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def discrepancy_visual_reinspection_binding(
    state: ImageOnlyInvestigationState,
    *,
    reviewed_evidence_ids: Sequence[str],
) -> Dict[str, Any]:
    """Resolve a visual proposal to one claim using recorded atomic fact binding."""

    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    fact_by_id = {item.fact_id: item for item in state.facts}
    task_by_id = {item.task_id: item for item in state.tasks}
    reviewed_ids = list(dict.fromkeys(reviewed_evidence_ids))
    candidates: List[Dict[str, Any]] = []
    for claim in state.image_claims:
        if claim.status not in {"open", "unresolved", "conflicted"}:
            continue
        anchor_fact_ids = [
            fact_id
            for fact_id in claim.anchor_fact_ids
            if fact_id in fact_by_id
            and fact_by_id[fact_id].origin.type in {"input_image", "ocr"}
        ][:6]
        grounding_by_task: Dict[str, Dict[str, List[str]]] = {}
        for evidence_id in reviewed_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.evidence_kind == "image_region":
                continue
            if evidence.evidence_kind == "web_span" and (
                evidence.directness != "direct"
                or evidence.claim_binding != "source_assertion"
                or evidence.relation_scope != "same_relation"
                or evidence.relation_stance not in {"supports", "contradicts"}
            ):
                continue
            task = task_by_id.get(evidence.task_id)
            if (
                claim.fact_id in evidence.fact_ids
                and task is not None
                and claim.claim_id in task.claim_ids
            ):
                bucket = grounding_by_task.setdefault(
                    evidence.task_id,
                    {"evidence_ids": [], "source_fragments": []},
                )
                bucket["evidence_ids"].append(evidence_id)
                bucket["source_fragments"].append(evidence.exact_text)
        if not anchor_fact_ids:
            continue
        for source_task_id, bucket in grounding_by_task.items():
            grounding_evidence_ids = bucket["evidence_ids"]
            if not grounding_evidence_ids:
                continue
            candidates.append(
                {
                    "claim_id": claim.claim_id,
                    "claim_fact_id": claim.fact_id,
                    "source_task_id": source_task_id,
                    "anchor_fact_ids": anchor_fact_ids,
                    "grounding_evidence_ids": grounding_evidence_ids[:8],
                    "source_evidence_excerpt": _clip_text(
                        " ".join(bucket["source_fragments"]),
                        1200,
                    ),
                }
            )
    status = (
        "available"
        if len(candidates) == 1
        else "ambiguous"
        if candidates
        else "unavailable"
    )
    return {
        "status": status,
        "candidates": candidates,
        "binding": candidates[0] if status == "available" else None,
    }


def evidence_serves_claim(
    state: ImageOnlyInvestigationState,
    evidence: InvestigationEvidence,
    *,
    claim_id: str,
    claim_fact_id: str,
    task_by_id: Mapping[str, ResearchTask] | None = None,
) -> bool:
    """Return whether Evidence is owned by and materially serves one Claim."""

    tasks = task_by_id or {item.task_id: item for item in state.tasks}
    task = tasks.get(evidence.task_id)
    if task is None or claim_id not in task.claim_ids:
        return False
    if claim_fact_id in evidence.fact_ids:
        return True
    return claim_owned_pixel_evidence_is_eligible(
        evidence,
        claim_id=claim_id,
        task_by_id=tasks,
    )


def claim_owned_pixel_evidence_is_eligible(
    evidence: InvestigationEvidence,
    *,
    claim_id: str,
    task_by_id: Mapping[str, ResearchTask],
    require_qualified: bool = False,
) -> bool:
    """Return whether Evidence is a claim-owned, usable pixel observation.

    The producing tool and any reinspection record are provenance metadata, not
    semantic eligibility conditions. ``require_qualified`` is used only where
    the observation is allowed to carry a material decision.
    """

    task = task_by_id.get(evidence.task_id)
    if (
        evidence.evidence_kind != "image_region"
        or evidence.claim_binding != "pixel_observation"
        or evidence.stance != "neutral"
        or evidence.directness != "direct"
        or evidence.visual_answer_status == "ambiguous"
        or task is None
        or claim_id not in task.claim_ids
    ):
        return False
    return not require_qualified or evidence_is_qualified(evidence)


def claim_owned_visual_evidence_requirements(
    state: ImageOnlyInvestigationState,
    *,
    reviewed_evidence_ids: Sequence[str],
    evidence_by_id: Mapping[str, InvestigationEvidence] | None = None,
) -> List[Dict[str, Any]]:
    """Return qualified claim-owned pixel Evidence that a Decision must address."""

    evidence_by_id = evidence_by_id or {
        item.evidence_id: item for item in state.evidence
    }
    task_by_id = {item.task_id: item for item in state.tasks}
    claim_by_id = {item.claim_id: item for item in state.image_claims}
    requirements: List[Dict[str, Any]] = []
    for evidence_id in dict.fromkeys(str(item) for item in reviewed_evidence_ids):
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        task = task_by_id.get(evidence.task_id)
        claim_ids = [
            claim_id
            for claim_id in (task.claim_ids if task is not None else [])
            if claim_id in claim_by_id
            and claim_owned_pixel_evidence_is_eligible(
                evidence,
                claim_id=claim_id,
                task_by_id=task_by_id,
                require_qualified=True,
            )
        ]
        if not claim_ids:
            continue
        requirements.append(
            {
                "evidence_id": evidence.evidence_id,
                "tool_name": evidence.tool_name,
                "claim_ids": claim_ids,
                "visual_scope": evidence.visual_scope,
                "visual_answer_status": evidence.visual_answer_status,
                "observation": evidence.exact_text[:1200],
                "required_action": {
                    "claim_ids": claim_ids,
                    "visual_evidence_ids": [evidence.evidence_id],
                    "same_claim_rule": (
                        "If this Decision updates one of these Claims, cite this "
                        "Evidence in that ClaimAssessment or MaterialDiscrepancy. "
                        "Otherwise, if this is reviewed qualified claim-owned "
                        "pixel Evidence that does not bear on the current "
                        "Claim/discrepancy, set "
                        "visual_evidence_disposition.disposition exactly to "
                        "'irrelevant_to_current_claim_or_discrepancy', list "
                        "this Evidence ID in "
                        "visual_evidence_disposition.evidence_ids, and explain "
                        "why it does not bear on the current Claim/discrepancy."
                    ),
                },
            }
        )
    return requirements


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _clip_text(value: str, limit: int) -> str:
    text = _one_line(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _source_text_for_visual_binding(
    evidence_by_id: Mapping[str, InvestigationEvidence],
    evidence_ids: Sequence[str],
) -> str:
    return " ".join(
        _one_line(evidence_by_id[evidence_id].exact_text)
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
        and _one_line(evidence_by_id[evidence_id].exact_text)
    )


def _evidence_introduces_integrity_question(source_text: str) -> bool:
    return bool(
        _semantic_request_tokens(source_text) & _INTEGRITY_EVIDENCE_TOKENS
    )


def _visual_discriminator_error(
    proposal: Any,
    *,
    claim: ImageClaim,
    image_account_summary: str,
    source_text: str,
) -> str:
    candidates = list(proposal.candidate_discriminators)
    if len(candidates) < 2:
        return (
            "visual_reinspection must compare at least two source-grounded "
            "candidate discriminators before selecting one"
        )
    if proposal.selected_discriminator_index >= len(candidates):
        return "selected visual discriminator is outside the supplied candidates"

    for index, candidate in enumerate(candidates):
        source_phrase = _one_line(candidate.source_phrase)
        if not source_phrase:
            return f"visual discriminator {index} source_phrase must be non-empty"
        informative_tokens = (
            _semantic_request_tokens(candidate.visible_property)
            - _GENERIC_VISUAL_REQUEST_TOKENS
            - _VISUAL_DISCRIMINATOR_STOP_TOKENS
        )
        if len(informative_tokens) < 2:
            return (
                f"visual discriminator {index} visible_property is too generic "
                "for a focused pixel check"
            )

    selected = candidates[proposal.selected_discriminator_index]
    if selected.already_in_claim:
        return (
            "selected visual discriminator is already asserted by the current "
            "ImageClaim or visual account; choose a higher-information candidate"
        )
    selected_property = _one_line(selected.visible_property)
    existing_account = _one_line(
        f"{claim.statement} {image_account_summary}"
    ).casefold()
    if selected_property.casefold() in existing_account:
        return (
            "selected visual discriminator merely repeats the current ImageClaim "
            "or visual account"
        )
    request_tokens = _semantic_request_tokens(
        f"{proposal.question} {selected_property}"
    )
    if (
        request_tokens & _INTEGRITY_REQUEST_TOKENS
        and not _evidence_introduces_integrity_question(source_text)
    ):
        return (
            "visual discriminator introduces an unsupported integrity or "
            "media-origin question"
        )
    return ""


def _runtime_specific_visual_request_payload(
    proposal: Any,
    *,
    claim: ImageClaim,
    image_account_summary: str,
    source_text: str,
) -> tuple[Dict[str, Any], str]:
    payload = proposal.model_dump(mode="json")
    discriminator_error = _visual_discriminator_error(
        proposal,
        claim=claim,
        image_account_summary=image_account_summary,
        source_text=source_text,
    )
    if discriminator_error:
        return {}, discriminator_error

    selected = proposal.candidate_discriminators[
        proposal.selected_discriminator_index
    ]
    payload["expected_property"] = _one_line(selected.visible_property)[:240]
    payload.pop("candidate_discriminators", None)
    payload.pop("selected_discriminator_index", None)
    return payload, ""


def bind_discrepancy_decision_runtime_ids(
    state: ImageOnlyInvestigationState,
    output: DiscrepancyDecisionProposalOutput,
    *,
    reviewed_evidence_ids: Sequence[str],
) -> tuple[DiscrepancyDecisionOutput | None, str]:
    """Build the persisted Decision object without asking the model to copy IDs."""

    payload = output.model_dump(mode="json")
    resolved_visual_requirements = (
        claim_owned_visual_evidence_requirements(
            state,
            reviewed_evidence_ids=reviewed_evidence_ids,
            evidence_by_id={
                item.evidence_id: item for item in state.evidence
            },
        )
    )
    resolved_visual_ids = {
        str(item["evidence_id"])
        for item in resolved_visual_requirements
    }
    selected_visual_ids = {
        str(evidence_id)
        for assessment in output.claim_assessments
        for evidence_id in assessment.selected_evidence_ids
    }
    if output.material_discrepancy is not None:
        selected_visual_ids.update(
            str(evidence_id)
            for evidence_id in output.material_discrepancy.evidence_ids
        )
    disposition = payload.get("visual_evidence_disposition")
    if (
        isinstance(disposition, Mapping)
        and disposition.get("disposition")
        == "irrelevant_to_current_claim_or_discrepancy"
    ):
        listed_ids = list(
            dict.fromkeys(
                str(item)
                for item in disposition.get("evidence_ids", []) or []
            )
        )
        target_ids = [
            evidence_id
            for evidence_id in (
                listed_ids
                or [
                    str(item["evidence_id"])
                    for item in resolved_visual_requirements
                ]
            )
            if evidence_id in resolved_visual_ids
        ]
        remaining_ids = [
            evidence_id
            for evidence_id in target_ids
            if evidence_id not in selected_visual_ids
        ]
        if remaining_ids:
            payload["visual_evidence_disposition"] = {
                **disposition,
                "evidence_ids": remaining_ids,
            }
        else:
            # Web/source Evidence is not part of the claim-owned pixel
            # consumption contract. If the model lists it here (for example
            # after finding an exact image match), ignore that bookkeeping note
            # instead of turning it into a protocol rejection.
            payload["visual_evidence_disposition"] = None
    discrepancy = output.material_discrepancy
    if discrepancy is not None:
        claim_by_id = {claim.claim_id: claim for claim in state.image_claims}
        unknown_claim_ids = [
            claim_id
            for claim_id in discrepancy.affected_claim_ids
            if claim_id not in claim_by_id
        ]
        if unknown_claim_ids:
            return None, (
                "runtime discrepancy binding cites unknown ImageClaim(s): "
                + ", ".join(unknown_claim_ids)
            )
        required_anchor_ids: List[str] = []
        remaining_anchor_ids: List[str] = []
        for claim_id in discrepancy.affected_claim_ids:
            allowed = list(dict.fromkeys(claim_by_id[claim_id].anchor_fact_ids))
            if not allowed:
                return None, (
                    "runtime discrepancy binding found no pixel/OCR anchor for "
                    f"ImageClaim {claim_id!r}"
                )
            required_anchor_ids.append(allowed[0])
            remaining_anchor_ids.extend(allowed[1:])
        visual_anchor_fact_ids = list(
            dict.fromkeys([*required_anchor_ids, *remaining_anchor_ids])
        )[:12]
        payload["material_discrepancy"] = {
            **discrepancy.model_dump(mode="json"),
            "visual_anchor_fact_ids": visual_anchor_fact_ids,
        }

    proposal = output.visual_reinspection
    if proposal is None:
        return DiscrepancyDecisionOutput.model_validate(payload), ""
    resolved = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=reviewed_evidence_ids,
    )
    candidates = resolved.get("candidates", [])
    binding = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("claim_id", "")) == proposal.claim_id
        ),
        None,
    )
    if not isinstance(binding, Mapping):
        eligible_claim_ids = [
            str(candidate.get("claim_id"))
            for candidate in candidates
            if isinstance(candidate, Mapping) and candidate.get("claim_id")
        ]
        return None, (
            f"visual_reinspection claim_id {proposal.claim_id!r} is not an eligible "
            "runtime visual reinspection candidate"
            + (
                "; choose one of " + ", ".join(eligible_claim_ids)
                if eligible_claim_ids
                else ""
            )
        )
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    claim_by_id = {claim.claim_id: claim for claim in state.image_claims}
    claim = claim_by_id.get(str(binding["claim_id"]))
    if claim is None:
        return None, "runtime visual reinspection binding cites an unknown Claim"
    source_text = _source_text_for_visual_binding(
        evidence_by_id,
        list(binding["grounding_evidence_ids"]),
    )
    request_payload, request_error = _runtime_specific_visual_request_payload(
        proposal,
        claim=claim,
        image_account_summary=state.image_account_summary,
        source_text=source_text,
    )
    if request_error:
        return None, request_error
    request_payload.pop("claim_id", None)
    payload["visual_reinspection"] = {
        **request_payload,
        "anchor_fact_ids": list(binding["anchor_fact_ids"]),
        "grounding_evidence_ids": list(binding["grounding_evidence_ids"]),
    }
    return DiscrepancyDecisionOutput.model_validate(payload), ""


def _runtime_hypothesis_tools(
    suggested_tools: Sequence[str],
    queries: Sequence[str],
) -> List[str]:
    """Derive executable capabilities from the hypothesis payload.

    A non-empty query list already declares a text-search route.  Keep that
    protocol fact in one place instead of requiring the policy to repeat it in
    ``suggested_tools``.  The persisted runtime list remains bounded by the
    SearchHypothesis schema.
    """

    tools = list(dict.fromkeys(suggested_tools))
    if queries and "text_search" not in tools:
        tools = [*tools[:3], "text_search"]
    return tools


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
    "depicts_relation",
}
_CORE_REFINEMENT_PREDICATES = {
    "appears_to_depict",
    "source_record_matches",
    "provenance_matches",
    "identified_as",
    "depicts_relation",
    "located_at",
    "occurred_at",
    "depicts_event",
}
_CORE_METADATA_PREDICATES = {
    "attributed_as",
    "created_by",
    "dated_as",
}
_SOURCE_BINDING_PREDICATES = {
    "source_record_matches",
    "provenance_matches",
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
    if state.image_claims and (
        not task.claim_ids
        or (
            task.hypothesis_id is None
            and not any(
                item.task_id == task.task_id
                for item in state.visual_reinspections
            )
        )
    ):
        raise RuntimeError(
            "v4 investigation actions require claim/hypothesis ownership or "
            "an accepted visual-reinspection task"
        )

    state.action_count += 1
    task.attempt_count += 1
    task.status = "active"
    hypothesis = next(
        (
            item
            for item in state.search_hypotheses
            if item.hypothesis_id == task.hypothesis_id
        ),
        None,
    )
    if hypothesis is not None:
        hypothesis.attempt_count += 1
        hypothesis.status = "active"
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
    recalled_candidate_ids: List[str] = []
    read_memory_ids: List[str] = []
    memory_action_error = ""

    if tool_name == "recall_evidence":
        if succeeded:
            recalled_candidate_ids = list(
                dict.fromkeys(
                    str(item.get("memory_id", "")).strip()
                    for item in data.get("candidates", []) or []
                    if isinstance(item, Mapping)
                    and str(item.get("memory_id", "")).strip()
                )
            )[:12]
            state.recalled_archive_memory_ids = list(
                dict.fromkeys(
                    [
                        *state.recalled_archive_memory_ids,
                        *recalled_candidate_ids,
                    ]
                )
            )[-120:]
            state.pending_archive_read_ids = recalled_candidate_ids
        else:
            memory_action_error = str(data.get("error", "archive recall failed"))
    elif tool_name == "read_evidence":
        if succeeded:
            memory_id = str(data.get("memory_id", "")).strip()
            if memory_id:
                read_memory_ids = [memory_id]
                state.read_archive_memory_ids = list(
                    dict.fromkeys([*state.read_archive_memory_ids, memory_id])
                )[-120:]
                # One exact selection completes the two-step recall batch. Other
                # candidates remain immutable in the archive and can be recalled
                # again if they become relevant later.
                state.pending_archive_read_ids = []
        else:
            memory_action_error = str(data.get("error", "archive read failed"))
    elif succeeded:
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
            tool_args=tool_args,
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
                    recoverable=tool_name != "focused_visual_inspection",
                )
            )
    else:
        failure_ids.append(
            _append_failure(
                state,
                task,
                call_id=call_id,
                tool_name=tool_name,
                code=_failure_code(
                    str(data.get("error", "")),
                    metadata=metadata,
                ),
                message=str(data.get("error", "tool call failed")),
                recoverable=tool_name != "focused_visual_inspection",
            )
        )

    route_payload = route_signature(tool_name, tool_args)
    route_payload["function_call_id"] = call_id
    created_evidence = [
        item
        for item in state.evidence
        if item.evidence_id in set(evidence_ids)
    ]
    route_payload["outcome"] = (
        "memory_read"
        if read_memory_ids
        else "memory_candidates"
        if recalled_candidate_ids
        else "memory_empty"
        if tool_name in {"recall_evidence", "read_evidence"} and succeeded
        else "memory_failed"
        if tool_name in {"recall_evidence", "read_evidence"}
        else
        "evidence"
        if any(
            item.quality in {"strong", "moderate"}
            for item in created_evidence
        )
        else "context"
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

    visual_reinspection = next(
        (
            item
            for item in state.visual_reinspections
            if item.task_id == task.task_id
        ),
        None,
    )
    if visual_reinspection is not None:
        visual_reinspection.evidence_ids = list(
            dict.fromkeys(
                [
                    *visual_reinspection.evidence_ids,
                    *evidence_ids,
                ]
            )
        )[:8]
        visual_reinspection.failure_ids = list(
            dict.fromkeys(
                [
                    *visual_reinspection.failure_ids,
                    *failure_ids,
                ]
            )
        )[:4]
        visual_reinspection.view_artifacts = _unique_visual_view_artifacts(
            [
                *visual_reinspection.view_artifacts,
                *[
                    item
                    for item in data.get("view_artifacts", []) or []
                    if isinstance(item, Mapping)
                ],
            ]
        )[:8]
        visual_reinspection.status = "resolved" if evidence_ids else "failed"

        _refresh_fact_states(state)
        if evidence_ids:
            task.status = "resolved"
            if hypothesis is not None:
                hypothesis.status = "exhausted"
        return {
            "action_count": state.action_count,
            "task_id": task.task_id,
            "task_status": task.status,
            "created_discovery_ids": discovery_ids,
            "created_evidence_ids": evidence_ids,
            "created_finding_ids": finding_ids,
            "created_failure_ids": failure_ids,
            "recalled_candidate_ids": recalled_candidate_ids,
            "read_memory_ids": read_memory_ids,
            "pending_archive_read_ids": list(state.pending_archive_read_ids),
            "memory_action_error": memory_action_error,
            "fact_statuses": {
                fact.fact_id: fact.status
                for fact in state.facts
                if fact.fact_id in task.fact_ids
            },
            "visual_view_artifacts": list(visual_reinspection.view_artifacts),
        }

    _refresh_fact_states(state)
    if finding_ids and task.claim_ids:
        # In v4, extractor Findings are candidate material for the sparse
        # Discrepancy Decision; they do not resolve an ImageClaim or its search
        # route by themselves.  Keep investigating while a bounded material
        # route remains, otherwise record ordinary route exhaustion.  Only the
        # semantic Decision reducer may assess the Claim or explicitly retire
        # its SearchHypothesis.
        task.finding_ids = list(dict.fromkeys([*task.finding_ids, *finding_ids]))
        task.status = (
            "active"
            if _task_has_remaining_material_route(state, task)
            else "exhausted"
        )
    elif finding_ids and not _task_has_unresolved_decisive_fact(state, task):
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
    if hypothesis is not None:
        if task.status in {"resolved", "exhausted", "blocked", "superseded"}:
            hypothesis.status = (
                "retired" if task.status == "superseded" else "exhausted"
            )
        elif not _task_has_remaining_material_route(state, task):
            hypothesis.status = "exhausted"
            if task.status in {"active", "pending"}:
                task.status = "exhausted"
    return {
        "action_count": state.action_count,
        "task_id": task.task_id,
        "task_status": task.status,
        "created_discovery_ids": discovery_ids,
        "created_evidence_ids": evidence_ids,
        "created_finding_ids": finding_ids,
        "created_failure_ids": failure_ids,
        "recalled_candidate_ids": recalled_candidate_ids,
        "read_memory_ids": read_memory_ids,
        "pending_archive_read_ids": list(state.pending_archive_read_ids),
        "memory_action_error": memory_action_error,
        "fact_statuses": {
            fact.fact_id: fact.status
            for fact in state.facts
            if fact.fact_id in task.fact_ids
        },
    }


def apply_image_account_planning(
    state: ImageOnlyInvestigationState,
    output: ImageAccountPlanningOutput,
) -> Dict[str, Any]:
    """Atomically install the image account, claims, hypotheses, and tasks."""

    candidate = state.model_copy(deep=True)
    if candidate.image_claims or candidate.search_hypotheses:
        return {
            "accepted": False,
            "rejected_reason": "image account planning has already been applied",
        }

    fact_by_id = {fact.fact_id: fact for fact in candidate.facts}
    if len(candidate.facts) + len(output.image_claims) > 72:
        return {"accepted": False, "rejected_reason": "VisualFact budget exhausted"}
    if len(output.image_claims) > MAX_IMAGE_CLAIMS:
        return {"accepted": False, "rejected_reason": "image claim budget exhausted"}
    if len(candidate.search_hypotheses) + len(output.search_hypotheses) > MAX_SEARCH_HYPOTHESES:
        return {"accepted": False, "rejected_reason": "search hypothesis budget exhausted"}
    if len(candidate.tasks) + len(output.search_hypotheses) > TOTAL_TASKS_MAX:
        return {"accepted": False, "rejected_reason": "total task budget exhausted"}

    high_claim_keys = {
        item.claim_key
        for item in output.image_claims
        if item.salience == "high"
    }
    if not high_claim_keys:
        return {
            "accepted": False,
            "rejected_reason": "image account requires a high-salience target fact",
        }
    if candidate.proposed_verdict in {"fake", "real"}:
        return {
            "accepted": False,
            "rejected_reason": "image account planning cannot run after verdict",
        }
    for index, proposal in enumerate(output.search_hypotheses):
        if proposal.route_focus == "media_origin":
            return {
                "accepted": False,
                "rejected_reason": (
                    "search hypothesis route_focus=media_origin is not "
                    "allowed; rewrite the route to test a depicted entity, "
                    "event, relation value, scene/world constraint, or "
                    "same-capture visual reference"
                ),
            }
        if any(
            _hypothesis_text_equivalent(prior.statement, proposal.statement)
            for prior in output.search_hypotheses[:index]
        ):
            return {
                "accepted": False,
                "rejected_reason": "planning contains duplicate search routes",
            }

    claim_by_key: Dict[str, ImageClaim] = {}
    new_fact_ids: List[str] = []
    new_claim_ids: List[str] = []
    new_hypothesis_ids: List[str] = []
    new_task_ids: List[str] = []
    for proposal in output.image_claims:
        anchors = [fact_by_id.get(fact_id) for fact_id in proposal.anchor_fact_ids]
        if any(anchor is None for anchor in anchors):
            return {
                "accepted": False,
                "rejected_reason": "image claim cites unknown visual anchor",
            }
        if any(
            anchor.origin.type not in {"input_image", "ocr"}
            for anchor in anchors
            if anchor is not None
        ):
            return {
                "accepted": False,
                "rejected_reason": "image claim anchor is not image/OCR grounded",
            }
        anchor = anchors[0]
        assert anchor is not None
        fact_id = stable_id(
            "vf", candidate.brief.case_id, "image-claim", proposal.claim_key
        )
        claim_id = stable_id(
            "claim", candidate.brief.case_id, proposal.claim_key
        )
        fact = VisualFact(
            fact_id=fact_id,
            kind=proposal.kind,
            statement=proposal.statement,
            subject_entity_id=anchor.subject_entity_id,
            predicate=proposal.predicate,
            object_entity_id=anchor.object_entity_id,
            status="active",
            basis_ids=list(dict.fromkeys(proposal.anchor_fact_ids))[:12],
            decision_relevance="supporting",
            origin=FactOrigin(
                type=(
                    "ocr"
                    if any(item is not None and item.origin.type == "ocr" for item in anchors)
                    else "input_image"
                ),
                origin_ids=list(dict.fromkeys(proposal.anchor_fact_ids))[:8],
            ),
        )
        claim = ImageClaim(
            claim_id=claim_id,
            fact_id=fact_id,
            statement=proposal.statement,
            anchor_fact_ids=list(dict.fromkeys(proposal.anchor_fact_ids)),
            salience=proposal.salience,
        )
        candidate.facts.append(fact)
        candidate.image_claims.append(claim)
        fact_by_id[fact_id] = fact
        claim_by_key[proposal.claim_key] = claim
        new_fact_ids.append(fact_id)
        new_claim_ids.append(claim_id)

    # Initial routes are planned independently of individual ImageClaims. They are
    # registered against the whole image account only after Planning so Evidence,
    # budgets, and stopping remain auditable without turning ownership into a
    # semantic constraint on what the model may investigate.
    claims = list(claim_by_key.values())
    claim_ids = [claim.claim_id for claim in claims]
    fact_ids = [claim.fact_id for claim in claims]
    owns_visual_integrity = any(
        fact_by_id[fact_id].predicate == "visual_integrity"
        for fact_id in fact_ids
    )
    for proposal in output.search_hypotheses:
        suggested_tools = _runtime_hypothesis_tools(
            proposal.suggested_tools,
            proposal.queries,
        )
        if not owns_visual_integrity:
            suggested_tools = [
                tool_name
                for tool_name in suggested_tools
                if tool_name
                not in {"check_consistency", "analyze_visual_anomalies"}
            ]
        if not suggested_tools:
            return {
                "accepted": False,
                "rejected_reason": (
                    "search hypothesis has no executable first-hop tool after "
                    "runtime capability filtering"
                ),
            }
        hypothesis_id = stable_id(
            "hypothesis", candidate.brief.case_id, proposal.hypothesis_key
        )
        task_id = stable_id("task", hypothesis_id, "claim-route")
        hypothesis = SearchHypothesis(
            hypothesis_id=hypothesis_id,
            claim_ids=claim_ids,
            route_focus=proposal.route_focus,
            statement=proposal.statement,
            queries=list(dict.fromkeys(proposal.queries)),
            expected_information=proposal.expected_information,
            suggested_tools=suggested_tools,
            priority=proposal.priority,
            task_id=task_id,
        )
        task = ResearchTask(
            task_id=task_id,
            fact_ids=fact_ids,
            claim_ids=claim_ids,
            hypothesis_id=hypothesis_id,
            question=proposal.statement,
            purpose=proposal.expected_information,
            priority=proposal.priority,
            status="active",
            origin_ids=list(dict.fromkeys([hypothesis_id, *claim_ids, *fact_ids]))[:12],
            suggested_tools=suggested_tools,
            suggested_queries=list(dict.fromkeys(proposal.queries)),
        )
        candidate.search_hypotheses.append(hypothesis)
        candidate.tasks.append(task)
        for claim in claims:
            if task_id not in claim.task_ids:
                claim.task_ids.append(task_id)
        new_hypothesis_ids.append(hypothesis_id)
        new_task_ids.append(task_id)

    candidate.image_account_summary = output.account_summary
    try:
        validated = ImageOnlyInvestigationState.model_validate(candidate.model_dump())
    except Exception as exc:
        return {
            "accepted": False,
            "rejected_reason": f"invalid image account state: {exc}",
        }
    _replace_state(state, validated)
    return {
        "accepted": True,
        "accepted_fact_ids": new_fact_ids,
        "accepted_claim_ids": new_claim_ids,
        "accepted_hypothesis_ids": new_hypothesis_ids,
        "accepted_task_ids": new_task_ids,
    }


def _core_target_fact(
    state: ImageOnlyInvestigationState,
    *,
    fact_by_id: Mapping[str, VisualFact] | None = None,
) -> VisualFact | None:
    facts = fact_by_id or {fact.fact_id: fact for fact in state.facts}
    if state.core_verdict_fact_id in facts:
        return facts[state.core_verdict_fact_id]
    # Compatibility fallback for v4 reducer fixtures and historical replays
    # created before core_verdict_fact_id became the semantic owner.
    for claim in state.image_claims:
        if claim.salience == "high" and claim.fact_id in facts:
            return facts[claim.fact_id]
    return None


def _claim_directional_chain_ids(
    state: ImageOnlyInvestigationState,
    *,
    claim_id: str,
    claim_fact_id: str,
    evidence_ids: Sequence[str],
    stance: str,
    evidence_by_id: Mapping[str, InvestigationEvidence],
    task_by_id: Mapping[str, ResearchTask],
) -> tuple[set[str], set[str]]:
    baseline_evidence_ids = {
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
        and evidence_is_qualified_for_stance(evidence_by_id[evidence_id], stance)
        and evidence_by_id[evidence_id].task_id in task_by_id
        and claim_id in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
    }
    qualified_evidence_ids = _claim_decision_evidence_ids(
        baseline_evidence_ids,
        evidence_by_id=evidence_by_id,
    )
    finding_ids = {
        finding.finding_id
        for finding in state.findings
        if finding.stance == stance
        and claim_fact_id in finding.fact_ids
        and finding.task_id in task_by_id
        and claim_id in task_by_id[finding.task_id].claim_ids
        and any(
            evidence_id in qualified_evidence_ids
            and evidence_by_id[evidence_id].task_id == finding.task_id
            for evidence_id in finding.evidence_ids
        )
    }
    if stance == "refute":
        composite_evidence_ids, composite_finding_ids = (
            _composite_source_visual_refute_finding_ids(
                state,
                claim_id=claim_id,
                claim_fact_id=claim_fact_id,
                evidence_ids=evidence_ids,
                evidence_by_id=evidence_by_id,
                task_by_id=task_by_id,
            )
        )
        qualified_evidence_ids.update(composite_evidence_ids)
        finding_ids.update(composite_finding_ids)
    return qualified_evidence_ids, finding_ids


def _claim_decision_evidence_ids(
    evidence_ids: Iterable[str],
    *,
    evidence_by_id: Mapping[str, InvestigationEvidence],
) -> set[str]:
    """Return Evidence strong enough to carry an ImageClaim decision.

    Exact spans from unknown or user-generated web sources are useful leads, but
    one such page must not by itself close a high-salience ImageClaim. It needs
    an independent second source, a trusted official/news source, or a non-web
    visual/reference chain.
    """

    ordered_ids = list(dict.fromkeys(evidence_ids))
    trusted_ids: set[str] = set()
    weak_web_ids: list[str] = []
    for evidence_id in ordered_ids:
        evidence = evidence_by_id[evidence_id]
        if evidence.evidence_kind != "web_span":
            trusted_ids.add(evidence_id)
        elif evidence.source_class in {"official", "news", "visual"}:
            trusted_ids.add(evidence_id)
        elif evidence.source_class in {"unknown", "ugc"}:
            weak_web_ids.append(evidence_id)
        else:
            trusted_ids.add(evidence_id)
    if trusted_ids:
        return trusted_ids

    independent_ids: list[str] = []
    seen_domains: set[str] = set()
    seen_families: set[str] = set()
    for evidence_id in weak_web_ids:
        evidence = evidence_by_id[evidence_id]
        family = evidence.source_family
        domain = (
            classify_source(evidence.source_url).registered_domain
            or family
        )
        if domain in seen_domains or family in seen_families:
            continue
        seen_domains.add(domain)
        seen_families.add(family)
        independent_ids.append(evidence_id)
        if len(independent_ids) == 2:
            return set(independent_ids)
    return set()


def _is_composite_source_visual_discrepancy_finding(
    finding: Finding,
) -> bool:
    return (
        finding.stance == "refute"
        and COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY
        in finding.source_family_ids
    )


def _visual_evidence_can_join_source_visual_chain(
    evidence: InvestigationEvidence,
    *,
    claim_id: str,
    task_by_id: Mapping[str, ResearchTask],
) -> bool:
    """Return whether one claim-owned pixel observation can join a refute chain.

    The Evidence contract, rather than the producing tool or a task transition,
    determines whether a visual observation is usable.
    """

    return claim_owned_pixel_evidence_is_eligible(
        evidence,
        claim_id=claim_id,
        task_by_id=task_by_id,
        require_qualified=True,
    )


def _composite_source_visual_refute_evidence_ids(
    state: ImageOnlyInvestigationState,
    *,
    claim_id: str,
    claim_fact_id: str,
    evidence_ids: Sequence[str],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    task_by_id: Mapping[str, ResearchTask],
) -> List[str]:
    """Return a narrow source+pixel conflict chain without relabeling Evidence.

    Neither Evidence record changes stance; the directional object is the
    composite Finding created by the Decision reducer. The pixel side may come
    from any direct, claim-owned image-region observation that satisfies the
    Evidence contract.
    """

    source_ids = [
        evidence_id
        for evidence_id in dict.fromkeys(evidence_ids)
        if evidence_id in evidence_by_id
        and evidence_by_id[evidence_id].task_id in task_by_id
        and claim_id
            in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
        and claim_fact_id in evidence_by_id[evidence_id].fact_ids
        if evidence_by_id[evidence_id].evidence_kind == "web_span"
        and evidence_by_id[evidence_id].directness == "direct"
        and evidence_by_id[evidence_id].claim_binding == "source_assertion"
        and evidence_by_id[evidence_id].quality in {"strong", "moderate"}
    ]
    visual_ids = [
        evidence_id
        for evidence_id in dict.fromkeys(evidence_ids)
        if evidence_id in evidence_by_id
        and _visual_evidence_can_join_source_visual_chain(
            evidence=evidence_by_id[evidence_id],
            claim_id=claim_id,
            task_by_id=task_by_id,
        )
    ]
    if not source_ids or not visual_ids:
        return []

    return list(dict.fromkeys([source_ids[0], visual_ids[0]]))


def _composite_source_visual_refute_finding_ids(
    state: ImageOnlyInvestigationState,
    *,
    claim_id: str,
    claim_fact_id: str,
    evidence_ids: Sequence[str],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    task_by_id: Mapping[str, ResearchTask],
) -> tuple[set[str], set[str]]:
    selected = set(
        _composite_source_visual_refute_evidence_ids(
            state,
            claim_id=claim_id,
            claim_fact_id=claim_fact_id,
            evidence_ids=evidence_ids,
            evidence_by_id=evidence_by_id,
            task_by_id=task_by_id,
        )
    )
    if not selected:
        return set(), set()
    finding_ids = {
        finding.finding_id
        for finding in state.findings
        if _is_composite_source_visual_discrepancy_finding(finding)
        and claim_fact_id in finding.fact_ids
        and set(finding.evidence_ids) <= set(evidence_ids)
        and selected <= set(finding.evidence_ids)
        and finding.task_id in task_by_id
        and claim_id in task_by_id[finding.task_id].claim_ids
    }
    return selected if finding_ids else set(), finding_ids


def _append_composite_source_visual_discrepancy_findings(
    state: ImageOnlyInvestigationState,
    output: DiscrepancyDecisionOutput,
    *,
    claim_by_id: Mapping[str, ImageClaim],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    task_by_id: Mapping[str, ResearchTask],
    decision_ordinal: int,
) -> List[str]:
    discrepancy = output.material_discrepancy
    if (
        discrepancy is None
        or discrepancy.status != "established"
        or discrepancy.materiality != "decisive"
    ):
        return []
    created_ids: List[str] = []
    for claim_id in discrepancy.affected_claim_ids:
        claim = claim_by_id.get(claim_id)
        if claim is None:
            continue
        selected_ids = _composite_source_visual_refute_evidence_ids(
            state,
            claim_id=claim_id,
            claim_fact_id=claim.fact_id,
            evidence_ids=discrepancy.evidence_ids,
            evidence_by_id=evidence_by_id,
            task_by_id=task_by_id,
        )
        if not selected_ids:
            continue
        source_task_id = evidence_by_id[selected_ids[0]].task_id
        finding_id = stable_id(
            "finding",
            state.brief.case_id,
            state.action_count,
            decision_ordinal,
            claim_id,
            "composite-source-visual-refute",
            selected_ids,
        )
        if any(item.finding_id == finding_id for item in state.findings):
            continue
        finding = Finding(
            finding_id=finding_id,
            task_id=source_task_id,
            fact_ids=[claim.fact_id],
            statement=discrepancy.statement[:1200],
            stance="refute",
            evidence_ids=selected_ids,
            source_family_ids=list(
                dict.fromkeys(
                    [
                        COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY,
                        *(
                            evidence_by_id[evidence_id].source_family
                            for evidence_id in selected_ids
                        ),
                    ]
                )
            )[:20],
            quality="decisive",
        )
        state.findings.append(finding)
        task = task_by_id.get(source_task_id)
        if task is not None and finding_id not in task.finding_ids:
            task.finding_ids.append(finding_id)
        created_ids.append(finding_id)
    return created_ids


def _format_visual_requirement_claim_map(
    requirements: Sequence[Mapping[str, Any]],
) -> str:
    parts: List[str] = []
    for requirement in requirements:
        evidence_id = str(requirement.get("evidence_id", ""))
        claim_ids = [
            str(claim_id)
            for claim_id in requirement.get("claim_ids", []) or []
        ]
        if evidence_id and claim_ids:
            parts.append(
                f"{evidence_id} -> claim_ids [{', '.join(claim_ids)}]"
            )
        elif evidence_id:
            parts.append(evidence_id)
    return "; ".join(parts)


def _discrepancy_contract_errors(
    state: ImageOnlyInvestigationState,
    output: DiscrepancyDecisionOutput,
    *,
    reviewed_evidence_ids: Sequence[str],
    claim_by_id: Mapping[str, ImageClaim],
    evidence_by_id: Mapping[str, InvestigationEvidence],
    fact_by_id: Mapping[str, VisualFact],
    task_by_id: Mapping[str, ResearchTask],
) -> list[str]:
    """Report independent reference/direction failures in one correction turn."""

    errors: list[str] = []
    reviewed_ids = set(reviewed_evidence_ids)
    required_visual_evidence_requirements = (
        claim_owned_visual_evidence_requirements(
            state,
            reviewed_evidence_ids=reviewed_evidence_ids,
            evidence_by_id=evidence_by_id,
        )
    )
    required_visual_evidence_ids = [
        str(item["evidence_id"])
        for item in required_visual_evidence_requirements
    ]
    required_visual_claim_ids_by_evidence = {
        str(item["evidence_id"]): {
            str(claim_id) for claim_id in item["claim_ids"]
        }
        for item in required_visual_evidence_requirements
    }
    selected_visual_evidence_ids = {
        evidence_id
        for proposal in output.claim_assessments
        for evidence_id in proposal.selected_evidence_ids
    }
    if output.material_discrepancy is not None:
        selected_visual_evidence_ids.update(
            output.material_discrepancy.evidence_ids
        )
    required_visual_ids = set(required_visual_evidence_ids)
    selected_required_visual_ids = (
        required_visual_ids & selected_visual_evidence_ids
    )
    decision_claim_ids = {
        proposal.claim_id for proposal in output.claim_assessments
    }
    if output.material_discrepancy is not None:
        decision_claim_ids.update(output.material_discrepancy.affected_claim_ids)
    disposition_ids: set[str] = set()
    if output.visual_evidence_disposition is not None:
        listed_ids = {
            str(item)
            for item in output.visual_evidence_disposition.evidence_ids
        }
        unknown_disposition_ids = sorted(
            evidence_id
            for evidence_id in listed_ids
            if evidence_id not in evidence_by_id
        )
        if unknown_disposition_ids:
            errors.append(
                "visual_evidence_disposition cites unknown Evidence: "
                + ", ".join(unknown_disposition_ids)
            )
        # This field is specifically for claim-owned pixel Evidence. Known
        # web/source Evidence may be mentioned by the model as redundant
        # bookkeeping (for example after an exact image match), but it is not
        # an unresolved visual obligation and must not block the Decision.
        disposition_ids = (
            listed_ids & required_visual_ids
            if listed_ids
            else required_visual_ids
        )
        overlap_ids = sorted(disposition_ids & selected_required_visual_ids)
        if overlap_ids:
            errors.append(
                "Decision must not both select and mark claim-owned visual "
                "Evidence irrelevant: "
                + ", ".join(overlap_ids)
            )
        disposed_claim_ids = {
            claim_id
            for evidence_id in disposition_ids & required_visual_ids
            for claim_id in required_visual_claim_ids_by_evidence[evidence_id]
        }
        same_claim_ids = sorted(disposed_claim_ids & decision_claim_ids)
        if same_claim_ids:
            errors.append(
                "Decision updates the same ImageClaim(s) as disposed visual "
                "Evidence; cite that Evidence instead: "
                + ", ".join(same_claim_ids)
            )
    unresolved_visual_ids = sorted(
        required_visual_ids - selected_required_visual_ids - disposition_ids
    )
    if unresolved_visual_ids:
        claim_map = _format_visual_requirement_claim_map(
            [
                item
                for item in required_visual_evidence_requirements
                if str(item["evidence_id"]) in unresolved_visual_ids
            ]
        )
        errors.append(
            "Decision must consume or explicitly disposition claim-owned visual "
            "Evidence: "
            + ", ".join(unresolved_visual_ids)
            + (f" ({claim_map})" if claim_map else "")
            + ". Cite each ID in claim_assessments[].selected_evidence_ids or "
            "material_discrepancy.evidence_ids, or list it in "
            "visual_evidence_disposition.evidence_ids with a rationale"
        )
    valid_assessments: dict[str, Any] = {}

    for proposal in output.claim_assessments:
        error_count_before = len(errors)
        claim = claim_by_id.get(proposal.claim_id)
        if claim is None:
            errors.append(
                f"assessment cites unknown ImageClaim {proposal.claim_id!r}; "
                "valid ImageClaim IDs: " + ", ".join(claim_by_id)
            )
            continue
        evidence_ids = list(dict.fromkeys(proposal.selected_evidence_ids))
        unknown_ids = [item for item in evidence_ids if item not in evidence_by_id]
        if unknown_ids:
            errors.append(
                f"assessment for ImageClaim {proposal.claim_id!r} cites unknown "
                f"Evidence: {', '.join(unknown_ids)}"
            )
        known_ids = [item for item in evidence_ids if item in evidence_by_id]
        unreviewed_ids = [item for item in known_ids if item not in reviewed_ids]
        if unreviewed_ids:
            errors.append(
                f"assessment for ImageClaim {proposal.claim_id!r} uses unreviewed "
                f"Evidence: {', '.join(unreviewed_ids)}"
            )
        outside_ids = [
            evidence_id
            for evidence_id in known_ids
            if not evidence_serves_claim(
                state,
                evidence_by_id[evidence_id],
                claim_id=proposal.claim_id,
                claim_fact_id=claim.fact_id,
                task_by_id=task_by_id,
            )
        ]
        if outside_ids:
            errors.append(
                f"assessment for ImageClaim {proposal.claim_id!r} uses Evidence "
                f"outside its owned tasks: {', '.join(outside_ids)}"
            )
        for stance in required_assessment_stances(proposal.assessment):
            qualified_ids, finding_ids = _claim_directional_chain_ids(
                state,
                claim_id=claim.claim_id,
                claim_fact_id=claim.fact_id,
                evidence_ids=known_ids,
                stance=stance,
                evidence_by_id=evidence_by_id,
                task_by_id=task_by_id,
            )
            composite_candidate_ids = (
                _composite_source_visual_refute_evidence_ids(
                    state,
                    claim_id=claim.claim_id,
                    claim_fact_id=claim.fact_id,
                    evidence_ids=known_ids,
                    evidence_by_id=evidence_by_id,
                    task_by_id=task_by_id,
                )
                if stance == "refute"
                else []
            )
            if not qualified_ids and not composite_candidate_ids:
                selected_directions = sorted(
                    {evidence_by_id[item].stance for item in known_ids}
                )
                owned_directional_ids = [
                    evidence_id
                    for evidence_id in known_ids
                    if evidence_is_qualified_for_stance(
                        evidence_by_id[evidence_id],
                        stance,
                    )
                    and evidence_by_id[evidence_id].task_id in task_by_id
                    and proposal.claim_id
                    in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
                ]
                if owned_directional_ids and stance in {"refute", "support"}:
                    source_details = []
                    for evidence_id in owned_directional_ids:
                        evidence = evidence_by_id[evidence_id]
                        source_details.append(
                            f"{evidence_id}="
                            f"{getattr(evidence, 'source_class', 'unknown')}/"
                            f"{getattr(evidence, 'source_family', 'unknown')}"
                        )
                    errors.append(
                        f"{proposal.assessment} assessment for ImageClaim "
                        f"{proposal.claim_id!r} requires owned qualified {stance} Evidence, "
                        "but its source support is insufficient for this high-salience "
                        f"{stance} decision: {', '.join(source_details)}. "
                        "This is not a missing Finding or Evidence-ID binding. For a "
                        "refute/support closure, add either two independent direct "
                        "qualified source families, one trusted official/news/visual "
                        "source, or an eligible same-capture comparison; otherwise "
                        "keep the assessment insufficient and verdict_proposal='continue'. "
                        "Do not resubmit the same terminal verdict."
                    )
                else:
                    errors.append(
                        f"{proposal.assessment} assessment for ImageClaim "
                        f"{proposal.claim_id!r} requires owned qualified {stance} "
                        "Evidence; selected Evidence records direction(s): "
                        f"{', '.join(selected_directions) or 'none'}. Omit the "
                        "assessment or keep it insufficient; do not relabel Evidence"
                    )
            elif not finding_ids and not composite_candidate_ids:
                errors.append(
                    f"{proposal.assessment} assessment for ImageClaim "
                    f"{proposal.claim_id!r} requires a {stance} Finding -> "
                    "Evidence chain. Omit this assessment unless a listed "
                    "reviewed directional chain serves this exact ImageClaim"
                )
        if len(errors) == error_count_before:
            valid_assessments[proposal.claim_id] = proposal

    discrepancy = output.material_discrepancy
    duplicate_discrepancy_id = ""
    if discrepancy is not None:
        duplicate_discrepancy_id = next(
            (
                item.discrepancy_id
                for item in state.material_discrepancies
                if set(item.affected_claim_ids)
                == set(discrepancy.affected_claim_ids)
                and set(item.evidence_ids) == set(discrepancy.evidence_ids)
                and item.materiality == discrepancy.materiality
                and item.status == discrepancy.status
            ),
            "",
        )
    if duplicate_discrepancy_id:
        errors.append(
            "material_discrepancy duplicates already-recorded discrepancy "
            f"{duplicate_discrepancy_id!r}; omit material_discrepancy because "
            "the prior canonical record remains active"
        )
        # Validate the rest of the proposed update as though the duplicate were
        # omitted. This avoids cascading requirements to restate the prior
        # ClaimAssessment while still rejecting the submitted duplicate.
        discrepancy = None
    valid_output_discrepancy = False
    if discrepancy is not None:
        discrepancy_error_count_before = len(errors)
        unknown_claim_ids = [
            item for item in discrepancy.affected_claim_ids if item not in claim_by_id
        ]
        if unknown_claim_ids:
            errors.append(
                "discrepancy cites unknown ImageClaim: "
                + ", ".join(unknown_claim_ids)
            )
        unknown_anchor_ids = [
            item for item in discrepancy.visual_anchor_fact_ids if item not in fact_by_id
        ]
        if unknown_anchor_ids:
            errors.append(
                "discrepancy cites unknown visual anchor: "
                + ", ".join(unknown_anchor_ids)
            )
        non_pixel_anchor_ids = [
            item
            for item in discrepancy.visual_anchor_fact_ids
            if item in fact_by_id
            and fact_by_id[item].origin.type not in {"input_image", "ocr"}
        ]
        if non_pixel_anchor_ids:
            errors.append(
                "discrepancy anchors are not image/OCR grounded: "
                + ", ".join(non_pixel_anchor_ids)
            )
        unknown_evidence_ids = [
            item for item in discrepancy.evidence_ids if item not in evidence_by_id
        ]
        if unknown_evidence_ids:
            errors.append(
                "discrepancy cites unknown Evidence: "
                + ", ".join(unknown_evidence_ids)
            )
        known_discrepancy_ids = [
            item for item in discrepancy.evidence_ids if item in evidence_by_id
        ]
        unreviewed_discrepancy_ids = [
            item for item in known_discrepancy_ids if item not in reviewed_ids
        ]
        if unreviewed_discrepancy_ids:
            errors.append(
                "discrepancy uses unreviewed Evidence: "
                + ", ".join(unreviewed_discrepancy_ids)
            )
        composite_discrepancy_ids = {
            evidence_id
            for claim_id in discrepancy.affected_claim_ids
            if claim_id in claim_by_id
            for evidence_id in _composite_source_visual_refute_evidence_ids(
                state,
                claim_id=claim_id,
                claim_fact_id=claim_by_id[claim_id].fact_id,
                evidence_ids=known_discrepancy_ids,
                evidence_by_id=evidence_by_id,
                task_by_id=task_by_id,
            )
        }
        if (
            discrepancy.materiality == "decisive"
            and discrepancy.status == "established"
            and not any(
                evidence_is_qualified_for_stance(evidence_by_id[item], "refute")
                for item in known_discrepancy_ids
            )
            and not composite_discrepancy_ids
        ):
            selected_directions = sorted(
                {evidence_by_id[item].stance for item in known_discrepancy_ids}
            )
            errors.append(
                "established decisive discrepancy requires qualified refute "
                "Evidence; selected Evidence records direction(s): "
                f"{', '.join(selected_directions) or 'none'}. Remove the "
                "discrepancy or select a qualified refute chain; do not relabel "
                "Evidence"
            )

        assessment_by_claim_id = {
            item.claim_id: item for item in output.claim_assessments
        }
        if discrepancy.materiality == "decisive":
            expected_assessment = (
                "refuted" if discrepancy.status == "established" else "conflicted"
            )
            mismatched_ids = [
                claim_id
                for claim_id in discrepancy.affected_claim_ids
                if claim_id not in assessment_by_claim_id
                or assessment_by_claim_id[claim_id].assessment
                != expected_assessment
            ]
            if mismatched_ids:
                errors.append(
                    f"decisive discrepancy status {discrepancy.status!r} requires "
                    f"{expected_assessment} assessment for affected ImageClaims: "
                    + ", ".join(mismatched_ids)
                )
        for claim_id in discrepancy.affected_claim_ids:
            claim = claim_by_id.get(claim_id)
            if claim is None:
                continue
            if not set(discrepancy.visual_anchor_fact_ids) & set(claim.anchor_fact_ids):
                errors.append(
                    "discrepancy anchor is outside the affected claim; allowed "
                    f"visual anchor IDs for {claim_id!r}: "
                    + ", ".join(claim.anchor_fact_ids)
                )
            owned_ids = [
                evidence_id
                for evidence_id in known_discrepancy_ids
                if evidence_by_id[evidence_id].task_id in task_by_id
                and claim_id
                in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
            ]
            if not owned_ids and known_discrepancy_ids:
                errors.append(
                    f"discrepancy Evidence is outside affected ImageClaim "
                    f"{claim_id!r}'s tasks"
                )
            if (
                discrepancy.materiality == "decisive"
                and discrepancy.status == "established"
                and owned_ids
                and not _claim_directional_chain_ids(
                    state,
                    claim_id=claim_id,
                    claim_fact_id=claim.fact_id,
                    evidence_ids=owned_ids,
                    stance="refute",
                    evidence_by_id=evidence_by_id,
                    task_by_id=task_by_id,
                )[1]
                and not _composite_source_visual_refute_evidence_ids(
                    state,
                    claim_id=claim_id,
                    claim_fact_id=claim.fact_id,
                    evidence_ids=owned_ids,
                    evidence_by_id=evidence_by_id,
                    task_by_id=task_by_id,
                )
            ):
                errors.append(
                    f"affected target fact {claim_id!r} requires an owned qualified "
                    "refute Finding -> Evidence discrepancy chain"
                )
        valid_output_discrepancy = len(errors) == discrepancy_error_count_before

    core_fact = _core_target_fact(state, fact_by_id=fact_by_id)
    core_claim_ids = {
        claim.claim_id
        for claim in state.image_claims
        if core_fact is not None and claim.fact_id == core_fact.fact_id
    }
    projected_core_status = core_fact.status if core_fact is not None else ""
    for claim_id in core_claim_ids:
        if claim_id in valid_assessments:
            projected_core_status = valid_assessments[claim_id].assessment
            break
    retired_ids = set(output.retire_hypothesis_ids)
    open_core_route_ids = [
        hypothesis.hypothesis_id
        for hypothesis in state.search_hypotheses
        if hypothesis.status in {"open", "active"}
        and hypothesis.hypothesis_id not in retired_ids
        and any(
            task.hypothesis_id == hypothesis.hypothesis_id
            and (
                core_fact is None
                or core_fact.fact_id in task.fact_ids
            )
            for task in task_by_id.values()
        )
    ]
    new_core_routes = [
        item.statement
        for item in output.new_hypotheses
        if any(
            claim_id in claim_by_id
            and core_fact is not None
            and claim_by_id[claim_id].fact_id == core_fact.fact_id
            for claim_id in item.claim_ids
        )
    ]
    current_decisive = [
        item
        for item in state.material_discrepancies
        if item.materiality == "decisive"
        and item.status in {"established", "conflicted"}
    ]
    output_established_core_discrepancy = bool(
        discrepancy is not None
        and valid_output_discrepancy
        and discrepancy.materiality == "decisive"
        and discrepancy.status == "established"
        and any(
            claim_id in claim_by_id
            and core_fact is not None
            and claim_by_id[claim_id].fact_id == core_fact.fact_id
            for claim_id in discrepancy.affected_claim_ids
        )
    )
    current_established_core_discrepancy = any(
        item.status == "established"
        and any(
            claim_id in claim_by_id
            and core_fact is not None
            and claim_by_id[claim_id].fact_id == core_fact.fact_id
            for claim_id in item.affected_claim_ids
        )
        for item in current_decisive
    )
    established_core_discrepancy = (
        current_established_core_discrepancy
        or output_established_core_discrepancy
    )
    if projected_core_status == "refuted" and not established_core_discrepancy:
        errors.append(
            "a refuted core image-grounded target fact requires a valid decisive "
            "established discrepancy and verdict_proposal='fake' in the same "
            "atomic update"
        )
    if output.verdict_proposal == "real" and (
        not core_fact
        or projected_core_status != "supported"
        or current_decisive
        or (discrepancy is not None and discrepancy.materiality == "decisive")
        or open_core_route_ids
        or new_core_routes
    ):
        detail: list[str] = []
        if not core_fact or projected_core_status != "supported":
            detail.append("core target fact is not supported")
        if open_core_route_ids:
            detail.append("open core routes: " + ", ".join(open_core_route_ids))
        if new_core_routes:
            detail.append("new core routes remain open")
        if current_decisive or (
            discrepancy is not None and discrepancy.materiality == "decisive"
        ):
            detail.append("decisive discrepancy remains")
        errors.append(
            "real verdict requires the core target fact to be supported, no "
            "decisive discrepancy, and no open core route; choose "
            "continue unless those conditions are already satisfied"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
    if output.verdict_proposal == "fake" and not established_core_discrepancy:
        errors.append(
            "fake verdict requires a valid decisive established discrepancy "
            "affecting the core target fact; otherwise choose continue"
        )
    if established_core_discrepancy and output.verdict_proposal != "fake":
        errors.append(
            "a valid established decisive high-salience discrepancy requires "
            "verdict_proposal='fake' in the same complete JSON object"
        )

    return list(dict.fromkeys(errors))


def apply_discrepancy_decision(
    state: ImageOnlyInvestigationState,
    output: DiscrepancyDecisionOutput,
    *,
    reviewed_evidence_ids: Sequence[str],
    trigger: str,
    source_access_policy: Any = None,
) -> Dict[str, Any]:
    """Validate a discrepancy checkpoint on a copy, then commit it atomically."""

    blocked_queries = [
        (query, reason)
        for proposal in output.new_hypotheses
        for query in proposal.queries
        if (reason := query_policy_violation(
            query,
            source_access_policy=source_access_policy,
        ))
    ]
    if blocked_queries:
        details = "; ".join(
            f"{reason}: {query!r}" for query, reason in blocked_queries[:3]
        )
        return {
            "accepted": False,
            "rejected_reason": (
                "new hypothesis queries must seek underlying facts or sources, "
                "not a ready-made fact-check verdict or an excluded source; "
                + details
            ),
        }

    candidate = state.model_copy(deep=True)
    if candidate.proposed_verdict in {"fake", "real"}:
        return {
            "accepted": False,
            "rejected_reason": "no discrepancy decision is allowed after verdict",
        }
    if not candidate.image_claims:
        return {
            "accepted": False,
            "rejected_reason": "image account planning must precede discrepancy decision",
        }
    claim_by_id = {claim.claim_id: claim for claim in candidate.image_claims}
    hypothesis_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in candidate.search_hypotheses
    }
    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in candidate.evidence
    }
    fact_by_id = {fact.fact_id: fact for fact in candidate.facts}
    task_by_id = {task.task_id: task for task in candidate.tasks}
    decision_ordinal = len(candidate.discrepancy_decisions)
    reviewed_ids = list(dict.fromkeys(reviewed_evidence_ids))
    if any(evidence_id not in evidence_by_id for evidence_id in reviewed_ids):
        return {"accepted": False, "rejected_reason": "checkpoint cites unknown Evidence"}
    if trigger not in {
        "qualified_evidence",
        "scheduled_boundary",
        "strategy_boundary",
        "before_unresolved",
    }:
        return {"accepted": False, "rejected_reason": "unknown discrepancy trigger"}
    if trigger == "qualified_evidence" and not reviewed_ids:
        return {
            "accepted": False,
            "rejected_reason": "qualified Evidence checkpoint requires Evidence",
        }
    effective_visual_disposition = False
    if output.visual_evidence_disposition is not None:
        required_visual_ids = {
            str(item["evidence_id"])
            for item in claim_owned_visual_evidence_requirements(
                candidate,
                reviewed_evidence_ids=reviewed_ids,
                evidence_by_id=evidence_by_id,
            )
        }
        listed_visual_ids = {
            str(item)
            for item in output.visual_evidence_disposition.evidence_ids
        }
        effective_visual_disposition = bool(
            required_visual_ids
            and (
                not listed_visual_ids
                or listed_visual_ids & required_visual_ids
            )
        )
    if (
        trigger == "qualified_evidence"
        and reviewed_ids
        and not output.claim_assessments
        and output.material_discrepancy is None
        and not output.retire_hypothesis_ids
        and not output.new_hypotheses
        and output.visual_reinspection is None
        and not effective_visual_disposition
        and output.verdict_proposal == "continue"
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "qualified Evidence checkpoint must record a Claim assessment, "
                "discrepancy, route update, visual reinspection, or explicit "
                "claim-owned visual Evidence disposition; an empty continue "
                "Decision would "
                "silently leave reviewed Evidence unconsumed"
            ),
        }
    if any(item not in hypothesis_by_id for item in output.retire_hypothesis_ids):
        return {"accepted": False, "rejected_reason": "decision cites unknown SearchHypothesis"}
    if len(output.new_hypotheses) > MAX_NEW_HYPOTHESES_PER_DECISION:
        return {"accepted": False, "rejected_reason": "new hypothesis decision budget exhausted"}
    if len(candidate.search_hypotheses) + len(output.new_hypotheses) > MAX_SEARCH_HYPOTHESES:
        return {"accepted": False, "rejected_reason": "search hypothesis budget exhausted"}
    if len(candidate.tasks) + len(output.new_hypotheses) > TOTAL_TASKS_MAX:
        return {"accepted": False, "rejected_reason": "total task budget exhausted"}

    contract_errors = _discrepancy_contract_errors(
        candidate,
        output,
        reviewed_evidence_ids=reviewed_ids,
        claim_by_id=claim_by_id,
        evidence_by_id=evidence_by_id,
        fact_by_id=fact_by_id,
        task_by_id=task_by_id,
    )
    if contract_errors:
        return {
            "accepted": False,
            "rejected_reason": "; ".join(contract_errors),
        }

    created_composite_finding_ids = (
        _append_composite_source_visual_discrepancy_findings(
            candidate,
            output,
            claim_by_id=claim_by_id,
            evidence_by_id=evidence_by_id,
            task_by_id=task_by_id,
            decision_ordinal=decision_ordinal,
        )
    )

    accepted_assessment_ids: List[str] = []
    for proposal in output.claim_assessments:
        evidence_ids = list(dict.fromkeys(proposal.selected_evidence_ids))
        if any(evidence_id not in evidence_by_id for evidence_id in evidence_ids):
            return {"accepted": False, "rejected_reason": "assessment cites unknown Evidence"}
        if proposal.assessment != "insufficient" and not evidence_ids:
            return {
                "accepted": False,
                "rejected_reason": (
                    "material assessment for ImageClaim "
                    f"{proposal.claim_id!r} requires Evidence; omit the "
                    "assessment when no reviewed owned Evidence serves it"
                ),
            }
        if not set(evidence_ids) <= set(reviewed_ids):
            return {
                "accepted": False,
                "rejected_reason": "assessment must use reviewed Evidence",
            }
        claim = claim_by_id[proposal.claim_id]
        for evidence_id in evidence_ids:
            evidence = evidence_by_id[evidence_id]
            if not evidence_serves_claim(
                candidate,
                evidence,
                claim_id=claim.claim_id,
                claim_fact_id=claim.fact_id,
                task_by_id=task_by_id,
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        f"Evidence {evidence_id!r} is outside ImageClaim "
                        f"{claim.claim_id!r}'s owned tasks; omit that claim "
                        "assessment or select reviewed owned Evidence"
                    ),
                }
        for stance in required_assessment_stances(proposal.assessment):
            qualified_evidence_ids, finding_ids = _claim_directional_chain_ids(
                candidate,
                claim_id=claim.claim_id,
                claim_fact_id=claim.fact_id,
                evidence_ids=evidence_ids,
                stance=stance,
                evidence_by_id=evidence_by_id,
                task_by_id=task_by_id,
            )
            if not qualified_evidence_ids:
                selected_directions = sorted(
                    {
                        evidence_by_id[evidence_id].stance
                        for evidence_id in evidence_ids
                        if evidence_id in evidence_by_id
                    }
                )
                return {
                    "accepted": False,
                    "rejected_reason": (
                        f"{proposal.assessment} assessment for ImageClaim "
                        f"{claim.claim_id!r} requires owned qualified {stance} "
                        "Evidence; selected Evidence records direction(s): "
                        f"{', '.join(selected_directions) or 'none'}. Omit the "
                        "assessment or keep it insufficient; do not relabel Evidence"
                    ),
                }
            if not finding_ids:
                return {
                    "accepted": False,
                    "rejected_reason": (
                        f"{proposal.assessment} assessment for ImageClaim "
                        f"{claim.claim_id!r} requires a {stance} Finding -> "
                        "Evidence chain"
                    ),
                }
        assessment_id = stable_id(
            "assessment",
            candidate.brief.case_id,
            candidate.action_count,
            decision_ordinal,
            proposal.claim_id,
            proposal.assessment,
            evidence_ids,
        )
        candidate.claim_assessments.append(
            ClaimAssessment(
                assessment_id=assessment_id,
                claim_id=proposal.claim_id,
                action_count=candidate.action_count,
                assessment=proposal.assessment,
                evidence_ids=evidence_ids,
                remaining_gap=proposal.remaining_gap,
                rationale=proposal.rationale,
            )
        )
        claim.status = (
            "unresolved" if proposal.assessment == "insufficient" else proposal.assessment
        )
        fact_by_id[claim.fact_id].status = (
            "active"
            if proposal.assessment == "insufficient"
            else proposal.assessment
        )
        accepted_assessment_ids.append(assessment_id)

    discrepancy_id = None
    discrepancy = output.material_discrepancy
    if discrepancy is not None:
        if any(claim_id not in claim_by_id for claim_id in discrepancy.affected_claim_ids):
            return {"accepted": False, "rejected_reason": "discrepancy cites unknown ImageClaim"}
        if any(fact_id not in fact_by_id for fact_id in discrepancy.visual_anchor_fact_ids):
            return {"accepted": False, "rejected_reason": "discrepancy cites unknown visual anchor"}
        if any(
            fact_by_id[fact_id].origin.type not in {"input_image", "ocr"}
            for fact_id in discrepancy.visual_anchor_fact_ids
        ):
            return {
                "accepted": False,
                "rejected_reason": "discrepancy anchor is not image/OCR grounded",
            }
        if any(evidence_id not in evidence_by_id for evidence_id in discrepancy.evidence_ids):
            return {"accepted": False, "rejected_reason": "discrepancy cites unknown Evidence"}
        if not set(discrepancy.evidence_ids) <= set(reviewed_ids):
            return {
                "accepted": False,
                "rejected_reason": "discrepancy must use reviewed Evidence",
            }
        affected_claim_ids = list(dict.fromkeys(discrepancy.affected_claim_ids))
        discrepancy_evidence_ids = list(dict.fromkeys(discrepancy.evidence_ids))
        has_discrepancy_refute_chain = any(
            _claim_directional_chain_ids(
                candidate,
                claim_id=claim_id,
                claim_fact_id=claim_by_id[claim_id].fact_id,
                evidence_ids=discrepancy_evidence_ids,
                stance="refute",
                evidence_by_id=evidence_by_id,
                task_by_id=task_by_id,
            )[1]
            for claim_id in affected_claim_ids
        )
        if (
            discrepancy.materiality == "decisive"
            and discrepancy.status == "established"
            and not has_discrepancy_refute_chain
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "established decisive discrepancy requires qualified Evidence"
                ),
            }
        assessment_by_claim_id = {
            item.claim_id: item for item in output.claim_assessments
        }
        if discrepancy.materiality == "decisive":
            expected_assessment = (
                "refuted"
                if discrepancy.status == "established"
                else "conflicted"
            )
            if any(
                claim_id not in assessment_by_claim_id
                or assessment_by_claim_id[claim_id].assessment
                != expected_assessment
                for claim_id in affected_claim_ids
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "decisive discrepancy status conflicts with claim assessment"
                    ),
                }
        for claim_id in discrepancy.affected_claim_ids:
            claim = claim_by_id[claim_id]
            if not set(discrepancy.visual_anchor_fact_ids) & set(claim.anchor_fact_ids):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "discrepancy anchor is outside the affected claim; "
                        f"allowed visual anchor IDs for {claim_id!r}: "
                        f"{', '.join(claim.anchor_fact_ids)}"
                    ),
                }
            if not any(
                (
                    task_by_id.get(evidence_by_id[evidence_id].task_id) is not None
                    and claim_id
                    in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
                )
                for evidence_id in discrepancy_evidence_ids
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "discrepancy Evidence is outside affected claim tasks"
                    ),
                }
            if (
                discrepancy.materiality == "decisive"
                and discrepancy.status == "established"
                and not _claim_directional_chain_ids(
                    candidate,
                    claim_id=claim_id,
                    claim_fact_id=claim.fact_id,
                    evidence_ids=discrepancy_evidence_ids,
                    stance="refute",
                    evidence_by_id=evidence_by_id,
                    task_by_id=task_by_id,
                )[1]
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "each affected claim requires an owned qualified refute "
                        "Finding -> Evidence discrepancy chain"
                    ),
                }
        for evidence_id in discrepancy_evidence_ids:
            task = task_by_id.get(evidence_by_id[evidence_id].task_id)
            if task is None or not set(task.claim_ids) & set(affected_claim_ids):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "discrepancy Evidence is outside affected claim tasks"
                    ),
                }
        discrepancy_id = stable_id(
            "discrepancy",
            candidate.brief.case_id,
            candidate.action_count,
            decision_ordinal,
            discrepancy.statement,
            discrepancy.affected_claim_ids,
            discrepancy.evidence_ids,
        )
        candidate.material_discrepancies.append(
            MaterialDiscrepancy(
                discrepancy_id=discrepancy_id,
                statement=discrepancy.statement,
                affected_claim_ids=affected_claim_ids,
                visual_anchor_fact_ids=list(dict.fromkeys(discrepancy.visual_anchor_fact_ids)),
                evidence_ids=discrepancy_evidence_ids,
                materiality=discrepancy.materiality,
                status=discrepancy.status,
                rationale=discrepancy.rationale,
            )
        )

    retired_ids: List[str] = []
    for hypothesis_id in output.retire_hypothesis_ids:
        hypothesis = hypothesis_by_id[hypothesis_id]
        hypothesis.status = "retired"
        if hypothesis.task_id is not None:
            task = task_by_id.get(hypothesis.task_id)
            if task is not None and task.status in {"pending", "active"}:
                task.status = "superseded"
        retired_ids.append(hypothesis_id)

    accepted_hypothesis_ids: List[str] = []
    for index, proposal in enumerate(output.new_hypotheses):
        if proposal.route_focus == "media_origin":
            return {
                "accepted": False,
                "rejected_reason": (
                    "new hypothesis route_focus=media_origin is not allowed; "
                    "rewrite the route to test a depicted entity, event, "
                    "relation value, scene/world constraint, or same-capture "
                    "visual reference"
                ),
            }
        if any(claim_id not in claim_by_id for claim_id in proposal.claim_ids):
            return {"accepted": False, "rejected_reason": "new hypothesis cites unknown ImageClaim"}
        proposal_claim_ids = list(dict.fromkeys(proposal.claim_ids))
        if any(
            set(item.claim_ids) == set(proposal_claim_ids)
            and _hypothesis_text_equivalent(item.statement, proposal.statement)
            for item in candidate.search_hypotheses
        ):
            return {
                "accepted": False,
                "rejected_reason": "new hypothesis duplicates an existing route",
            }
        hypothesis_id = stable_id(
            "hypothesis",
            candidate.brief.case_id,
            candidate.action_count,
            decision_ordinal,
            index,
            proposal.statement,
        )
        task_id = stable_id("task", hypothesis_id, "decision-route")
        suggested_tools = _runtime_hypothesis_tools(
            proposal.suggested_tools,
            proposal.queries,
        )
        hypothesis = SearchHypothesis(
            hypothesis_id=hypothesis_id,
            claim_ids=proposal_claim_ids,
            route_focus=proposal.route_focus,
            statement=proposal.statement,
            queries=list(dict.fromkeys(proposal.queries)),
            expected_information=proposal.expected_information,
            suggested_tools=suggested_tools,
            priority=proposal.priority,
            task_id=task_id,
        )
        candidate.search_hypotheses.append(hypothesis)
        candidate.tasks.append(
            ResearchTask(
                task_id=task_id,
                fact_ids=[claim_by_id[item].fact_id for item in hypothesis.claim_ids],
                claim_ids=hypothesis.claim_ids,
                hypothesis_id=hypothesis_id,
                question=proposal.statement,
                purpose=proposal.expected_information,
                priority=proposal.priority,
                origin_ids=list(dict.fromkeys([hypothesis_id, *hypothesis.claim_ids]))[:12],
                suggested_tools=hypothesis.suggested_tools,
                suggested_queries=hypothesis.queries,
            )
        )
        for claim_id in hypothesis.claim_ids:
            claim_by_id[claim_id].task_ids.append(task_id)
        accepted_hypothesis_ids.append(hypothesis_id)

    accepted_visual_question_id = None
    if output.visual_reinspection is not None:
        request = output.visual_reinspection
        if candidate.action_count >= MAX_TOOL_ACTIONS:
            return {
                "accepted": False,
                "rejected_reason": (
                    "the tool-action budget cannot execute another visual "
                    "reinspection"
                ),
            }
        if len(candidate.visual_reinspections) >= MAX_V4_VISUAL_REINSPECTIONS:
            return {"accepted": False, "rejected_reason": "visual reinspection budget exhausted"}
        if len(candidate.tasks) >= TOTAL_TASKS_MAX:
            return {
                "accepted": False,
                "rejected_reason": "task budget cannot hold the visual reinspection",
            }
        anchor_ids = list(dict.fromkeys(request.anchor_fact_ids))
        anchors = [fact_by_id.get(fact_id) for fact_id in anchor_ids]
        if any(anchor is None for anchor in anchors) or any(
            anchor is not None
            and anchor.origin.type not in {"input_image", "ocr"}
            for anchor in anchors
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection anchors must be existing pixel/OCR facts"
                ),
            }
        grounding_ids = list(dict.fromkeys(request.grounding_evidence_ids))
        if any(evidence_id not in evidence_by_id for evidence_id in grounding_ids):
            return {
                "accepted": False,
                "rejected_reason": "visual reinspection cites unknown Evidence",
            }
        if not set(grounding_ids) <= set(reviewed_ids):
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection grounding must use reviewed Evidence"
                ),
            }
        grounding_task_ids = list(
            dict.fromkeys(
                evidence_by_id[evidence_id].task_id
                for evidence_id in grounding_ids
            )
        )
        if len(grounding_task_ids) != 1:
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection grounding Evidence must belong to one "
                    "ResearchTask"
                ),
            }
        parent_task_id = grounding_task_ids[0]
        # A visual request is allowed to resolve an unresolved Claim even when
        # this checkpoint deliberately omits a ClaimAssessment.  In
        # particular, a model may first discover that all reviewed Evidence is
        # neutral and therefore cannot legally label the Claim ``refuted``;
        # it can still request a focused pixel check for the still-open Claim.
        # Use the candidate's state as the source of truth, while retaining the
        # output assessments as a narrow refinement for newly introduced
        # ``insufficient``/``conflicted`` records.
        unresolved_assessment_claim_ids = {
            claim.claim_id
            for claim in candidate.image_claims
            if claim.status in {"open", "unresolved", "conflicted"}
        }
        unresolved_assessment_claim_ids.update(
            item.claim_id
            for item in output.claim_assessments
            if item.assessment in {"insufficient", "conflicted"}
        )
        visual_claim_ids = [
            claim_id
            for claim_id in unresolved_assessment_claim_ids
            if set(anchor_ids) & set(claim_by_id[claim_id].anchor_fact_ids)
            and any(
                claim_by_id[claim_id].fact_id
                in evidence_by_id[evidence_id].fact_ids
                and claim_id
                in task_by_id[evidence_by_id[evidence_id].task_id].claim_ids
                for evidence_id in grounding_ids
                if evidence_by_id[evidence_id].task_id in task_by_id
            )
        ]
        if not visual_claim_ids:
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection must serve a reviewed unresolved claim"
                ),
            }
        if any(
            _visual_reinspection_requests_equivalent(prior.request, request)
            for prior in candidate.visual_reinspections
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "an equivalent visual question has already been requested"
                ),
            }
        visual_fact_ids = [claim_by_id[item].fact_id for item in visual_claim_ids]
        accepted_visual_question_id = stable_id(
            "visual-question",
            candidate.brief.case_id,
            visual_fact_ids,
            request.model_dump(mode="json"),
        )
        visual_task_id = stable_id(
            "task",
            accepted_visual_question_id,
            "focused-visual-inspection",
        )
        visual_task = ResearchTask(
            task_id=visual_task_id,
            fact_ids=visual_fact_ids,
            claim_ids=visual_claim_ids,
            question=request.question,
            purpose=(
                "Reinspect the original pixels after Evidence introduced a "
                "concrete visible hypothesis."
            ),
            priority=1,
            status="active",
            parent_task_id=parent_task_id,
            origin_ids=list(
                dict.fromkeys(
                    [*visual_fact_ids, *visual_claim_ids, *anchor_ids, *grounding_ids]
                )
            )[:12],
            suggested_tools=["focused_visual_inspection"],
        )
        candidate.tasks.append(visual_task)
        task_by_id[visual_task_id] = visual_task
        for claim_id in visual_claim_ids:
            claim_by_id[claim_id].task_ids.append(visual_task_id)
        candidate.visual_reinspections.append(
            VisualReinspectionRecord(
                visual_question_id=accepted_visual_question_id,
                task_id=visual_task_id,
                fact_id=visual_fact_ids[0],
                created_action_count=candidate.action_count,
                request=request,
                anchor_regions=_visual_reinspection_anchor_regions(
                    candidate,
                    anchor_ids,
                ),
            )
        )
        candidate.recommended_next_task_ids = list(
            dict.fromkeys(
                [visual_task_id, *candidate.recommended_next_task_ids]
            )
        )[:4]

    decisive = [
        item for item in candidate.material_discrepancies
        if item.materiality == "decisive" and item.status in {"established", "conflicted"}
    ]
    established_decisive = [
        item for item in decisive if item.status == "established"
    ]
    core_fact = _core_target_fact(candidate)
    established_core_discrepancy = any(
        any(
            core_fact is not None
            and claim_id in claim_by_id
            and claim_by_id[claim_id].fact_id == core_fact.fact_id
            for claim_id in item.affected_claim_ids
        )
        for item in established_decisive
    )
    if core_fact is not None and core_fact.status == "refuted" and not established_core_discrepancy:
        return {
            "accepted": False,
            "rejected_reason": (
                "a refuted core image-grounded target fact requires a decisive "
                "established discrepancy and fake verdict in the same atomic update"
            ),
        }
    if established_core_discrepancy and output.verdict_proposal != "fake":
        return {
            "accepted": False,
            "rejected_reason": (
                "an established decisive core discrepancy requires fake"
            ),
        }
    if output.verdict_proposal == "fake" and not established_core_discrepancy:
        return {"accepted": False, "rejected_reason": "fake verdict requires a decisive established discrepancy"}
    if output.verdict_proposal == "real" and (
        core_fact is None
        or core_fact.status != "supported"
        or decisive
        or remaining_claim_hypothesis_routes(candidate)
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "real verdict requires the core target fact to be supported, "
                "no decisive discrepancy, and no open core route"
            ),
        }
    candidate.proposed_verdict = output.verdict_proposal
    decision_id = stable_id(
        "discrepancy-decision",
        candidate.brief.case_id,
        candidate.action_count,
        trigger,
        reviewed_ids,
        decision_ordinal,
    )
    candidate.discrepancy_decisions.append(
        DiscrepancyDecisionRecord(
            decision_id=decision_id,
            action_count=candidate.action_count,
            trigger=trigger,
            reviewed_evidence_ids=reviewed_ids,
            output=output,
            accepted_assessment_ids=accepted_assessment_ids,
            accepted_discrepancy_id=discrepancy_id,
            accepted_hypothesis_ids=accepted_hypothesis_ids,
            retired_hypothesis_ids=retired_ids,
            accepted_visual_question_id=accepted_visual_question_id,
        )
    )
    try:
        validated = ImageOnlyInvestigationState.model_validate(candidate.model_dump())
    except Exception as exc:
        return {"accepted": False, "rejected_reason": f"invalid discrepancy state: {exc}"}
    _replace_state(state, validated)
    return {
        "accepted": True,
        "decision_id": decision_id,
        "accepted_assessment_ids": accepted_assessment_ids,
        "accepted_discrepancy_id": discrepancy_id,
        "created_composite_finding_ids": created_composite_finding_ids,
        "accepted_hypothesis_ids": accepted_hypothesis_ids,
        "retired_hypothesis_ids": retired_ids,
        "accepted_visual_question_id": accepted_visual_question_id,
        "accepted_visual_evidence_disposition": bool(
            output.visual_evidence_disposition
        ),
        "verdict_proposal": output.verdict_proposal,
    }


def _replace_state(
    state: ImageOnlyInvestigationState,
    replacement: ImageOnlyInvestigationState,
) -> None:
    """Replace a validated mutable Pydantic state without exposing partial writes."""

    for field_name in ImageOnlyInvestigationState.model_fields:
        setattr(state, field_name, getattr(replacement, field_name))


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
        unobserved_statement_values = _unobserved_named_values(
            proposal.statement,
            (),
            grounding_text,
        )
        invalid_initial_scope = (
            proposal.predicate != "visual_integrity"
            and _target_describes_unobserved_media_state(
                proposal.statement
            )
        )
        if invalid_initial_scope:
            rejected_reasons.append(
                "target describes an unseen original, unaltered, or "
                "counterfactual media state instead of the visible positive "
                "subject-object, person-product, place, or event relation. "
                "Preserve the visible relation; do not replace it with hidden "
                "source-image details or incidental OCR metadata"
            )
            continue
        proposal = _normalize_initial_visual_identity_predicate(
            proposal,
            parents=parents,
        )
        proposal = _normalize_single_subject_visual_relation(
            proposal,
            state=state,
            parents=parents,
        )
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
        relation_entities = None
        if proposal.predicate == "depicts_relation":
            relation_entities = _visible_relation_entities(state, parents)
            if proposal.kind != "relation" or relation_entities is None:
                rejected_reasons.append(
                    "depicts_relation must bind two different visible entities "
                    "from image-grounded parent facts"
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
            and not unobserved_statement_values
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
            subject_entity_id = parent.subject_entity_id
            object_entity_id = parent.object_entity_id
            if relation_entities is not None:
                subject_entity_id, object_entity_id = relation_entities
            elif proposal.predicate == "identified_as":
                visible_subject_ids = _visible_parent_entity_ids(
                    state,
                    parents,
                )
                if len(visible_subject_ids) == 1:
                    subject_entity_id = visible_subject_ids[0]
                    object_entity_id = None
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
                subject_entity_id=subject_entity_id,
                predicate=proposal.predicate,
                object_entity_id=object_entity_id,
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


def _normalize_initial_visual_identity_predicate(
    proposal: TargetFactProposal,
    *,
    parents: Sequence[VisualFact],
) -> TargetFactProposal:
    """Keep visible scene identity separate from source-record attribution.

    A policy may use ``source_record_matches`` while stating only what the image
    visibly depicts. Without a visible OCR/source anchor, that proposition is a
    scene or subject identity, not a source-record claim. Normalize the predicate
    instead of rejecting an otherwise grounded target or allowing unseen
    publisher metadata to own the verdict.
    """

    if proposal.predicate != "source_record_matches":
        return proposal
    if any(
        parent.origin.type == "ocr" or parent.predicate == "reads"
        for parent in parents
    ):
        return proposal
    statement = " ".join(str(proposal.statement or "").casefold().split())
    source_scope = (
        "source record",
        "official record",
        "public record",
        "published by",
        "released by",
        "posted by",
        "uploaded by",
        "source page",
        "web page",
        "website",
        "account",
        "post text",
    )
    if any(phrase in statement for phrase in source_scope):
        return proposal
    return proposal.model_copy(
        update={
            "predicate": "identified_as",
            "question": (
                "Does reliable evidence support this image-grounded visual "
                f"identity: {proposal.statement.rstrip('.')}?"
            )[:800],
            "purpose": "Verify the visible subject or scene identity.",
        }
    )


def _normalize_single_subject_visual_relation(
    proposal: TargetFactProposal,
    *,
    state: ImageOnlyInvestigationState,
    parents: Sequence[VisualFact],
) -> TargetFactProposal:
    """Represent a one-subject scene hypothesis without inventing a second entity."""

    if proposal.predicate != "depicts_relation":
        return proposal
    visible_entity_ids = _visible_parent_entity_ids(state, parents)
    if len(visible_entity_ids) != 1:
        return proposal
    all_visible_entity_ids = {
        item.entity_id
        for item in state.entities
        if item.origin == "input_image"
        and item.entity_type.casefold() != "image"
    }
    if all_visible_entity_ids != set(visible_entity_ids):
        return proposal
    return proposal.model_copy(
        update={
            "kind": "attribute",
            "predicate": "identified_as",
            "purpose": (
                proposal.purpose
                or "Verify the visible subject or scene identity."
            ),
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


def _visible_relation_entities(
    state: ImageOnlyInvestigationState,
    parents: Sequence[VisualFact],
) -> tuple[str, str] | None:
    """Bind a visible relation to two actual image entities, not the canvas."""

    visible_entity_ids = _visible_parent_entity_ids(state, parents)
    if len(visible_entity_ids) < 2:
        return None
    entities = {item.entity_id: item for item in state.entities}

    def subject_rank(entity_id: str) -> tuple[int, int]:
        entity_type = entities[entity_id].entity_type.casefold()
        return (
            0 if entity_type in {"person", "animal", "organization"} else 1,
            visible_entity_ids.index(entity_id),
        )

    subject_entity_id = min(visible_entity_ids, key=subject_rank)
    object_entity_id = next(
        entity_id
        for entity_id in visible_entity_ids
        if entity_id != subject_entity_id
    )
    return subject_entity_id, object_entity_id


def _visible_parent_entity_ids(
    state: ImageOnlyInvestigationState,
    parents: Sequence[VisualFact],
) -> List[str]:
    """Return non-canvas visible entities explicitly owned by parent facts."""

    entities = {item.entity_id: item for item in state.entities}
    return list(
        dict.fromkeys(
            parent.subject_entity_id
            for parent in parents
            if parent.predicate == "visible_in"
            and parent.origin.type == "input_image"
            and parent.subject_entity_id in entities
            and entities[parent.subject_entity_id].entity_type != "image"
        )
    )


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


def pending_query_replan_evidence_ids(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return core Evidence not yet exposed to a search-direction replan."""

    core_id = state.core_verdict_fact_id
    if not core_id:
        return []
    reviewed = {
        evidence_id
        for item in state.query_replans
        for evidence_id in item.new_evidence_ids
    }
    return [
        item.evidence_id
        for item in state.evidence
        if core_id in item.fact_ids
        and item.evidence_id not in reviewed
    ]


def pending_visual_reinspection(
    state: ImageOnlyInvestigationState,
) -> VisualReinspectionRecord | None:
    """Return the oldest accepted visual question that still needs execution."""

    return next(
        (
            item
            for item in state.visual_reinspections
            if item.status == "pending"
        ),
        None,
    )


def _visual_reinspection_anchor_regions(
    state: ImageOnlyInvestigationState,
    anchor_fact_ids: Sequence[str],
) -> List[List[float]]:
    """Resolve stable detail regions from the request's pixel/OCR fact anchors."""

    fact_by_id = {item.fact_id: item for item in state.facts}
    entity_by_id = {item.entity_id: item for item in state.entities}
    retrieval_by_id = {
        item.anchor_id: item
        for item in state.retrieval_anchors
    }
    regions: List[List[float]] = []
    for fact_id in anchor_fact_ids:
        fact = fact_by_id.get(fact_id)
        if fact is None:
            continue
        candidate_ids = [
            fact.subject_entity_id,
            fact.object_entity_id or "",
            *fact.basis_ids,
            *fact.origin.origin_ids,
        ]
        for candidate_id in candidate_ids:
            entity = entity_by_id.get(candidate_id)
            region = (
                list(entity.region)
                if entity is not None and entity.region is not None
                else None
            )
            retrieval = retrieval_by_id.get(candidate_id)
            if region is None and retrieval is not None:
                region = (
                    list(retrieval.region)
                    if retrieval.region is not None
                    else None
                )
            if region is None or region == [0.0, 0.0, 1.0, 1.0]:
                continue
            normalized = [round(float(item), 6) for item in region]
            if normalized not in regions:
                regions.append(normalized)
    return regions[:4]


def _visual_reinspection_requests_equivalent(
    left: VisualReinspectionRequest,
    right: VisualReinspectionRequest,
) -> bool:
    """Reject repeated visual questions without constraining their semantics."""

    if left.scope != right.scope:
        return False
    left_anchors = set(left.anchor_fact_ids)
    right_anchors = set(right.anchor_fact_ids)
    if left_anchors and right_anchors and not left_anchors & right_anchors:
        return False
    left_tokens = _semantic_request_tokens(
        left.question + " " + left.expected_property
    )
    right_tokens = _semantic_request_tokens(
        right.question + " " + right.expected_property
    )
    if not left_tokens or not right_tokens:
        return False
    return (
        len(left_tokens & right_tokens)
        / len(left_tokens | right_tokens)
        >= 0.65
    )


def _hypothesis_text_equivalent(left: str, right: str) -> bool:
    """Reject semantically repeated bounded routes without comparing IDs."""

    left_tokens = _semantic_request_tokens(left)
    right_tokens = _semantic_request_tokens(right)
    if not left_tokens or not right_tokens:
        return str(left or "").strip().casefold() == str(right or "").strip().casefold()
    return (
        len(left_tokens & right_tokens)
        / len(left_tokens | right_tokens)
        >= 0.8
    )


def _semantic_request_tokens(value: str) -> set[str]:
    text = str(value or "").casefold()
    tokens = {
        token
        for token in re.findall(
            r"[\w]+",
            text,
            flags=re.UNICODE,
        )
        if len(token) > 1
    }
    cjk = "".join(
        character
        for character in text
        if "\u3400" <= character <= "\u9fff"
    )
    tokens.update(
        cjk[index : index + 3]
        for index in range(max(0, len(cjk) - 2))
    )
    return tokens


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
    if any(
        item.tool_name == "visit"
        and item.stance == "refute"
        and item.quality in {"strong", "moderate"}
        for item in new_rows
    ):
        # The extractor only proposes a query-relative stance. An explicit
        # contradiction is important enough to send to the semantic checkpoint
        # even when the selected passage was marked indirect; Gemini still owns
        # the actual proposition-level verdict.
        return "decisive_evidence"
    if any(
        item.tool_name == "visit"
        and item.evidence_kind == "web_span"
        and item.directness == "direct"
        and item.quality in {"strong", "moderate"}
        for item in new_rows
    ):
        # One directly inspected exact span is a material semantic boundary.
        # The checkpoint may accept, reject, contextualize, or replan it; the
        # extractor's source class and query-relative stance do not decide.
        return "decisive_evidence"
    directly_inspected = [
        item
        for item in new_rows
        if item.tool_name
        in {
            "visit",
            "compare_with_reference",
            "crop_and_inspect",
            "focused_visual_inspection",
        }
        and item.directness == "direct"
        and item.quality in {"strong", "moderate"}
    ]
    if directly_inspected and prior is None:
        return "decisive_evidence"
    if any(
        item.tool_name == "focused_visual_inspection"
        for item in directly_inspected
    ):
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
    if (
        output.assessment in {"insufficient", "conflicted"}
        and output.binding_requirement == "same_capture_required"
        and core.predicate not in _SOURCE_BINDING_PREDICATES
    ):
        return {
            "accepted": False,
            "rejected_reason": (
                "same-capture Evidence may resolve a visible-world proposition "
                "when it has already been found, but exact source-image recovery "
                "cannot become the mandatory remaining gap for this active "
                f"{core.predicate!r} fact. Keep the gap on the visible-world "
                "relation, request focused visual reinspection when a newly "
                "discovered candidate has visible discriminators, or resolve "
                "the fact from eligible Evidence."
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
    accepted_visual_question_id = ""
    accepted_visual_task_id = ""
    visual_request = output.visual_reinspection
    if visual_request is not None:
        if output.refinement is not None:
            return {
                "accepted": False,
                "rejected_reason": (
                    "one Evidence decision cannot both replace the active fact "
                    "and request visual reinspection; inspect the pixels before "
                    "deciding whether a refinement is warranted"
                ),
            }
        if output.assessment not in {"insufficient", "conflicted"}:
            return {
                "accepted": False,
                "rejected_reason": (
                    "a resolved fact must stop instead of opening visual "
                    "reinspection"
                ),
            }
        if state.action_count >= MAX_TOOL_ACTIONS:
            return {
                "accepted": False,
                "rejected_reason": (
                    "the tool-action budget cannot execute another visual "
                    "reinspection"
                ),
            }
        if len(state.visual_reinspections) >= MAX_VISUAL_REINSPECTIONS:
            return {
                "accepted": False,
                "rejected_reason": (
                    "the bounded visual-reinspection budget is exhausted"
                ),
            }
        if len(state.tasks) >= TOTAL_TASKS_MAX:
            return {
                "accepted": False,
                "rejected_reason": (
                    "task budget cannot hold the visual reinspection"
                ),
            }
        anchor_ids = list(dict.fromkeys(visual_request.anchor_fact_ids))
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
                    "visual reinspection anchors must be existing pixel/OCR facts"
                ),
            }
        grounding_ids = list(
            dict.fromkeys(visual_request.grounding_evidence_ids)
        )
        if not set(grounding_ids) <= set(reviewed_ids):
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection grounding must use newly reviewed Evidence"
                ),
            }
        if any(
            core.fact_id not in evidence_by_id[item].fact_ids
            for item in grounding_ids
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "visual reinspection grounding is outside the active core fact"
                ),
            }
        if any(
            _visual_reinspection_requests_equivalent(
                prior.request,
                visual_request,
            )
            for prior in state.visual_reinspections
        ):
            return {
                "accepted": False,
                "rejected_reason": (
                    "an equivalent visual question has already been requested"
                ),
            }

        anchor_regions = _visual_reinspection_anchor_regions(
            state,
            anchor_ids,
        )
        visual_question_id = stable_id(
            "visual-question",
            state.brief.case_id,
            core.fact_id,
            visual_request.model_dump(mode="json"),
        )
        visual_task = ResearchTask(
            task_id=stable_id(
                "task",
                visual_question_id,
                "focused-visual-inspection",
            ),
            fact_ids=[core.fact_id],
            question=visual_request.question,
            purpose=(
                "Reinspect the original pixels after web investigation introduced "
                "a concrete visible hypothesis."
            ),
            priority=1,
            status="active",
            parent_task_id=None,
            origin_ids=list(
                dict.fromkeys(
                    [
                        core.fact_id,
                        *anchor_ids,
                        *grounding_ids,
                    ]
                )
            )[:12],
            suggested_tools=["focused_visual_inspection"],
            suggested_queries=[],
        )
        state.tasks.append(visual_task)
        state.visual_reinspections.append(
            VisualReinspectionRecord(
                visual_question_id=visual_question_id,
                task_id=visual_task.task_id,
                fact_id=core.fact_id,
                created_action_count=state.action_count,
                request=visual_request,
                anchor_regions=anchor_regions,
            )
        )
        state.recommended_next_task_ids = list(
            dict.fromkeys(
                [
                    visual_task.task_id,
                    *state.recommended_next_task_ids,
                ]
            )
        )[:4]
        accepted_visual_question_id = visual_question_id
        accepted_visual_task_id = visual_task.task_id

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
                    "image-world relation; refine the visible subject, object "
                    "category, place, or event rather than switching to source "
                    "record, platform, provenance, or other metadata attribution"
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
        removable_named_values: set[str] = set()
        if refinement.slot == "object_category":
            object_entity = next(
                (
                    item
                    for item in state.entities
                    if item.entity_id == core.object_entity_id
                ),
                None,
            )
            if (
                object_entity is None
                or not any(
                    item.origin.type == "input_image"
                    and item.subject_entity_id == object_entity.entity_id
                    for item in anchors
                )
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "object_category refinement must be anchored to the "
                        "same visible object entity"
                    ),
                }
            removable_named_values = _attribution_tokens(
                object_entity.name
            )
            grounding_text = " ".join(
                evidence_by_id[item].exact_text
                for item in grounding_ids
                if item in evidence_by_id
            )
            if not _object_category_is_evidence_grounded(
                core.statement,
                refinement.statement,
                grounding_text,
            ):
                return {
                    "accepted": False,
                    "rejected_reason": (
                        "object_category refinement introduces category terms "
                        "that are not grounded in the newly reviewed exact "
                        "Evidence spans"
                    ),
                }
        if not _refinement_preserves_core_scope(
            core.statement,
            refinement.statement,
            removable_named_values=removable_named_values,
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
        inherited_discovery_ids = _inherit_refinement_discoveries(
            state,
            refinement_task,
            refinement_fact_id=refinement_fact.fact_id,
            grounding_evidence_ids=grounding_ids,
        )
        refinement_task.origin_ids = list(
            dict.fromkeys(
                [
                    *refinement_task.origin_ids,
                    *inherited_discovery_ids,
                ]
            )
        )[:12]
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
                        statement=(
                            "Semantic Evidence Decision: the active proposition "
                            f"was {output.assessment} by the selected Evidence."
                        ),
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
        accepted_visual_question_id=(
            accepted_visual_question_id or None
        ),
        accepted_visual_task_id=(
            accepted_visual_task_id or None
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
        "accepted_visual_question_id": accepted_visual_question_id,
        "accepted_visual_task_id": accepted_visual_task_id,
        "reviewed_evidence_ids": reviewed_ids,
    }


def apply_evidence_decision_with_refinement_fallback(
    state: ImageOnlyInvestigationState,
    output: EvidenceDecisionOutput,
    *,
    reviewed_evidence_ids: Sequence[str],
    trigger: str,
) -> Dict[str, Any]:
    """Preserve a valid unresolved decision when only its optional refinement fails."""

    update = apply_evidence_decision(
        state,
        output,
        reviewed_evidence_ids=reviewed_evidence_ids,
        trigger=trigger,
    )
    if (
        update.get("accepted", False)
        or output.refinement is None
        or output.visual_reinspection is not None
        or output.assessment not in {"insufficient", "conflicted"}
    ):
        return update

    rejected_reason = str(
        update.get("rejected_reason", "optional refinement was rejected")
    )
    stripped_output = output.model_copy(update={"refinement": None})
    fallback = apply_evidence_decision(
        state,
        stripped_output,
        reviewed_evidence_ids=reviewed_evidence_ids,
        trigger=trigger,
    )
    if not fallback.get("accepted", False):
        return update
    fallback["discarded_refinement_reason"] = rejected_reason
    fallback["discarded_refinement"] = output.refinement.model_dump(mode="json")
    return fallback


def _valid_visual_refinement_transition(
    current_predicate: str,
    proposed_predicate: str,
) -> bool:
    if proposed_predicate == current_predicate:
        return True
    return (
        current_predicate in {"appears_to_depict", "depicts_relation"}
        and proposed_predicate
        in {
            "identified_as",
            "depicts_relation",
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
        "object_category": (
            {"depicts_relation"}
            if current_predicate == "depicts_relation"
            else set()
        ),
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
    *,
    removable_named_values: set[str] | None = None,
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
    required_named_values = preserved_named_values - set(
        removable_named_values or set()
    )
    return required_named_values <= proposed_tokens


def _object_category_is_evidence_grounded(
    current_statement: str,
    proposed_statement: str,
    grounding_text: str,
) -> bool:
    """Require every newly introduced category term to occur in exact Evidence."""

    current = _semantic_token_roots(current_statement)
    proposed = _semantic_token_roots(proposed_statement)
    evidence = _semantic_token_roots(grounding_text)
    new_terms = proposed - current - _REFINEMENT_SCOPE_STOPWORDS
    return bool(new_terms) and new_terms <= evidence


def _semantic_token_roots(value: str) -> set[str]:
    roots: set[str] = set()
    for token in re.findall(
        r"[a-z0-9]+",
        str(value or "").casefold(),
    ):
        if not token:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        roots.add(token)
    return roots


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
    trigger: str = "interval",
) -> ReflectionRecord:
    """Validate and apply a bounded Reflection delta."""

    if len(state.reflections) >= MAX_REFLECTIONS:
        raise RuntimeError("maximum image-only Reflection count is exhausted")
    if trigger not in {"interval", "saturation"}:
        raise ValueError(f"unsupported Reflection trigger: {trigger}")
    if trigger == "saturation" and any(
        item.trigger == "saturation" for item in state.reflections
    ):
        raise RuntimeError("the one saturation Reflection is already exhausted")
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
            rejected.append(f"{update.task_id} proposed no state change")
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
    accepted_strategy = "continue"
    accepted_replan_query = ""
    strategy_rejected_reason = ""
    strategy = bounded_output.strategy_decision
    core = fact_by_id.get(state.core_verdict_fact_id or "")
    latest_coverage = (
        state.coverage_audits[-1] if state.coverage_audits else None
    )
    low_gain_intervals = (
        latest_coverage.low_gain_intervals if latest_coverage is not None else 0
    )

    if strategy == "replan":
        task = task_by_id.get(bounded_output.strategy_task_id or "")
        if task is None:
            strategy_rejected_reason = (
                "strategy replan must name an existing task"
            )
        elif state.core_verdict_fact_id not in task.fact_ids:
            strategy_rejected_reason = (
                "strategy replan task must own the unresolved core fact"
            )
        elif task.status not in {"active", "pending", "exhausted"}:
            strategy_rejected_reason = (
                "strategy replan task is not available for investigation"
            )
        elif task.query_replan_count >= 1:
            strategy_rejected_reason = (
                "the task already used its one semantic replan"
            )
        elif "text_search" not in runtime_task_tool_names(state, task):
            strategy_rejected_reason = (
                "strategy replan requires a text-search-capable task"
            )
        elif low_gain_intervals < 2 and trigger != "saturation":
            strategy_rejected_reason = (
                "strategy replan requires sustained low information gain"
            )
        else:
            accepted_replan_query = _novel_replan_query(
                state,
                task,
                bounded_output.replacement_query,
            )
            if not accepted_replan_query:
                strategy_rejected_reason = (
                    "strategy replan proposed no genuinely new semantic query"
                )
            else:
                task.query_replan_count += 1
                task.suggested_queries = [accepted_replan_query]
                task.status = "active"
                state.recommended_next_task_ids = list(
                    dict.fromkeys(
                        [
                            task.task_id,
                            *state.recommended_next_task_ids,
                        ]
                    )
                )[:4]
                _abandon_stale_discoveries(
                    state,
                    task,
                    reason=(
                        bounded_output.strategy_rationale
                        or "Reflection replaced a stalled search direction."
                    ),
                )
                accepted_strategy = "replan"
                state.stop_reason = ""
    elif strategy == "stop_unresolved":
        if core is not None and core.status in {"supported", "refuted"}:
            strategy_rejected_reason = (
                "an already resolved core fact cannot stop as unresolved"
            )
        elif low_gain_intervals < 2 and trigger != "saturation":
            strategy_rejected_reason = (
                "stop_unresolved requires sustained low information gain"
            )
        else:
            accepted_strategy = "stop_unresolved"
            state.stop_reason = "information_saturated"
    else:
        routes = (
            remaining_material_routes(
                state,
                fact_id=state.core_verdict_fact_id or "",
            )
            if state.core_verdict_fact_id
            else []
        )
        if trigger == "saturation" and not routes:
            strategy_rejected_reason = (
                "saturation Reflection cannot continue without an executable route"
            )
        else:
            accepted_strategy = "continue"
            if routes and state.stop_reason == "information_saturated":
                state.stop_reason = ""

    if strategy_rejected_reason:
        rejected.append(strategy_rejected_reason)
    record = ReflectionRecord(
        reflection_id=stable_id(
            "reflection",
            state.brief.case_id,
            state.action_count,
            len(state.reflections) + 1,
        ),
        action_count=state.action_count,
        trigger=trigger,
        output=bounded_output,
        accepted_task_update_ids=accepted_updates,
        accepted_new_task_ids=accepted_new,
        rejected_reasons=rejected,
        accepted_strategy_decision=accepted_strategy,
        accepted_replan_query=accepted_replan_query,
        strategy_rejected_reason=strategy_rejected_reason,
        evidence_gain=evidence_gain,
        decision_gain=decision_gain,
    )
    state.reflections.append(record)
    state.reflection_failure_streak = 0
    return record


def _novel_replan_query(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    proposed: str,
) -> str:
    attempted = _attempted_routes_by_task(state).get(task.task_id, [])
    prior_queries: List[str] = list(task.suggested_queries)
    for route in attempted:
        if str(route.get("tool", "")).strip() != "text_search":
            continue
        values = route.get("queries", []) or []
        if isinstance(values, str):
            values = [values]
        prior_queries.extend(str(value) for value in values)

    query = " ".join(str(proposed).split())
    if not query:
        return ""
    if any(
        routes_semantically_equivalent(
            "text_search",
            {"queries": [query]},
            "text_search",
            {"queries": [prior]},
        )
        for prior in prior_queries
        if prior
    ):
        return ""
    return query


def _normalized_replan_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[\w]+",
            str(value or "").casefold(),
            flags=re.UNICODE,
        )
        if len(token) > 1
    }


def _normalized_replan_text(value: str) -> str:
    return " ".join(
        re.findall(
            r"[\w]+",
            str(value or "").casefold(),
            flags=re.UNICODE,
        )
    )


def query_concept_extraction_error(
    state: ImageOnlyInvestigationState,
    output: QueryConceptExtractionOutput,
    *,
    task_id: str,
    new_evidence_ids: Sequence[str],
) -> str:
    """Validate that extracted concepts are traceable to supplied exact Evidence."""

    if output.task_id != task_id:
        return "query concepts must use the supplied task_id"
    valid_evidence = {
        item.evidence_id: item
        for item in state.evidence
        if state.core_verdict_fact_id in item.fact_ids
        and item.evidence_id in set(new_evidence_ids)
    }
    core = next(
        (
            fact
            for fact in state.facts
            if fact.fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    known_texts = [
        core.statement if core is not None else "",
    ]
    for route in _attempted_routes_by_task(state).get(task_id, []):
        if str(route.get("tool", "")).strip() != "text_search":
            continue
        values = route.get("queries", []) or []
        if isinstance(values, str):
            values = [values]
        known_texts.extend(str(value) for value in values)
    normalized_known = [
        _normalized_replan_text(value)
        for value in known_texts
        if _normalized_replan_text(value)
    ]
    seen_concept_ids: set[str] = set()
    for concept in output.concepts:
        if concept.concept_id in seen_concept_ids:
            return f"duplicate query concept id {concept.concept_id}"
        seen_concept_ids.add(concept.concept_id)
        evidence = valid_evidence.get(concept.evidence_id)
        if evidence is None:
            return (
                f"query concept {concept.concept_id} cites Evidence that was not "
                "supplied for the active fact"
            )
        phrase = _normalized_replan_text(concept.evidence_phrase)
        if not phrase or phrase not in _normalized_replan_text(evidence.exact_text):
            return (
                f"query concept {concept.concept_id} does not copy an exact "
                "Evidence phrase"
            )
        if any(phrase in known for known in normalized_known):
            return (
                f"query concept {concept.concept_id} repeats information already "
                "explicit in the proposition or attempted queries"
            )
    return ""


def _compose_query_replan(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    concept_extraction: QueryConceptExtractionOutput,
    output: QueryReplanOutput,
    *,
    new_evidence_ids: Sequence[str],
) -> tuple[str, str]:
    """Validate one evidence-grounded query without interpreting its domain."""

    if concept_extraction.task_id != task.task_id:
        return "", "query concepts must belong to the replanned task"
    concepts_by_id = {
        item.concept_id: item
        for item in concept_extraction.concepts
    }
    selected = concepts_by_id.get(output.selected_concept_id)
    if selected is None:
        return "", "selected query concept was not supplied"
    if selected.evidence_id not in set(new_evidence_ids):
        return "", "selected query concept is not grounded in new Evidence"

    query = " ".join(output.replacement_query.split())
    concept_term = _normalized_replan_text(output.concept_term)
    if not concept_term or concept_term not in _normalized_replan_text(query):
        return "", "replacement query does not use its declared concept term"
    novel = _novel_replan_query(state, task, query)
    if not novel:
        return "", "proposed no genuinely new semantic query"
    return novel, ""


def _query_replan_error(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    proposed_queries: Sequence[str],
    *,
    trigger: str,
) -> str:
    if trigger != "evidence_boundary":
        return "query replan requires an evidence boundary"
    if task.query_replan_count >= 1:
        return "already used its one semantic query replan"
    if "text_search" not in runtime_task_tool_names(state, task):
        return "cannot replan queries because text_search is not enabled"
    if state.core_verdict_fact_id not in task.fact_ids:
        return "cannot replan queries for a task outside the open core fact"
    core = next(
        (
            fact
            for fact in state.facts
            if fact.fact_id == state.core_verdict_fact_id
        ),
        None,
    )
    if core is None or core.status in {"supported", "refuted"}:
        return "cannot replan queries after the core fact is resolved"
    if not proposed_queries:
        return "must propose one genuinely new query"

    attempts = _attempted_routes_by_task(state).get(task.task_id, [])
    text_search_count = sum(
        str(route.get("tool", "")).strip() == "text_search"
        for route in attempts
    )
    if text_search_count < MAX_TEXT_SEARCH_ROUTES_PER_TASK:
        return (
            "cannot replan before the initial text-search allowance is "
            "exhausted"
        )
    latest_text_search_index = max(
        (
            index
            for index, route in enumerate(attempts)
            if str(route.get("tool", "")).strip() == "text_search"
        ),
        default=-1,
    )
    latest_search_outcome = (
        str(attempts[latest_text_search_index].get("outcome", "")).strip()
        if latest_text_search_index >= 0
        else ""
    )
    inspected_after_latest_search = any(
        str(route.get("tool", "")).strip()
        in {"visit", "compare_with_reference"}
        for route in attempts[latest_text_search_index + 1 :]
    )
    if (
        latest_search_outcome == "discovery"
        and not inspected_after_latest_search
    ):
        return (
            "cannot replan away from the latest search batch before inspecting at "
            "least one candidate"
        )
    return ""


def apply_query_replan(
    state: ImageOnlyInvestigationState,
    concept_extraction: QueryConceptExtractionOutput,
    output: QueryReplanOutput,
    *,
    trigger: str,
    new_evidence_ids: Sequence[str],
    source_access_policy: Any = None,
) -> QueryReplanRecord:
    """Apply one model-proposed search-direction update without judging facts."""

    task = _task_by_id(state, output.task_id)
    accepted_query = ""
    rejected_reason = ""
    if task is None:
        rejected_reason = f"unknown task {output.task_id}"
    elif task.query_replan_count >= 1:
        rejected_reason = "already used its one semantic query replan"
    else:
        rejected_reason = query_concept_extraction_error(
            state,
            concept_extraction,
            task_id=task.task_id,
            new_evidence_ids=new_evidence_ids,
        )
        if not rejected_reason:
            accepted_query, rejected_reason = _compose_query_replan(
                state,
                task,
                concept_extraction,
                output,
                new_evidence_ids=new_evidence_ids,
            )
        if not rejected_reason and accepted_query:
            violation = query_policy_violation(
                accepted_query,
                source_access_policy=source_access_policy,
            )
            if violation:
                rejected_reason = (
                    "replacement query must seek underlying facts or sources, "
                    "not a ready-made fact-check verdict or an excluded source; "
                    f"{violation}: {accepted_query!r}"
                )
        if not rejected_reason:
            rejected_reason = _query_replan_error(
                state,
                task,
                [accepted_query] if accepted_query else [],
                trigger=trigger,
            )

    valid_evidence_ids = {
        item.evidence_id
        for item in state.evidence
        if state.core_verdict_fact_id in item.fact_ids
    }
    grounded_evidence_ids = [
        item
        for item in dict.fromkeys(new_evidence_ids)
        if item in valid_evidence_ids
    ][:40]
    if trigger == "evidence_boundary" and not grounded_evidence_ids:
        rejected_reason = rejected_reason or (
            "evidence-boundary replan requires new core Evidence"
        )

    if task is not None and not rejected_reason:
        task.query_replan_count += 1
        if accepted_query:
            task.suggested_queries = [accepted_query]
            task.status = "active"
            state.recommended_next_task_ids = list(
                dict.fromkeys(
                    [
                        task.task_id,
                        *state.recommended_next_task_ids,
                    ]
                )
            )[:4]
            _abandon_stale_discoveries(
                state,
                task,
                reason=(
                    output.rationale
                    or "Query Replan replaced the stalled search direction."
                ),
            )
    record = QueryReplanRecord(
        replan_id=stable_id(
            "query-replan",
            state.brief.case_id,
            state.action_count,
            len(state.query_replans) + 1,
            output.model_dump(mode="json"),
        ),
        action_count=state.action_count,
        trigger=trigger,
        new_evidence_ids=grounded_evidence_ids,
        concept_extraction=concept_extraction,
        output=output,
        accepted_queries=[accepted_query] if accepted_query and not rejected_reason else [],
        rejected_reason=rejected_reason,
    )
    state.query_replans.append(record)
    return record


def route_local_replan_candidate(
    state: ImageOnlyInvestigationState,
    *,
    observation_update: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return one task-local boundary that merits a bounded model replan.

    This is intentionally structural rather than semantic.  It observes whether a
    candidate batch was exhausted, a source failed, or an inspected image/source
    produced material but unresolved information.  The model—not this helper—then
    chooses whether to change query, add a visual inspection, continue, or retire
    the route.
    """

    task_id = str(observation_update.get("task_id", "")).strip()
    task = _task_by_id(state, task_id)
    if (
        task is None
        or task.route_replan_count >= 1
        or task.status not in {"active", "pending", "exhausted"}
        or not _task_serves_open_route_local_target(state, task)
    ):
        return None
    if _route_local_target_is_resolved(state, task):
        return None

    attempts = _attempted_routes_by_task(state).get(task.task_id, [])
    if not attempts:
        return None
    latest = attempts[-1]
    tool_name = str(latest.get("tool", "")).strip()
    outcome = str(latest.get("outcome", "")).strip()

    failure_ids = {
        str(item)
        for item in observation_update.get("created_failure_ids", []) or []
    }
    failures = {
        item.failure_id: item
        for item in state.failures
        if item.task_id == task.task_id
    }
    if any(
        failure_id in failures
        and failures[failure_id].recoverable
        and failures[failure_id].code
        in {
            "tool_error",
            "rate_limited",
            "provider_unavailable",
            "access_limited",
            "malformed_result",
            "timeout",
        }
        for failure_id in failure_ids
    ):
        return task.task_id, "source_failure"

    created_evidence_ids = {
        str(item)
        for item in observation_update.get("created_evidence_ids", []) or []
    }
    if tool_name in {
        "crop_and_inspect",
        "focused_visual_inspection",
        "check_consistency",
        "analyze_visual_anomalies",
    }:
        if created_evidence_ids or outcome in {"empty", "context"}:
            return task.task_id, "visual_signal"

    if tool_name in {"visit", "compare_with_reference"}:
        if created_evidence_ids:
            return task.task_id, "related_unclosed"
        pending_pages = _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="visit",
        )
        pending_references = _pending_inspection_batches(
            state,
            task,
            attempts,
            tool_name="compare_with_reference",
        )
        if not pending_pages and not pending_references:
            return task.task_id, "candidate_exhausted"

    if tool_name == "text_search" and outcome in {"empty", "failed"}:
        return task.task_id, "candidate_exhausted"
    return None


def _task_serves_open_route_local_target(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    """Accept both legacy core facts and v4 image-claim target ownership."""

    core_id = state.core_verdict_fact_id or ""
    if core_id:
        return core_id in task.fact_ids
    open_claims = [
        item
        for item in state.image_claims
        if item.status in {"open", "unresolved", "conflicted"}
    ]
    if not open_claims:
        return False
    open_claim_ids = {item.claim_id for item in open_claims}
    open_fact_ids = {item.fact_id for item in open_claims}
    return bool(
        set(task.claim_ids) & open_claim_ids
        or set(task.fact_ids) & open_fact_ids
    )


def _route_local_target_is_resolved(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    """Return whether the route's active target has already closed."""

    if state.proposed_verdict in {"fake", "real"}:
        return True
    core_id = state.core_verdict_fact_id or ""
    if core_id:
        core = next((item for item in state.facts if item.fact_id == core_id), None)
        return bool(
            core is not None
            and core.status in {"supported", "refuted", "conflicted"}
        )
    claims_by_id = {item.claim_id: item for item in state.image_claims}
    route_claims = [
        claims_by_id[item]
        for item in task.claim_ids
        if item in claims_by_id
    ]
    return bool(route_claims) and all(
        item.status in {"supported", "refuted"} for item in route_claims
    )


def apply_route_local_replan(
    state: ImageOnlyInvestigationState,
    output: RouteLocalReplanOutput,
    *,
    trigger: str,
    source_access_policy: Any = None,
) -> RouteLocalReplanRecord:
    """Apply one free but bounded route-local replanning decision.

    The initial Planning graph is unchanged.  This only changes an existing route
    after the runtime has reached an observed retrieval/inspection boundary.
    """

    task = _task_by_id(state, output.task_id)
    accepted_strategy = "rejected"
    accepted_query = ""
    rejected_reason = ""
    valid_triggers = {
        "related_unclosed",
        "candidate_exhausted",
        "source_failure",
        "visual_signal",
    }

    if trigger not in valid_triggers:
        rejected_reason = "route-local replan requires a recognized route boundary"
    elif task is None:
        rejected_reason = f"unknown task {output.task_id}"
    elif task.route_replan_count >= 1:
        rejected_reason = "the task already used its one route-local replan"
    elif task.status not in {"active", "pending", "exhausted"}:
        rejected_reason = "route-local replan task is not available"
    elif not _task_serves_open_route_local_target(state, task):
        rejected_reason = "route-local replan task must own an open target fact"
    elif _route_local_target_is_resolved(state, task):
        rejected_reason = "cannot replan after the route target is resolved"

    if not rejected_reason and task is not None:
        if output.strategy == "replace_query":
            if task.query_replan_count >= 1:
                rejected_reason = "the task already has one replacement query"
            elif "text_search" not in runtime_task_tool_names(state, task):
                rejected_reason = "route cannot replace a query without text_search"
            else:
                accepted_query = _novel_replan_query(
                    state,
                    task,
                    output.replacement_query,
                )
                if not accepted_query:
                    rejected_reason = "replacement query is empty or semantically repeated"
                else:
                    violation = query_policy_violation(
                        accepted_query,
                        source_access_policy=source_access_policy,
                    )
                    if violation:
                        rejected_reason = (
                            "replacement query must seek underlying facts or sources; "
                            f"{violation}: {accepted_query!r}"
                        )
            if not rejected_reason:
                task.query_replan_count += 1
                task.suggested_queries = [accepted_query]
                task.status = "active"
                task.route_replan_focus = ""
                _abandon_stale_discoveries(
                    state,
                    task,
                    reason=output.rationale,
                )
                accepted_strategy = "replace_query"
        elif output.strategy == "add_visual_route":
            focus = " ".join(output.visual_focus.split())
            attempts = _attempted_routes_by_task(state).get(task.task_id, [])
            already_inspected = any(
                str(item.get("tool", "")).strip() == "crop_and_inspect"
                for item in attempts
            )
            if not focus:
                rejected_reason = "visual route requires a concrete visible focus"
            elif already_inspected:
                rejected_reason = "route already used crop_and_inspect"
            else:
                task.suggested_tools = list(
                    dict.fromkeys([*task.suggested_tools, "crop_and_inspect"])
                )
                task.route_replan_focus = focus
                task.status = "active"
                _abandon_stale_discoveries(
                    state,
                    task,
                    reason=output.rationale,
                )
                accepted_strategy = "add_visual_route"
        elif output.strategy == "continue":
            if task.status == "exhausted":
                rejected_reason = "cannot continue an exhausted route without a new action"
            else:
                task.route_replan_focus = ""
                accepted_strategy = "continue"
        elif output.strategy == "stop_route":
            task.status = "exhausted"
            task.route_replan_focus = ""
            hypothesis = next(
                (
                    item
                    for item in state.search_hypotheses
                    if item.hypothesis_id == task.hypothesis_id
                ),
                None,
            )
            if hypothesis is not None and not any(
                other.task_id != task.task_id
                and other.hypothesis_id == task.hypothesis_id
                and other.status in {"active", "pending"}
                for other in state.tasks
            ):
                hypothesis.status = "exhausted"
            accepted_strategy = "stop_route"

    if accepted_strategy != "rejected" and task is not None:
        task.route_replan_count += 1
        state.recommended_next_task_ids = list(
            dict.fromkeys(
                [
                    task.task_id,
                    *state.recommended_next_task_ids,
                ]
            )
        )[:4]

    record = RouteLocalReplanRecord(
        replan_id=stable_id(
            "route-local-replan",
            state.brief.case_id,
            state.action_count,
            len(state.route_local_replans) + 1,
            output.model_dump(mode="json"),
        ),
        action_count=max(1, state.action_count),
        trigger=trigger,
        task_id=output.task_id,
        output=output,
        accepted_strategy=accepted_strategy,
        accepted_query=accepted_query if accepted_strategy == "replace_query" else "",
        rejected_reason=rejected_reason,
    )
    state.route_local_replans.append(record)
    return record


def _inherit_refinement_discoveries(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    refinement_fact_id: str,
    grounding_evidence_ids: Sequence[str],
) -> List[str]:
    """Carry source pages linked by grounding Evidence into a refinement task."""

    evidence_by_id = {
        item.evidence_id: item
        for item in state.evidence
    }
    grounding = [
        evidence_by_id[item]
        for item in grounding_evidence_ids
        if item in evidence_by_id
    ]
    created: List[str] = []
    existing = {item.discovery_id for item in state.discoveries}
    for evidence in grounding:
        evidence_url = canonicalize_url(evidence.source_url)
        if not evidence_url:
            continue
        for source in list(state.discoveries):
            if source.task_id != evidence.task_id or source.abandoned:
                continue
            if evidence_url not in {
                canonicalize_url(source.candidate_url),
                canonicalize_url(source.reference_image_url),
            }:
                continue
            discovery_id = stable_id(
                "discovery",
                "refinement-inheritance",
                task.task_id,
                source.discovery_id,
            )
            if discovery_id in existing:
                continue
            state.discoveries.append(
                InvestigationDiscovery(
                    discovery_id=discovery_id,
                    task_id=task.task_id,
                    fact_ids=[refinement_fact_id],
                    function_call_id=source.function_call_id,
                    tool_name=source.tool_name,
                    candidate_url=source.candidate_url,
                    reference_image_url="",
                    title=source.title,
                    snippet=source.snippet,
                    candidate_type=source.candidate_type,
                )
            )
            existing.add(discovery_id)
            created.append(discovery_id)
    return created


def _abandon_stale_discoveries(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    reason: str,
) -> None:
    for discovery in state.discoveries:
        if (
            discovery.task_id == task.task_id
            and not discovery.abandoned
        ):
            discovery.abandoned = True
            discovery.abandonment_reason = reason[:800]


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
                            str(
                                item.get("snippet")
                                or item.get("content_preview", "")
                            ),
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
    tool_args: Mapping[str, Any],
    data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    image_sha256: str,
) -> tuple[List[str], List[str]]:
    evidence_ids: List[str] = []
    finding_ids: List[str] = []
    # Task ownership is an auditable eligibility boundary, not a semantic
    # judgment about which owned Claim the material ultimately affects. A web
    # inspection may select one owned Claim as its atomic stance target; other
    # tools keep the whole task-owned set available to the Decision checkpoint.
    selected_claim_id = str(tool_args.get("__claim_id", "")).strip()
    claim_fact_by_id = {
        claim.claim_id: claim.fact_id
        for claim in state.image_claims
        if claim.claim_id in task.claim_ids
    }
    owned_fact_ids = (
        [claim_fact_by_id[selected_claim_id]]
        if selected_claim_id in claim_fact_by_id
        else list(task.fact_ids)
    )

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
        relation_scope = str(
            record.get("relation_scope", "unclear")
        ).strip().lower()
        relation_stance = str(
            record.get("relation_stance", "unclear")
        ).strip().lower()
        if relation_scope not in {
            "same_relation",
            "partial_relation",
            "different_instance",
            "unclear",
        }:
            relation_scope = "unclear"
        if relation_stance not in {
            "supports",
            "contradicts",
            "background",
            "unclear",
        }:
            relation_stance = "unclear"
        directional_relation = (
            relation_scope == "same_relation"
            and relation_stance in {"supports", "contradicts"}
        )
        context_only = bool(record.get("context_only", False)) or not (
            directional_relation
        )
        stance = (
            {
                "supports": "support",
                "contradicts": "refute",
            }.get(relation_stance, "neutral")
            if not context_only
            else "neutral"
        )
        relevance = str(record.get("relevance", "")).strip().lower()
        quality = (
            "weak"
            if context_only
            else
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
                    fact_ids=owned_fact_ids,
                    function_call_id=function_call_id,
                    tool_name=tool_name,
                    evidence_kind="web_span",
                    source_url=source_url,
                    source_family=identity.source_family,
                    source_class=identity.source_class,
                    exact_text=evidence,
                    image_claim=str(record.get("image_claim", "")),
                    retrieval_goal=str(record.get("retrieval_goal", "")),
                    span_start=int(span["start"]),
                    span_end=int(span["end"]),
                    artifact_sha256=artifact,
                    retrieved_at=str(record.get("retrieved_at", "")),
                    stance=stance,
                    quality=quality,
                    directness=(
                        "direct"
                        if (
                            not context_only
                            and str(record.get("directness", "")).strip().lower()
                            == "direct"
                        )
                        else "indirect"
                    ),
                    claim_binding="source_assertion",
                    relation_scope=relation_scope,
                    relation_stance=relation_stance,
                    temporal_alignment=str(
                        record.get("temporal_alignment", "")
                    ).strip(),
                    risk_flags=list(identity.risk_flags),
                )
            )
            evidence_ids.append(evidence_id)
        if stance in {"support", "refute"} and owned_fact_ids:
            finding_id = stable_id(
                "finding",
                task.task_id,
                stance,
                evidence_id,
                owned_fact_ids,
            )
            if finding_id not in {item.finding_id for item in state.findings}:
                state.findings.append(
                    Finding(
                        finding_id=finding_id,
                        task_id=task.task_id,
                        fact_ids=owned_fact_ids,
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
        fact_ids=owned_fact_ids,
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
            # A visual match binds source context to the input pixels. It can
            # support image-grounded visual content/identity claims, but not
            # world-context claims such as location, event, date, authorship, or
            # provenance.
            if (
                claim_binding == "same_capture"
                and _task_owns_same_capture_supportable_fact(state, task)
            ):
                stance = "support"
    elif tool_name == "crop_and_inspect":
        observations = data.get("observations")
        if not isinstance(observations, list):
            # Compatibility for traces produced before observations replaced
            # the old findings field. The legacy answer field is intentionally
            # excluded because it was an unconstrained judgment, not an image
            # observation.
            observations = data.get("findings", [])
        statement = "\n".join(
            item
            for item in [
                str(data.get("description", "")).strip(),
                *[
                    str(observation).strip()
                    for observation in observations or []
                    if str(observation).strip()
                ],
                *[
                    str(anomaly).strip()
                    for anomaly in data.get("anomalies", []) or []
                    if str(anomaly).strip()
                ],
            ]
            if item
        ).strip()
    elif tool_name == "focused_visual_inspection":
        summary = str(data.get("summary", "")).strip()
        observations = [
            str(item.get("statement", "")).strip()
            for item in data.get("observations", []) or []
            if isinstance(item, Mapping)
            and str(item.get("statement", "")).strip()
        ]
        statement = "\n".join(
            dict.fromkeys([summary, *observations])
        ).strip()
        region = [0.0, 0.0, 1.0, 1.0]
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
        visual_question_id=(
            str(data.get("visual_question_id", "")).strip() or None
            if tool_name == "focused_visual_inspection"
            else None
        ),
        visual_scope=(
            str(data.get("scope", "")).strip() or None
            if tool_name == "focused_visual_inspection"
            else None
        ),
        visual_answer_status=(
            str(data.get("answer_status", "")).strip() or None
            if tool_name == "focused_visual_inspection"
            else None
        ),
        visual_observations=(
            [
                VisualObservation.model_validate(item)
                for item in data.get("observations", []) or []
                if isinstance(item, Mapping)
            ]
            if tool_name == "focused_visual_inspection"
            else []
        ),
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
        bundled = value.get("evidence_records")
        if isinstance(bundled, list) and bundled:
            for record in bundled:
                if isinstance(record, Mapping):
                    yield record
            return
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


def _refresh_fact_states(state: ImageOnlyInvestigationState) -> None:
    findings_by_fact: Dict[str, List[Finding]] = {}
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    for finding in state.findings:
        for fact_id in finding.fact_ids:
            findings_by_fact.setdefault(fact_id, []).append(finding)
    for fact in state.facts:
        claim = next(
            (item for item in state.image_claims if item.fact_id == fact.fact_id),
            None,
        )
        if fact.fact_id == state.core_verdict_fact_id:
            # The core VisualFact is the semantic owner. A legacy bookkeeping
            # row may mirror its status for replay, but an unresolved row must
            # not downgrade an already adjudicated core fact.
            if fact.status in {"supported", "refuted", "conflicted"}:
                continue
            if claim is not None and claim.status in {
                "supported",
                "refuted",
                "conflicted",
            }:
                fact.status = claim.status
            else:
                fact.status = "active"
            continue
        if claim is not None:
            # Non-core legacy rows remain serializable for old v4 traces.
            fact.status = {
                "open": "active",
                "unresolved": "active",
                "supported": "supported",
                "refuted": "refuted",
                "conflicted": "conflicted",
            }[claim.status]
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


def record_route_selection_exhaustion(
    state: ImageOnlyInvestigationState,
    *,
    task_id: str,
    request_id: str,
) -> Dict[str, Any]:
    """Record a bounded policy-routing failure without inventing an action.

    The model has already consumed the stage's protocol-correction budget while
    proposing only rejected calls.  Blocking this task prevents a zero-action
    loop; a subsequent Discrepancy Decision may still add a different hypothesis
    or proceed with an explicitly unresolved gap.
    """

    task = _task_by_id(state, task_id)
    if task is None or task.status not in {"active", "pending"}:
        raise ValueError(
            "route-selection exhaustion requires one active runtime task"
        )
    failure_id = _append_failure(
        state,
        task,
        call_id=request_id or stable_id("route-request", task_id),
        tool_name="route_controller",
        code="protocol_error",
        message=(
            "The policy exhausted its bounded protocol-correction chain without "
            "selecting a new executable tool action."
        ),
        recoverable=False,
    )
    task.status = "blocked"
    hypothesis = next(
        (
            item
            for item in state.search_hypotheses
            if item.hypothesis_id == task.hypothesis_id
        ),
        None,
    )
    if hypothesis is not None:
        hypothesis.status = "exhausted"
    # A blocked task cannot own the next exact archive read. The recalled memory
    # remains durable and may be selected again by a later hypothesis; only the
    # ephemeral pending selector is cleared to avoid a zero-action retry loop.
    state.pending_archive_read_ids = []
    state.recommended_next_task_ids = [
        item for item in state.recommended_next_task_ids if item != task.task_id
    ]
    return {
        "action_count": state.action_count,
        "task_id": task.task_id,
        "task_status": task.status,
        "created_failure_ids": [failure_id],
        "pending_archive_read_ids": [],
        "route_selection_exhausted": True,
    }


def _append_failure(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    *,
    call_id: str,
    tool_name: str,
    code: str,
    message: str,
    recoverable: bool = True,
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
                recoverable=recoverable,
                message=message[:4000],
            )
        )
    return failure_id


def _failure_code(
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    lowered = str(message or "").lower()
    metadata = metadata or {}
    exception_name = str(metadata.get("tool_exception", "")).casefold()
    if any(
        token in lowered
        for token in (
            "budget exhausted",
            "correction budget",
            "task exhausted",
            "tool budget",
        )
    ):
        return "budget_exhausted"
    if "timeout" in lowered or "timeout" in exception_name:
        return "timeout"
    if any(
        token in lowered
        for token in (
            "http 429",
            "status 429",
            "status=429",
            "too many requests",
            "rate limit",
            "rate-limit",
            "resource_exhausted",
        )
    ):
        return "rate_limited"
    if any(
        token in lowered
        for token in (
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "status 500",
            "status 502",
            "status 503",
            "status 504",
            "status=500",
            "status=502",
            "status=503",
            "status=504",
        )
    ):
        return "provider_unavailable"
    if any(
        token in lowered
        for token in ("blocked", "access", "captcha", "download")
    ):
        return "access_limited"
    if any(token in lowered for token in ("provider", "unavailable")):
        return "provider_unavailable"
    if any(
        token in lowered
        for token in (
            "toolresultcontracterror",
            "tool result is not",
            "tool result is missing",
            "invalid json",
            "malformed",
            "answer_status is missing or invalid",
            "summary is required",
            "response schema",
        )
    ) or exception_name == "toolresultcontracterror":
        return "malformed_result"
    if any(token in lowered for token in ("argument", "unknown", "schema")):
        return "protocol_error"
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
    if tool_name == "focused_visual_inspection":
        return not (
            str(data.get("summary", "")).strip()
            or any(
                isinstance(item, Mapping)
                and str(item.get("statement", "")).strip()
                for item in data.get("observations", []) or []
            )
        )
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


def _task_owns_same_capture_supportable_fact(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
) -> bool:
    facts = {fact.fact_id: fact for fact in state.facts}
    return any(
        facts.get(fact_id) is not None
        and same_capture_can_support_visual_claim(facts[fact_id])
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
    routes: List[str] = []
    for task in tasks:
        routes.extend(
            _remaining_task_material_routes(
                state,
                task,
                attempted.get(task.task_id, []),
            )
        )
    return _deduplicate_shared_image_routes(routes)


def remaining_claim_hypothesis_routes(
    state: ImageOnlyInvestigationState,
    *,
    task_ids: set[str] | None = None,
) -> List[str]:
    """Return untried routes for the image-grounded target fact.

    The function name is retained for replay/API compatibility. Route
    eligibility is owned by task fact IDs and active hypotheses; the status of
    a legacy ImageClaim bookkeeping row must not close or reopen an
    investigation route.
    """

    known_hypothesis_ids = {
        hypothesis.hypothesis_id
        for hypothesis in state.search_hypotheses
        if hypothesis.status in {"open", "active"}
    }
    core_fact_id = state.core_verdict_fact_id
    tasks = [
        task
        for task in state.tasks
        if task.status in {"active", "pending"}
        and task.hypothesis_id in known_hypothesis_ids
        and (core_fact_id is None or core_fact_id in task.fact_ids)
        and (task_ids is None or task.task_id in task_ids)
    ]
    attempted = _attempted_routes_by_task(state)
    routes: List[str] = []
    for task in tasks:
        routes.extend(
            _remaining_task_material_routes(
                state,
                task,
                attempted.get(task.task_id, []),
            )
        )
    return _deduplicate_shared_image_routes(routes)


def archive_recall_available(
    state: ImageOnlyInvestigationState,
    *,
    task_ids: set[str],
) -> bool:
    """Allow bounded optional recall only after a task has produced archive material."""

    attempted = _attempted_routes_by_task(state)
    for task_id in task_ids:
        routes = attempted.get(task_id, [])
        has_investigation_material = any(
            str(route.get("tool", "")).strip()
            not in {"recall_evidence", "read_evidence"}
            for route in routes
        )
        recall_count = sum(
            str(route.get("tool", "")).strip() == "recall_evidence"
            for route in routes
        )
        if (
            has_investigation_material
            and recall_count < MAX_ARCHIVE_RECALL_ROUTES_PER_TASK
        ):
            return True
    return False


def pending_discrepancy_evidence_ids(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return v4 Evidence not yet reviewed by a Discrepancy Decision."""

    reviewed = {
        evidence_id
        for decision in state.discrepancy_decisions
        for evidence_id in decision.reviewed_evidence_ids
    }
    claim_fact_ids = {claim.fact_id for claim in state.image_claims}
    claim_task_ids = {
        task.task_id
        for task in state.tasks
        if task.claim_ids
    }
    return [
        item.evidence_id
        for item in state.evidence
        if item.evidence_id not in reviewed
        and item.task_id in claim_task_ids
        and bool(set(item.fact_ids) & claim_fact_ids)
    ]


def discrepancy_decision_evidence_ids(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return the bounded new-and-prior Evidence context for a v4 checkpoint."""

    claim_fact_ids = {claim.fact_id for claim in state.image_claims}
    claim_task_ids = {
        task.task_id
        for task in state.tasks
        if task.claim_ids
    }
    owned = [
        item.evidence_id
        for item in state.evidence
        if item.task_id in claim_task_ids
        and bool(set(item.fact_ids) & claim_fact_ids)
    ]
    return owned[-40:]


def discrepancy_decision_checkpoint_reason(
    state: ImageOnlyInvestigationState,
    *,
    update: Mapping[str, Any] | None = None,
    before_unresolved: bool = False,
) -> str:
    """Choose sparse v4 checkpoints without deciding Evidence semantics."""

    pending_ids = set(pending_discrepancy_evidence_ids(state))
    if before_unresolved:
        return "before_unresolved"
    created_ids = {
        str(item)
        for item in (update or {}).get("created_evidence_ids", []) or []
        if str(item) in pending_ids
    }
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    if created_ids:
        rows = [evidence_by_id[item] for item in created_ids]
        if any(
            item.directness == "direct"
            and item.quality in {"strong", "moderate"}
            and (
                item.stance in {"support", "refute"}
                or item.evidence_kind == "reference_comparison"
            )
            for item in rows
        ):
            return "qualified_evidence"
        qualified_since_prior = [
            evidence_by_id[item]
            for item in pending_ids
            if evidence_by_id[item].directness == "direct"
            and evidence_by_id[item].quality in {"strong", "moderate"}
        ]
        if len(qualified_since_prior) >= 2:
            return "scheduled_boundary"

    # Two consecutive non-substantive actions are enough to ask whether the
    # current hypothesis should be retired or replaced. This checkpoint does
    # not settle the case or relax any verdict precondition.
    latest_progress = state.progress_events[-1] if state.progress_events else None
    latest_strategy_action = max(
        (
            decision.action_count
            for decision in state.discrepancy_decisions
            if decision.trigger == "strategy_boundary"
        ),
        default=-1,
    )
    if (
        not state.pending_archive_read_ids
        and latest_progress is not None
        and latest_progress.gain
        not in {
            "evidence_gain",
            "decision_gain",
            "visual_understanding_gain",
        }
        and state.no_substantive_gain_streak >= 2
        and (
            latest_strategy_action < 0
            or state.action_count - latest_strategy_action >= 2
        )
    ):
        return "strategy_boundary"
    return ""


def query_replan_candidate_task_ids(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return unresolved core tasks eligible for one evidence-led replan."""

    core_id = state.core_verdict_fact_id
    core = next(
        (fact for fact in state.facts if fact.fact_id == core_id),
        None,
    )
    if core is None or core.status in {"supported", "refuted"}:
        return []

    attempts_by_task = _attempted_routes_by_task(state)
    eligible: List[str] = []
    for task in state.tasks:
        if (
            core_id not in task.fact_ids
            or task.status not in {"active", "pending", "exhausted"}
            or task.query_replan_count >= 1
            or "text_search" not in runtime_task_tool_names(state, task)
        ):
            continue
        attempts = attempts_by_task.get(task.task_id, [])
        text_search_indexes = [
            index
            for index, route in enumerate(attempts)
            if str(route.get("tool", "")).strip() == "text_search"
        ]
        if len(text_search_indexes) < MAX_TEXT_SEARCH_ROUTES_PER_TASK:
            continue
        latest_index = text_search_indexes[-1]
        latest_outcome = str(
            attempts[latest_index].get("outcome", "")
        ).strip()
        if latest_outcome == "discovery" and not any(
            str(route.get("tool", "")).strip()
            in {"visit", "compare_with_reference"}
            for route in attempts[latest_index + 1 :]
        ):
            continue
        eligible.append(task.task_id)
    return eligible


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
    facts = {fact.fact_id: fact for fact in state.facts}
    owns_visual_integrity = any(
        facts.get(fact_id) is not None
        and facts[fact_id].predicate == "visual_integrity"
        for fact_id in task.fact_ids
    )
    if not owns_visual_integrity:
        allowed.difference_update(
            {"check_consistency", "analyze_visual_anomalies"}
        )
    for discovery in state.discoveries:
        if discovery.task_id != task.task_id or discovery.abandoned:
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


def _remaining_task_material_routes(
    state: ImageOnlyInvestigationState,
    task: ResearchTask,
    attempts: Sequence[Mapping[str, Any]],
) -> List[str]:
    allowed = runtime_task_tool_names(state, task)
    if not allowed:
        return []

    text_search_count = 0
    one_shot_tools: set[str] = set()
    available_root_image_reverse_branches = (
        remaining_root_image_reverse_branches(state)
    )
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
    if (
        "text_search" in allowed
        and task.query_replan_count
        and _current_replan_query_is_pending(task, attempts)
    ):
        # A newly accepted semantic replan is the next action, rather than one
        # more optional branch from the stale plan. Once attempted, the normal
        # route inventory resumes.
        return [f"text_search:{task.task_id}"]
    if (
        "reverse_image_search" in allowed
        and available_root_image_reverse_branches
    ):
        routes.append(
            f"reverse_image_search:{ROOT_IMAGE_TARGET}:{task.task_id}"
        )
    if (
        "text_search" in allowed
        and not task.query_replan_count
        and text_search_count < MAX_TEXT_SEARCH_ROUTES_PER_TASK
    ):
        routes.append(f"text_search:{task.task_id}")
    for tool_name in (
        "ocr_with_position",
        "crop_and_inspect",
        "focused_visual_inspection",
        "check_consistency",
        "analyze_visual_anomalies",
    ):
        if tool_name in allowed and tool_name not in one_shot_tools:
            routes.append(f"{tool_name}:{task.task_id}")
    return routes


def _attempted_root_image_reverse_branches(
    state: ImageOnlyInvestigationState,
) -> set[str]:
    """Return root-image reverse branches already used in this investigation."""

    return {
        str(route.get("branch", "lens")).strip().lower() or "lens"
        for route in _iter_attempted_routes(state)
        if str(route.get("tool", "")).strip() == "reverse_image_search"
    }


def remaining_root_image_reverse_branches(
    state: ImageOnlyInvestigationState,
) -> List[str]:
    """Return distinct reverse-search branches that remain for the root image.

    Lens and semantic search are different retrieval routes. A branch is
    consumed only after that exact branch has run; the stage-level tool budget
    still bounds the total number of reverse-image calls.
    """

    attempted = _attempted_root_image_reverse_branches(state)
    return [
        branch
        for branch in REVERSE_IMAGE_SEARCH_BRANCHES
        if branch not in attempted
    ]


def _deduplicate_shared_image_routes(routes: Sequence[str]) -> List[str]:
    """Keep one route per shared image target while retaining task ownership."""

    seen_targets: set[str] = set()
    deduplicated: List[str] = []
    for route in routes:
        parts = route.split(":", 2)
        if parts[0] == "reverse_image_search" and len(parts) == 3:
            target = parts[1]
            if target in seen_targets:
                continue
            seen_targets.add(target)
        if route not in deduplicated:
            deduplicated.append(route)
    return deduplicated


def _current_replan_query_is_pending(
    task: ResearchTask,
    attempts: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether the one accepted replacement query still needs execution."""

    if task.query_replan_count <= 0 or not task.suggested_queries:
        return False
    replacement = task.suggested_queries[0]
    for route in attempts:
        if str(route.get("tool", "")).strip() != "text_search":
            continue
        queries = route.get("queries", []) or []
        if isinstance(queries, str):
            queries = [queries]
        if any(
            routes_semantically_equivalent(
                "text_search",
                {"queries": [replacement]},
                "text_search",
                {"queries": [str(query)]},
            )
            for query in queries
            if str(query).strip()
        ):
            return False
    return True


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
        if discovery.task_id == task.task_id and not discovery.abandoned:
            discovery_batches.setdefault(
                discovery.function_call_id,
                [],
            ).append(discovery)

    pending: List[List[str]] = []
    for discoveries in discovery_batches.values():
        discovery_by_inspection_url = {
            canonicalize_url(
                item.candidate_url
                if tool_name == "visit"
                else item.reference_image_url
            ): item
            for item in discoveries
            if canonicalize_url(
                item.candidate_url
                if tool_name == "visit"
                else item.reference_image_url
            )
        }
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
        )
        reverse_batch = any(
            item.candidate_type in {"reverse_image", "visual_reference"}
            for item in discoveries
        )
        candidate_urls = candidate_urls[
            :(
                MAX_REVERSE_SEARCH_CANDIDATES_PER_BATCH
                if reverse_batch
                else MAX_TEXT_SEARCH_CANDIDATES_PER_BATCH
            )
        ]
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
            if tool_name == "visit":
                # Fetched source text resolves or materially advances sibling
                # pages in the batch.  It does not bind an accompanying
                # reference image to the input pixels; keep reference-image
                # comparisons eligible so source captions cannot substitute for
                # visual alignment.
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
        remaining = [
            url for url in candidate_urls if url not in attempted_outcomes
        ]
        unsuccessful_candidates = [
            discovery_by_inspection_url[url]
            for url in candidate_urls
            if url in attempted_outcomes
            and attempted_outcomes[url] in {"empty", "failed", "context"}
            and url in discovery_by_inspection_url
        ]
        if unsuccessful_candidates:
            remaining = [
                url
                for url in remaining
                if not any(
                    _same_inspection_source_family(
                        discovery_by_inspection_url[url],
                        attempted,
                        tool_name=tool_name,
                    )
                    for attempted in unsuccessful_candidates
                    if url in discovery_by_inspection_url
                )
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


def _same_inspection_source_family(
    left: InvestigationDiscovery,
    right: InvestigationDiscovery,
    *,
    tool_name: str,
) -> bool:
    """Spend a reference fallback on a different source family.

    Reverse-image batches often contain several visually unrelated product
    results served by one CDN. After one failed comparison, another asset from
    that same family is not a distinct recovery route. Page visits retain their
    existing bounded sibling behavior because two pages on one site can contain
    materially different assertions.
    """

    if tool_name != "compare_with_reference":
        return False
    left_resource = (
        left.reference_image_url
    )
    right_resource = (
        right.reference_image_url
    )
    left_identity = classify_source(left_resource)
    right_identity = classify_source(right_resource)
    return bool(
        left_identity.source_family
        and left_identity.source_family == right_identity.source_family
        and left.candidate_type == right.candidate_type
    )
