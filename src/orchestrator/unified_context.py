"""Compact model-visible context for the active unified ReAct policy."""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.orchestrator.evidence_semantics import evidence_is_qualified_for_stance
from src.orchestrator.investigation_models import ImageOnlyInvestigationState
from src.orchestrator.task_store import (
    MAX_SEARCH_HYPOTHESES,
    MAX_TOOL_ACTIONS,
    MAX_V4_VISUAL_REINSPECTIONS,
    claim_owned_visual_evidence_requirements,
    discrepancy_visual_reinspection_binding,
    evidence_serves_claim,
    remaining_claim_hypothesis_routes,
)


def _semantic_evidence(item: Any) -> dict[str, Any]:
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
        "likely_different_original_capture": item.likely_different_original_capture,
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


def compile_fact_check_evidence_citations(
    state: ImageOnlyInvestigationState,
    basis: Any,
) -> list[dict[str, Any]]:
    """Compile terminal report citations from the immutable verdict basis."""

    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    return [
        {
            "evidence_id": evidence_id,
            "source_url": item.source_url,
            "source_family": item.source_family,
            "evidence_kind": item.evidence_kind,
            "relation_stance": item.relation_stance,
            "excerpt": item.exact_text[:2400],
        }
        for evidence_id in basis.evidence_ids
        if (item := evidence_by_id.get(evidence_id)) is not None
    ]


def _attempted_routes(state: ImageOnlyInvestigationState) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in state.attempted_routes[-20:]:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def render_unified_react_context(
    state: ImageOnlyInvestigationState,
    *,
    route_local_replan_boundary: tuple[str, str] | None = None,
) -> str:
    """Render only the state needed to choose the next action."""

    completed = list(state.unified_react_bootstrap_tools_completed)
    completed_set = set(completed)
    missing = [
        name
        for name in ("perceive_scene", "ocr_with_position")
        if name not in completed_set
    ]
    if missing:
        payload: dict[str, Any] = {
            "phase": "visual_bootstrap",
            "case_objective": state.brief.objective,
            "workspace": "empty",
            "completed_visual_tools": completed,
            "required_next_visual_tools": missing,
            "instruction": "Select exactly one missing visual bootstrap tool.",
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    visual_facts = [
        {
            "fact_id": item.fact_id,
            "kind": item.kind,
            "statement": item.statement,
            "predicate": item.predicate,
            "origin": item.origin.type,
            "basis_ids": item.basis_ids,
        }
        for item in state.facts
        if item.origin.type in {"input_image", "ocr"}
    ][:36]
    if not state.target_facts:
        payload = {
            "phase": "first_investigation_action",
            "case_objective": state.brief.objective,
            "visual_bootstrap_completed": completed,
            "visual_entities": [
                item.model_dump(mode="json")
                for item in state.entities[:16]
            ],
            "visual_or_ocr_anchor_facts": visual_facts,
            "visible_scene_details": list(state.visual_notable_details[:12]),
            "visual_uncertainties": list(state.visual_uncertainties[:6]),
            "retrieval_anchors": [
                item.model_dump(mode="json")
                for item in state.retrieval_anchors[:32]
            ],
            "first_action_contract": {
                "required": "investigation_intent",
                "target_fact": (
                    "one positive image-grounded world relation using existing "
                    "anchor_fact_ids"
                ),
                "route": (
                    "one primary route for this action, with route_focus, "
                    "expected_information, and priority"
                ),
                "alternate_route_focuses": (
                    "optional distinct route focuses; runtime expands the primary "
                    "route into two or three material routes and runs only primary"
                ),
            },
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    facts = {item.fact_id: item for item in state.facts}
    payload = {
        "phase": "investigation",
        "image_account_summary": state.image_account_summary,
        "visual_entities": [
            item.model_dump(mode="json")
            for item in state.entities[:16]
        ],
        "visible_scene_details": list(state.visual_notable_details[:12]),
        "visual_uncertainties": list(state.visual_uncertainties[:6]),
        "target_facts": [
            item.model_dump(mode="json") for item in state.target_facts
        ],
        "target_fact_details": [
            facts[item.fact_id].model_dump(mode="json")
            for item in state.target_facts
            if item.fact_id in facts
        ],
        "active_routes": [
            {
                **item.model_dump(mode="json"),
                "owned_target_facts": [
                    facts[fact_id].statement
                    for fact_id in item.fact_ids
                    if fact_id in facts
                ],
            }
            for item in state.tasks
            if item.status in {"active", "pending"}
        ],
        "recent_discoveries": [
            item.model_dump(mode="json")
            for item in state.discoveries[-12:]
            if not item.abandoned
        ],
        "recent_evidence": [
            _semantic_evidence(item) for item in state.evidence[-12:]
        ],
        "recent_failures": [
            item.model_dump(mode="json") for item in state.failures[-8:]
        ],
        "open_gaps": [
            item.model_dump(mode="json")
            for item in state.evidence_gaps
            if item.status == "open"
        ],
        "remaining_routes": remaining_claim_hypothesis_routes(state)[:16],
        "attempted_routes": _attempted_routes(state),
        "action_budget": {
            "used": state.action_count,
            "remaining": max(0, MAX_TOOL_ACTIONS - state.action_count),
        },
    }
    if route_local_replan_boundary is not None:
        task_id, trigger = route_local_replan_boundary
        payload["route_local_replan_boundary"] = {
            "task_id": task_id,
            "trigger": trigger,
            "instruction": (
                "The runtime exposed route_local_replan as the next ReAct "
                "control action. Decide whether this route should continue, "
                "use one concrete new query, add one concrete visual route, "
                "or stop only this route. Do not change the target merely "
                "because new context appeared."
            ),
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_unified_reflection_context(
    state: ImageOnlyInvestigationState,
) -> str:
    latest_decision = (
        state.discrepancy_decisions[-1].model_dump(mode="json")
        if state.discrepancy_decisions
        else None
    )
    payload = {
        "action_count": state.action_count,
        "target_facts": [
            item.model_dump(mode="json") for item in state.target_facts
        ],
        "active_task_count": sum(
            item.status in {"active", "pending"} for item in state.tasks
        ),
        "remaining_route_count": len(remaining_claim_hypothesis_routes(state)),
        "open_gaps": [
            item.model_dump(mode="json")
            for item in state.evidence_gaps
            if item.status == "open"
        ],
        "recent_evidence": [
            _semantic_evidence(item) for item in state.evidence[-8:]
        ],
        "recent_failures": [
            item.model_dump(mode="json") for item in state.failures[-8:]
        ],
        "recent_decision": latest_decision,
        "no_substantive_gain_streak": state.no_substantive_gain_streak,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_unified_discrepancy_decision_context(
    state: ImageOnlyInvestigationState,
    *,
    reviewed_evidence_ids: Iterable[str],
    trigger: str,
    allow_new_hypotheses: bool = False,
) -> str:
    """Render the bounded semantic-decision handoff without legacy field names."""

    reviewed = list(dict.fromkeys(str(item) for item in reviewed_evidence_ids))
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    task_by_id = {item.task_id: item for item in state.tasks}
    claim_by_id = {item.claim_id: item for item in state.target_facts}
    reviewed_ownership: list[dict[str, Any]] = []
    reviewable_claim_ids: list[str] = []
    for evidence_id in reviewed:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        task = task_by_id.get(evidence.task_id)
        claim_ids = []
        if task is not None:
            claim_ids = [
                claim_id
                for claim_id in task.claim_ids
                if claim_id in claim_by_id
                and evidence_serves_claim(
                    state,
                    evidence,
                    claim_id=claim_id,
                    claim_fact_id=claim_by_id[claim_id].fact_id,
                    task_by_id=task_by_id,
                )
            ]
        reviewable_claim_ids.extend(claim_ids)
        reviewed_ownership.append(
            {
                "evidence_id": evidence_id,
                "task_id": evidence.task_id,
                "claim_ids": claim_ids,
                "hypothesis_id": task.hypothesis_id if task else None,
            }
        )

    visual_binding = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=reviewed,
    )
    visual_requirements = claim_owned_visual_evidence_requirements(
        state,
        reviewed_evidence_ids=reviewed,
        evidence_by_id=evidence_by_id,
    )
    reviewed_chains = []
    for finding in state.findings:
        task = task_by_id.get(finding.task_id)
        if task is None or finding.stance not in {"support", "refute"}:
            continue
        ids = [
            evidence_id
            for evidence_id in finding.evidence_ids
            if evidence_id in reviewed
            and evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].task_id == finding.task_id
            and evidence_is_qualified_for_stance(
                evidence_by_id[evidence_id], finding.stance
            )
        ]
        if ids:
            reviewed_chains.append(
                {
                    "finding_id": finding.finding_id,
                    "task_id": finding.task_id,
                    "stance": finding.stance,
                    "evidence_ids": ids,
                    "target_fact_claim_ids": list(task.claim_ids),
                }
            )

    payload = {
        "checkpoint": "unified_discrepancy_decision",
        "trigger": trigger,
        "runtime_id_registry": {
            "target_fact_claim_ids": sorted(claim_by_id),
            "reviewed_evidence_ids": [
                evidence_id for evidence_id in reviewed if evidence_id in evidence_by_id
            ],
            "visual_fact_ids": [item.fact_id for item in state.facts],
            "search_hypothesis_ids": [
                item.hypothesis_id for item in state.search_hypotheses
            ],
        },
        "target_facts": [
            item.model_dump(mode="json") for item in state.target_facts
        ],
        "reviewed_evidence": [
            {
                **evidence_by_id[evidence_id].model_dump(mode="json"),
                "admissible_stances": [
                    stance
                    for stance in ("support", "refute")
                    if evidence_is_qualified_for_stance(
                        evidence_by_id[evidence_id], stance
                    )
                ],
            }
            for evidence_id in reviewed
            if evidence_id in evidence_by_id
        ],
        "reviewed_evidence_ownership": reviewed_ownership,
        "reviewable_target_fact_claim_ids": list(dict.fromkeys(reviewable_claim_ids)),
        "reviewed_directional_chains": reviewed_chains,
        "claim_owned_visual_evidence_requirements": visual_requirements[:12],
        "runtime_visual_reinspection_binding": visual_binding,
        "prior_claim_assessments": [
            item.model_dump(mode="json") for item in state.claim_assessments[-12:]
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
        "remaining_routes": remaining_claim_hypothesis_routes(state)[:16],
        "action_count": state.action_count,
        "remaining_action_budget": max(0, MAX_TOOL_ACTIONS - state.action_count),
        "remaining_hypothesis_budget": max(
            0, MAX_SEARCH_HYPOTHESES - len(state.search_hypotheses)
        ),
        "remaining_visual_reinspection_budget": max(
            0, MAX_V4_VISUAL_REINSPECTIONS - len(state.visual_reinspections)
        ),
        "allow_new_hypotheses": bool(allow_new_hypotheses),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_unified_judgment_context(
    state: ImageOnlyInvestigationState,
    compiled_verdict: str,
    basis: Any,
    *,
    final_visual_audit: Any = None,
    image_is_attached: bool = False,
) -> str:
    """Render only the compiled decision material for the terminal call."""

    claims = {item.claim_id: item for item in state.target_facts}
    discrepancies = {
        item.discrepancy_id: item for item in state.material_discrepancies
    }
    findings = {item.finding_id: item for item in state.findings}
    evidence = {item.evidence_id: item for item in state.evidence}
    facts = {item.fact_id: item for item in state.facts}
    if isinstance(final_visual_audit, dict):
        visual_policy = (
            "Use only the structured visual audit below for pixel observations."
        )
    elif image_is_attached:
        visual_policy = "The original image is attached to this judgment request."
    else:
        visual_policy = (
            "The original image is not attached. Use only recorded visual anchors "
            "and visual-tool observations."
        )
    payload: dict[str, Any] = {
        "compiled_verdict": compiled_verdict,
        "final_visual_audit": final_visual_audit,
        "visual_input_policy": visual_policy,
        "terminal_visual_rationale_required": not bool(compiled_verdict),
        "terminal_visual_rationale_contract": (
            {
                "target_visible_property": "The target entity, relation, value, or observable condition.",
                "observed_property": "The recorded visual observation establishing that value or a competing value.",
                "counterfactual_difference": "The target-specific visible correspondence or contradiction.",
                "relation_to_verdict": ["supports_real", "supports_fake"],
            }
            if not compiled_verdict
            else None
        ),
        "compiled_basis": basis.model_dump(mode="json"),
        "fact_check_report_contract": {
            "purpose": (
                "Write a concise reader-facing fact-check report using only "
                "the compiled material below."
            ),
            "required_sections": [
                "headline",
                "claim_under_review",
                "verdict_summary",
                "key_findings",
                "evidence_summary",
                "remaining_uncertainties",
            ],
            "citation_policy": (
                "Do not invent sources, URLs, Evidence IDs, observations, or "
                "facts. The runtime attaches the authoritative citation "
                "inventory from selected Evidence IDs."
            ),
        },
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
            _semantic_evidence(evidence[item])
            for item in basis.evidence_ids
            if item in evidence
        ],
        "unresolved_diagnostic_findings": [
            findings[item].model_dump(mode="json")
            for item in basis.diagnostic_finding_ids
            if item in findings and item not in basis.finding_ids
        ],
        "unresolved_diagnostic_evidence": [
            {
                "evidence_id": evidence[item].evidence_id,
                "tool_name": evidence[item].tool_name,
                "evidence_kind": evidence[item].evidence_kind,
                "exact_text": evidence[item].exact_text[:1000],
                "claim_binding": evidence[item].claim_binding,
                "relation_scope": evidence[item].relation_scope,
                "relation_stance": evidence[item].relation_stance,
                "visual_scope": evidence[item].visual_scope,
                "visual_answer_status": evidence[item].visual_answer_status,
            }
            for item in basis.diagnostic_evidence_ids
            if item in evidence and item not in basis.evidence_ids
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
