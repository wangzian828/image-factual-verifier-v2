from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.orchestrator.investigation_models import (
    ClaimAssessmentProposal,
    DiscrepancyDecisionProposalOutput,
    DiscrepancyDecisionOutput,
    FactOrigin,
    Finding,
    ImageAccountPlanningOutput,
    ImageClaimProposal,
    ImageOnlyInvestigationState,
    InvestigationDiscovery,
    InvestigationBrief,
    InvestigationEvidence,
    MaterialDiscrepancy,
    MaterialDiscrepancyDraft,
    MaterialDiscrepancyProposal,
    NewSearchHypothesis,
    SearchHypothesisProposal,
    VisualEntity,
    VisualEvidenceDisposition,
    VisualDiscriminatorCandidate,
    VisualFact,
    VisualReinspectionProposal,
    VisualReinspectionRequest,
)
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.image_only_prompts import (
    DISCREPANCY_DECISION_SYSTEM_PROMPT,
    render_discrepancy_decision_context,
)
from src.orchestrator.task_store import (
    MAX_ARCHIVE_RECALL_ROUTES_PER_TASK,
    apply_discrepancy_decision,
    apply_image_account_planning,
    archive_recall_available,
    bind_discrepancy_decision_runtime_ids,
    discrepancy_decision_checkpoint_reason,
    discrepancy_visual_reinspection_binding,
    record_route_selection_exhaustion,
    record_tool_observation,
    remaining_claim_hypothesis_routes,
    runtime_task_tool_names,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.progress_control import record_action_progress
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import ImageOnlyRuntimeCase, VerificationState


def _state() -> ImageOnlyInvestigationState:
    entity = VisualEntity(
        entity_id="entity-person",
        name="a presenter",
        entity_type="person",
        region=[0.1, 0.1, 0.8, 0.9],
        confidence=0.98,
    )
    anchor = VisualFact(
        fact_id="fact-visible-person",
        kind="attribute",
        statement="A presenter is visible holding a product packet.",
        subject_entity_id=entity.entity_id,
        predicate="visible_in",
        status="active",
        basis_ids=[entity.entity_id],
        decision_relevance="supporting",
        origin=FactOrigin(type="input_image", origin_ids=[entity.entity_id]),
    )
    return ImageOnlyInvestigationState(
        brief=InvestigationBrief(
            brief_id="brief-v4-reducer",
            case_id="case-v4-reducer",
        ),
        entities=[entity],
        facts=[anchor],
    )


def _planning_output() -> ImageAccountPlanningOutput:
    return ImageAccountPlanningOutput(
        account_summary="The image presents a person-product relationship.",
        image_claims=[
            ImageClaimProposal(
                claim_key="person-product",
                statement="The presenter is holding and promoting the shown product.",
                kind="relation",
                predicate="depicts_relation",
                anchor_fact_ids=["fact-visible-person"],
                salience="high",
            )
        ],
        search_hypotheses=[
            SearchHypothesisProposal(
                hypothesis_key="source-photo",
                route_focus="same_capture_reference",
                statement="A source photograph may show what the presenter held.",
                queries=["presenter source photograph product"],
                expected_information="A traceable source image or report.",
                suggested_tools=["text_search", "compare_with_reference"],
            )
        ],
    )


def test_image_account_requires_exactly_one_high_salience_claim() -> None:
    payload = _planning_output().model_dump(mode="json")
    payload["image_claims"].append(
        {
            "claim_key": "separate-visible-fragment",
            "statement": "A separate visible label appears in the image.",
            "kind": "text_claim",
            "predicate": "reads",
            "anchor_fact_ids": ["fact-visible-person"],
            "salience": "high",
        }
    )

    with pytest.raises(
        ValidationError,
        match="exactly one high-salience central target fact",
    ):
        ImageAccountPlanningOutput.model_validate(payload)


def test_image_account_limits_search_hypotheses_to_three() -> None:
    payload = _planning_output().model_dump(mode="json")
    first = payload["search_hypotheses"][0]
    payload["search_hypotheses"] = [
        {
            **first,
            "hypothesis_key": f"route-{index}",
            "statement": f"Distinct route {index} may clarify the visible fact.",
        }
        for index in range(4)
    ]

    with pytest.raises(ValidationError):
        ImageAccountPlanningOutput.model_validate(payload)


def test_planning_derives_text_search_from_nonempty_queries() -> None:
    payload = _planning_output().model_dump(mode="json")
    payload["search_hypotheses"][0]["suggested_tools"] = [
        "reverse_image_search"
    ]

    output = ImageAccountPlanningOutput.model_validate(payload)
    state = _state()
    update = apply_image_account_planning(state, output)

    assert update["accepted"] is True
    assert state.search_hypotheses[0].queries == [
        "presenter source photograph product"
    ]
    assert state.search_hypotheses[0].suggested_tools == [
        "reverse_image_search",
        "text_search",
    ]
    assert state.tasks[0].suggested_tools == [
        "reverse_image_search",
        "text_search",
    ]


def _planned_state() -> ImageOnlyInvestigationState:
    state = _state()
    update = apply_image_account_planning(state, _planning_output())
    assert update["accepted"] is True
    return state


def test_page_evidence_keeps_paired_reference_image_comparison_pending() -> None:
    state = _planned_state()
    task = state.tasks[0]
    source_url = "https://example.org/source-photo"
    reference_url = "https://images.example.org/reference-photo.jpg"
    state.discoveries.append(
        InvestigationDiscovery(
            discovery_id="discovery-source-with-reference-image",
            task_id=task.task_id,
            fact_ids=list(task.fact_ids),
            function_call_id="call-reverse-with-reference",
            tool_name="reverse_image_search",
            candidate_url=source_url,
            reference_image_url=reference_url,
            title="Source photo caption",
            snippet="A source page with an accompanying reference image.",
            candidate_type="reverse_image",
        )
    )
    state.attempted_routes.append(
        json.dumps(
            {
                "tool": "visit",
                "task_id": task.task_id,
                "urls": [source_url],
                "outcome": "evidence",
            }
        )
    )

    routes = remaining_claim_hypothesis_routes(
        state,
        task_ids={task.task_id},
    )

    assert routes == [
        f"compare_with_reference:{task.task_id}:{reference_url}"
    ]


def test_route_selection_exhaustion_blocks_task_without_counting_an_action() -> None:
    state = _planned_state()
    task = state.tasks[0]
    state.pending_archive_read_ids = ["memory-pending"]
    state.recommended_next_task_ids = [task.task_id]

    update = record_route_selection_exhaustion(
        state,
        task_id=task.task_id,
        request_id="req-route-selection",
    )

    assert state.action_count == 0
    assert update["route_selection_exhausted"] is True
    assert update["task_status"] == "blocked"
    assert task.status == "blocked"
    assert state.search_hypotheses[0].status == "exhausted"
    assert state.failures[0].code == "protocol_error"
    assert state.failures[0].recoverable is False
    assert state.pending_archive_read_ids == []
    assert state.recommended_next_task_ids == []
    assert remaining_claim_hypothesis_routes(state) == []


def test_discrepancy_checkpoint_noop_is_an_accepted_nonterminal_update() -> None:
    state = _planned_state()
    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale=(
                "No atomic Decision update was accepted; continue with the "
                "recorded workspace and unresolved gaps."
            ),
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert update["accepted"] is True
    assert update["verdict_proposal"] == "continue"
    assert state.proposed_verdict == "continue"
    assert state.discrepancy_decisions[-1].trigger == "scheduled_boundary"


def test_discrepancy_new_hypothesis_rejects_fact_check_query() -> None:
    state = _planned_state()
    before = state.model_dump(mode="json")
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            new_hypotheses=[
                NewSearchHypothesis(
                    claim_ids=[claim.claim_id],
                    route_focus="scene_world_constraints",
                    statement=(
                        "Identify the miniature craft shown in the image from "
                        "independent source records."
                    ),
                    queries=["miniature baby sloth tweet debunks fact check"],
                    expected_information=(
                        "An original source or product record identifying the "
                        "depicted miniature."
                    ),
                    suggested_tools=["text_search"],
                )
            ],
            verdict_proposal="continue",
            rationale="Open a source-grounded route for the unresolved claim.",
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert update["accepted"] is False
    assert "fact-check-oriented query" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_qualified_evidence_checkpoint_rejects_empty_continue() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale="Leave the reviewed Evidence untouched.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "empty continue Decision" in update["rejected_reason"]
    assert "reviewed Evidence unconsumed" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_decision_exhaustion_fallback_records_reviewed_evidence_only() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]

    fallback = Orchestrator._decision_correction_exhaustion_fallback(
        state,
        reviewed_evidence_ids=[evidence.evidence_id],
        rejected_reason=(
            "fake verdict requires a valid decisive established discrepancy; "
            "visual reinspection request duplicates an existing route"
        ),
    )

    assert fallback is not None
    update = apply_discrepancy_decision(
        state,
        fallback,
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True
    assert update["verdict_proposal"] == "continue"
    assert state.proposed_verdict == "continue"
    assert state.material_discrepancies == []
    assert state.claim_assessments[-1].claim_id == claim.claim_id
    assert state.claim_assessments[-1].assessment == "insufficient"
    assert state.claim_assessments[-1].evidence_ids == [evidence.evidence_id]
    assert "without inferring support" in state.claim_assessments[-1].rationale


def test_decision_exhaustion_fallback_does_not_spread_evidence_across_task_claims() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    sibling_fact = fact.model_copy(
        update={
            "fact_id": "fact-unrelated-sibling",
            "statement": "An unrelated visible label appears in the image.",
        }
    )
    sibling_claim = claim.model_copy(
        update={
            "claim_id": "claim-unrelated-sibling",
            "fact_id": sibling_fact.fact_id,
            "statement": sibling_fact.statement,
            "salience": "medium",
        }
    )
    state.facts.append(sibling_fact)
    state.image_claims.append(sibling_claim)
    state.tasks[0].fact_ids.append(sibling_fact.fact_id)
    state.tasks[0].claim_ids.append(sibling_claim.claim_id)

    fallback = Orchestrator._decision_correction_exhaustion_fallback(
        state,
        reviewed_evidence_ids=[evidence.evidence_id],
        rejected_reason="Decision correction budget exhausted.",
    )

    assert fallback is not None
    assert [item.claim_id for item in fallback.claim_assessments] == [
        claim.claim_id
    ]


def test_decision_context_and_reducer_reject_task_only_claim_ownership() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    sibling_fact = fact.model_copy(
        update={
            "fact_id": "fact-context-sibling",
            "statement": "A separate visible label appears in the image.",
        }
    )
    sibling_claim = claim.model_copy(
        update={
            "claim_id": "claim-context-sibling",
            "fact_id": sibling_fact.fact_id,
            "statement": sibling_fact.statement,
            "salience": "medium",
        }
    )
    state.facts.append(sibling_fact)
    state.image_claims.append(sibling_claim)
    state.tasks[0].fact_ids.append(sibling_fact.fact_id)
    state.tasks[0].claim_ids.append(sibling_claim.claim_id)
    context = json.loads(
        render_discrepancy_decision_context(
            state,
            reviewed_evidence_ids=[evidence.evidence_id],
            trigger="qualified_evidence",
        )
    )

    assert context["reviewable_claim_ids"] == [claim.claim_id]
    assert context["reviewed_evidence_ownership"][0]["claim_ids"] == [
        claim.claim_id
    ]
    before = state.model_dump(mode="json")
    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=sibling_claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="Task ownership alone must not bind this Evidence.",
                )
            ],
            verdict_proposal="continue",
            rationale="Reject the fact-mismatched Evidence assignment.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "outside its owned tasks" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_protocol_exhaustion_reason_uses_last_validator_feedback() -> None:
    steps = [
        StageStep(
            action_type="output_rejected",
            metadata={"rejection_reason": "first rejection"},
        ),
        StageStep(
            action_type="output_rejected",
            metadata={"rejection_reason": "final validator feedback"},
        ),
        StageStep(
            action_type="output",
            metadata={"protocol_correction_exhaustion_boundary": True},
        ),
    ]

    assert (
        Orchestrator._protocol_exhaustion_rejection_reason(steps)
        == "final validator feedback"
    )


def test_planning_query_can_establish_the_underlying_fact_independently() -> None:
    state = _state()
    output = _planning_output()
    hypothesis = output.search_hypotheses[0]
    hypothesis.statement = (
        "Independent event records may establish what the presenter actually held."
    )
    hypothesis.queries = ["what did the presenter hold during the event"]
    hypothesis.expected_information = (
        "A reliable account of the object actually present at the event."
    )

    update = apply_image_account_planning(state, output)

    assert update["accepted"] is True
    assert state.search_hypotheses[0].queries == [
        "what did the presenter hold during the event"
    ]
    assert "claim_keys" not in SearchHypothesisProposal.model_json_schema()[
        "properties"
    ]
    assert "verification_question" not in ImageClaimProposal.model_json_schema()[
        "properties"
    ]
    assert state.tasks[0].question == hypothesis.statement


def _semantic_safety_fixture() -> dict[str, object]:
    path = (
        Path(__file__).parent
        / "test_fixtures"
        / "v4_semantic_safety_regressions.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    assert isinstance(payload[0], dict)
    return payload[0]


def _append_evidence(state: ImageOnlyInvestigationState) -> InvestigationEvidence:
    claim = state.image_claims[0]
    task = next(task for task in state.tasks if claim.claim_id in task.claim_ids)
    evidence = InvestigationEvidence(
        evidence_id="evidence-source-comparison",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-source-comparison",
        tool_name="compare_with_reference",
        evidence_kind="reference_comparison",
        source_url="https://example.org/reference.jpg",
        source_family="domain:example.org",
        exact_text="The source capture shows the presenter holding a microphone.",
        artifact_sha256="a" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="same_capture",
        same_subject_or_scene=True,
        same_capture_or_near_duplicate=True,
        likely_different_original_capture=False,
        edit_evidence_present=True,
        confidence=0.99,
    )
    state.evidence.append(evidence)
    finding = Finding(
        finding_id="finding-source-comparison",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        statement="The qualified comparison refutes the visible relation.",
        stance="refute",
        evidence_ids=[evidence.evidence_id],
        source_family_ids=[evidence.source_family],
        quality="decisive",
    )
    state.findings.append(finding)
    task.finding_ids.append(finding.finding_id)
    return evidence


def _append_web_refute_evidence(
    state: ImageOnlyInvestigationState,
    *,
    suffix: str,
    source_class: str,
    source_family: str,
) -> InvestigationEvidence:
    claim = state.image_claims[0]
    task = next(task for task in state.tasks if claim.claim_id in task.claim_ids)
    evidence = InvestigationEvidence(
        evidence_id=f"evidence-web-refute-{suffix}",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id=f"call-web-refute-{suffix}",
        tool_name="visit",
        evidence_kind="web_span",
        source_url=f"https://{suffix}.example/source",
        source_family=source_family,
        source_class=source_class,
        exact_text="The source directly contradicts the depicted relationship.",
        span_start=0,
        span_end=60,
        artifact_sha256=("b" if suffix == "one" else "c") * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="moderate",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="contradicts",
    )
    state.evidence.append(evidence)
    finding = Finding(
        finding_id=f"finding-web-refute-{suffix}",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        statement="The reviewed source contradicts the depicted relationship.",
        stance="refute",
        evidence_ids=[evidence.evidence_id],
        source_family_ids=[evidence.source_family],
        quality="decisive",
    )
    state.findings.append(finding)
    task.finding_ids.append(finding.finding_id)
    return evidence


def _microphone_discriminators() -> list[VisualDiscriminatorCandidate]:
    return [
        VisualDiscriminatorCandidate(
            source_phrase=(
                "The source capture shows the presenter holding a microphone."
            ),
            visible_property="presenter holding a microphone",
            why_discriminative=(
                "The held object distinguishes the source relation from a "
                "generic presenter scene."
            ),
            already_in_claim=False,
            expected_if_source_matches=(
                "The presenter visibly grips a microphone."
            ),
        ),
        VisualDiscriminatorCandidate(
            source_phrase=(
                "The source capture shows the presenter holding a microphone."
            ),
            visible_property="microphone visible in the presenter's hands",
            why_discriminative=(
                "The object should be attached to the visible hand relation, "
                "not merely present elsewhere in the frame."
            ),
            already_in_claim=False,
            expected_if_source_matches=(
                "A microphone is visibly in the presenter’s hands."
            ),
        ),
    ]


def _append_source_visual_conflict_pair(
    state: ImageOnlyInvestigationState,
    *,
    visual_answer_status: str = "observed",
    link_visual_to_reinspection: bool = True,
) -> tuple[InvestigationEvidence, InvestigationEvidence]:
    claim = state.image_claims[0]
    task = next(task for task in state.tasks if claim.claim_id in task.claim_ids)
    source = InvestigationEvidence(
        evidence_id="evidence-source-bare-hand",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-source-bare-hand",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/official-caption",
        source_family="domain:example.org",
        exact_text="The source caption says the handshake happened without wearing gloves.",
        span_start=0,
        span_end=68,
        artifact_sha256="b" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="support",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="supports",
    )
    state.evidence.append(source)

    visual_request = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_reinspection=VisualReinspectionRequest(
                reason="relation",
                scope="relation",
                question=(
                    "Is the hand used in the visible handshake bare or covered by "
                    "a white glove?"
                ),
                expected_property="the handshake hand is covered by a white glove",
                anchor_fact_ids=claim.anchor_fact_ids,
                grounding_evidence_ids=[source.evidence_id],
            ),
            verdict_proposal="continue",
            rationale="A source visible-property claim needs focused pixel review.",
        ),
        reviewed_evidence_ids=[source.evidence_id],
        trigger="qualified_evidence",
    )
    assert visual_request["accepted"] is True, visual_request
    record = state.visual_reinspections[-1]
    visual = InvestigationEvidence(
        evidence_id=f"evidence-visual-glove-{visual_answer_status}",
        task_id=record.task_id,
        fact_ids=[claim.fact_id],
        function_call_id=f"call-visual-glove-{visual_answer_status}",
        tool_name="focused_visual_inspection",
        evidence_kind="image_region",
        source_family="visual:focused_visual_inspection",
        exact_text=(
            "Focused visual inspection observes that the handshake hand is "
            "covered by a white glove."
        ),
        image_region=[0.25, 0.25, 0.75, 0.75],
        artifact_sha256="c" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="neutral",
        quality="strong",
        directness="direct",
        claim_binding="pixel_observation",
        visual_question_id=(
            record.visual_question_id
            if link_visual_to_reinspection
            else "visual-question-unrelated"
        ),
        visual_scope="relation",
        visual_answer_status=visual_answer_status,
    )
    state.evidence.append(visual)
    record.status = "resolved"
    record.evidence_ids = [visual.evidence_id] if link_visual_to_reinspection else []
    visual_task = next(item for item in state.tasks if item.task_id == record.task_id)
    visual_task.status = "resolved"
    assert visual_task.parent_task_id == task.task_id
    return source, visual


def test_archive_memory_actions_update_memory_state_without_fact_failure() -> None:
    state = _planned_state()
    task = state.tasks[0]
    recall = StageStep(
        action_type="tool_call",
        tool_name="recall_evidence",
        tool_args={"query": "source object", "__question_id": task.task_id},
        tool_result=json.dumps(
            {
                "status": "success",
                "candidates": [
                    {"memory_id": "memory-a"},
                    {"memory_id": "memory-b"},
                ],
            }
        ),
        metadata={"function_call_id": "call-recall"},
    )

    recall_update = record_tool_observation(
        state,
        recall,
        image_sha256="a" * 64,
    )

    assert recall_update["recalled_candidate_ids"] == ["memory-a", "memory-b"]
    assert recall_update["created_failure_ids"] == []
    assert state.pending_archive_read_ids == ["memory-a", "memory-b"]
    assert state.failures == []

    read = StageStep(
        action_type="tool_call",
        tool_name="read_evidence",
        tool_args={"memory_id": "memory-a", "__question_id": task.task_id},
        tool_result=json.dumps(
            {
                "status": "success",
                "memory_id": "memory-a",
                "content": "exact archived span",
            }
        ),
        metadata={"function_call_id": "call-read"},
    )
    read_update = record_tool_observation(
        state,
        read,
        image_sha256="a" * 64,
    )

    assert read_update["read_memory_ids"] == ["memory-a"]
    assert read_update["created_failure_ids"] == []
    assert state.pending_archive_read_ids == []
    assert state.read_archive_memory_ids == ["memory-a"]
    assert state.failures == []


def test_archive_recall_is_optional_and_bounded_per_task() -> None:
    state = _planned_state()
    task = state.tasks[0]

    assert archive_recall_available(state, task_ids={task.task_id}) is False
    state.attempted_routes.append(
        json.dumps({"tool": "text_search", "task_id": task.task_id})
    )
    assert archive_recall_available(state, task_ids={task.task_id}) is True

    for index in range(MAX_ARCHIVE_RECALL_ROUTES_PER_TASK):
        state.attempted_routes.append(
            json.dumps(
                {
                    "tool": "recall_evidence",
                    "task_id": task.task_id,
                    "args": {"query": f"archive direction {index}"},
                }
            )
        )
    assert archive_recall_available(state, task_ids={task.task_id}) is False


def test_v4_finding_does_not_resolve_claim_route_before_semantic_decision() -> None:
    state = _planned_state()
    task = state.tasks[0]
    hypothesis = state.search_hypotheses[0]
    statement = "The presenter appears at the event before the product segment."
    step = StageStep(
        action_type="tool_call",
        tool_name="visit",
        tool_args={"url": "https://example.org/event", "__question_id": task.task_id},
        tool_result=json.dumps(
            {
                "status": "success",
                "selected_url": "https://example.org/event",
                "url": "https://example.org/event",
                "evidence": statement,
                "summary": statement,
                "relevance": "medium",
                "stance": "refute",
                "relation_scope": "same_relation",
                "relation_stance": "contradicts",
                "directness": "indirect",
                "temporal_alignment": "not_applicable",
                "artifact_sha256": "b" * 64,
                "evidence_span": {"start": 0, "end": len(statement)},
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "injection_flags": [],
                "evidence_eligible": True,
            }
        ),
        metadata={"function_call_id": "call-indirect-v4-finding"},
    )

    update = record_tool_observation(
        state,
        step,
        image_sha256="a" * 64,
    )

    assert update["created_finding_ids"]
    assert state.claim_assessments == []
    assert task.finding_ids == update["created_finding_ids"]
    assert task.status == "active"
    assert hypothesis.status == "active"


def test_web_evidence_keeps_mixed_text_and_relation_claims_reviewable() -> None:
    state = _state()
    output = _planning_output()
    output.image_claims.append(
        ImageClaimProposal(
            claim_key="visible-endorsement-text",
            statement="Visible text says the presenter endorses the shown product.",
            kind="text_claim",
            predicate="states_endorsement",
            anchor_fact_ids=["fact-visible-person"],
            salience="high",
        )
    )
    update = apply_image_account_planning(state, output)
    assert update["accepted"] is True
    task = state.tasks[0]
    statement = "The presenter says she has never endorsed the advertised product."
    step = StageStep(
        action_type="tool_call",
        tool_name="visit",
        tool_args={
            "url": "https://example.org/statement",
            "__question_id": task.task_id,
            "__claim_id": next(
                claim.claim_id
                for claim in state.image_claims
                if next(
                    fact
                    for fact in state.facts
                    if fact.fact_id == claim.fact_id
                ).kind
                == "text_claim"
            ),
        },
        tool_result=json.dumps(
            {
                "status": "success",
                "selected_url": "https://example.org/statement",
                "url": "https://example.org/statement",
                "evidence": statement,
                "relevance": "high",
                "stance": "refute",
                "relation_scope": "same_relation",
                "relation_stance": "contradicts",
                "directness": "direct",
                "temporal_alignment": "not_applicable",
                "artifact_sha256": "c" * 64,
                "evidence_span": {"start": 0, "end": len(statement)},
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "injection_flags": [],
                "evidence_eligible": True,
            }
        ),
        metadata={"function_call_id": "call-mixed-claim-web-evidence"},
    )

    observation = record_tool_observation(
        state,
        step,
        image_sha256="a" * 64,
    )

    owned_claim_fact_ids = {claim.fact_id for claim in state.image_claims}
    evidence = next(
        item
        for item in state.evidence
        if item.evidence_id in observation["created_evidence_ids"]
    )
    finding = next(
        item
        for item in state.findings
        if item.finding_id in observation["created_finding_ids"]
    )
    text_claim_fact_id = next(
        fact_id
        for fact_id in owned_claim_fact_ids
        if next(
            fact for fact in state.facts if fact.fact_id == fact_id
        ).kind
        == "text_claim"
    )
    assert evidence.fact_ids == [text_claim_fact_id]
    assert finding.fact_ids == [text_claim_fact_id]


def test_visual_evidence_is_scoped_to_the_selected_claim() -> None:
    state = _state()
    output = _planning_output()
    output.image_claims.append(
        ImageClaimProposal(
            claim_key="visible-label",
            statement="A visible label appears on the shown product.",
            kind="text_claim",
            predicate="reads",
            anchor_fact_ids=["fact-visible-person"],
            salience="medium",
        )
    )
    assert apply_image_account_planning(state, output)["accepted"] is True
    task = state.tasks[0]
    selected_claim = next(
        claim
        for claim in state.image_claims
        if next(
            fact for fact in state.facts if fact.fact_id == claim.fact_id
        ).kind
        == "text_claim"
    )
    step = StageStep(
        action_type="tool_call",
        tool_name="crop_and_inspect",
        tool_args={
            "__question_id": task.task_id,
            "__claim_id": selected_claim.claim_id,
            "bbox": [0.1, 0.1, 0.8, 0.8],
            "focus_question": "Is the visible label present?",
        },
        tool_result=json.dumps(
            {
                "status": "success",
                "answer": "A visible label is present on the product.",
                "bbox": [0.1, 0.1, 0.8, 0.8],
            }
        ),
        metadata={"function_call_id": "call-claim-scoped-visual-evidence"},
    )

    observation = record_tool_observation(
        state,
        step,
        image_sha256="a" * 64,
    )

    evidence = next(
        item
        for item in state.evidence
        if item.evidence_id in observation["created_evidence_ids"]
    )
    assert evidence.fact_ids == [selected_claim.fact_id]


def _record_same_capture_reference(
    state: ImageOnlyInvestigationState,
) -> tuple[InvestigationEvidence, Finding | None, dict[str, object]]:
    claim = state.image_claims[0]
    task = state.tasks[0]
    step = StageStep(
        action_type="tool_call",
        tool_name="compare_with_reference",
        tool_args={
            "__question_id": task.task_id,
            "__claim_id": claim.claim_id,
            "reference_url": (
                "https://www.noaa.gov/sites/default/files/styles/"
                "landscape_width_1275/public/2026-06/"
                "NOAA-Ship-Henry-B-Bigelow-in-waters-near-Newport-RI-"
                "homeport.jpg"
            ),
        },
        tool_result=json.dumps(
            {
                "status": "success",
                "overall_observation": (
                    "Both images are identical captures showing the NOAA "
                    "vessel HENRY B. BIGELOW on the water against a distant "
                    "shoreline."
                ),
                "resolved_reference_url": (
                    "https://www.noaa.gov/sites/default/files/styles/"
                    "landscape_width_1275/public/2026-06/"
                    "NOAA-Ship-Henry-B-Bigelow-in-waters-near-Newport-RI-"
                    "homeport.jpg"
                ),
                "same_subject_or_scene": True,
                "same_capture_or_near_duplicate": True,
                "likely_different_original_capture": False,
                "edit_evidence_present": False,
                "confidence": 0.98,
            }
        ),
        metadata={"function_call_id": "call-noaa-same-capture-reference"},
    )
    observation = record_tool_observation(
        state,
        step,
        image_sha256="b" * 64,
    )
    evidence = next(
        item
        for item in state.evidence
        if item.evidence_id in observation["created_evidence_ids"]
    )
    finding = next(
        (
            item
            for item in state.findings
            if item.finding_id in observation["created_finding_ids"]
        ),
        None,
    )
    return evidence, finding, observation


@pytest.mark.parametrize(
    "predicate",
    [
        "depicts_vessel",
        "shows_ship",
        "vessel_identity",
        "visible_hull_markings",
    ],
)
def test_same_capture_reference_supports_free_visual_predicate_and_verdict_chain(
    predicate: str,
) -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    fact.statement = (
        "The image depicts the NOAA fisheries research ship Henry B. Bigelow "
        "(hull number R 225) navigating on coastal waters."
    )
    fact.predicate = predicate
    claim.statement = fact.statement

    evidence, finding, _observation = _record_same_capture_reference(state)

    assert evidence.stance == "support"
    assert evidence.claim_binding == "same_capture"
    assert evidence.source_class == "official"
    assert finding is not None
    assert finding.stance == "support"
    assert finding.evidence_ids == [evidence.evidence_id]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale=(
                        "The official same-capture reference directly supports "
                        "the image-grounded vessel identity claim."
                    ),
                )
            ],
            retire_hypothesis_ids=[state.search_hypotheses[0].hypothesis_id],
            verdict_proposal="real",
            rationale="The single high-salience visual claim is supported.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True, update
    coverage = audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.stop_reason == "verdict_determined"
    assert verdict == "real"
    assert basis.evidence_ids == [evidence.evidence_id]
    assert basis.finding_ids == [finding.finding_id]


@pytest.mark.parametrize(
    "predicate",
    [
        "located_at",
        "depicts_event",
        "dated_as",
        "provenance_matches",
    ],
)
def test_same_capture_reference_does_not_support_world_context_predicate(
    predicate: str,
) -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    fact.statement = "The image was captured near Newport, Rhode Island."
    fact.predicate = predicate
    claim.statement = fact.statement

    evidence, finding, observation = _record_same_capture_reference(state)

    assert evidence.stance == "neutral"
    assert evidence.claim_binding == "same_capture"
    assert finding is None
    assert observation["created_finding_ids"] == []


def test_image_account_planning_creates_stable_owned_graph() -> None:
    first = _state()
    second = _state()

    first_update = apply_image_account_planning(first, _planning_output())
    second_update = apply_image_account_planning(second, _planning_output())

    assert first_update["accepted"] is True
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    claim = first.image_claims[0]
    hypothesis = first.search_hypotheses[0]
    task = first.tasks[0]
    assert hypothesis.claim_ids == [claim.claim_id]
    assert task.claim_ids == [claim.claim_id]
    assert task.hypothesis_id == hypothesis.hypothesis_id
    assert claim.task_ids == [task.task_id]
    assert ImageOnlyInvestigationState.model_validate_json(
        first.model_dump_json()
    ) == first


def test_image_account_planning_rejects_unknown_anchor_atomically() -> None:
    state = _state()
    before = state.model_dump(mode="json")
    output = _planning_output().model_copy(deep=True)
    output.image_claims[0].anchor_fact_ids = ["unknown-anchor"]

    update = apply_image_account_planning(state, output)

    assert update["accepted"] is False
    assert "unknown visual anchor" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_search_hypothesis_requires_an_executable_first_hop() -> None:
    payload = _planning_output().model_dump(mode="json")
    payload["search_hypotheses"][0]["queries"] = []
    payload["search_hypotheses"][0]["suggested_tools"] = ["visit"]

    with pytest.raises(ValidationError, match="executable first-hop tool"):
        ImageAccountPlanningOutput.model_validate(payload)


def test_image_account_planning_rejects_media_origin_route_focus_atomically() -> None:
    state = _state()
    output = _planning_output().model_copy(deep=True)
    output.search_hypotheses[0].route_focus = "media_origin"
    before = state.model_dump(mode="json")

    update = apply_image_account_planning(state, output)

    assert update["accepted"] is False
    assert "route_focus=media_origin" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_image_account_planning_requires_one_open_route() -> None:
    payload = _planning_output().model_dump(mode="json")
    payload["search_hypotheses"] = []

    with pytest.raises(ValidationError, match="at least 1 item"):
        ImageAccountPlanningOutput.model_validate(payload)


def test_non_integrity_task_does_not_expose_integrity_only_tools() -> None:
    state = _planned_state()
    task = state.tasks[0]
    task.suggested_tools.extend(
        ["check_consistency", "analyze_visual_anomalies"]
    )

    allowed = runtime_task_tool_names(state, task)

    assert "text_search" in allowed
    assert "check_consistency" not in allowed
    assert "analyze_visual_anomalies" not in allowed


def test_planning_rejects_route_with_no_authorized_first_hop() -> None:
    state = _state()
    output = _planning_output()
    output.search_hypotheses[0].queries = []
    output.search_hypotheses[0].suggested_tools = [
        "analyze_visual_anomalies"
    ]
    before = state.model_dump(mode="json")

    update = apply_image_account_planning(state, output)

    assert update["accepted"] is False
    assert "no executable first-hop tool" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_establishes_fake_atomically() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                {
                    "claim_id": claim.claim_id,
                    "assessment": "refuted",
                    "selected_evidence_ids": [evidence.evidence_id],
                    "rationale": "The same capture contradicts the depicted relation.",
                }
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The product packet replaces the microphone in the source capture.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="decisive",
                rationale="The edit changes the image's high-salience relationship.",
            ),
            verdict_proposal="fake",
            rationale="A decisive visually anchored discrepancy is established.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True
    assert state.proposed_verdict == "fake"
    assert state.image_claims[0].status == "refuted"
    assert state.material_discrepancies[0].evidence_ids == [evidence.evidence_id]
    assert state.discrepancy_decisions[0].accepted_discrepancy_id
    assert ImageOnlyInvestigationState.model_validate_json(
        state.model_dump_json()
    ) == state

    coverage = audit_discrepancy_coverage(
        state,
        decision_checkpoint=True,
    )
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.complete is True
    assert coverage.stop_reason == "verdict_determined"
    assert verdict == "fake"
    assert basis.claim_ids == [claim.claim_id]
    assert basis.discrepancy_ids == [
        state.material_discrepancies[0].discrepancy_id
    ]
    assert basis.evidence_ids == [evidence.evidence_id]
    assert basis.visual_anchor_fact_ids == claim.anchor_fact_ids
    assert state.stop_reason == "verdict_determined"


def test_single_unknown_web_span_cannot_close_high_salience_claim() -> None:
    state = _planned_state()
    evidence = _append_web_refute_evidence(
        state,
        suffix="one",
        source_class="unknown",
        source_family="domain:unknown-one.example",
    )
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The reviewed span contradicts the relation.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The reviewed span contradicts the depicted relation.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The contradiction would be decisive if the source chain were sufficient.",
            ),
            verdict_proposal="fake",
            rationale="Close the case from the reviewed span.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "requires owned qualified refute Evidence" in update["rejected_reason"]
    assert "source support is insufficient" in update["rejected_reason"]
    assert "not a missing Finding or Evidence-ID binding" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_two_independent_unknown_web_spans_can_close_high_salience_claim() -> None:
    state = _planned_state()
    first = _append_web_refute_evidence(
        state,
        suffix="one",
        source_class="unknown",
        source_family="domain:unknown-one.example",
    )
    second = _append_web_refute_evidence(
        state,
        suffix="two",
        source_class="unknown",
        source_family="domain:unknown-two.example",
    )
    claim = state.image_claims[0]
    evidence_ids = [first.evidence_id, second.evidence_id]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=evidence_ids,
                    rationale="Two independent reviewed sources contradict the relation.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="Independent reviewed sources contradict the depicted relation.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=evidence_ids,
                materiality="decisive",
                status="established",
                rationale="The independent source chain is sufficient.",
            ),
            verdict_proposal="fake",
            rationale="Close the case from independent reviewed sources.",
        ),
        reviewed_evidence_ids=evidence_ids,
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True, update
    assert state.proposed_verdict == "fake"
    assert state.material_discrepancies[0].evidence_ids == evidence_ids
    audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert verdict == "fake"
    assert basis.evidence_ids == evidence_ids
    assert basis.finding_ids == [
        "finding-web-refute-one",
        "finding-web-refute-two",
    ]


def test_refuted_high_salience_claim_cannot_be_downgraded_to_supporting() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The qualified evidence fully refutes this claim.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source contradicts the high-salience image claim.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="supporting",
                rationale="The contradiction is recorded but incorrectly downgraded.",
            ),
            verdict_proposal="continue",
            rationale="Continue despite fully refuting a high-salience claim.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "refuted core image-grounded target fact" in update["rejected_reason"]
    assert "verdict_proposal='fake'" in update["rejected_reason"]
    assert "same atomic update" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_established_high_discrepancy_requires_fake_in_same_object() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The qualified evidence refutes the central claim.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source contradicts the central image relation.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The contradiction is decisive and fully grounded.",
            ),
            verdict_proposal="continue",
            rationale="This intentionally omits the required terminal proposal.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "verdict_proposal='fake'" in update["rejected_reason"]
    assert "same complete JSON object" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_structured_duplicate_without_cascade() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    claim.salience = "medium"
    prior = MaterialDiscrepancy(
        discrepancy_id="discrepancy-already-recorded",
        statement="The source contradicts the depicted relation.",
        affected_claim_ids=[claim.claim_id],
        visual_anchor_fact_ids=claim.anchor_fact_ids,
        evidence_ids=[evidence.evidence_id],
        materiality="decisive",
        status="established",
        rationale="This canonical discrepancy was accepted earlier.",
    )
    state.material_discrepancies.append(prior)
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="Different prose for the same recorded contradiction.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="decisive",
                status="established",
                rationale="This should not restate prior canonical state.",
            ),
            verdict_proposal="continue",
            rationale="The high-salience route remains open.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert prior.discrepancy_id in update["rejected_reason"]
    assert "omit material_discrepancy" in update["rejected_reason"]
    assert "requires refuted assessment" not in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


@pytest.mark.parametrize(
    ("assessment", "stance", "expected_direction"),
    [
        ("supported", "refute", "support"),
        ("refuted", "support", "refute"),
        ("conflicted", "refute", "support"),
    ],
)
def test_discrepancy_decision_rejects_assessment_without_required_direction(
    assessment: str,
    stance: str,
    expected_direction: str,
) -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.stance = stance
    state.findings[0].stance = stance
    if stance == "support":
        evidence.edit_evidence_present = False
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment=assessment,
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="Attempt to promote Evidence in the wrong direction.",
                )
            ],
            rationale="This semantic mismatch must fail closed.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert f"qualified {expected_direction} Evidence" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_neutral_different_capture_canary_state() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    fixture = _semantic_safety_fixture()
    frozen_evidence = fixture["evidence"]
    rejected_proposal = fixture["rejected_proposal"]
    assert isinstance(frozen_evidence, dict)
    assert isinstance(rejected_proposal, dict)
    for field, value in frozen_evidence.items():
        setattr(evidence, field, value)
    state.findings.clear()
    state.tasks[0].finding_ids.clear()
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment=str(rejected_proposal["assessment"]),
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale=(
                        "Incorrectly treat a different unedited capture as proof "
                        "of compositing."
                    ),
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The image was digitally composited.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality=str(rejected_proposal["discrepancy_materiality"]),
                status=str(rejected_proposal["discrepancy_status"]),
                rationale="This conclusion is absent from the Evidence.",
            ),
            verdict_proposal=str(rejected_proposal["verdict"]),
            rationale="This mirrors the rejected real-canary semantic upgrade.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert str(fixture["expected_rejection_contains"]) in update[
        "rejected_reason"
    ]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_forged_reference_stance() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.claim_binding = "same_subject"
    evidence.same_capture_or_near_duplicate = False
    evidence.likely_different_original_capture = True
    evidence.edit_evidence_present = True
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="A forced refute label cannot override capture metadata.",
                )
            ],
            rationale="Reject incoherent reference-comparison semantics.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "qualified refute Evidence" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_requires_finding_evidence_chain() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    state.findings.clear()
    state.tasks[0].finding_ids.clear()
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="Evidence without a Finding cannot own a verdict.",
                )
            ],
            rationale="Reject the incomplete provenance chain.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "Finding -> Evidence chain" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_unknown_evidence_without_partial_state() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                {
                    "claim_id": claim.claim_id,
                    "assessment": "refuted",
                    "selected_evidence_ids": ["unknown-evidence"],
                    "rationale": "The claim is contradicted.",
                }
            ],
            verdict_proposal="continue",
            rationale="Continue after recording the assessment.",
        ),
        reviewed_evidence_ids=["unknown-evidence"],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "unknown Evidence" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_reports_independent_contract_errors_together() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.stance = "support"
    evidence.edit_evidence_present = False
    state.findings[0].stance = "support"
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=["unknown-evidence"],
                    rationale="This proposal contains an invalid reference.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source contradicts the central relation.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                materiality="decisive",
                status="established",
                rationale="This proposal also uses Evidence in the wrong direction.",
            ),
            verdict_proposal="continue",
            rationale="The complete contract feedback should be returned at once.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "unknown Evidence" in update["rejected_reason"]
    assert "qualified refute Evidence" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_reports_unknown_claim_and_real_gate_together() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.stance = "support"
    evidence.edit_evidence_present = False
    state.findings[0].stance = "support"
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id="claim-typo",
                    assessment="supported",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The Claim ID is invalid.",
                )
            ],
            verdict_proposal="real",
            rationale="The route is still open and the high Claim is unsupported.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "unknown ImageClaim 'claim-typo'" in update["rejected_reason"]
    assert claim.claim_id in update["rejected_reason"]
    assert "real verdict requires" in update["rejected_reason"]
    assert "choose continue" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_accepts_bounded_visual_reinspection() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                {
                    "claim_id": claim.claim_id,
                    "assessment": "insufficient",
                    "selected_evidence_ids": [evidence.evidence_id],
                    "remaining_gap": "Inspect the visible object boundary.",
                    "rationale": "The comparison motivates a focused pixel check.",
                }
            ],
            visual_reinspection=VisualReinspectionRequest(
                reason="relation",
                scope="relation",
                question="Does the product boundary blend naturally with the hand?",
                expected_property="A coherent hand-object boundary.",
                anchor_fact_ids=claim.anchor_fact_ids,
                grounding_evidence_ids=[evidence.evidence_id],
            ),
            verdict_proposal="continue",
            rationale="Request one focused visual check.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True
    assert len(state.visual_reinspections) == 1
    record = state.visual_reinspections[0]
    task = next(item for item in state.tasks if item.task_id == record.task_id)
    assert task.claim_ids == [claim.claim_id]
    assert task.suggested_tools == ["focused_visual_inspection"]
    assert task.parent_task_id == evidence.task_id
    assert state.discrepancy_decisions[0].accepted_visual_question_id == (
        record.visual_question_id
    )
    assert ImageOnlyInvestigationState.model_validate_json(
        state.model_dump_json()
    ) == state


def test_visual_reinspection_can_target_open_claim_without_assessment() -> None:
    """Neutral Evidence may motivate pixels without a speculative assessment."""

    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_reinspection=VisualReinspectionRequest(
                reason="relation",
                scope="relation",
                question="Does the visible object actually have the claimed relation?",
                expected_property="A directly observable relation in the original pixels.",
                anchor_fact_ids=claim.anchor_fact_ids,
                grounding_evidence_ids=[evidence.evidence_id],
            ),
            verdict_proposal="continue",
            rationale=(
                "The reviewed material is neutral, so keep the open Claim and "
                "inspect the relevant pixels before drawing a directional conclusion."
            ),
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True
    assert len(state.visual_reinspections) == 1
    assert state.discrepancy_decisions[0].accepted_visual_question_id


def test_discrepancy_decision_accepts_source_visual_composite_refute() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id, visual.evidence_id],
                    rationale=(
                        "The source says the visible handshake hand is bare, "
                        "while focused inspection observes a white glove."
                    ),
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement=(
                    "The source-pixel account conflicts on whether the visible "
                    "handshake hand is bare or gloved."
                ),
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id, visual.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The conflict changes a high-salience visible relation.",
            ),
            verdict_proposal="fake",
            rationale="A decisive source-pixel discrepancy is established.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True, update
    assert source.stance == "support"
    assert visual.stance == "neutral"
    assert len(update["created_composite_finding_ids"]) == 1
    composite = next(
        item
        for item in state.findings
        if item.finding_id == update["created_composite_finding_ids"][0]
    )
    assert composite.stance == "refute"
    assert composite.evidence_ids == [source.evidence_id, visual.evidence_id]
    assert "composite:source_visual_discrepancy" in composite.source_family_ids

    coverage = audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.complete is True
    assert verdict == "fake"
    assert basis.finding_ids == [composite.finding_id]
    assert basis.evidence_ids == [source.evidence_id, visual.evidence_id]


def test_source_visual_composite_allows_reinspection_specific_visual_fact() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]
    visual_record = state.visual_reinspections[-1]
    visual_task = next(item for item in state.tasks if item.task_id == visual_record.task_id)
    detail_fact = state.facts[0].model_copy(deep=True)
    detail_fact.fact_id = "fact-focused-visual-detail"
    detail_fact.statement = (
        "Focused visual inspection records the visible handshake-hand covering."
    )
    state.facts.append(detail_fact)
    visual.fact_ids = [detail_fact.fact_id]
    visual_record.fact_id = detail_fact.fact_id
    visual_task.fact_ids = [detail_fact.fact_id]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id, visual.evidence_id],
                    rationale=(
                        "The source and focused pixel observation conflict on the "
                        "visible relation."
                    ),
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement=(
                    "The source-pixel account conflicts on the visible "
                    "handshake-hand property."
                ),
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id, visual.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The pixel observation is linked through the runtime task.",
            ),
            verdict_proposal="fake",
            rationale="A decisive source-pixel discrepancy is established.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True, update
    assert len(update["created_composite_finding_ids"]) == 1
    audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert verdict == "fake"
    assert basis.evidence_ids == [source.evidence_id, visual.evidence_id]


def test_source_visual_composite_accepts_claim_owned_crop_evidence() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(
        state,
        link_visual_to_reinspection=False,
    )
    visual.tool_name = "crop_and_inspect"
    visual.source_family = "visual:crop_and_inspect"
    visual.visual_question_id = None
    visual.visual_answer_status = None
    visual.visual_scope = None
    # Keep the accepted request record because the prior Decision references
    # it, but leave its evidence_ids empty: the reducer must not require a
    # focused-reinspection link for claim-owned pixel Evidence.
    claim = state.image_claims[0]

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id, visual.evidence_id],
                    rationale=(
                        "The source measurement and claim-owned crop observation "
                        "jointly refute the depicted living animal."
                    ),
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement=(
                    "The source measurements conflict with the miniature object "
                    "shown in the claim-owned crop."
                ),
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id, visual.evidence_id],
                materiality="decisive",
                status="established",
                rationale="Both Evidence records are direct and claim-owned.",
            ),
            verdict_proposal="fake",
            rationale="A decisive source-pixel discrepancy is established.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is True, update
    assert len(update["created_composite_finding_ids"]) == 1


def test_source_visual_composite_rejects_ambiguous_visual_result() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(
        state,
        visual_answer_status="ambiguous",
    )
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id, visual.evidence_id],
                    rationale="The focused visual result is ambiguous.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="An ambiguous visual result must not establish a conflict.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id, visual.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The pixel side is not resolved.",
            ),
            verdict_proposal="fake",
            rationale="This must fail closed.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "qualified refute" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_source_visual_composite_requires_visual_evidence_in_discrepancy() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id, visual.evidence_id],
                    rationale="The assessment includes both sides of the conflict.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The discrepancy omits the visual side of the conflict.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The proposed discrepancy is incomplete.",
            ),
            verdict_proposal="fake",
            rationale="This must fail closed.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "qualified refute" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_runtime_binds_visual_proposal_to_atomic_evidence_fact() -> None:
    """A multi-claim task must not broaden a fact-bound visual reinspection."""

    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.stance = "neutral"
    evidence.claim_binding = "source_assertion"
    evidence.relation_scope = "same_relation"
    evidence.relation_stance = "background"
    evidence.same_capture_or_near_duplicate = None
    evidence.edit_evidence_present = None
    state.findings.clear()
    state.tasks[0].finding_ids.clear()
    claim = state.image_claims[0]
    task = next(item for item in state.tasks if claim.claim_id in item.claim_ids)
    claim_fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    medium_fact = claim_fact.model_copy(
        update={
            "fact_id": "fact-visible-hand-covering",
            "statement": "The visible person is wearing gloves.",
            "status": "active",
        }
    )
    medium_claim = claim.model_copy(
        update={
            "claim_id": "claim-visible-hand-covering",
            "fact_id": medium_fact.fact_id,
            "statement": medium_fact.statement,
            "salience": "medium",
            "status": "open",
            "task_ids": [task.task_id],
        }
    )
    state.facts.append(medium_fact)
    state.image_claims.append(medium_claim)
    task.fact_ids.append(medium_fact.fact_id)
    task.claim_ids.append(medium_claim.claim_id)
    state.search_hypotheses[0].claim_ids.append(medium_claim.claim_id)
    medium_evidence = evidence.model_copy(
        update={
            "evidence_id": "evidence-visible-hand-covering",
            "fact_ids": [medium_claim.fact_id],
            "function_call_id": "call-visible-hand-covering",
            "exact_text": (
                "The source says the visible person is without wearing gloves."
            ),
        }
    )
    state.evidence.append(medium_evidence)
    binding = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=[evidence.evidence_id, medium_evidence.evidence_id],
    )
    assert binding["status"] == "ambiguous"

    proposal = DiscrepancyDecisionProposalOutput(
        visual_reinspection=VisualReinspectionProposal(
            claim_id=claim.claim_id,
            reason="relation",
            scope="relation",
            question="Is the visible object on the support or beside it?",
            expected_property="on versus beside",
            candidate_discriminators=_microphone_discriminators(),
        ),
        verdict_proposal="continue",
        rationale="Reinspect the one fact-bound high-salience relation.",
    )
    bound, reason = bind_discrepancy_decision_runtime_ids(
        state,
        proposal,
        reviewed_evidence_ids=[evidence.evidence_id, medium_evidence.evidence_id],
    )

    assert reason == ""
    assert bound is not None
    assert bound.visual_reinspection is not None
    assert bound.visual_reinspection.anchor_fact_ids == claim.anchor_fact_ids
    assert bound.visual_reinspection.grounding_evidence_ids == [
        evidence.evidence_id
    ]
    update = apply_discrepancy_decision(
        state,
        bound,
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert update["accepted"] is True, update
    visual_task = next(
        item
        for item in state.tasks
        if item.task_id == state.visual_reinspections[0].task_id
    )
    assert visual_task.claim_ids == [claim.claim_id]
    proposal_schema = VisualReinspectionProposal.model_json_schema()["properties"]
    assert "anchor_fact_ids" not in proposal_schema
    assert "grounding_evidence_ids" not in proposal_schema
    assert "candidate_discriminators" in proposal_schema
    assert "selected_discriminator_index" in proposal_schema


def test_runtime_binds_material_discrepancy_visual_anchors() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    proposal = DiscrepancyDecisionProposalOutput(
        claim_assessments=[
            ClaimAssessmentProposal(
                claim_id=claim.claim_id,
                assessment="refuted",
                selected_evidence_ids=[evidence.evidence_id],
                rationale="The qualified source comparison refutes the Claim.",
            )
        ],
        material_discrepancy=MaterialDiscrepancyDraft(
            statement="The source comparison contradicts the visible relation.",
            affected_claim_ids=[claim.claim_id],
            evidence_ids=[evidence.evidence_id],
            materiality="decisive",
            status="established",
            rationale="The changed relation is decisive.",
        ),
        verdict_proposal="fake",
        rationale="A decisive high-salience discrepancy is established.",
    )

    bound, reason = bind_discrepancy_decision_runtime_ids(
        state,
        proposal,
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert reason == ""
    assert bound is not None
    assert bound.material_discrepancy is not None
    assert bound.material_discrepancy.visual_anchor_fact_ids == claim.anchor_fact_ids
    update = apply_discrepancy_decision(
        state,
        bound,
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert update["accepted"] is True, update
    draft_schema = MaterialDiscrepancyDraft.model_json_schema()["properties"]
    assert "visual_anchor_fact_ids" not in draft_schema


def test_discrepancy_context_flags_evidence_to_visual_alignment_candidate() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    task = next(task for task in state.tasks if claim.claim_id in task.claim_ids)
    evidence = InvestigationEvidence(
        evidence_id="evidence-without-gloves",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-without-gloves",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/diana",
        source_family="domain:example.org",
        exact_text=(
            "The archived caption says the handshake happened without wearing "
            "gloves."
        ),
        span_start=0,
        span_end=72,
        artifact_sha256="d" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="support",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="supports",
    )
    state.evidence.append(evidence)
    state.findings.append(
        Finding(
            finding_id="finding-without-gloves",
            task_id=task.task_id,
            fact_ids=[claim.fact_id],
            statement=evidence.exact_text,
            stance="support",
            evidence_ids=[evidence.evidence_id],
            source_family_ids=[evidence.source_family],
            quality="supporting",
        )
    )

    context = json.loads(
        render_discrepancy_decision_context(
            state,
            reviewed_evidence_ids=[evidence.evidence_id],
            trigger="qualified_evidence",
        )
    )

    candidates = context["evidence_to_visual_alignment_candidates"]
    assert candidates == [
        {
            "evidence_id": evidence.evidence_id,
            "evidence_text": evidence.exact_text,
            "claim_id": claim.claim_id,
            "claim_statement": claim.statement,
            "claim_status": "open",
            "source_evidence_excerpt": candidates[0][
                "source_evidence_excerpt"
            ],
            "current_image_account": state.image_account_summary,
            "allowed_visual_anchors": [
                {
                    "fact_id": claim.anchor_fact_ids[0],
                    "statement": state.facts[0].statement,
                }
            ],
            "required_review": candidates[0]["required_review"],
        }
    ]
    assert "candidate discriminators" in candidates[0]["required_review"]
    assert "absolute size or weight" in DISCREPANCY_DECISION_SYSTEM_PROMPT
    assert "media origin from style" in DISCREPANCY_DECISION_SYSTEM_PROMPT
    assert context["runtime_visual_reinspection_binding"] == {
        "status": "available",
        "candidates": [
                {
                    "claim_id": claim.claim_id,
                    "claim_fact_id": claim.fact_id,
                    "source_task_id": task.task_id,
                    "anchor_fact_ids": claim.anchor_fact_ids,
                    "grounding_evidence_ids": [evidence.evidence_id],
                    "source_evidence_excerpt": evidence.exact_text,
            }
        ],
        "binding": {
            "claim_id": claim.claim_id,
            "claim_fact_id": claim.fact_id,
            "source_task_id": task.task_id,
            "anchor_fact_ids": claim.anchor_fact_ids,
            "grounding_evidence_ids": [evidence.evidence_id],
            "source_evidence_excerpt": evidence.exact_text,
        },
    }


def test_runtime_binding_rewrites_generic_visual_reinspection_text() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            visual_reinspection=VisualReinspectionProposal(
                claim_id=state.image_claims[0].claim_id,
                reason="text",
                scope="relation",
                question="text",
                expected_property="text",
                candidate_discriminators=_microphone_discriminators(),
            ),
            verdict_proposal="continue",
            rationale="The source-introduced visible property needs a pixel check.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert error == ""
    assert bound is not None
    request = bound.visual_reinspection
    assert request is not None
    assert request.question == "text"
    assert request.expected_property == "presenter holding a microphone"
    assert request.anchor_fact_ids == state.image_claims[0].anchor_fact_ids
    assert request.grounding_evidence_ids == [evidence.evidence_id]


def test_runtime_binding_rewrites_misdirected_integrity_visual_question() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            visual_reinspection=VisualReinspectionProposal(
                claim_id=state.image_claims[0].claim_id,
                reason="integrity",
                scope="integrity",
                question="Does the image show generic AI artifacts?",
                expected_property="generic AI artifact inspection",
                candidate_discriminators=_microphone_discriminators(),
            ),
            verdict_proposal="continue",
            rationale=(
                "The source introduces a visible object property, not an "
                "integrity claim."
            ),
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert bound is None
    assert "unsupported integrity" in error


def test_mixed_visual_decision_is_rejected_instead_of_silently_projected() -> None:
    payload = {
        "claim_assessments": [
            {
                "claim_id": "claim-guess",
                "assessment": "refuted",
                "selected_evidence_ids": ["evidence-guess"],
                "rationale": "This semantic conclusion must not be accepted yet.",
            }
        ],
        "material_discrepancy": {
            "statement": "A speculative discrepancy.",
            "affected_claim_ids": ["claim-guess"],
            "evidence_ids": ["evidence-guess"],
            "rationale": "Pixels have not yet been inspected.",
        },
        "new_hypotheses": [
            {
                "claim_ids": ["claim-guess"],
                "route_focus": "relation_value",
                "statement": "A speculative new route.",
                "queries": ["speculative route"],
                "expected_information": "Speculative information.",
                "suggested_tools": ["text_search"],
            }
        ],
        "visual_reinspection": {
            "claim_id": "claim-guess",
            "reason": "identity",
            "scope": "subject",
            "question": "Does the animal have an orange patch around its mouth?",
            "expected_property": "orange patch around the mouth",
        },
        "verdict_proposal": "real",
        "rationale": "The visual check must be isolated first.",
    }

    with pytest.raises(ValueError, match="visual reinspection proposal"):
        DiscrepancyDecisionProposalOutput.model_validate(payload)


def test_runtime_binding_requires_model_selected_visual_discriminators() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.evidence_kind = "web_span"
    evidence.span_start = 0
    evidence.span_end = len(evidence.exact_text)
    evidence.exact_text = (
        "The source reports that a surgical team works in an operating room."
    )
    evidence.claim_binding = "source_assertion"
    evidence.relation_scope = "same_relation"
    evidence.relation_stance = "supports"

    binding = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=[evidence.evidence_id],
    )
    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            visual_reinspection=VisualReinspectionProposal(
                claim_id=state.image_claims[0].claim_id,
                reason="identity",
                scope="subject",
                question="Does the foreground show a specimen or an instrument?",
                expected_property="foreground object type",
            ),
            rationale="A broad source statement must not authorize this check.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert binding["status"] == "available"
    assert bound is None
    assert "at least two source-grounded" in error


def test_runtime_binding_accepts_recomposed_visual_discriminator_phrase() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.evidence_kind = "web_span"
    evidence.span_start = 0
    evidence.span_end = len(evidence.exact_text)
    evidence.claim_binding = "source_assertion"
    evidence.relation_scope = "same_relation"
    evidence.relation_stance = "supports"
    candidates = _microphone_discriminators()
    candidates[0] = candidates[0].model_copy(
        update={
            "source_phrase": (
                "The source describes the presenter with a hand-held microphone."
            )
        }
    )

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            visual_reinspection=VisualReinspectionProposal(
                claim_id=state.image_claims[0].claim_id,
                reason="relation",
                scope="relation",
                question="Does the source-grounded relation appear in the image?",
                expected_property="presenter holding a microphone",
                candidate_discriminators=candidates,
            ),
            rationale="The selected pixel check must retain source provenance.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert error == ""
    assert bound is not None


def test_visual_reinspection_binding_ignores_neutral_indirect_source_titles() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    task = next(item for item in state.tasks if claim.claim_id in item.claim_ids)
    evidence = InvestigationEvidence(
        evidence_id="evidence-neutral-title",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-neutral-title",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://x.com/example/status/1",
        source_family="domain:x.com",
        exact_text='Derren Brown on X: "Quick sloth baby before bed if you need one."',
        span_start=0,
        span_end=72,
        artifact_sha256="7" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="neutral",
        quality="weak",
        directness="indirect",
        claim_binding="source_assertion",
        relation_scope="unclear",
        relation_stance="unclear",
    )
    state.evidence.append(evidence)

    binding = discrepancy_visual_reinspection_binding(
        state,
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert binding == {
        "status": "unavailable",
        "candidates": [],
        "binding": None,
    }


def test_runtime_binding_rewrites_long_source_text_to_visible_property() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.evidence_kind = "web_span"
    evidence.span_start = 0
    evidence.span_end = 182
    evidence.exact_text = (
        "Facebook metadata and a New York Times excerpt describe Likweli as a "
        "new monkey with an orange patch around its nose and mouth, followed by "
        "unrelated publication metadata and a long source attribution."
    )
    evidence.claim_binding = "source_assertion"
    evidence.relation_scope = "same_relation"
    evidence.relation_stance = "supports"

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            visual_reinspection=VisualReinspectionProposal(
                claim_id=state.image_claims[0].claim_id,
                    reason="identity",
                    scope="subject",
                    question=(
                        "Does the original image visibly show the source-described "
                        "orange patch around the nose and mouth?"
                    ),
                expected_property=(
                    "distribution_of_orange_pigmentation_on_cheeks_vs_nose_bridge"
                ),
                candidate_discriminators=[
                    VisualDiscriminatorCandidate(
                        source_phrase=(
                            "Facebook metadata and a New York Times excerpt "
                            "describe Likweli as a new monkey with an orange "
                            "patch around its nose and mouth"
                        ),
                        visible_property=(
                            "orange patch around the nose and mouth"
                        ),
                        why_discriminative=(
                            "The patch location distinguishes the described "
                            "animal relation from other facial markings."
                        ),
                        already_in_claim=False,
                        expected_if_source_matches=(
                            "An orange patch is visible around the nose and mouth."
                        ),
                    ),
                    VisualDiscriminatorCandidate(
                        source_phrase=(
                            "Facebook metadata and a New York Times excerpt "
                            "describe Likweli as a new monkey with an orange "
                            "patch around its nose and mouth"
                        ),
                        visible_property=(
                            "orange facial marking around the mouth"
                        ),
                        why_discriminative=(
                            "The location of the marking is more informative "
                            "than simply checking whether an animal is present."
                        ),
                        already_in_claim=False,
                        expected_if_source_matches=(
                            "The facial marking is visible around the mouth."
                        ),
                    ),
                ],
            ),
            rationale="Inspect the source-grounded animal property.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
    )

    assert error == ""
    assert bound is not None
    request = bound.visual_reinspection
    assert request is not None
    assert request.expected_property == "orange patch around the nose and mouth"
    assert "cheeks" not in request.expected_property.casefold()
    assert "bridge" not in request.expected_property.casefold()
    assert len(request.expected_property) <= 240
    assert request.reason == "identity"
    assert request.scope == "subject"


def test_decision_must_consume_claim_owned_visual_evidence_or_explain_irrelevance() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]
    assert (
        '"irrelevant_to_current_claim_or_discrepancy"'
        in DISCREPANCY_DECISION_SYSTEM_PROMPT
    )
    assert "If the Evidence is irrelevant" not in DISCREPANCY_DECISION_SYSTEM_PROMPT
    before = state.model_dump(mode="json")
    decision_context = json.loads(
        render_discrepancy_decision_context(
            state,
            reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
            trigger="qualified_evidence",
        )
    )
    requirements = decision_context["claim_owned_visual_evidence_requirements"]
    assert len(requirements) == 1
    assert requirements[0]["claim_ids"] == [claim.claim_id]
    assert requirements[0]["required_action"]["claim_ids"] == [claim.claim_id]
    assert requirements[0]["required_action"]["visual_evidence_ids"] == [
        visual.evidence_id
    ]

    ignored = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[source.evidence_id],
                    rationale="This source-only conclusion improperly ignores pixels.",
                )
            ],
            verdict_proposal="continue",
            rationale="The focused Evidence must be addressed.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert ignored["accepted"] is False
    assert "claim-owned visual Evidence" in ignored["rejected_reason"]
    assert visual.evidence_id in ignored["rejected_reason"]
    assert claim.claim_id in ignored["rejected_reason"]
    assert state.model_dump(mode="json") == before

    same_claim_disposition = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    rationale=(
                        "This still updates the Claim that the focused inspection "
                        "was created to resolve."
                    ),
                )
            ],
            visual_evidence_disposition=VisualEvidenceDisposition(
                disposition="irrelevant_to_current_claim_or_discrepancy",
                evidence_ids=[visual.evidence_id],
                rationale=(
                    "The visual observation was already considered elsewhere, "
                    "but this Decision would still update the same Claim."
                ),
            ),
            verdict_proposal="continue",
            rationale="This disposition must not bypass same-Claim consumption.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert same_claim_disposition["accepted"] is False
    assert "updates the same ImageClaim" in same_claim_disposition["rejected_reason"]
    assert state.model_dump(mode="json") == before

    disposition = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_evidence_disposition=VisualEvidenceDisposition(
                disposition="irrelevant_to_current_claim_or_discrepancy",
                evidence_ids=[visual.evidence_id],
                rationale=(
                    "The focused hand-covering observation does not answer the "
                    "separate scene-location claim currently under review."
                ),
            ),
            verdict_proposal="continue",
            rationale="Record the non-use of the resolved visual observation.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )

    assert disposition["accepted"] is True, disposition
    assert state.discrepancy_decisions[-1].output.visual_evidence_disposition


def test_decision_context_requires_claim_owned_crop_visual_evidence() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(
        state,
        link_visual_to_reinspection=False,
    )
    visual.tool_name = "crop_and_inspect"
    visual.source_family = "visual:crop_and_inspect"
    visual.visual_question_id = None
    visual.visual_answer_status = None
    visual.visual_scope = None
    claim = state.image_claims[0]

    context = json.loads(
        render_discrepancy_decision_context(
            state,
            reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
            trigger="qualified_evidence",
        )
    )
    requirements = context["claim_owned_visual_evidence_requirements"]
    assert [item["evidence_id"] for item in requirements] == [visual.evidence_id]
    assert requirements[0]["tool_name"] == "crop_and_inspect"

    ignored = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[source.evidence_id],
                    rationale="This source-only update must not drop the crop observation.",
                )
            ],
            verdict_proposal="continue",
            rationale="The visual Evidence still requires explicit handling.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )
    assert ignored["accepted"] is False
    assert visual.evidence_id in ignored["rejected_reason"]

    disposed = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_evidence_disposition=VisualEvidenceDisposition(
                disposition="irrelevant_to_current_claim_or_discrepancy",
                evidence_ids=[visual.evidence_id],
                rationale=(
                    "This crop observation concerns a separate visual property "
                    "and is not relevant to the current decision."
                ),
            ),
            verdict_proposal="continue",
            rationale="Record the explicit non-use of the crop Evidence.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )
    assert disposed["accepted"] is True, disposed
    assert state.discrepancy_decisions[-1].output.visual_evidence_disposition


def test_runtime_binding_prefers_selected_visual_evidence_over_conflicting_irrelevance() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[visual.evidence_id],
                    rationale="Consume the resolved focused visual Evidence.",
                )
            ],
            visual_evidence_disposition=VisualEvidenceDisposition(
                disposition="irrelevant_to_current_claim_or_discrepancy",
                rationale="Contradictory model branch that must be removed.",
            ),
            verdict_proposal="continue",
            rationale="The pixel Evidence is the auditable branch.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
    )

    assert error == ""
    assert bound is not None
    assert bound.visual_evidence_disposition is None
    update = apply_discrepancy_decision(
        state,
        bound,
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )
    assert update["accepted"] is True, update
    assert visual.evidence_id in bound.claim_assessments[0].selected_evidence_ids


def test_runtime_binding_ignores_web_evidence_in_visual_disposition() -> None:
    state = _planned_state()
    source, visual = _append_source_visual_conflict_pair(state)
    claim = state.image_claims[0]

    bound, error = bind_discrepancy_decision_runtime_ids(
        state,
        DiscrepancyDecisionProposalOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[visual.evidence_id],
                    rationale="Consume the reviewed pixel observation.",
                )
            ],
            visual_evidence_disposition=VisualEvidenceDisposition(
                disposition="irrelevant_to_current_claim_or_discrepancy",
                evidence_ids=[source.evidence_id],
                rationale=(
                    "The source page is redundant after the exact image match "
                    "and is not itself a pixel observation."
                ),
            ),
            verdict_proposal="continue",
            rationale="The source note must not block the pixel Evidence update.",
        ),
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
    )

    assert error == ""
    assert bound is not None
    assert bound.visual_evidence_disposition is None
    update = apply_discrepancy_decision(
        state,
        bound,
        reviewed_evidence_ids=[source.evidence_id, visual.evidence_id],
        trigger="qualified_evidence",
    )
    assert update["accepted"] is True, update


def test_source_assertion_can_be_decided_without_runtime_selected_reinspection() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    task = next(
        item for item in state.tasks if claim.claim_id in item.claim_ids
    )
    source = InvestigationEvidence(
        evidence_id="evidence-source-visible-property",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-source-visible-property",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/source",
        source_family="domain:example.org",
        source_class="news",
        exact_text=(
            "The source says the presenter is holding a silver microphone."
        ),
        span_start=0,
        span_end=60,
        artifact_sha256="d" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="contradicts",
    )
    state.evidence.append(source)
    state.findings.append(
        Finding(
            finding_id="finding-source-visible-property",
            task_id=task.task_id,
            fact_ids=[claim.fact_id],
            statement="The source contradicts the visible held object.",
            stance="refute",
            evidence_ids=[source.evidence_id],
            source_family_ids=[source.source_family],
            quality="decisive",
        )
    )
    source_only = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id],
                    rationale="The reviewed source directly refutes the relation.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source names a different held object.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The source evidence is sufficient for this relation.",
            ),
            verdict_proposal="fake",
            rationale="Accept the qualified source-based discrepancy.",
        ),
        reviewed_evidence_ids=[source.evidence_id],
        trigger="qualified_evidence",
    )

    assert source_only["accepted"] is True, source_only
    assert state.proposed_verdict == "fake"


def test_failed_optional_visual_does_not_create_a_source_gate() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    task = next(
        item for item in state.tasks if claim.claim_id in item.claim_ids
    )
    source = InvestigationEvidence(
        evidence_id="evidence-source-after-failed-visual",
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id="call-source-after-failed-visual",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/source",
        source_family="domain:example.org",
        source_class="news",
        exact_text=(
            "The source says the presenter is holding a silver microphone."
        ),
        span_start=0,
        span_end=60,
        artifact_sha256="e" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="contradicts",
    )
    state.evidence.append(source)
    state.findings.append(
        Finding(
            finding_id="finding-source-after-failed-visual",
            task_id=task.task_id,
            fact_ids=[claim.fact_id],
            statement="The source refutes the visible held-object relation.",
            stance="refute",
            evidence_ids=[source.evidence_id],
            source_family_ids=[source.source_family],
            quality="decisive",
        )
    )
    request = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_reinspection=VisualReinspectionRequest(
                reason="relation",
                scope="relation",
                question=(
                    "Is the presenter visibly holding a silver microphone?"
                ),
                expected_property="holding a silver microphone",
                anchor_fact_ids=claim.anchor_fact_ids,
                grounding_evidence_ids=[source.evidence_id],
            ),
            verdict_proposal="continue",
            rationale="Request the required pixel check.",
        ),
        reviewed_evidence_ids=[source.evidence_id],
        trigger="qualified_evidence",
    )
    assert request["accepted"] is True, request
    record = state.visual_reinspections[-1]
    record.status = "failed"
    record.evidence_ids = []

    source_only = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source.evidence_id],
                    rationale="The source evidence independently refutes the relation.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source contradicts the visible held-object relation.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[source.evidence_id],
                materiality="decisive",
                status="established",
                rationale="The failed optional visual check does not erase the "
                "qualified source direction.",
            ),
            verdict_proposal="fake",
            rationale="Accept the qualified source discrepancy.",
        ),
        reviewed_evidence_ids=[source.evidence_id],
        trigger="qualified_evidence",
    )

    assert source_only["accepted"] is True, source_only


def test_selected_visual_evidence_remains_scoped_to_the_pending_claim() -> None:
    state = _state()
    planning = ImageAccountPlanningOutput(
        account_summary="The image presents a person and a visible label.",
        image_claims=[
            ImageClaimProposal(
                claim_key="person-product",
                statement="The presenter is holding the shown product.",
                kind="relation",
                predicate="depicts_relation",
                anchor_fact_ids=["fact-visible-person"],
                salience="high",
            ),
            ImageClaimProposal(
                claim_key="visible-label",
                statement="A visible label appears on the product.",
                kind="text_claim",
                predicate="reads",
                anchor_fact_ids=["fact-visible-person"],
                salience="medium",
            ),
        ],
        search_hypotheses=[
            SearchHypothesisProposal(
                hypothesis_key="source-check",
                route_focus="same_capture_reference",
                statement="A source may clarify the visible properties.",
                queries=["source visible properties"],
                expected_information="A direct source.",
                suggested_tools=["text_search"],
            )
        ],
    )
    assert apply_image_account_planning(state, planning)["accepted"] is True
    claim_a, claim_b = state.image_claims
    task = state.tasks[0]
    source_a = InvestigationEvidence(
        evidence_id="evidence-pending-claim-a",
        task_id=task.task_id,
        fact_ids=[claim_a.fact_id],
        function_call_id="call-pending-claim-a",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/a",
        source_family="domain:example.org",
        source_class="news",
        exact_text="The source says the presenter is holding a silver microphone.",
        span_start=0,
        span_end=60,
        artifact_sha256="f" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="contradicts",
    )
    source_b = InvestigationEvidence(
        evidence_id="evidence-resolved-claim-b",
        task_id=task.task_id,
        fact_ids=[claim_b.fact_id],
        function_call_id="call-resolved-claim-b",
        tool_name="visit",
        evidence_kind="web_span",
        source_url="https://example.org/b",
        source_family="domain:example.org",
        source_class="news",
        exact_text="The source says the product has a red label.",
        span_start=0,
        span_end=48,
        artifact_sha256="9" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="support",
        quality="strong",
        directness="direct",
        claim_binding="source_assertion",
        relation_scope="same_relation",
        relation_stance="supports",
    )
    state.evidence.extend([source_a, source_b])
    request = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            visual_reinspection=VisualReinspectionRequest(
                reason="text",
                scope="text",
                question="Is a red label visibly present on the product?",
                expected_property="a red label on the product",
                anchor_fact_ids=claim_b.anchor_fact_ids,
                grounding_evidence_ids=[source_b.evidence_id],
            ),
            verdict_proposal="continue",
            rationale="Inspect Claim B pixels.",
        ),
        reviewed_evidence_ids=[source_b.evidence_id],
        trigger="qualified_evidence",
    )
    assert request["accepted"] is True, request
    record = state.visual_reinspections[-1]
    visual_b = InvestigationEvidence(
        evidence_id="evidence-pixel-claim-b",
        task_id=record.task_id,
        fact_ids=[claim_b.fact_id],
        function_call_id="call-pixel-claim-b",
        tool_name="focused_visual_inspection",
        evidence_kind="image_region",
        source_family="visual:focused_visual_inspection",
        exact_text="The focused inspection observed the label.",
        image_region=[0.1, 0.1, 0.5, 0.5],
        artifact_sha256="8" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="neutral",
        quality="strong",
        directness="direct",
        claim_binding="pixel_observation",
        visual_question_id=record.visual_question_id,
        visual_scope="text",
        visual_answer_status="observed",
    )
    state.evidence.append(visual_b)
    record.status = "resolved"
    record.evidence_ids = [visual_b.evidence_id]
    state.findings.append(
        Finding(
            finding_id="finding-pending-claim-a",
            task_id=task.task_id,
            fact_ids=[claim_a.fact_id],
            statement="The source refutes Claim A.",
            stance="refute",
            evidence_ids=[source_a.evidence_id],
            source_family_ids=[source_a.source_family],
            quality="decisive",
        )
    )

    mixed = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim_a.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[source_a.evidence_id],
                    rationale="The source directly refutes Claim A.",
                ),
                ClaimAssessmentProposal(
                    claim_id=claim_b.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[visual_b.evidence_id],
                    rationale="Consume Claim B's unrelated visual Evidence.",
                ),
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The source contradicts Claim A.",
                affected_claim_ids=[claim_a.claim_id],
                visual_anchor_fact_ids=claim_a.anchor_fact_ids,
                evidence_ids=[source_a.evidence_id],
                materiality="decisive",
                status="established",
                rationale="Claim A is refuted by its owned source evidence.",
            ),
            verdict_proposal="fake",
            rationale="Claim A's source evidence is owned and qualified.",
        ),
        reviewed_evidence_ids=[
            source_a.evidence_id,
            source_b.evidence_id,
            visual_b.evidence_id,
        ],
        trigger="qualified_evidence",
    )

    assert mixed["accepted"] is True, mixed


def test_source_visual_gate_cannot_use_generic_decision_exhaustion_fallback() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)

    fallback = Orchestrator._decision_correction_exhaustion_fallback(
        state,
        reviewed_evidence_ids=[evidence.evidence_id],
        rejected_reason=(
            "Decision must request targeted visual_reinspection before "
            "semantically using source Evidence with a concrete visible property"
        ),
    )

    assert fallback is None


@pytest.mark.parametrize(
    ("tool_payload", "metadata", "expected_code"),
    [
        (
            {
                "status": "error",
                "error": (
                    "ToolActionTimeout: focused_visual_inspection exceeded 0.1s"
                ),
            },
            {
                "tool_success": False,
                "tool_exception": "ToolActionTimeout",
                "tool_timeout_seconds": 0.1,
            },
            "timeout",
        ),
        (
            {
                "status": "error",
                "error": (
                    "GeminiInteractionsHTTPError: request failed with HTTP 429 "
                    "after 3 retries"
                ),
            },
            {
                "tool_success": False,
                "tool_exception": "GeminiInteractionsHTTPError",
            },
            "rate_limited",
        ),
        (
            {
                "status": "error",
                "error": (
                    "GeminiInteractionsHTTPError: request failed with HTTP 503 "
                    "after 3 retries"
                ),
            },
            {
                "tool_success": False,
                "tool_exception": "GeminiInteractionsHTTPError",
            },
            "provider_unavailable",
        ),
        (
            {
                "status": "error",
                "error": (
                    "Focused visual inspection failed: ValueError: "
                    "answer_status is missing or invalid"
                ),
            },
            {"tool_success": False},
            "malformed_result",
        ),
        (
            {
                "status": "success",
                "summary": "",
                "observations": [],
            },
            {"tool_success": True},
            "no_results",
        ),
        (
            {
                "status": "error",
                "error": "provider unavailable during focused inspection",
            },
            {"tool_success": False, "tool_exception": "ProviderUnavailable"},
            "provider_unavailable",
        ),
        (
            {
                "status": "error",
                "error": "focused visual task exhausted its correction budget",
            },
            {
                "tool_success": False,
                "correction_budget_exhausted": True,
            },
            "budget_exhausted",
        ),
        (
            {
                "status": "error",
                "error": "focused inspection returned an explicit failure payload",
            },
            {"tool_success": False},
            "tool_error",
        ),
    ],
)
def test_focused_visual_failure_guard_blocks_source_only_follow_up(
    tmp_path: Path,
    tool_payload: dict[str, object],
    metadata: dict[str, object],
    expected_code: str,
) -> None:
    state = _planned_state()
    source, _ = _append_source_visual_conflict_pair(
        state,
        link_visual_to_reinspection=True,
    )
    initial_evidence_ids = {
        evidence.evidence_id for evidence in state.evidence
    }
    record = state.visual_reinspections[-1]
    record.status = "pending"
    record.evidence_ids = []
    visual_task = next(task for task in state.tasks if task.task_id == record.task_id)
    visual_task.status = "active"
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"focused-visual-failure")
    runtime_case = ImageOnlyRuntimeCase(
        case_id=state.brief.case_id,
        image_path=str(image_path),
        image_sha256="f" * 64,
    )
    verification = VerificationState(
        image_id=runtime_case.case_id,
        image_path=runtime_case.image_path,
        runtime_case=runtime_case,
        input_mode="image_only",
        investigation_state=state,
    )
    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled",
        validate_startup=False,
    )

    async def failing_tool(*_args: object, **_kwargs: object) -> tuple[str, dict[str, object]]:
        return (json.dumps(tool_payload), metadata)

    orchestrator._execute_tool = failing_tool  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match="refusing to continue into a source-only follow-up Decision",
    ):
        asyncio.run(
            orchestrator._run_image_only_visual_reinspection(
                verification,
                state,
                image_path=str(image_path),
                runtime_case=runtime_case,
                visual_question_id=record.visual_question_id,
            )
        )

    assert state.visual_reinspections[-1].status == "failed"
    assert state.visual_reinspections[-1].evidence_ids == []
    assert state.failures[-1].tool_name == "focused_visual_inspection"
    assert state.failures[-1].code == expected_code
    assert state.failures[-1].recoverable is False
    assert not any(
        evidence.evidence_id not in initial_evidence_ids
        and evidence.tool_name == "focused_visual_inspection"
        for evidence in state.evidence
    )
    assert source.evidence_id in state.visual_reinspections[-1].request.grounding_evidence_ids


def test_focused_visual_outer_timeout_uses_real_tool_deadline(
    tmp_path: Path,
) -> None:
    state = _planned_state()
    _append_source_visual_conflict_pair(
        state,
        link_visual_to_reinspection=True,
    )
    record = state.visual_reinspections[-1]
    record.status = "pending"
    record.evidence_ids = []
    visual_task = next(
        task for task in state.tasks if task.task_id == record.task_id
    )
    visual_task.status = "active"
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"focused-visual-timeout")
    runtime_case = ImageOnlyRuntimeCase(
        case_id=state.brief.case_id,
        image_path=str(image_path),
        image_sha256="e" * 64,
    )
    verification = VerificationState(
        image_id=runtime_case.case_id,
        image_path=runtime_case.image_path,
        runtime_case=runtime_case,
        input_mode="image_only",
        investigation_state=state,
    )
    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled",
        validate_startup=False,
    )

    class HangingFocusedTool:
        parameters = {
            "properties": {
                "image_input": {"type": "string"},
            }
        }

        async def call_async(
            self,
            _params: dict[str, object],
        ) -> dict[str, object]:
            await asyncio.sleep(0.2)
            return {"status": "success", "summary": "too late"}

    orchestrator.all_tools["focused_visual_inspection"] = HangingFocusedTool()
    orchestrator.tool_action_timeout_seconds = 0.01

    with pytest.raises(
        RuntimeError,
        match="Tool failure \\[timeout\\]",
    ):
        asyncio.run(
            orchestrator._run_image_only_visual_reinspection(
                verification,
                state,
                image_path=str(image_path),
                runtime_case=runtime_case,
                visual_question_id=record.visual_question_id,
            )
        )

    assert state.visual_reinspections[-1].status == "failed"
    assert state.failures[-1].code == "timeout"
    assert state.failures[-1].recoverable is False
    timeout_step = verification.all_steps[-1]
    assert timeout_step.metadata["tool_exception"] == "ToolActionTimeout"
    assert timeout_step.metadata["tool_timeout_seconds"] == 0.01
    assert timeout_step.metadata["focused_visual_failure"][
        "source_only_follow_up_blocked"
    ] is True


def test_discrepancy_decision_rejects_fake_without_decisive_discrepancy() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                {
                    "claim_id": claim.claim_id,
                    "assessment": "refuted",
                    "selected_evidence_ids": [evidence.evidence_id],
                    "rationale": "The evidence refutes the claim.",
                }
            ],
            verdict_proposal="fake",
            rationale="Fake is proposed without a discrepancy record.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "fake verdict requires" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_unreviewed_evidence_atomically() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    task = next(item for item in state.tasks if item.task_id == evidence.task_id)
    other = evidence.model_copy(
        update={
            "evidence_id": "evidence-unreviewed",
            "function_call_id": "call-unreviewed",
            "task_id": task.task_id,
        }
    )
    state.evidence.append(other)
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id, other.evidence_id],
                    rationale="Both rows are cited, but only one was reviewed.",
                )
            ],
            verdict_proposal="continue",
            rationale="This checkpoint must fail closed.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )

    assert update["accepted"] is False
    assert "reviewed Evidence" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_duplicate_hypothesis_atomically() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    hypothesis = state.search_hypotheses[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            new_hypotheses=[
                NewSearchHypothesis(
                    claim_ids=[claim.claim_id],
                    route_focus="same_capture_reference",
                    statement=hypothesis.statement,
                    queries=["another wording"],
                    expected_information="The same already-open source route.",
                    suggested_tools=["text_search"],
                )
            ],
            verdict_proposal="continue",
            rationale="No semantic route should be duplicated.",
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert update["accepted"] is False
    assert "duplicates an existing route" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_media_origin_new_hypothesis_atomically() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    before = state.model_dump(mode="json")

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            new_hypotheses=[
                NewSearchHypothesis(
                    claim_ids=[claim.claim_id],
                    route_focus="media_origin",
                    statement="Recover the creator and publishing context of the image.",
                    queries=["image creator publishing context"],
                    expected_information="Creator, publisher, and publication history.",
                    suggested_tools=["text_search"],
                )
            ],
            verdict_proposal="continue",
            rationale="This media-origin continuation should be rejected.",
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert update["accepted"] is False
    assert "route_focus=media_origin" in update["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_derives_text_search_for_new_query_route() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    query = "what object did the presenter hold during the event"

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            new_hypotheses=[
                NewSearchHypothesis(
                    claim_ids=[claim.claim_id],
                    route_focus="relation_value",
                    statement=(
                        "Independent event records may identify the object that "
                        "was actually held."
                    ),
                    queries=[query],
                    expected_information=(
                        "A reliable event record naming the held object."
                    ),
                    suggested_tools=["reverse_image_search"],
                )
            ],
            verdict_proposal="continue",
            rationale="A distinct underlying-fact route remains open.",
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert update["accepted"] is True
    hypothesis = next(
        item
        for item in state.search_hypotheses
        if item.hypothesis_id in update["accepted_hypothesis_ids"]
    )
    task = next(item for item in state.tasks if item.task_id == hypothesis.task_id)
    assert hypothesis.queries == [query]
    assert hypothesis.suggested_tools == [
        "reverse_image_search",
        "text_search",
    ]
    assert task.suggested_queries == [query]
    assert task.suggested_tools == hypothesis.suggested_tools


def test_discrepancy_decision_rejects_post_verdict_update_atomically() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    first = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="refuted",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The same capture refutes the depicted relation.",
                )
            ],
            material_discrepancy=MaterialDiscrepancyProposal(
                statement="The visible product replaces the source microphone.",
                affected_claim_ids=[claim.claim_id],
                visual_anchor_fact_ids=claim.anchor_fact_ids,
                evidence_ids=[evidence.evidence_id],
                rationale="The edit changes the high-salience relationship.",
            ),
            verdict_proposal="fake",
            rationale="The decisive discrepancy closes the case.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert first["accepted"] is True
    before = state.model_dump(mode="json")

    second = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale="This must not reopen a terminal state.",
        ),
        reviewed_evidence_ids=[],
        trigger="scheduled_boundary",
    )

    assert second["accepted"] is False
    assert "after verdict" in second["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_decision_rejects_second_visual_reinspection_atomically() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    request = VisualReinspectionRequest(
        reason="relation",
        scope="relation",
        question="Does the object boundary blend naturally with the hand?",
        expected_property="A coherent hand-object boundary.",
        anchor_fact_ids=claim.anchor_fact_ids,
        grounding_evidence_ids=[evidence.evidence_id],
    )
    first = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The comparison motivates pixel review.",
                )
            ],
            visual_reinspection=request,
            rationale="Request the bounded visual check.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert first["accepted"] is True
    before = state.model_dump(mode="json")

    second = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="A second check is not budgeted.",
                )
            ],
            visual_reinspection=request.model_copy(
                update={
                    "question": "Is there a halo around the visible product?",
                    "expected_property": "No compositing halo.",
                }
            ),
            rationale="Attempt another visual check.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="scheduled_boundary",
    )

    assert second["accepted"] is False
    assert "budget exhausted" in second["rejected_reason"]
    assert state.model_dump(mode="json") == before


def test_discrepancy_coverage_compiles_real_only_after_routes_close() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    evidence.stance = "support"
    evidence.exact_text = "The direct source supports the depicted relationship."
    evidence.edit_evidence_present = False
    state.findings[0].stance = "support"
    state.findings[0].statement = "The qualified comparison supports the visible relation."
    claim = state.image_claims[0]
    hypothesis_id = state.search_hypotheses[0].hypothesis_id

    open_update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The direct source supports the claim.",
                )
            ],
            verdict_proposal="continue",
            rationale="The claim is supported, but its route remains open.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert open_update["accepted"] is True
    assert audit_discrepancy_coverage(
        state,
        decision_checkpoint=True,
    ).complete is False

    close_update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The direct source supports the claim.",
                )
            ],
            retire_hypothesis_ids=[hypothesis_id],
            verdict_proposal="real",
            rationale="All high-salience claims are supported and routes close.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="scheduled_boundary",
    )
    assert close_update["accepted"] is True
    coverage = audit_discrepancy_coverage(
        state,
        decision_checkpoint=True,
    )
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.complete is True
    assert verdict == "real"
    assert basis.claim_ids == [claim.claim_id]
    assert basis.evidence_ids == [evidence.evidence_id]


def test_remaining_discrepancy_routes_ignore_legacy_claim_status() -> None:
    state = _planned_state()
    state.image_claims[0].status = "supported"

    assert remaining_claim_hypothesis_routes(state)
    audit = audit_discrepancy_coverage(state, decision_checkpoint=True)
    assert audit.stop_reason == "continue"


def test_discrepancy_coverage_uses_core_fact_not_legacy_claim_status() -> None:
    state = _planned_state()
    claim = state.image_claims[0]
    fact = next(item for item in state.facts if item.fact_id == claim.fact_id)
    fact.status = "supported"
    state.core_verdict_fact_id = fact.fact_id
    state.decisive_fact_ids = [fact.fact_id]
    claim.status = "unresolved"
    state.search_hypotheses[0].status = "retired"
    state.proposed_verdict = "real"

    audit = audit_discrepancy_coverage(state, decision_checkpoint=True)

    assert audit.complete is True
    assert audit.stop_reason == "verdict_determined"


def test_discrepancy_coverage_preserves_gap_for_binary_judgment_after_routes_close() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    hypothesis_id = state.search_hypotheses[0].hypothesis_id

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    remaining_gap="No source binds the depicted held object.",
                    rationale="The comparison is not decisive.",
                )
            ],
            retire_hypothesis_ids=[hypothesis_id],
            verdict_proposal="continue",
            rationale="The high-salience claim remains unresolved after saturation.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="before_unresolved",
    )
    assert update["accepted"] is True
    coverage = audit_discrepancy_coverage(
        state,
        decision_checkpoint=True,
    )
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.complete is False
    assert coverage.stop_reason == "meaningful_routes_exhausted"
    assert verdict == ""
    assert basis.decision_mode == "bounded_binary_judgment"
    assert basis.claim_ids == [claim.claim_id]
    assert basis.unresolved_gaps == [
        "No source binds the depicted held object."
    ]


def test_bounded_basis_does_not_select_evidence_without_a_finding_chain() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    state.findings.clear()
    state.tasks[0].finding_ids.clear()
    claim = state.image_claims[0]
    hypothesis_id = state.search_hypotheses[0].hypothesis_id

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    remaining_gap="The reviewed material has no Finding chain.",
                    rationale=(
                        "Neutral or otherwise non-directional material may remain "
                        "context, but cannot enter the selected verdict basis."
                    ),
                )
            ],
            retire_hypothesis_ids=[hypothesis_id],
            verdict_proposal="continue",
            rationale="Close the exhausted route without inventing a Finding.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="before_unresolved",
    )

    assert update["accepted"] is True
    audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert verdict == ""
    assert basis.decision_mode == "bounded_binary_judgment"
    assert basis.finding_ids == []
    assert basis.evidence_ids == []


def test_bounded_basis_keeps_supported_high_and_unresolved_medium_claims() -> None:
    state = _planned_state()
    high_claim = state.image_claims[0]
    high_fact = next(item for item in state.facts if item.fact_id == high_claim.fact_id)
    medium_fact = high_fact.model_copy(deep=True)
    medium_fact.fact_id = "fact-medium-open"
    medium_fact.statement = "A secondary visible attribute remains unverified."
    state.facts.append(medium_fact)
    medium_claim = high_claim.model_copy(deep=True)
    medium_claim.claim_id = "claim-medium-open"
    medium_claim.fact_id = medium_fact.fact_id
    medium_claim.statement = medium_fact.statement
    medium_claim.salience = "medium"
    state.image_claims.append(medium_claim)
    hypothesis = state.search_hypotheses[0]
    hypothesis.claim_ids.append(medium_claim.claim_id)
    task = state.tasks[0]
    task.claim_ids.append(medium_claim.claim_id)
    task.fact_ids.append(medium_fact.fact_id)
    medium_claim.task_ids.append(task.task_id)

    evidence = _append_evidence(state)
    evidence.stance = "support"
    evidence.edit_evidence_present = False
    state.findings[0].stance = "support"
    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=high_claim.claim_id,
                    assessment="supported",
                    selected_evidence_ids=[evidence.evidence_id],
                    rationale="The high-salience relation is directly supported.",
                )
            ],
            retire_hypothesis_ids=[hypothesis.hypothesis_id],
            verdict_proposal="continue",
            rationale="A secondary Claim remains unresolved at route exhaustion.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="before_unresolved",
    )

    assert update["accepted"] is True
    coverage = audit_discrepancy_coverage(state, decision_checkpoint=True)
    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert coverage.stop_reason == "meaningful_routes_exhausted"
    assert verdict == ""
    assert basis.claim_ids == [high_claim.claim_id, medium_claim.claim_id]
    assert basis.evidence_ids == [evidence.evidence_id]
    assert basis.visual_anchor_fact_ids == high_claim.anchor_fact_ids
    assert basis.unresolved_gaps == [
        f"ImageClaim {medium_claim.claim_id} remains unresolved."
    ]


def test_discrepancy_coverage_does_not_stop_before_pending_archive_read() -> None:
    state = _planned_state()
    evidence = _append_evidence(state)
    claim = state.image_claims[0]
    hypothesis_id = state.search_hypotheses[0].hypothesis_id
    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            claim_assessments=[
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=[evidence.evidence_id],
                    remaining_gap="The archived source needs exact reading.",
                    rationale="The current comparison is not decisive.",
                )
            ],
            retire_hypothesis_ids=[hypothesis_id],
            verdict_proposal="continue",
            rationale="Read the selected archive item before settlement.",
        ),
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="before_unresolved",
    )
    assert update["accepted"] is True
    state.pending_archive_read_ids = ["memory-decisive"]

    audit = audit_discrepancy_coverage(state, decision_checkpoint=True)

    assert audit.stop_reason == "continue"
    assert state.stop_reason == ""


def test_discrepancy_coverage_does_not_settle_from_no_gain_streak() -> None:
    state = _planned_state()
    state.action_count = 4
    state.no_substantive_gain_streak = 4

    audit = audit_discrepancy_coverage(state, decision_checkpoint=True)

    assert audit.complete is False
    assert audit.stop_reason == "continue"
    assert state.stop_reason == ""
    assert state.discrepancy_coverage_audits[-1].action_count == 4


def test_two_no_gain_actions_open_a_strategy_boundary_without_settling() -> None:
    state = _planned_state()
    state.action_count = 1
    record_action_progress(state, {})
    assert discrepancy_decision_checkpoint_reason(state, update={}) == ""

    state.action_count = 2
    record_action_progress(state, {})
    assert (
        discrepancy_decision_checkpoint_reason(state, update={})
        == "strategy_boundary"
    )

    update = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale=(
                "No-gain actions do not settle the Claim; a remaining route "
                "may still be attempted."
            ),
        ),
        reviewed_evidence_ids=[],
        trigger="strategy_boundary",
    )

    assert update["accepted"] is True
    assert state.proposed_verdict == "continue"
    assert state.stop_reason == ""


def test_two_empty_strategy_boundaries_do_not_end_investigation() -> None:
    state = _planned_state()

    for action_count in (1, 2):
        state.action_count = action_count
        record_action_progress(state, {})
    first = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale="The first strategy checkpoint found no substantive update.",
        ),
        reviewed_evidence_ids=[],
        trigger="strategy_boundary",
    )
    assert first["accepted"] is True

    for action_count in (3, 4):
        state.action_count = action_count
        record_action_progress(state, {})
    second = apply_discrepancy_decision(
        state,
        DiscrepancyDecisionOutput(
            verdict_proposal="continue",
            rationale="The second strategy checkpoint found no substantive update.",
        ),
        reviewed_evidence_ids=[],
        trigger="strategy_boundary",
    )
    assert second["accepted"] is True

    audit = audit_discrepancy_coverage(state, decision_checkpoint=True)

    assert audit.complete is False
    assert audit.stop_reason == "continue"
    assert state.stop_reason == ""
