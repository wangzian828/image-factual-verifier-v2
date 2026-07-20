from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.orchestrator.investigation_models import (
    ClaimAssessmentProposal,
    DiscrepancyDecisionOutput,
    FactOrigin,
    Finding,
    ImageAccountPlanningOutput,
    ImageClaimProposal,
    ImageOnlyInvestigationState,
    InvestigationBrief,
    InvestigationEvidence,
    MaterialDiscrepancyProposal,
    NewSearchHypothesis,
    SearchHypothesisProposal,
    VisualEntity,
    VisualFact,
    VisualReinspectionRequest,
)
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.task_store import (
    apply_discrepancy_decision,
    apply_image_account_planning,
    record_tool_observation,
    runtime_task_tool_names,
)
from src.orchestrator.stage_runner import StageStep


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
                statement="A source photograph may show what the presenter held.",
                queries=["presenter source photograph product"],
                expected_information="A traceable source image or report.",
                suggested_tools=["text_search", "compare_with_reference"],
            )
        ],
    )


def _planned_state() -> ImageOnlyInvestigationState:
    state = _state()
    update = apply_image_account_planning(state, _planning_output())
    assert update["accepted"] is True
    return state


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
    payload["search_hypotheses"][0]["suggested_tools"] = ["visit"]

    with pytest.raises(ValidationError, match="executable first-hop tool"):
        ImageAccountPlanningOutput.model_validate(payload)


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
    assert state.discrepancy_decisions[0].accepted_visual_question_id == (
        record.visual_question_id
    )
    assert ImageOnlyInvestigationState.model_validate_json(
        state.model_dump_json()
    ) == state


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
