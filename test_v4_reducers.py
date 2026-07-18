from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.orchestrator.investigation_models import (
    ClaimAssessmentProposal,
    DiscrepancyDecisionOutput,
    FactOrigin,
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
)


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
                verification_question="Is the depicted person-product relationship genuine?",
            )
        ],
        search_hypotheses=[
            SearchHypothesisProposal(
                hypothesis_key="source-photo",
                claim_keys=["person-product"],
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
    return evidence


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


def test_discrepancy_coverage_compiles_unverifiable_gap_after_routes_close() -> None:
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
            verdict_proposal="unverifiable",
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
    assert coverage.complete is True
    assert verdict == "unverifiable"
    assert basis.claim_ids == [claim.claim_id]
    assert basis.unresolved_gaps == [
        "No source binds the depicted held object."
    ]
