from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.audit_real_trace import audit_trace
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.investigation_models import (
    DiscrepancyDecisionOutput,
    DiscrepancyJudgment,
    FactOrigin,
    Finding,
    ImageAccountPlanningOutput,
    ImageClaimProposal,
    ImageOnlyInvestigationState,
    InvestigationBrief,
    InvestigationEvidence,
    MaterialDiscrepancyProposal,
    SearchHypothesisProposal,
    VisualEntity,
    VisualFact,
)
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import VerificationState
from src.orchestrator.task_store import (
    apply_discrepancy_decision,
    apply_image_account_planning,
    stable_id,
)


FIXTURE_PATH = (
    Path(__file__).parent / "test_fixtures" / "v4_historical_replay.json"
)
def _fixtures() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def _planning_output(fixture: dict[str, object]) -> ImageAccountPlanningOutput:
    return ImageAccountPlanningOutput(
        account_summary=str(fixture["claim"]),
        image_claims=[
            ImageClaimProposal(
                claim_key="primary-visible-relation",
                statement=str(fixture["claim"]),
                kind="relation",
                predicate="depicts_relation",
                anchor_fact_ids=["fact-visible-anchor"],
                salience="high",
                verification_question=(
                    "Does provenance-complete Evidence support the visible relation?"
                ),
            )
        ],
        search_hypotheses=[
            SearchHypothesisProposal(
                hypothesis_key="historical-evidence-route",
                claim_keys=["primary-visible-relation"],
                statement=str(fixture["hypothesis"]),
                queries=[],
                expected_information=str(fixture["expected_information"]),
                suggested_tools=["text_search"],
            )
        ],
    )


def _base_state(fixture: dict[str, object]) -> ImageOnlyInvestigationState:
    anchor = fixture["anchor"]
    assert isinstance(anchor, dict)
    entity = VisualEntity(
        entity_id="entity-visible-anchor",
        name=str(anchor["entity_name"]),
        entity_type=str(anchor["entity_type"]),
        region=[0.05, 0.05, 0.95, 0.95],
        confidence=0.98,
    )
    fact = VisualFact(
        fact_id="fact-visible-anchor",
        kind="relation",
        statement=str(anchor["statement"]),
        subject_entity_id=entity.entity_id,
        predicate="visible_in",
        status="active",
        basis_ids=[entity.entity_id],
        decision_relevance="supporting",
        origin=FactOrigin(type="input_image", origin_ids=[entity.entity_id]),
    )
    return ImageOnlyInvestigationState(
        brief=InvestigationBrief(
            brief_id=stable_id("brief", fixture["case_id"], "v4-replay"),
            case_id=str(fixture["case_id"]),
        ),
        entities=[entity],
        facts=[fact],
    )


def _append_frozen_evidence(
    state: ImageOnlyInvestigationState,
    fixture: dict[str, object],
) -> InvestigationEvidence:
    claim = state.image_claims[0]
    task = state.tasks[0]
    exact_text = str(fixture["exact_text"])
    evidence_kind = str(fixture["evidence_kind"])
    call_id = stable_id("call", fixture["case_id"], "frozen-evidence")
    evidence = InvestigationEvidence(
        evidence_id=stable_id(
            "evidence", fixture["case_id"], fixture["artifact_sha256"]
        ),
        task_id=task.task_id,
        fact_ids=[claim.fact_id],
        function_call_id=call_id,
        tool_name=str(fixture["tool_name"]),
        evidence_kind=evidence_kind,
        source_url=str(fixture["source_url"]),
        source_family=str(fixture["source_family"]),
        source_class=str(fixture["source_class"]),
        exact_text=exact_text,
        span_start=0 if evidence_kind == "web_span" else None,
        span_end=len(exact_text) if evidence_kind == "web_span" else None,
        artifact_sha256=str(fixture["artifact_sha256"]),
        retrieved_at=str(fixture["retrieved_at"]),
        stance=str(fixture["stance"]),
        quality="strong",
        directness="direct",
        claim_binding=str(fixture["claim_binding"]),
        same_subject_or_scene=fixture.get("same_subject_or_scene"),
        same_capture_or_near_duplicate=fixture.get(
            "same_capture_or_near_duplicate"
        ),
        likely_different_original_capture=fixture.get(
            "likely_different_original_capture"
        ),
        edit_evidence_present=fixture.get("edit_evidence_present"),
        confidence=0.99,
    )
    state.evidence.append(evidence)
    state.findings.append(
        Finding(
            finding_id=stable_id("finding", fixture["case_id"], evidence.evidence_id),
            task_id=task.task_id,
            fact_ids=[claim.fact_id],
            statement=str(fixture["assessment_rationale"]),
            stance=str(fixture["stance"]),
            evidence_ids=[evidence.evidence_id],
            source_family_ids=[evidence.source_family],
            quality="decisive",
        )
    )
    task.finding_ids.append(state.findings[-1].finding_id)
    state.action_count = 1
    return evidence


def _decision_output(
    state: ImageOnlyInvestigationState,
    fixture: dict[str, object],
    evidence: InvestigationEvidence,
) -> DiscrepancyDecisionOutput:
    claim = state.image_claims[0]
    verdict = str(fixture["expected_verdict"])
    assessment = "supported" if verdict == "real" else "refuted"
    hypothesis = state.search_hypotheses[0]
    discrepancy = None
    if verdict == "fake":
        discrepancy = MaterialDiscrepancyProposal(
            statement=str(fixture["discrepancy"]),
            affected_claim_ids=[claim.claim_id],
            visual_anchor_fact_ids=claim.anchor_fact_ids,
            evidence_ids=[evidence.evidence_id],
            materiality="decisive",
            status="established",
            rationale=str(fixture["assessment_rationale"]),
        )
    return DiscrepancyDecisionOutput(
        claim_assessments=[
            {
                "claim_id": claim.claim_id,
                "assessment": assessment,
                "selected_evidence_ids": [evidence.evidence_id],
                "rationale": str(fixture["assessment_rationale"]),
            }
        ],
        material_discrepancy=discrepancy,
        retire_hypothesis_ids=(
            [hypothesis.hypothesis_id] if verdict == "real" else []
        ),
        verdict_proposal=verdict,
        rationale=str(fixture["assessment_rationale"]),
    )


def _judgment(
    verdict: str,
    basis: object,
    fixture: dict[str, object],
) -> DiscrepancyJudgment:
    return DiscrepancyJudgment(
        verdict=verdict,
        confidence=0.99,
        selected_claim_ids=list(basis.claim_ids),
        selected_discrepancy_ids=list(basis.discrepancy_ids),
        selected_visual_anchor_fact_ids=list(basis.visual_anchor_fact_ids),
        selected_finding_ids=list(basis.finding_ids),
        selected_evidence_ids=list(basis.evidence_ids),
        overall_assessment=str(fixture["assessment_rationale"]),
        unresolved_gaps=list(basis.unresolved_gaps),
    )


def _steps(
    state: ImageOnlyInvestigationState,
    evidence: InvestigationEvidence,
) -> list[StageStep]:
    return [
        StageStep(
            round=1,
            stage_name="image_account_planning",
            action_type="output",
            metadata={
                "native_interactions": True,
                "interaction_id": "replay-planning",
                "previous_interaction_id": None,
            },
        ),
        StageStep(
            round=1,
            stage_name="image_only_discrepancy_investigation",
            action_type="tool_call",
            tool_name=evidence.tool_name,
            tool_result=json.dumps(
                {"status": "success", "artifact_sha256": evidence.artifact_sha256}
            ),
            metadata={
                "native_interactions": True,
                "interaction_id": "replay-evidence",
                "previous_interaction_id": "replay-planning",
                "function_call_id": evidence.function_call_id,
                "tool_success": True,
            },
        ),
        StageStep(
            round=1,
            stage_name="image_only_discrepancy_decision",
            action_type="output",
            metadata={
                "native_interactions": True,
                "interaction_id": "replay-decision",
                "previous_interaction_id": "replay-evidence",
            },
        ),
        StageStep(
            round=1,
            stage_name="image_only_discrepancy_judgment",
            action_type="output",
            metadata={
                "native_interactions": True,
                "interaction_id": "replay-judgment",
                "previous_interaction_id": "replay-decision",
            },
        ),
    ]


def _write_trace(
    tmp_path: Path,
    fixture: dict[str, object],
    state: ImageOnlyInvestigationState,
    verdict: str,
    judgment: DiscrepancyJudgment,
    steps: list[StageStep],
) -> Path:
    verification = VerificationState(
        image_id=str(fixture["case_id"]),
        input_mode="image_only",
        decision_policy_version="discrepancy-first-v4",
        investigation_state=state,
        judgment=judgment,
        all_steps=steps,
        termination="success",
    )
    basis = state.discrepancy_verdict_basis
    assert basis is not None
    trace = {
        "image_id": str(fixture["case_id"]),
        "input_mode": "image_only",
        "decision_policy_version": "discrepancy-first-v4",
        "verdict": verdict,
        "verdict_basis": basis.model_dump(mode="json"),
        "judgment": judgment.model_dump(mode="json"),
        "termination": "success",
        "token_usage": {"prompt": 0, "completion": 0, "thought": 0},
        "state": verification.to_dict(),
    }
    path = tmp_path / f"{fixture['case_id']}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda row: row["name"])
def test_v4_historical_trace_replay(
    tmp_path: Path,
    fixture: dict[str, object],
) -> None:
    state = _base_state(fixture)
    planning = apply_image_account_planning(state, _planning_output(fixture))
    assert planning["accepted"] is True

    claim = state.image_claims[0]
    hypothesis = state.search_hypotheses[0]
    task = state.tasks[0]
    assert hypothesis.claim_ids == [claim.claim_id]
    assert task.claim_ids == [claim.claim_id]
    assert task.hypothesis_id == hypothesis.hypothesis_id
    assert claim.task_ids == [task.task_id]

    evidence = _append_frozen_evidence(state, fixture)
    output = _decision_output(state, fixture, evidence)
    decision = apply_discrepancy_decision(
        state,
        output,
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert decision["accepted"] is True

    terminal = audit_discrepancy_coverage(state, decision_checkpoint=True)
    assert terminal.complete is True
    assert terminal.stop_reason == "verdict_determined"
    assert terminal.action_count == 1

    verdict, basis = compile_discrepancy_verdict_basis(state)
    assert verdict == fixture["expected_verdict"]
    assert basis.claim_ids == [claim.claim_id]
    assert basis.finding_ids == [state.findings[0].finding_id]
    assert basis.evidence_ids == [evidence.evidence_id]
    assert basis.visual_anchor_fact_ids == claim.anchor_fact_ids
    if verdict == "fake":
        assert len(basis.discrepancy_ids) == 1
    else:
        assert basis.discrepancy_ids == []

    judgment = _judgment(verdict, basis, fixture)
    state.discrepancy_judgment = judgment
    trace_path = _write_trace(
        tmp_path,
        fixture,
        state,
        verdict,
        judgment,
        _steps(state, evidence),
    )
    report = audit_trace(trace_path)
    assert not report.failures(strict_scheduler=True)
    assert report.stats["v4_actions"] == 1

    before = state.model_dump(mode="json")
    rejected = apply_discrepancy_decision(
        state,
        output,
        reviewed_evidence_ids=[evidence.evidence_id],
        trigger="scheduled_boundary",
    )
    assert rejected == {
        "accepted": False,
        "rejected_reason": "no discrepancy decision is allowed after verdict",
    }
    assert state.model_dump(mode="json") == before

    second = _base_state(fixture)
    second_planning = apply_image_account_planning(
        second,
        _planning_output(fixture),
    )
    assert second_planning["accepted"] is True
    second_evidence = _append_frozen_evidence(second, fixture)
    second_output = _decision_output(second, fixture, second_evidence)
    second_decision = apply_discrepancy_decision(
        second,
        second_output,
        reviewed_evidence_ids=[second_evidence.evidence_id],
        trigger="qualified_evidence",
    )
    assert second_decision["accepted"] is True
    audit_discrepancy_coverage(second, decision_checkpoint=True)
    compile_discrepancy_verdict_basis(second)
    second.discrepancy_judgment = _judgment(
        verdict,
        second.discrepancy_verdict_basis,
        fixture,
    )
    assert second.model_dump(mode="json") == state.model_dump(mode="json")


def test_historical_replay_fixture_is_redacted_and_deterministic() -> None:
    raw = FIXTURE_PATH.read_text(encoding="utf-8")
    lowered = raw.casefold()
    assert "api_key" not in lowered
    assert "password" not in lowered
    assert "gold" not in lowered
    assert "interaction_id" not in lowered
    for fixture in _fixtures():
        assert datetime.fromisoformat(str(fixture["retrieved_at"])).tzinfo is not None
