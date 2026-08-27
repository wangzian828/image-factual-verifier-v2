from __future__ import annotations

import json

from src.orchestrator.context_workspace import (
    build_explicit_workspace,
    build_stage_handoff,
    fit_stage_handoff_to_budget,
    render_stage_request,
)
from src.orchestrator.investigation_models import (
    ClaimAssessment,
    FactOrigin,
    ImageClaim,
    ImageOnlyInvestigationState,
    InvestigationBrief,
    ResearchTask,
    SearchHypothesis,
    VisualEntity,
    VisualFact,
)


def _state() -> ImageOnlyInvestigationState:
    entity = VisualEntity(
        entity_id="entity-1",
        name="a vehicle",
        entity_type="vehicle",
        region=[0.1, 0.1, 0.9, 0.9],
        confidence=0.9,
    )
    fact = VisualFact(
        fact_id="fact-1",
        kind="attribute",
        statement="A ceremonial vehicle is visible.",
        subject_entity_id=entity.entity_id,
        predicate="visible_in",
        status="active",
        basis_ids=[entity.entity_id],
        decision_relevance="decisive",
        origin=FactOrigin(type="input_image", origin_ids=[entity.entity_id]),
    )
    claim = ImageClaim(
        claim_id="claim-1",
        fact_id=fact.fact_id,
        statement="The image identifies the ceremonial vehicle correctly.",
        anchor_fact_ids=[fact.fact_id],
        salience="high",
        task_ids=["task-1"],
    )
    hypothesis = SearchHypothesis(
        hypothesis_id="hypothesis-1",
        claim_ids=[claim.claim_id],
        statement="The event record may identify the actual vehicle.",
        queries=["event official transport"],
        expected_information="An exact official description of the vehicle.",
        suggested_tools=["text_search", "visit"],
        task_id="task-1",
    )
    task = ResearchTask(
        task_id="task-1",
        fact_ids=[fact.fact_id],
        claim_ids=[claim.claim_id],
        hypothesis_id=hypothesis.hypothesis_id,
        question="What vehicle was used at the event?",
        purpose="Check the visible relation.",
        origin_ids=[claim.claim_id, hypothesis.hypothesis_id],
        suggested_tools=["text_search", "visit"],
    )
    assessment = ClaimAssessment(
        assessment_id="assessment-1",
        claim_id=claim.claim_id,
        action_count=1,
        assessment="insufficient",
        remaining_gap="The official event account has not been read.",
        rationale="The current material is only a lead.",
    )
    return ImageOnlyInvestigationState(
        brief=InvestigationBrief(brief_id="brief-1", case_id="case-1"),
        entities=[entity],
        facts=[fact],
        target_facts=[claim],
        search_hypotheses=[hypothesis],
        tasks=[task],
        claim_assessments=[assessment],
        action_count=1,
    )


def test_workspace_protects_claim_assessment_and_open_routes() -> None:
    workspace = build_explicit_workspace(_state())

    assert workspace.protected_context.claim_ids == ["claim-1"]
    assert workspace.protected_context.assessment_ids == ["assessment-1"]
    assert workspace.protected_context.task_ids == ["task-1"]
    assert workspace.protected_context.hypothesis_ids == ["hypothesis-1"]
    assert workspace.open_questions == [
        "The official event account has not been read.",
        "An exact official description of the vehicle.",
    ]


def test_handoff_is_stable_and_has_complete_protected_coverage() -> None:
    state = _state()
    first = build_stage_handoff(
        state,
        target_stage="verification",
        stage_input="continue",
        available_tools=["visit", "text_search"],
    )
    second = build_stage_handoff(
        state,
        target_stage="verification",
        stage_input="continue",
        available_tools=["text_search", "visit"],
    )

    assert first.handoff_id == second.handoff_id
    assert first.protected_coverage == 1.0
    assert not first.missing_protected_ids
    assert first.available_tools == ["text_search", "visit"]


def test_budgeter_removes_background_but_never_protected_state() -> None:
    packet = build_stage_handoff(
        _state(),
        target_stage="verification",
        stage_input="continue",
        available_tools=["text_search"],
    )
    packet.workspace.recent_discoveries = [
        {"discovery_id": f"discovery-{index}", "snippet": "x" * 2000}
        for index in range(8)
    ]
    result = fit_stage_handoff_to_budget(packet, target_tokens=1024)

    assert result.removed_item_ids
    assert result.all_protected_items_reachable is True
    assert result.packet.protected_coverage == 1.0
    assert set(result.retained_protected_ids) == set(packet.protected_ids)
    assert result.packet.workspace.claims == packet.workspace.claims
    assert result.packet.workspace.open_tasks == packet.workspace.open_tasks
    assert result.packet.workspace.active_hypotheses == packet.workspace.active_hypotheses


def test_budgeter_reports_protected_overflow_without_silent_deletion() -> None:
    packet = build_stage_handoff(
        _state(),
        target_stage="verification",
        stage_input="y" * 8000,
        available_tools=["text_search"],
    )
    result = fit_stage_handoff_to_budget(packet, target_tokens=1024)

    assert result.protected_context_overflow is True
    assert result.all_protected_items_reachable is True
    assert result.packet.stage_input == packet.stage_input
    assert result.packet.protected_ids == packet.protected_ids


def test_initial_planning_request_does_not_duplicate_workspace() -> None:
    packet = build_stage_handoff(
        _state(),
        target_stage="unified_react",
        stage_input={"bootstrap_tasks": [{"task_id": "task-1"}]},
    )

    rendered = render_stage_request(packet)
    payload = json.loads(rendered)

    assert '"workspace_version"' in rendered
    assert '"workspace"' not in rendered
    assert payload["bootstrap_tasks"][0]["task_id"] == "task-1"


def test_complete_stage_requests_keep_only_their_workspace_addendum() -> None:
    state = _state()
    packet = build_stage_handoff(
        state,
        target_stage="unified_reflection",
        stage_input={
            "target_facts": [item.model_dump(mode="json") for item in state.target_facts],
            "active_tasks": [item.model_dump(mode="json") for item in state.tasks],
        },
        available_tools=["text_search"],
    )
    packet.workspace.recent_discoveries = [
        {"discovery_id": "discovery-1", "snippet": "background"}
    ]
    packet.workspace.attempted_routes = [
        {"tool": "text_search", "query": "background"}
    ]

    rendered = render_stage_request(packet)

    assert '"workspace_projection"' in rendered
    assert '"protected_evidence"' not in rendered
    assert '"recent_discoveries"' not in rendered
    assert '"attempted_routes"' not in rendered
    # The full workspace remains on the packet for archive/audit and recall.
    assert packet.workspace.recent_discoveries
    assert packet.workspace.attempted_routes


def test_stage_owned_contexts_do_not_receive_full_workspace() -> None:
    packet = build_stage_handoff(
        _state(),
        target_stage="unified_react",
        stage_input={
            "route_boundary": "candidate_exhausted",
            "active_target": {"fact_id": "fact-1"},
        },
        available_tools=[],
    )

    rendered = render_stage_request(packet)

    assert '"workspace_projection"' in rendered
    assert '"workspace": {' not in rendered


def test_bounded_judgment_request_omits_general_workspace() -> None:
    packet = build_stage_handoff(
        _state(),
        target_stage="unified_judgment",
        stage_input={"compiled_verdict": "fake", "compiled_basis": {"claim_ids": ["claim-1"]}},
    )

    rendered = render_stage_request(packet)
    payload = json.loads(rendered)

    assert '"workspace_projection"' in rendered
    assert (
        payload["runtime_handoff"]["workspace_projection"][
            "full_workspace_archived"
        ]
        is True
    )
    assert '"workspace": {' not in rendered
