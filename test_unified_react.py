from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.bootstrap import build_visual_bootstrap
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.unified_react import (
    RouteLocalReplanTool,
    UnifiedReactToolAdapter,
    available_unified_react_tool_names,
    build_unified_react_tools,
    new_unified_react_state,
    reduce_unified_react_action,
    reduce_visual_bootstrap_action,
    route_local_replan_candidate,
    validate_unified_react_action,
)
from src.orchestrator.task_store import (
    _failure_code,
    remaining_claim_hypothesis_routes,
)
from src.tools.base import BaseTool
from src.trajectory.exporter import (
    export_trajectory_action_only_example,
    export_trajectory_sft_example,
    unified_react_training_buckets,
)


def _case() -> ImageOnlyRuntimeCase:
    return ImageOnlyRuntimeCase(
        case_id="unified-fixture",
        image_path="fixture.jpg",
        image_sha256="a" * 64,
    )


def _step(
    *,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Dict[str, Any],
    call_id: str,
    thought: str = "Choose the next grounded action.",
) -> StageStep:
    return StageStep(
        stage_name="unified_react",
        action_type="tool_call",
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=json.dumps(tool_result, ensure_ascii=False),
        thought=thought,
        metadata={
            "stage": "unified_react",
            "function_call_id": call_id,
            "interaction_id": f"interaction-{call_id}",
            "policy_action": {
                "type": "tool_call",
                "name": tool_name,
                "arguments": dict(tool_args),
            },
            "policy_input": {
                "system_instruction": "unified test",
                "input_payload": {"turn": call_id},
                "tools": [],
                "response_format": None,
            },
        },
    )


def _bootstrap_state() -> tuple[Any, ImageOnlyRuntimeCase]:
    case = _case()
    state = new_unified_react_state(case)
    scene_step = _step(
        tool_name="perceive_scene",
        tool_args={},
        tool_result={
            "status": "success",
            "entities": [
                {
                    "name": "red bridge",
                    "entity_type": "building",
                    "bbox": [0.1, 0.2, 0.8, 0.7],
                    "confidence": 0.9,
                }
            ],
            "scene_description": "A red bridge spans a river.",
            "image_type": "photo",
        },
        call_id="scene",
    )
    scene_report = PerceptionReport(
        entities=[
            Entity(
                name="red bridge",
                entity_type="building",
                bbox=[0.1, 0.2, 0.8, 0.7],
                confidence=0.9,
            )
        ],
        scene_description="A red bridge spans a river.",
        image_type="photo",
    )
    scene_update = reduce_visual_bootstrap_action(
        state,
        step=scene_step,
        runtime_case=case,
        perception=scene_report,
    )
    scene_step.metadata["unified_react_delta"] = {"state_update": scene_update}

    ocr_step = _step(
        tool_name="ocr_with_position",
        tool_args={},
        tool_result={
            "status": "success",
            "text_regions": [
                {
                    "text": "RIVERFEST",
                    "bbox_quad": [
                        [0.2, 0.2],
                        [0.5, 0.2],
                        [0.5, 0.3],
                        [0.2, 0.3],
                    ],
                    "confidence": 0.95,
                    "language": "en",
                }
            ],
        },
        call_id="ocr",
    )
    full_report = PerceptionReport(
        entities=scene_report.entities,
        scene_description=scene_report.scene_description,
        image_type="photo",
        text_regions=[
            TextRegion(
                text="RIVERFEST",
                bbox_quad=[
                    [0.2, 0.2],
                    [0.5, 0.2],
                    [0.5, 0.3],
                    [0.2, 0.3],
                ],
                confidence=0.95,
                language="en",
            )
        ],
    )
    ocr_update = reduce_visual_bootstrap_action(
        state,
        step=ocr_step,
        runtime_case=case,
        perception=full_report,
    )
    ocr_step.metadata["unified_react_delta"] = {"state_update": ocr_update}
    return (state, case, [scene_step, ocr_step])


def test_unified_react_requires_model_selected_visual_bootstrap() -> None:
    state = new_unified_react_state(_case())
    assert available_unified_react_tool_names(state) == [
        "perceive_scene",
        "ocr_with_position",
    ]
    assert (
        validate_unified_react_action(
            state,
            tool_name="text_search",
            tool_args={"queries": "red bridge"},
        )
        == "text_search is not available in the current unified-ReAct state"
    )

    state, _case_value, _steps = _bootstrap_state()
    assert state.tasks == []
    assert state.target_facts == []
    assert state.search_hypotheses == []
    assert state.facts
    assert "text_search" in available_unified_react_tool_names(state)


def test_first_real_action_creates_target_and_route_only_after_bootstrap() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "routes": [
            {
                "route_focus": "entity_event_identity",
                "expected_information": (
                    "Whether public event information connects Riverfest to the "
                    "pictured bridge."
                ),
                "queries": ["Riverfest red bridge"],
                "suggested_tools": ["text_search", "visit"],
                "priority": 1,
            },
            {
                "route_focus": "visual_consistency",
                "expected_information": (
                    "Whether the bridge structure and visible event text are "
                    "consistent with the claimed event context."
                ),
                "suggested_tools": ["focused_visual_inspection"],
                "priority": 2,
            },
        ],
    }
    step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": intent,
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [],
                    "provider": "fixture",
                }
            ],
        },
        call_id="search",
    )
    assert not validate_unified_react_action(
        state,
        tool_name="text_search",
        tool_args=step.tool_args,
    )
    update = reduce_unified_react_action(
        state,
        step=step,
        runtime_case=case,
    )

    assert update["accepted"] is True
    assert len(state.target_facts) == 1
    assert len(state.search_hypotheses) == 2
    assert len(state.tasks) == 2
    assert step.tool_args["task_id"] == state.tasks[0].task_id
    assert "investigation_intent" in step.tool_args
    assert update["created_target_fact_ids"]


def test_compact_provider_intent_expands_to_distinct_routes() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    compact_intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "route": {
            "route_focus": "entity_event_identity",
            "expected_information": (
                "Whether public event information connects Riverfest to the pictured bridge."
            ),
            "priority": 1,
        },
        "alternate_route_focuses": ["visual_consistency"],
    }
    step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": compact_intent,
        },
        tool_result={
            "status": "success",
            "queries": [{"query": "Riverfest red bridge", "results": []}],
        },
        call_id="compact-search",
    )

    assert not validate_unified_react_action(
        state,
        tool_name="text_search",
        tool_args=step.tool_args,
    )
    update = reduce_unified_react_action(state, step=step, runtime_case=case)

    assert update["accepted"] is True
    assert len(state.search_hypotheses) == 3
    assert len(state.tasks) == 3
    assert state.search_hypotheses[0].suggested_tools[0] == "text_search"


def test_route_local_replan_is_a_dynamic_react_control_tool() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "routes": [
            {
                "route_focus": "entity_event_identity",
                "expected_information": "Whether the event uses this bridge.",
                "queries": ["Riverfest red bridge"],
                "suggested_tools": ["text_search"],
                "priority": 1,
            },
            {
                "route_focus": "visual_consistency",
                "expected_information": "Whether the bridge structure is consistent.",
                "suggested_tools": ["focused_visual_inspection"],
                "priority": 2,
            },
        ],
    }
    search_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": intent,
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [],
                }
            ],
        },
        call_id="route-boundary-search",
    )
    reduce_unified_react_action(state, step=search_step, runtime_case=case)
    boundary = route_local_replan_candidate(
        state,
        observation_update={
            "task_id": state.tasks[0].task_id,
            "created_evidence_ids": [],
        },
    )

    assert boundary == (state.tasks[0].task_id, "candidate_exhausted")
    tools = build_unified_react_tools(
        state,
        {"text_search": _RecordingTool()},
        route_local_replan_boundary=boundary,
    )
    assert len(tools) == 1
    assert isinstance(tools[0], RouteLocalReplanTool)
    assert [tool.name for tool in tools] == ["route_local_replan"]

    task = state.tasks[0]
    before_action_count = state.action_count
    args = {
        "task_id": task.task_id,
        "strategy": "replace_query",
        "replacement_query": "Riverfest bridge event location",
        "rationale": "The first query returned no candidates; use the same relation with a concrete event term.",
    }
    assert not validate_unified_react_action(
        state,
        tool_name="route_local_replan",
        tool_args=args,
        route_local_replan_boundary=boundary,
    )
    control_step = _step(
        tool_name="route_local_replan",
        tool_args=args,
        tool_result={
            "status": "success",
            "control": "route_local_replan",
        },
        call_id="route-boundary-control",
    )
    control_step.metadata["route_local_replan_trigger"] = boundary[1]
    update = reduce_unified_react_action(
        state,
        step=control_step,
        runtime_case=case,
    )

    assert update["accepted"] is True
    assert update["accepted_strategy"] == "replace_query"
    assert state.action_count == before_action_count
    assert task.query_replan_count == 1
    assert task.route_replan_count == 1
    assert task.suggested_queries == ["Riverfest bridge event location"]
    assert f"text_search:{task.task_id}" in remaining_claim_hypothesis_routes(
        state,
        task_ids={task.task_id},
    )


def test_external_fetch_failures_use_existing_recoverable_failure_categories() -> None:
    assert (
        _failure_code(
            "RuntimeError: Configured Jina page fetch failed: "
            "[SSL] record layer failure (_ssl.c:2590)",
            tool_name="visit",
        )
        == "access_limited"
    )
    assert (
        _failure_code(
            "Page visit was blocked by security verification: captcha",
            tool_name="visit",
        )
        == "access_limited"
    )
    assert (
        _failure_code(
            "Reference image access failed for https://example.test/image.jpg",
            tool_name="compare_with_reference",
        )
        == "access_limited"
    )


def test_compare_contract_failure_is_fatal_but_external_failure_is_not() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "routes": [
            {
                "route_focus": "entity_event_identity",
                "expected_information": "Whether public event information connects Riverfest to the pictured bridge.",
                "queries": ["Riverfest red bridge"],
                "suggested_tools": ["text_search", "visit"],
                "priority": 1,
            },
            {
                "route_focus": "visual_consistency",
                "expected_information": "Whether the bridge structure and visible event text are consistent with the claimed event context.",
                "suggested_tools": ["focused_visual_inspection"],
                "priority": 2,
            },
        ],
    }
    first_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": intent,
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [
                        {
                            "url": "https://example.test/riverfest",
                            "title": "Riverfest",
                            "snippet": "An event page.",
                        }
                    ],
                    "provider": "fixture",
                }
            ],
        },
        call_id="search-for-failure-test",
    )
    reduce_unified_react_action(state, step=first_step, runtime_case=case)
    task_id = state.tasks[0].task_id

    external_step = _step(
        tool_name="visit",
        tool_args={
            "task_id": task_id,
            "url": ["https://example.test/blocked"],
        },
        tool_result={
            "status": "error",
            "error": "Page visit was blocked by security verification: captcha",
        },
        call_id="external-failure",
    )
    external_update = reduce_unified_react_action(
        state,
        step=external_step,
        runtime_case=case,
    )
    assert state.stop_reason == ""
    assert external_update["fatal_engineering_error"] is False
    assert state.failures[-1].code == "access_limited"
    assert state.failures[-1].recoverable is True
    assert json.loads(state.attempted_routes[-1])["outcome"] == "failed"

    compare_step = _step(
        tool_name="compare_with_reference",
        tool_args={
            "task_id": task_id,
            "reference_url": "https://example.test/reference.jpg",
        },
        tool_result={
            "status": "error",
            "error": (
                "Gemini Interactions comparison failed: ValueError: "
                "edit_evidence_strength must be 'none' without edit evidence"
            ),
        },
        call_id="malformed-compare",
    )
    compare_update = reduce_unified_react_action(
        state,
        step=compare_step,
        runtime_case=case,
    )
    assert state.stop_reason == "engineering_error"
    assert compare_update["fatal_engineering_error"] is True
    assert state.failures[-1].code == "malformed_result"
    assert state.failures[-1].recoverable is False


class _RecordingTool(BaseTool):
    name = "fixture_tool"
    description = "fixture"
    parameters = {
        "type": "object",
        "properties": {
            "payload": {"type": "string"},
        },
        "required": ["payload"],
    }

    def __init__(self) -> None:
        self.received: Dict[str, Any] = {}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.received = dict(params)
        return {"status": "success"}


class _CandidateVisitTool(BaseTool):
    name = "visit"
    description = "Visit a runtime-selected page."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
            }
        },
        "required": ["url"],
    }

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "params": dict(params)}


def test_unified_tool_adapter_never_forwards_runtime_intent() -> None:
    delegate = _RecordingTool()
    adapter = UnifiedReactToolAdapter(
        delegate=delegate,
        task_ids=("task-1",),
        require_initial_intent=True,
    )
    adapter.call(
        {
            "payload": "visible to provider",
            "task_id": "task-1",
            "investigation_intent": {"target_fact": {}, "route": {}},
        }
    )
    assert delegate.received == {"payload": "visible to provider"}


class _ImagePathRecordingTool(BaseTool):
    name = "image_path_fixture"
    description = "fixture"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.image_path = ""
        self.received_paths: list[str] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.received_paths.append(self.image_path)
        return {"status": "success", "params": dict(params)}

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.received_paths.append(self.image_path)
        return {"status": "success", "params": dict(params)}


def test_unified_tool_adapter_binds_image_path_without_mutating_shared_tool() -> None:
    delegate = _ImagePathRecordingTool()
    adapter = UnifiedReactToolAdapter(delegate=delegate, image_path="case-a.jpg")

    asyncio.run(adapter.call_async({}))

    assert delegate.received_paths == ["case-a.jpg"]
    assert delegate.image_path == ""


def test_unified_visit_schema_lists_current_candidates_without_fragile_enum() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "routes": [
            {
                "route_focus": "entity_event_identity",
                "expected_information": "Whether the event uses this bridge.",
                "queries": ["Riverfest red bridge"],
                "suggested_tools": ["text_search", "visit"],
                "priority": 1,
            },
            {
                "route_focus": "visual_consistency",
                "expected_information": "Whether the visible bridge is consistent.",
                "suggested_tools": ["focused_visual_inspection"],
                "priority": 2,
            },
        ],
    }
    search_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": intent,
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [
                        {
                            "url": "https://example.test/riverfest",
                            "title": "Riverfest",
                            "snippet": "An event page.",
                        }
                    ],
                }
            ],
        },
        call_id="schema-candidate-search",
    )
    reduce_unified_react_action(state, step=search_step, runtime_case=case)

    tools = build_unified_react_tools(
        state,
        {"visit": _CandidateVisitTool()},
    )
    schema = tools[0].parameters["properties"]["url"]

    assert "https://example.test/riverfest" in schema["description"]
    assert "enum" not in schema


def test_stop_route_is_rejected_before_control_tool_execution() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    search_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": {
                "target_fact": {
                    "statement": (
                        "The pictured bridge is associated with the Riverfest event."
                    ),
                    "kind": "relation",
                    "predicate": "depicts_relation",
                    "anchor_fact_ids": [anchor_id],
                },
                "routes": [
                    {
                        "route_focus": "entity_event_identity",
                        "expected_information": "Whether the event uses this bridge.",
                        "queries": ["Riverfest red bridge"],
                        "suggested_tools": ["text_search", "visit"],
                        "priority": 1,
                    },
                    {
                        "route_focus": "visual_consistency",
                        "expected_information": "Whether the bridge structure and visible event text are consistent with the claimed event context.",
                        "suggested_tools": ["focused_visual_inspection"],
                        "priority": 2,
                    },
                ],
            },
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [
                        {
                            "url": "https://example.test/riverfest",
                            "title": "Riverfest",
                            "snippet": "An event page.",
                        }
                    ],
                    "provider": "fixture",
                }
            ],
        },
        call_id="search-with-page",
    )
    reduce_unified_react_action(state, step=search_step, runtime_case=case)

    reason = validate_unified_react_action(
        state,
        tool_name="stop_route",
        tool_args={
            "task_id": state.tasks[0].task_id,
            "rationale": "No further work is needed.",
        },
    )

    assert reason == "stop_route is not available in the current unified-ReAct state"
    assert "stop_route" not in available_unified_react_tool_names(state)


def test_visit_rejects_stale_runtime_candidate_url() -> None:
    state, case, _steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    intent = {
        "target_fact": {
            "statement": "The pictured bridge is associated with the Riverfest event.",
            "kind": "relation",
            "predicate": "depicts_relation",
            "anchor_fact_ids": [anchor_id],
        },
        "routes": [
            {
                "route_focus": "entity_event_identity",
                "expected_information": "Whether the event uses this bridge.",
                "queries": ["Riverfest red bridge"],
                "suggested_tools": ["text_search", "visit"],
                "priority": 1,
            },
            {
                "route_focus": "visual_consistency",
                "expected_information": "Whether the visible bridge and event text are consistent.",
                "suggested_tools": ["focused_visual_inspection"],
                "priority": 2,
            },
        ],
    }
    search_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": intent,
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [
                        {
                            "url": "https://example.test/riverfest",
                            "title": "Riverfest",
                            "snippet": "An event page.",
                        }
                    ],
                }
            ],
        },
        call_id="candidate-search",
    )
    reduce_unified_react_action(state, step=search_step, runtime_case=case)
    task_id = state.tasks[0].task_id

    reason = validate_unified_react_action(
        state,
        tool_name="visit",
        tool_args={
            "task_id": task_id,
            "url": ["https://example.test/stale"],
        },
    )

    assert "stale or unavailable url" in reason
    assert not validate_unified_react_action(
        state,
        tool_name="visit",
        tool_args={
            "task_id": task_id,
            "url": ["https://example.test/riverfest/"],
        },
    )


def test_visual_bootstrap_keeps_all_fact_anchor_dependencies() -> None:
    case = _case()
    perception = PerceptionReport(
        entities=[
            Entity(
                name=f"object {index}",
                entity_type="object",
                bbox=[0.01 * index, 0.1, 0.3, 0.4],
                confidence=0.9,
            )
            for index in range(8)
        ],
        scene_description="A crowded event display.",
        text_regions=[
            TextRegion(
                text=f"VISIBLE TEXT {index}",
                bbox_quad=[
                    [0.1, 0.1],
                    [0.2, 0.1],
                    [0.2, 0.2],
                    [0.1, 0.2],
                ],
                confidence=0.95,
                language="en",
            )
            for index in range(16)
        ],
    )

    bootstrap = build_visual_bootstrap(case, perception)
    anchor_ids = {item.anchor_id for item in bootstrap.retrieval_anchors}
    entity_ids = {item.entity_id for item in bootstrap.entities}

    assert len(bootstrap.retrieval_anchors) == 25
    for fact in bootstrap.facts:
        assert set(fact.basis_ids) <= anchor_ids | entity_ids | {
            item.fact_id for item in bootstrap.facts
        }


def test_visual_bootstrap_materializes_attributes_relations_and_limits() -> None:
    case = _case()
    perception = PerceptionReport(
        scene_description=(
            "A speaker stands behind a podium in front of a blue summit screen. "
            "Several audience members face the stage."
        ),
        entities=[
            Entity(
                name="speaker",
                entity_type="person",
                bbox=[0.2, 0.1, 0.5, 0.8],
                confidence=0.94,
                attributes={
                    "role_or_action": "speaks behind a podium",
                    "clothing": "dark suit",
                },
            ),
            Entity(
                name="podium",
                entity_type="object",
                bbox=[0.25, 0.55, 0.55, 0.95],
                confidence=0.91,
                attributes={"position": "in front of the speaker"},
            ),
        ],
        relations=[
            {
                "subject": "speaker",
                "predicate": "stands_behind",
                "object": "podium",
                "description": "The speaker stands behind the podium.",
                "confidence": 0.92,
            }
        ],
        notable_details=["Blue summit text is visible on the stage screen."],
        uncertainties=["The exact identity of the speaker is not visually resolved."],
    )

    bootstrap = build_visual_bootstrap(case, perception)
    speaker = next(item for item in bootstrap.entities if item.name == "speaker")
    assert speaker.attributes["role_or_action"] == "speaks behind a podium"
    assert any(
        fact.kind == "relation"
        and fact.predicate == "stands_behind"
        and fact.object_entity_id is not None
        and "speaker stands behind" in fact.statement
        for fact in bootstrap.facts
    )
    assert bootstrap.notable_details == [
        "Blue summit text is visible on the stage screen."
    ]
    assert bootstrap.uncertainties == [
        "The exact identity of the speaker is not visually resolved."
    ]


def test_unified_export_uses_qwen_think_and_tool_call_and_rejects_missing_thought() -> None:
    state, case, bootstrap_steps = _bootstrap_state()
    anchor_id = state.facts[0].fact_id
    search_step = _step(
        tool_name="text_search",
        tool_args={
            "queries": "Riverfest red bridge",
            "investigation_intent": {
                "target_fact": {
                    "statement": (
                        "The pictured bridge is associated with the Riverfest event."
                    ),
                    "kind": "relation",
                    "predicate": "depicts_relation",
                    "anchor_fact_ids": [anchor_id],
                },
                "routes": [
                    {
                        "route_focus": "entity_event_identity",
                        "expected_information": "Whether the event uses this bridge.",
                        "queries": ["Riverfest red bridge"],
                        "suggested_tools": ["text_search", "visit"],
                        "priority": 1,
                    },
                    {
                        "route_focus": "visual_consistency",
                        "expected_information": "Whether the bridge structure and visible event text are consistent with the claimed event context.",
                        "suggested_tools": ["focused_visual_inspection"],
                        "priority": 2,
                    },
                ],
            },
        },
        tool_result={
            "status": "success",
            "queries": [
                {
                    "query": "Riverfest red bridge",
                    "results": [],
                    "provider": "fixture",
                }
            ],
        },
        call_id="search",
        thought="OCR gives a concrete event anchor, so search that relation.",
    )
    update = reduce_unified_react_action(
        state,
        step=search_step,
        runtime_case=case,
    )
    search_step.metadata["unified_react_delta"] = {"state_update": update}
    final_step = {
        "stage": "unified_discrepancy_decision",
        "action_type": "output",
        "thought": "No qualified Evidence resolves the target yet.",
        "metadata": {
            "policy_input": {
                "system_instruction": "decision",
                "input_payload": {"phase": "decision"},
                "tools": [],
                "response_format": {"type": "json"},
            },
            "policy_action": {"verdict_proposal": "continue", "rationale": "none"},
        },
    }
    trace = {
        "image_id": case.case_id,
        "input_mode": "image_only",
        "decision_policy_version": "unified-react-v1",
        "termination": "success",
        "state": {
            "image_id": case.case_id,
            "input_mode": "image_only",
            "runtime_case": case.model_dump(mode="json"),
            "investigation_state": state.model_dump(mode="json"),
            "all_steps": [
                {
                    "stage": item.stage_name,
                    "action_type": item.action_type,
                    "tool_name": item.tool_name,
                    "tool_args": item.tool_args,
                    "tool_result": item.tool_result,
                    "thought": item.thought,
                    "metadata": item.metadata,
                }
                for item in [*bootstrap_steps, search_step]
            ]
            + [final_step],
        },
    }
    exported = export_trajectory_sft_example(trace)
    assert exported.trajectory_version == "ifv-trajectory-sft-v3"
    assert "<think>" in exported.messages[1]["content"] or any(
        "<think>" in message["content"]
        for message in exported.messages
        if message["role"] == "assistant"
    )
    assert any(
        message["role"] == "tool_call"
        and json.loads(message["content"])["name"] == "perceive_scene"
        for message in exported.messages
    )

    broken = json.loads(json.dumps(trace))
    broken["state"]["all_steps"][0]["thought"] = ""
    with pytest.raises(ValueError, match="provider-visible thought"):
        export_trajectory_sft_example(broken)
    action_only = export_trajectory_action_only_example(broken)
    assert action_only.trajectory_version == "ifv-trajectory-action-only-v1"
    assert unified_react_training_buckets(broken) == [
        "action_only",
        "rl_candidate",
    ]
