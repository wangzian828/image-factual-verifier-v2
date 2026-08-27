from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.unified_react import (
    UnifiedReactToolAdapter,
    available_unified_react_tool_names,
    new_unified_react_state,
    reduce_unified_react_action,
    reduce_visual_bootstrap_action,
    validate_unified_react_action,
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
        "route": {
            "route_focus": "entity_event_identity",
            "expected_information": (
                "Whether public event information connects Riverfest to the "
                "pictured bridge."
            ),
            "priority": 1,
        },
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
    assert len(state.search_hypotheses) == 1
    assert len(state.tasks) == 1
    assert step.tool_args["task_id"] == state.tasks[0].task_id
    assert "investigation_intent" in step.tool_args
    assert update["created_target_fact_ids"]


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
                "route": {
                    "route_focus": "entity_event_identity",
                    "expected_information": "Whether the event uses this bridge.",
                    "priority": 1,
                },
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
    assert exported.trajectory_version == "ifv-trajectory-sft-v4"
    assert "<think>" in exported.messages[1]["content"] or any(
        "<think>" in message["content"]
        for message in exported.messages
        if message["role"] == "assistant"
    )
    assert any(
        "<function=perceive_scene>" in message["content"]
        for message in exported.messages
        if message["role"] == "assistant"
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
