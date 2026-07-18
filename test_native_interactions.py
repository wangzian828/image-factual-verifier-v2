from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, Field

from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.stage_runner import (
    InteractionSession,
    StageRunner,
    StageStep,
)
from src.tools.base import BaseTool
from src.tools.visit import VisitTool
from test_support_models import ToolStageOutput


class NativeFakeBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Interactions request")
        return self.responses.pop(0)

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs: Any):
        raise AssertionError("Native tool stages must not use get_response")


class RecordingTool(BaseTool):
    name = "text_search"
    description = "Search the web for text evidence."
    parameters = {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
    }

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        return {
            "status": "success",
            "query": params["queries"][0],
            "selected_url": "https://example.org/source",
            "summary": "The source directly answers the investigation question.",
            "evidence": "The source directly answers the investigation question.",
            "results": [],
        }


class VisualInspectTool(BaseTool):
    name = "crop_and_inspect"
    description = "Inspect a visual crop."
    parameters = {
        "type": "object",
        "properties": {
            "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "focus_question": {"type": "string"},
            "visual_question_id": {"type": "string"},
            "source_evidence_id": {"type": "string"},
            "expected_property": {"type": "string"},
        },
        "required": ["bbox", "focus_question"],
    }

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        return {
            "status": "success",
            "answer": "The cropped region matches the expected property.",
            "findings": [],
            "anomalies": [],
        }


class NativeStructuredOutput(BaseModel):
    value: str


class BoundedNativeOutput(BaseModel):
    values: List[str] = Field(default_factory=list, max_length=2)


def _function_call_response() -> Dict[str, Any]:
    return {
        "id": "interaction-1",
        "status": "requires_action",
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_thought_tokens": 0,
        },
        "steps": [
            {
                "id": "call-1",
                "type": "function_call",
                "name": "text_search",
                "arguments": {"question_id": "q1", "queries": ["direct source"]},
            }
        ],
    }


def _completed_response() -> Dict[str, Any]:
    output = {
        "evidence": [
            {
                "function_call_id": "call-1",
                "source": "https://example.org/source",
                "summary": "The source directly answers the investigation question.",
                "raw_excerpt": "The source directly answers the investigation question.",
                "direction": "supports",
                "quality": "moderate",
                "tool_used": "text_search",
                "related_question": "q1",
            }
        ],
        "visual_anomalies": [],
        "authenticity_assessment": "authentic",
        "key_findings": ["The direct source supports q1."],
        "source_findings": [],
        "visual_evidence": [],
        "world_model": {},
        "question_resolutions": [],
        "coverage_complete": False,
        "unresolved_priority_questions": [],
        "exhausted_priority_questions": [],
        "iteration_count": 1,
    }
    return {
        "id": "interaction-2",
        "status": "completed",
        "usage": {"total_input_tokens": 140, "total_output_tokens": 60},
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(output)}],
            }
        ],
    }


def test_native_function_call_round_trip() -> None:
    backend = NativeFakeBackend([_function_call_response(), _completed_response()])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(
        runner.run("Investigation questions:\n- [q0] first question\n- [q1] second question")
    )

    assert steps[0].tokens["thought"] == 0
    assert parsed is not None
    assert parsed.authenticity_assessment == "authentic"
    assert [step.action_type for step in steps] == ["tool_call", "output"]
    assert steps[0].tool_args["__question_id"] == "q1"
    assert steps[0].metadata["native_interactions"] is True
    assert steps[0].metadata["previous_interaction_id"] is None
    assert steps[0].metadata["interaction_id"] == "interaction-1"
    assert steps[1].metadata["previous_interaction_id"] == "interaction-1"
    assert steps[1].metadata["interaction_id"] == "interaction-2"
    assert tool.calls == [{"queries": ["direct source"]}]

    first_request, second_request = backend.requests
    assert first_request["previous_interaction_id"] is None
    assert first_request["tools"][0]["type"] == "function"
    assert first_request["tools"][0]["name"] == "text_search"
    assert "question_id" in first_request["tools"][0]["parameters"]["required"]
    assert first_request["tools"][0]["parameters"]["properties"]["question_id"]["enum"] == ["q0", "q1"]
    assert "image_input" not in first_request["tools"][0]["parameters"]["properties"]
    assert second_request["previous_interaction_id"] == "interaction-1"
    assert second_request["previous_interaction_id"] == steps[0].metadata["interaction_id"]
    assert second_request["previous_interaction_id"] == steps[1].metadata["previous_interaction_id"]
    assert second_request["tools"] == first_request["tools"]
    assert second_request["system_instruction"] == first_request["system_instruction"]
    assert first_request["response_format"] is None
    assert second_request["response_format"] is None
    assert first_request["generation_config"]["tool_choice"] == "any"
    assert "tool_choice" not in second_request["generation_config"]

    function_result = second_request["input_payload"][0]
    assert function_result["type"] == "function_result"
    assert function_result["name"] == "text_search"
    assert function_result["call_id"] == "call-1"
    returned = json.loads(function_result["result"][0]["text"])
    assert returned["question_id"] == "q1"
    assert returned["function_call_id"] == "call-1"
    assert "directly answers" in json.dumps(returned["result"])
    assert "is_error" not in function_result


def test_native_tool_schema_uses_runtime_owned_task_ids_without_bracket_hints() -> None:
    function_call = _function_call_response()
    function_call["steps"][0]["arguments"]["question_id"] = "task-live"
    backend = NativeFakeBackend([function_call, _completed_response()])
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        question_claims={"task-live": "The runtime-owned claim."},
    )

    parsed, steps = asyncio.run(
        runner.run('{"active_tasks":[{"task_id":"task-live"}]}')
    )

    assert parsed is not None
    assert steps[0].tool_args["__question_id"] == "task-live"
    schema = backend.requests[0]["tools"][0]["parameters"]
    assert schema["properties"]["question_id"]["enum"] == ["task-live"]
    assert "q0" not in schema["properties"]["question_id"]["description"]


def test_shared_session_carries_pending_function_result_into_next_stage() -> None:
    backend = NativeFakeBackend(
        [
            _function_call_response(),
            {
                "id": "checkpoint-interaction",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 25,
                    "total_output_tokens": 10,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"value": "reviewed tool observation"}
                                ),
                            }
                        ],
                    }
                ],
            },
        ]
    )
    session = InteractionSession(previous_interaction_id="planning-interaction")
    tool = RecordingTool()
    react = StageRunner(
        llm=backend,
        system_prompt="Choose one investigation tool.",
        tools=[tool],
        output_schema=NativeStructuredOutput,
        max_rounds=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        interaction_session=session,
        force_tool_each_round=True,
        should_stop=lambda steps: any(
            step.action_type == "tool_call" for step in steps
        ),
        stop_output_factory=lambda: NativeStructuredOutput(value="boundary"),
    )
    checkpoint = StageRunner(
        llm=backend,
        system_prompt="Review the tool observation.",
        tools=[],
        output_schema=NativeStructuredOutput,
        max_rounds=1,
        stage_name="checkpoint",
        attach_image=False,
        interaction_session=session,
    )

    react_output, react_steps = asyncio.run(
        react.run("Investigation questions:\n- [q1] inspect the source")
    )
    assert react_output is not None
    assert react_steps[0].action_type == "tool_call"
    assert session.previous_interaction_id == "interaction-1"
    assert len(session.pending_input) == 1

    checkpoint_output, _ = asyncio.run(
        checkpoint.run("Current evidence checkpoint context")
    )

    assert checkpoint_output is not None
    react_request, checkpoint_request = backend.requests
    assert react_request["previous_interaction_id"] == "planning-interaction"
    assert checkpoint_request["previous_interaction_id"] == "interaction-1"
    checkpoint_input = checkpoint_request["input_payload"]
    assert [item["type"] for item in checkpoint_input] == [
        "function_result",
        "user_input",
    ]
    assert checkpoint_input[0]["call_id"] == "call-1"
    assert checkpoint_input[1]["content"] == [
        {"type": "text", "text": "Current evidence checkpoint context"}
    ]
    assert session.previous_interaction_id == "checkpoint-interaction"
    assert session.pending_input == []


def test_native_schema_rejects_non_numeric_array_items() -> None:
    tool = RecordingTool()
    tool.parameters = {
        "type": "object",
        "properties": {
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            }
        },
        "required": ["bbox"],
    }
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="",
        tools=[tool],
        stage_name="verification",
    )
    runner.active_question_ids = ["q1"]

    error = runner._validate_native_tool_args(
        "text_search",
        {"question_id": "q1", "bbox": ["0.1", "0.2", "0.8", "0.9"]},
    )

    assert "bbox' for text_search[0] must be a number" in error


def test_native_response_schema_drops_max_items_but_local_validation_keeps_it() -> None:
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="",
        tools=[],
        output_schema=BoundedNativeOutput,
        stage_name="reflection",
    )

    assert "maxItems" in json.dumps(runner._normalized_output_schema())
    assert "maxItems" not in json.dumps(runner._native_response_format())
    assert runner._validate_output({"values": ["a", "b", "c"]}) is None


def test_usage_records_thought_tokens() -> None:
    assert StageRunner._usage_tokens(
        {
            "total_input_tokens": 11,
            "total_output_tokens": 7,
            "total_thought_tokens": 3,
        }
    ) == {"prompt": 11, "completion": 7, "thought": 3}


def test_native_structured_output_steps_record_request_parent_and_response_id() -> None:
    invalid = {
        "id": "structured-root",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"wrong": "value"})}],
            }
        ],
    }
    corrected = {
        "id": "structured-child",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"value": "ok"})}],
            }
        ],
    }
    backend = NativeFakeBackend([invalid, corrected])
    runner = StageRunner(
        llm=backend,
        system_prompt="Return structured output.",
        tools=[],
        output_schema=NativeStructuredOutput,
        max_rounds=1,
        stage_name="planning",
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("Create the plan."))

    assert parsed is not None
    assert parsed.value == "ok"
    assert steps[0].metadata["previous_interaction_id"] is None
    assert steps[0].metadata["interaction_id"] == "structured-root"
    assert steps[1].metadata["previous_interaction_id"] == "structured-root"
    assert steps[1].metadata["interaction_id"] == "structured-child"
    assert backend.requests[1]["previous_interaction_id"] == steps[1].metadata["previous_interaction_id"]


def test_native_output_before_tools_is_rejected() -> None:
    premature = _completed_response()
    premature["id"] = "interaction-0"
    backend = NativeFakeBackend([premature, _function_call_response(), _completed_response()])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify the source"))

    assert parsed is not None
    assert steps[0].action_type == "output_rejected"
    assert "at least 1 tool calls" in steps[0].metadata["rejection_reason"]
    assert steps[0].metadata["react_action_turn"] == 0
    assert steps[0].metadata["protocol_corrections_used"] == 1
    assert steps[1].action_type == "tool_call"
    assert steps[1].metadata["react_action_turn"] == 1
    assert backend.requests[1]["previous_interaction_id"] == "interaction-0"


def test_invalid_output_before_required_tool_gets_tool_only_correction() -> None:
    invalid = {
        "id": "interaction-invalid",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": '{"not_the_required_schema":true}',
                    }
                ],
            }
        ],
    }
    call = _function_call_response()
    call["id"] = "interaction-tool"
    completed = _completed_response()
    completed["id"] = "interaction-complete"
    backend = NativeFakeBackend([invalid, call, completed])
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=2,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify"))

    assert parsed is not None
    assert [step.action_type for step in steps] == [
        "output_rejected",
        "tool_call",
        "output",
    ]
    assert "Do not return output" in backend.requests[1]["input_payload"]
    assert "Invoke exactly one available function" in (
        backend.requests[1]["input_payload"]
    )
    assert "requires at least 1 executable function call" in (
        backend.requests[0]["system_instruction"]
    )


def test_force_tool_each_round_defers_output_to_forced_request() -> None:
    second_call = _function_call_response()
    second_call["id"] = "interaction-2"
    second_call["steps"][0]["id"] = "call-2"
    second_call["steps"][0]["arguments"] = {
        "question_id": "q0",
        "queries": ["another direct source"],
    }
    forced = _completed_response()
    forced["id"] = "interaction-forced"
    backend = NativeFakeBackend(
        [_function_call_response(), second_call, forced]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=2,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        force_tool_each_round=True,
    )

    parsed, steps = asyncio.run(
        runner.run("- [q0] first\n- [q1] second")
    )

    assert parsed is not None
    assert [step.action_type for step in steps] == [
        "tool_call",
        "tool_call",
        "output",
    ]
    assert backend.requests[0]["generation_config"]["tool_choice"] == "any"
    assert backend.requests[1]["generation_config"]["tool_choice"] == "any"
    assert "tool_choice" not in backend.requests[2]["generation_config"]
    assert backend.requests[2]["tools"] == []
    assert "Final JSON is requested separately" in (
        backend.requests[0]["system_instruction"]
    )


def test_native_output_requires_initial_attempt_for_each_required_question() -> None:
    first_call = _function_call_response()
    first_call["id"] = "interaction-q0"
    first_call["steps"][0]["id"] = "call-q0"
    first_call["steps"][0]["arguments"] = {
        "question_id": "q0",
        "queries": ["first direct source"],
    }
    premature = _completed_response()
    premature["id"] = "interaction-premature"
    premature["steps"][0]["content"][0]["text"] = json.dumps(
        {
            **json.loads(
                _completed_response()["steps"][0]["content"][0]["text"]
            ),
            "evidence": [
                {
                    "function_call_id": "call-q0",
                    "source": "https://example.org/q0",
                    "summary": "The first source answers q0.",
                    "raw_excerpt": "The first source answers q0.",
                    "direction": "supports",
                    "quality": "moderate",
                    "tool_used": "text_search",
                    "related_question": "q0",
                }
            ],
        }
    )
    second_call = _function_call_response()
    second_call["id"] = "interaction-q1"
    second_call["steps"][0]["id"] = "call-q1"
    second_call["steps"][0]["arguments"] = {
        "question_id": "q1",
        "queries": ["second direct source"],
    }
    completed = _completed_response()
    completed["id"] = "interaction-complete"
    backend = NativeFakeBackend([first_call, premature, second_call, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        priority_question_ids=["q0"],
        supporting_question_ids=["q1"],
    )

    parsed, steps = asyncio.run(runner.run("- [q0] first\n- [q1] second"))

    assert parsed is not None
    assert [step.action_type for step in steps] == [
        "tool_call",
        "output_rejected",
        "tool_call",
        "output",
    ]
    assert "untouched P2: q1" in steps[1].metadata["rejection_reason"]
    assert backend.requests[2]["previous_interaction_id"] == "interaction-premature"
    assert tool.calls == [
        {"queries": ["first direct source"]},
        {"queries": ["second direct source"]},
    ]


def test_native_missing_question_id_is_not_silently_assigned() -> None:
    missing_id = _function_call_response()
    missing_id["steps"][0]["arguments"].pop("question_id")
    corrected = _function_call_response()
    corrected["id"] = "interaction-2"
    corrected["steps"][0]["id"] = "call-2"
    completed = _completed_response()
    completed["id"] = "interaction-3"
    backend = NativeFakeBackend([missing_id, corrected, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q0] first\n- [q1] second"))

    assert parsed is not None
    assert steps[0].action_type == "format_error"
    assert steps[0].metadata["invalid_tool_arguments"] is True
    assert tool.calls == [{"queries": ["direct source"]}]
    first_result = backend.requests[1]["input_payload"][0]
    assert "Missing required argument(s) for text_search: question_id" in first_result["result"][0]["text"]
    assert first_result["is_error"] is True


def test_native_unknown_question_id_is_rejected() -> None:
    unknown = _function_call_response()
    unknown["steps"][0]["arguments"]["question_id"] = "q99"
    corrected = _function_call_response()
    corrected["id"] = "interaction-2"
    corrected["steps"][0]["id"] = "call-2"
    completed = _completed_response()
    completed["id"] = "interaction-3"
    backend = NativeFakeBackend([unknown, corrected, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q0] first\n- [q1] second"))

    assert parsed is not None
    assert steps[0].action_type == "format_error"
    assert steps[0].metadata["invalid_tool_arguments"] is True
    error = backend.requests[1]["input_payload"][0]["result"][0]["text"]
    assert "must be one of: q0, q1" in error
    assert tool.calls == [{"queries": ["direct source"]}]


def test_native_executes_all_parallel_calls_and_returns_all_results() -> None:
    parallel = _function_call_response()
    parallel["steps"].append(
        {
            "id": "call-parallel",
            "type": "function_call",
            "name": "text_search",
            "arguments": {"question_id": "q0", "queries": ["another source"]},
        }
    )
    completed = _completed_response()
    completed["id"] = "interaction-2"
    backend = NativeFakeBackend([parallel, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate with tools.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q0] first\n- [q1] second"))

    assert parsed is not None
    assert [step.action_type for step in steps[:2]] == ["tool_call", "tool_call"]
    assert all(step.metadata["function_call_count"] == 2 for step in steps[:2])
    assert [step.metadata["react_action_turn"] for step in steps[:2]] == [1, 1]
    assert len(backend.requests[1]["input_payload"]) == 2
    assert tool.calls == [
        {"queries": ["direct source"]},
        {"queries": ["another source"]},
    ]
    assert all(
        item["type"] == "function_result"
        for item in backend.requests[1]["input_payload"]
    )


def test_native_tool_schema_hides_server_image_path() -> None:
    class ImageTool(RecordingTool):
        name = "reverse_image_search"
        parameters = {
            "type": "object",
            "properties": {
                "image_input": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": ["image_input"],
        }

        def call(self, params):
            self.calls.append(dict(params))
            return {"status": "success", "candidate_page_urls": []}

    backend = NativeFakeBackend([])
    tool = ImageTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=1,
        image_path="D:/private/input.png",
        stage_name="verification",
        attach_image=False,
    )
    runner.active_question_ids = ["q0"]
    schema = runner._build_native_tool_schemas()[0]["parameters"]
    assert "image_input" not in schema["properties"]
    assert "image_input" not in schema["required"]
    assert schema["properties"]["question_id"]["enum"] == ["q0"]

    serialized, _ = asyncio.run(
        runner._execute_tool(
            "reverse_image_search",
            {"image_input": "D:/model/injected.png", "__question_id": "q0"},
        )
    )
    assert json.loads(serialized)["status"] == "success"
    assert tool.calls == [{"image_input": "D:/private/input.png"}]


def test_native_tool_schema_constrains_array_items() -> None:
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="Inspect the selected candidate.",
        tools=[VisitTool()],
        stage_name="verification",
        tool_argument_constraints={
            "visit": {
                "url": ["https://example.org/pending"],
            }
        },
    )
    runner.active_question_ids = ["q0"]

    schema = runner._build_native_tool_schemas()[0]["parameters"]

    assert schema["properties"]["url"]["items"]["enum"] == [
        "https://example.org/pending"
    ]
    assert runner._validate_native_tool_args(
        "visit",
        {
            "question_id": "q0",
            "url": ["https://example.org/pending"],
            "goal": "Check the claim.",
        },
    ) == ""
    assert "must be one of" in runner._validate_native_tool_args(
        "visit",
        {
            "question_id": "q0",
            "url": ["https://example.org/unowned"],
            "goal": "Check the claim.",
        },
    )


def test_canonical_reverse_image_result_preserves_validated_references() -> None:
    reference_url = "https://example.org/reference.jpg"

    result = StageRunner._canonical_reverse_image_result(
        {
            "status": "success",
            "branch": "lens",
            "candidate_page_urls": ["https://example.org/page"],
            "reference_image_candidates": [reference_url],
            "lens_results": [
                {
                    "title": "Reference",
                    "url": "https://example.org/page",
                    "image_url": reference_url,
                }
            ],
        }
    )

    assert result["reference_image_candidates"] == [reference_url]
    assert result["lens_results"][0]["image_url"] == reference_url


def test_tool_internal_llm_usage_is_private_and_attached_to_step_metadata() -> None:
    class ToolWithMetrics(RecordingTool):
        name = "reverse_image_search"
        parameters = {"type": "object", "properties": {}, "required": []}

        def call(self, params):
            return {
                "status": "success",
                "candidate_page_urls": [],
                "__runtime_metrics__": {
                    "llm_api_calls": 2,
                    "tokens": {"prompt": 120, "completion": 30, "thought": 0},
                },
                "nested": {
                    "__runtime_metrics__": {
                        "llm_api_calls": 2,
                        "tokens": {"prompt": 120, "completion": 30, "thought": 0},
                    }
                },
            }

    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="Investigate.",
        tools=[ToolWithMetrics()],
        output_schema=ToolStageOutput,
        stage_name="verification",
        attach_image=False,
    )

    serialized, metadata = asyncio.run(
        runner._execute_tool("reverse_image_search", {})
    )

    assert "__runtime_metrics__" not in json.loads(serialized)
    assert metadata["tool_llm_api_calls"] == 2
    assert metadata["tool_tokens"] == {
        "prompt": 120,
        "completion": 30,
        "thought": 0,
    }


def test_priority_two_can_be_resampled_after_all_required_questions_are_served() -> None:
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        stage_name="verification",
        attach_image=False,
        prior_steps=[
            StageStep(
                action_type="tool_call",
                tool_name="text_search",
                tool_args={"__question_id": "q0", "queries": ["primary statement"]},
            ),
            StageStep(
                action_type="tool_call",
                tool_name="text_search",
                tool_args={"__question_id": "q1", "queries": ["image source"]},
            ),
        ],
        priority_question_ids=["q0"],
        supporting_question_ids=["q1"],
    )

    assert runner._priority_coverage_error({"__question_id": "q1"}, []) == ""


@pytest.mark.parametrize(
    "blocked_query",
    [
        "politician joined party fact check",
        "politician joined party fake news",
        "politician party claim hoax",
    ],
)
def test_evaluation_rejects_fact_check_query_before_search_and_continues(
    blocked_query: str,
) -> None:
    rejected = _function_call_response()
    rejected["steps"][0]["arguments"]["queries"] = [blocked_query]
    corrected = _function_call_response()
    corrected["id"] = "interaction-corrected"
    corrected["steps"][0]["id"] = "call-corrected"
    corrected["steps"][0]["arguments"]["queries"] = ["politician official party statement"]
    completed = _completed_response()
    completed["id"] = "interaction-completed"
    backend = NativeFakeBackend([rejected, corrected, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        source_access_policy=SourceAccessPolicy(
            policy_id="evaluation",
            excluded_domains=frozenset({"factcrescendo.com"}),
        ),
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify the party claim"))

    assert parsed is not None
    assert steps[0].action_type == "format_error"
    assert steps[0].metadata["error_class"] == "protocol_error"
    assert steps[0].metadata["search_policy_rejection"] is True
    assert tool.calls == [{"queries": ["politician official party statement"]}]
    returned = backend.requests[1]["input_payload"][0]
    assert returned["is_error"] is True
    assert "removed every query" in returned["result"][0]["text"]


def test_mixed_search_queries_execute_only_policy_eligible_subset() -> None:
    mixed = _function_call_response()
    mixed["steps"][0]["arguments"]["queries"] = [
        "politician joined party fake news",
        "politician official party statement",
    ]
    completed = _completed_response()
    completed["id"] = "interaction-completed"
    backend = NativeFakeBackend([mixed, completed])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=2,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        source_access_policy=SourceAccessPolicy(
            policy_id="evaluation",
            excluded_domains=frozenset({"factcrescendo.com"}),
        ),
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify the party claim"))

    assert parsed is not None
    assert steps[0].action_type == "tool_call"
    assert steps[0].tool_args["queries"] == [
        "politician official party statement"
    ]
    assert steps[0].metadata["policy_filtered_query_count"] == 1
    assert "fake news" not in json.dumps(steps[0].tool_args)
    assert tool.calls == [{"queries": ["politician official party statement"]}]


def test_protocol_correction_exhaustion_is_a_hard_failure_without_forced_output() -> None:
    first = _completed_response()
    first["id"] = "interaction-rejected-1"
    second = _completed_response()
    second["id"] = "interaction-rejected-2"
    backend = NativeFakeBackend([first, second])
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=1,
        max_protocol_corrections=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    with pytest.raises(RuntimeError, match="protocol correction budget") as captured:
        asyncio.run(runner.run("- [q1] verify"))

    partial = getattr(captured.value, "stage_steps", [])
    assert len(backend.requests) == 2
    assert [step.action_type for step in partial] == [
        "output_rejected",
        "output_rejected",
    ]
    assert partial[-1].metadata["correction_budget_exhausted"] is True
    assert partial[-1].metadata["termination_reason"] == (
        "protocol_correction_budget_exhausted"
    )


def test_native_invalid_final_schema_is_rejected() -> None:
    invalid = _completed_response()
    invalid["steps"][0]["content"][0]["text"] = json.dumps(
        {
            "evidence": [],
            "visual_anomalies": [],
            "authenticity_assessment": "definitely_real",
            "key_findings": [],
            "source_findings": [],
            "visual_evidence": [],
            "world_model": {},
            "question_resolutions": [],
            "coverage_complete": False,
            "unresolved_priority_questions": [],
            "exhausted_priority_questions": [],
            "iteration_count": 1,
            "unexpected": "must be forbidden",
        }
    )
    backend = NativeFakeBackend([_function_call_response(), invalid, _completed_response()])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("- [q1] verify"))
    assert parsed is not None
    assert steps[1].action_type == "output_rejected"
    assert steps[1].output["authenticity_assessment"] == "definitely_real"
    assert len(backend.requests) == 3


@pytest.mark.parametrize(
    "response",
    [
        {"status": "completed", "steps": []},
        {"id": "interaction-bad", "status": "incomplete", "steps": []},
        {
            "id": "interaction-bad",
            "status": "completed",
            "steps": [
                {
                    "id": "call-1",
                    "type": "function_call",
                    "name": "text_search",
                    "arguments": {"question_id": "q1", "queries": ["source"]},
                }
            ],
        },
    ],
)
def test_native_invalid_response_envelope_is_a_hard_failure(response) -> None:
    backend = NativeFakeBackend([response])
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=1,
        stage_name="verification",
        attach_image=False,
    )
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run("- [q1] verify"))


def test_native_failure_exposes_completed_steps_for_error_trace() -> None:
    backend = NativeFakeBackend(
        [
            _function_call_response(),
            {"id": "interaction-incomplete", "status": "incomplete", "steps": []},
        ]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=2,
        stage_name="verification",
        attach_image=False,
    )

    with pytest.raises(RuntimeError) as captured:
        asyncio.run(runner.run("- [q1] verify"))

    partial = getattr(captured.value, "stage_steps", [])
    assert len(partial) == 1
    assert partial[0].action_type == "tool_call"
    assert partial[0].metadata["function_call_id"] == "call-1"


def test_forced_output_keeps_function_results_as_step_array() -> None:
    forced = _completed_response()
    forced["id"] = "interaction-forced"
    backend = NativeFakeBackend([_function_call_response(), forced])
    tool = RecordingTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        max_output_tokens=16384,
        generation_config={"temperature": 0.0},
        final_output_max_tokens=32768,
        final_output_generation_config={"thinking_level": "minimal"},
    )
    parsed, steps = asyncio.run(runner.run("- [q1] verify"))
    assert parsed is not None
    forced_request = backend.requests[1]
    assert forced_request["previous_interaction_id"] == "interaction-1"
    assert steps[0].metadata["previous_interaction_id"] is None
    assert steps[0].metadata["interaction_id"] == "interaction-1"
    assert steps[-1].metadata["forced_output"] is True
    assert steps[-1].metadata["previous_interaction_id"] == "interaction-1"
    assert steps[-1].metadata["interaction_id"] == "interaction-forced"
    assert forced_request["previous_interaction_id"] == steps[-1].metadata["previous_interaction_id"]
    assert forced_request["tools"] == []
    assert backend.requests[0]["max_tokens"] == 16384
    assert backend.requests[0]["generation_config"] == {
        "temperature": 0.0,
        "tool_choice": "any",
    }
    assert forced_request["max_tokens"] == 32768
    assert forced_request["generation_config"] == {
        "temperature": 0.0,
        "thinking_level": "minimal",
    }
    assert steps[-1].metadata["max_output_tokens"] == 32768
    assert steps[-1].metadata["thinking_level"] == "minimal"
    assert "No more tool turns remain" in forced_request["system_instruction"]
    assert all(
        item["type"] == "function_result"
        for item in forced_request["input_payload"]
    )


def test_forced_output_rejection_gets_one_structured_correction() -> None:
    rejected = _completed_response()
    rejected["id"] = "interaction-rejected"
    corrected = _completed_response()
    corrected["id"] = "interaction-corrected"
    corrected_payload = json.loads(
        corrected["steps"][0]["content"][0]["text"]
    )
    corrected_payload["coverage_complete"] = True
    corrected["steps"][0]["content"][0]["text"] = json.dumps(
        corrected_payload
    )
    backend = NativeFakeBackend(
        [_function_call_response(), rejected, corrected]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        output_validator=lambda parsed, _steps: (
            parsed.coverage_complete,
            "coverage is incomplete",
        ),
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify"))

    assert parsed is not None
    assert parsed.coverage_complete is True
    assert [step.action_type for step in steps] == [
        "tool_call",
        "output_rejected",
        "output",
    ]
    correction_request = backend.requests[2]
    assert correction_request["previous_interaction_id"] == (
        "interaction-rejected"
    )
    assert isinstance(correction_request["input_payload"], str)
    assert "Output rejected: coverage is incomplete" in (
        correction_request["input_payload"]
    )
    assert steps[1].metadata["forced_output_correction"] is False
    assert steps[2].metadata["forced_output_correction"] is True


def test_forced_output_failure_preserves_completed_tool_steps() -> None:
    backend = NativeFakeBackend(
        [
            _function_call_response(),
            {"id": "interaction-incomplete", "status": "incomplete", "steps": []},
        ]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        max_rounds=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        final_output_max_tokens=32768,
        final_output_generation_config={"thinking_level": "minimal"},
    )

    with pytest.raises(RuntimeError) as captured:
        asyncio.run(runner.run("- [q1] verify"))

    partial = getattr(captured.value, "stage_steps", [])
    assert len(partial) == 1
    assert partial[0].action_type == "tool_call"
    assert backend.requests[1]["max_tokens"] == 32768
    assert backend.requests[1]["generation_config"] == {
        "thinking_level": "minimal"
    }


def test_truncated_function_result_keeps_provenance_id() -> None:
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="Investigate.",
        tools=[RecordingTool()],
        output_schema=ToolStageOutput,
        stage_name="verification",
        attach_image=False,
        tool_response_max_chars=1200,
    )
    item = runner._build_native_function_result(
        call_id="call-large",
        tool_name="text_search",
        tool_args={"__question_id": "q1"},
        result=json.dumps(
            {
                "status": "success",
                "queries": [
                    {
                        "query": "source",
                        "summary": "x" * 5000,
                        "evidence": "y" * 5000,
                    }
                ],
            }
        ),
    )

    returned = json.loads(item["result"][0]["text"])
    assert returned["function_call_id"] == "call-large"
    assert returned["question_id"] == "q1"


def test_native_visual_question_runtime_binds_required_crop_args() -> None:
    visual_call = {
        "id": "interaction-visual-1",
        "status": "requires_action",
        "steps": [
            {
                "id": "call-visual-1",
                "type": "function_call",
                "name": "crop_and_inspect",
                "arguments": {"question_id": "q1", "visual_question_id": "vq0"},
            }
        ],
    }
    completed = _completed_response()
    completed["id"] = "interaction-visual-2"
    completed["steps"][0]["content"][0]["text"] = json.dumps(
        {
            "evidence": [
                {
                    "function_call_id": "call-visual-1",
                    "source": "crop_and_inspect",
                    "summary": "The cropped region matches the expected property.",
                    "raw_excerpt": "The cropped region matches the expected property.",
                    "direction": "neutral",
                    "quality": "moderate",
                    "tool_used": "crop_and_inspect",
                    "related_question": "q1",
                }
            ],
            "visual_anomalies": [],
            "authenticity_assessment": "uncertain",
            "key_findings": ["Visual revisit completed."],
            "source_findings": [],
            "visual_evidence": [],
            "world_model": {},
            "question_resolutions": [],
            "coverage_complete": False,
            "unresolved_priority_questions": [],
            "exhausted_priority_questions": [],
            "iteration_count": 1,
        }
    )
    backend = NativeFakeBackend([visual_call, completed])
    tool = VisualInspectTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate.",
        tools=[tool],
        output_schema=ToolStageOutput,
        max_rounds=2,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
        visual_call_validator=lambda _tool_name, _args: "",
    )
    runner._control_steps = [
        StageStep(
            action_type="tool_call",
            tool_name="visit",
            tool_args={"__question_id": "q1"},
            metadata={
                "investigation_state_update": {
                    "created_visual_questions": [
                        {
                            "visual_question_id": "vq0",
                            "claim_id": "claim-q1",
                            "source_evidence_id": "ev-1",
                            "target_bbox": [0.1, 0.2, 0.8, 0.9],
                            "expected_property": "Whether the target image region is consistent with: cited text",
                            "recommended_tools": ["crop_and_inspect"],
                            "status": "pending",
                        }
                    ],
                    "resolved_visual_questions": [],
                }
            },
        )
    ]

    parsed, steps = asyncio.run(runner.run("- [q1] inspect region"))

    assert parsed is not None
    assert steps[0].action_type == "tool_call"
    assert steps[0].tool_args["bbox"] == [0.1, 0.2, 0.8, 0.9]
    assert steps[0].tool_args["source_evidence_id"] == "ev-1"
    assert steps[0].tool_args["focus_question"].startswith("Whether the target image region is consistent with:")
    assert tool.calls == [
        {
            "bbox": [0.1, 0.2, 0.8, 0.9],
            "focus_question": "Whether the target image region is consistent with: cited text",
            "visual_question_id": "vq0",
            "source_evidence_id": "ev-1",
            "expected_property": "Whether the target image region is consistent with: cited text",
        }
    ]


def test_native_schema_allows_visual_question_without_explicit_bound_crop_fields() -> None:
    runner = StageRunner(
        llm=NativeFakeBackend([]),
        system_prompt="",
        tools=[VisualInspectTool()],
        stage_name="verification",
    )
    runner.active_question_ids = ["q1"]

    error = runner._validate_native_tool_args(
        "crop_and_inspect",
        {"question_id": "q1", "visual_question_id": "vq0"},
    )

    assert error == ""


if __name__ == "__main__":
    test_native_function_call_round_trip()
    test_native_output_before_tools_is_rejected()
    test_native_missing_question_id_is_not_silently_assigned()
    test_native_unknown_question_id_is_rejected()
    test_native_executes_all_parallel_calls_and_returns_all_results()
    test_native_tool_schema_hides_server_image_path()
    test_native_invalid_final_schema_is_rejected()
    print("Native Interactions tests passed.")
