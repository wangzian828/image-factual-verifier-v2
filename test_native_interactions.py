from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from src.orchestrator.stage_runner import StageRunner
from src.orchestrator.state import VerificationResult
from src.tools.base import BaseTool


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


class NativeStructuredOutput(BaseModel):
    value: str


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
        output_schema=VerificationResult,
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
    assert second_request["response_format"]["mime_type"] == "application/json"
    assert "evidence" in second_request["response_format"]["schema"]["properties"]

    function_result = second_request["input_payload"][0]
    assert function_result["type"] == "function_result"
    assert function_result["name"] == "text_search"
    assert function_result["call_id"] == "call-1"
    returned = json.loads(function_result["result"][0]["text"])
    assert returned["question_id"] == "q1"
    assert returned["function_call_id"] == "call-1"
    assert "directly answers" in json.dumps(returned["result"])
    assert "is_error" not in function_result


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
        output_schema=VerificationResult,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q1] verify the source"))

    assert parsed is not None
    assert steps[0].action_type == "output_rejected"
    assert "at least 1 tool calls" in steps[0].metadata["rejection_reason"]
    assert steps[1].action_type == "tool_call"
    assert backend.requests[1]["previous_interaction_id"] == "interaction-0"


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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
        max_rounds=3,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("- [q0] first\n- [q1] second"))

    assert parsed is not None
    assert [step.action_type for step in steps[:2]] == ["tool_call", "tool_call"]
    assert all(step.metadata["function_call_count"] == 2 for step in steps[:2])
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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
    assert backend.requests[0]["generation_config"] == {"temperature": 0.0}
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
        output_schema=VerificationResult,
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
        output_schema=VerificationResult,
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


if __name__ == "__main__":
    test_native_function_call_round_trip()
    test_native_output_before_tools_is_rejected()
    test_native_missing_question_id_is_not_silently_assigned()
    test_native_unknown_question_id_is_rejected()
    test_native_executes_all_parallel_calls_and_returns_all_results()
    test_native_tool_schema_hides_server_image_path()
    test_native_invalid_final_schema_is_rejected()
    print("Native Interactions tests passed.")
