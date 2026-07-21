from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from src.orchestrator.llm_backend import APIBackend, LLMResponse
from src.orchestrator.stage_runner import StageRunner
from src.tools.base import BaseTool


class QwenFakeBackend:
    provider = "qwen_local"
    wire_api = "chat_completions"
    model_name = "ifv-qwen3-vl-8b-thinking-smoke"

    def __init__(self, responses: List[LLMResponse]):
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def get_response(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        self.requests.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected Qwen request")
        return self.responses.pop(0)


class LookupTool(BaseTool):
    name = "lookup_fact"
    description = "Look up one fact."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        return {"status": "success", "answer": "ceremonial coach"}


class AnswerOutput(BaseModel):
    answer: str


def _tool_response() -> LLMResponse:
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-qwen-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_fact",
                                "arguments": json.dumps(
                                    {"query": "actual transport"}
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }
    return LLMResponse(
        text=APIBackend._extract_chat_completion_text(raw["choices"][0]),
        prompt_tokens=30,
        completion_tokens=6,
        raw=raw,
    )


def _output_response() -> LLMResponse:
    output = {"answer": "ceremonial coach"}
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(output),
                }
            }
        ]
    }
    return LLMResponse(
        text=json.dumps(output),
        prompt_tokens=55,
        completion_tokens=8,
        raw=raw,
    )


def test_qwen_native_function_round_trip_uses_tool_role() -> None:
    backend = QwenFakeBackend([_tool_response(), _output_response()])
    tool = LookupTool()
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate the relevant fact.",
        tools=[tool],
        output_schema=AnswerOutput,
        max_rounds=2,
        min_tool_calls=1,
        stage_name="planning",
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("Find the actual transport."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert [step.action_type for step in steps] == ["tool_call", "output"]
    assert tool.calls == [{"query": "actual transport"}]
    assert steps[0].metadata["function_call_id"] == "call-qwen-1"
    assert steps[0].metadata["policy_action"] == {
        "type": "tool_call",
        "name": "lookup_fact",
        "arguments": {"query": "actual transport"},
    }
    assert steps[1].metadata["policy_action"] == {
        "answer": "ceremonial coach"
    }

    first, second = backend.requests
    assert first["tool_choice"] == "required"
    assert first["tools"][0]["name"] == "lookup_fact"
    assert first["tools"][0]["parameters"]["additionalProperties"] is False
    assert first["messages"][0]["role"] == "system"
    assert "<tool_call>" not in first["messages"][0]["content"]
    assert second["tool_choice"] == "auto"
    assistant = next(
        item for item in second["messages"] if item["role"] == "assistant"
        and item.get("tool_calls")
    )
    tool_result = next(
        item for item in second["messages"] if item["role"] == "tool"
    )
    assert assistant["tool_calls"][0]["id"] == "call-qwen-1"
    assert tool_result["tool_call_id"] == "call-qwen-1"
    assert json.loads(tool_result["content"])["function_call_id"] == "call-qwen-1"


def test_qwen_no_tool_stage_uses_json_schema_and_redacts_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
        )
    )
    backend = QwenFakeBackend([_output_response()])
    runner = StageRunner(
        llm=backend,
        system_prompt="Return the image account.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=1,
        image_path=str(image_path),
        stage_name="image_account_planning",
        attach_image=True,
    )

    parsed, steps = asyncio.run(runner.run("Inspect the image."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    request = backend.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "answer" in request["response_format"]["json_schema"]["schema"][
        "properties"
    ]
    wire_payload = json.dumps(request["messages"])
    assert "data:image/" in wire_payload
    snapshot = steps[0].metadata["policy_input"]
    assert snapshot["input_payload"][0]["content"][0] == {
        "type": "image_url",
        "runtime_image": True,
    }
    assert "data:image/" not in json.dumps(snapshot)


def test_qwen_native_tool_call_wins_over_reasoning_and_blank_content() -> None:
    choice = {
        "message": {
            "role": "assistant",
            "content": "\n\n",
            "reasoning_content": "I should call the tool.",
            "tool_calls": [
                {
                    "id": "call-real-server",
                    "type": "function",
                    "function": {
                        "name": "lookup_fact",
                        "arguments": '{"query":"actual transport"}',
                    },
                }
            ],
        }
    }

    assert APIBackend._extract_chat_completion_text(choice) == (
        '<tool_call>{"name": "lookup_fact", '
        '"arguments": {"query": "actual transport"}}</tool_call>'
    )


def test_qwen_reasoning_fallback_is_limited_to_schema_bound_requests() -> None:
    choice = {
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": '{"answer":"coach"}',
            "tool_calls": None,
        }
    }

    assert APIBackend._extract_chat_completion_text(choice) == ""
    assert APIBackend._extract_chat_completion_text(
        choice,
        allow_reasoning_fallback=True,
    ) == '{"answer":"coach"}'
