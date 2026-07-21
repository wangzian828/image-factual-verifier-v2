from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel, Field

from src.orchestrator.llm_backend import APIBackend, LLMResponse
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.orchestrator.stage_runner import StageRunner, StageStep
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


class OptionalReferenceTool(BaseTool):
    name = "compare_with_reference"
    description = "Compare against one runtime-selected reference."
    parameters = {
        "type": "object",
        "properties": {
            "reference_url": {"type": "string"},
            "focus": {"type": "string"},
        },
        "required": [],
    }

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "reference_url": params["reference_url"]}


class AnswerOutput(BaseModel):
    answer: str


def _tool_response(
    query: str = "actual transport",
    call_id: str = "call-qwen-1",
    question_id: str = "",
) -> LLMResponse:
    arguments = {"query": query}
    if question_id:
        arguments["question_id"] = question_id
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "lookup_fact",
                                "arguments": json.dumps(arguments),
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
        max_output_tokens=512,
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
    assert first["max_tokens"] == 512
    assert second["max_tokens"] == 512
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


def test_qwen_protocol_correction_does_not_consume_action_round() -> None:
    backend = QwenFakeBackend(
        [
            _tool_response(
                query="already attempted",
                call_id="call-duplicate",
                question_id="task-1",
            ),
            _tool_response(
                query="new route",
                call_id="call-corrected",
                question_id="task-1",
            ),
        ]
    )
    tool = LookupTool()
    prior = StageStep(
        action_type="tool_call",
        tool_name="lookup_fact",
        tool_args={
            "query": "already attempted",
            "__question_id": "task-1",
        },
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate the relevant fact.",
        tools=[tool],
        output_schema=AnswerOutput,
        max_rounds=1,
        max_protocol_corrections=1,
        force_tool_each_round=True,
        should_stop=lambda steps: any(
            step.action_type == "tool_call" for step in steps
        ),
        stop_output_factory=lambda: AnswerOutput(answer="action boundary"),
        prior_steps=[prior],
        stage_name="verification",
        question_claims={"task-1": "Verify the transport."},
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("Find a new route."))

    assert parsed == AnswerOutput(answer="action boundary")
    assert [step.action_type for step in steps] == [
        "format_error",
        "tool_call",
        "output",
    ]
    assert steps[0].metadata["duplicate_tool_call"] is True
    assert steps[0].metadata["protocol_corrections_used"] == 1
    assert steps[1].metadata["react_action_turn"] == 1
    assert tool.calls == [{"query": "new route"}]
    assert len(backend.requests) == 2
    correction = backend.requests[1]["messages"][-1]["content"]
    assert "materially different" in correction
    assert "already attempted" in correction
    assert steps[0].metadata["rejection_reason"] == correction


def test_qwen_forced_tool_stage_boundary_hides_tools() -> None:
    backend = QwenFakeBackend(
        [
            _tool_response(
                query="already attempted",
                call_id="call-duplicate",
                question_id="task-1",
            ),
            _output_response(),
        ]
    )
    prior = StageStep(
        action_type="tool_call",
        tool_name="lookup_fact",
        tool_args={
            "query": "already attempted",
            "__question_id": "task-1",
        },
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Investigate the relevant fact.",
        tools=[LookupTool()],
        output_schema=AnswerOutput,
        max_rounds=1,
        max_protocol_corrections=0,
        force_tool_each_round=True,
        prior_steps=[prior],
        stage_name="verification",
        question_claims={"task-1": "Verify the transport."},
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("Find a new route."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert [step.action_type for step in steps] == [
        "format_error",
        "output",
    ]
    assert "tools" not in backend.requests[1]
    assert backend.requests[1]["response_format"]["type"] == "json_schema"


def test_runtime_constrained_optional_tool_selector_becomes_required() -> None:
    reference_url = "https://example.org/pending-reference.jpg"
    runner = StageRunner(
        llm=QwenFakeBackend([]),
        system_prompt="Inspect the selected reference.",
        tools=[OptionalReferenceTool()],
        stage_name="verification",
        question_claims={"task-1": "The image matches the reference."},
        tool_argument_constraints={
            "compare_with_reference": {
                "reference_url": [reference_url],
                "question_id": ["task-1"],
            }
        },
        attach_image=False,
    )
    runner.active_question_ids = ["task-1"]

    schema = runner._build_native_tool_schemas()[0]["parameters"]

    assert schema["properties"]["reference_url"]["enum"] == [reference_url]
    assert "reference_url" in schema["required"]
    assert "question_id" in schema["required"]


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
        generation_config={"enable_thinking": False},
    )

    parsed, steps = asyncio.run(runner.run("Inspect the image."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    request = backend.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["generation_config"] == {"enable_thinking": False}
    assert "minLength" not in json.dumps(request["response_format"])
    assert "maxLength" not in json.dumps(request["response_format"])
    assert "pattern" not in json.dumps(request["response_format"])
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


def test_qwen_schema_keeps_structural_and_numeric_constraints() -> None:
    class RankedOutput(BaseModel):
        label: str = Field(min_length=1, max_length=80)
        priority: int = Field(ge=1, le=3)
        values: List[str] = Field(min_length=1, max_length=3)

    backend = QwenFakeBackend([_output_response()])
    runner = StageRunner(
        llm=backend,
        system_prompt="Return a ranked result.",
        tools=[],
        output_schema=RankedOutput,
        max_rounds=1,
        attach_image=False,
    )
    schema = runner._openai_response_format()["json_schema"]["schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"label", "priority", "values"}
    assert schema["properties"]["values"]["type"] == "array"
    assert schema["properties"]["values"]["minItems"] == 1
    assert schema["properties"]["values"]["maxItems"] == 3
    assert schema["properties"]["priority"]["minimum"] == 1
    assert schema["properties"]["priority"]["maximum"] == 3


def test_qwen_recovers_only_missing_top_level_open_brace() -> None:
    output = '"answer": "ceremonial coach"}'
    raw = {"choices": [{"message": {"role": "assistant", "content": output}}]}
    backend = QwenFakeBackend(
        [
            LLMResponse(
                text=output,
                prompt_tokens=10,
                completion_tokens=5,
                raw=raw,
            )
        ]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Return one answer.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=1,
        attach_image=False,
    )

    parsed, steps = asyncio.run(runner.run("Answer the question."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert steps[0].metadata["format_repair"] == (
        "prepended_missing_top_level_open_brace"
    )
    assert StageRunner._try_parse_bare_json('"answer": 1') is None


def test_qwen_schema_correction_receives_validation_reason(
    tmp_path: Path,
) -> None:
    invalid_text = "{}"
    invalid_raw = {
        "choices": [
            {"message": {"role": "assistant", "content": invalid_text}}
        ]
    }
    backend = QwenFakeBackend(
        [
            LLMResponse(
                text=invalid_text,
                prompt_tokens=10,
                completion_tokens=2,
                raw=invalid_raw,
            ),
            _output_response(),
        ]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Return one answer.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=2,
        attach_image=False,
        runtime_store=CaseRuntimeStore(
            tmp_path,
            case_id="schema-correction",
            attempt_id="attempt",
        ),
    )

    parsed, steps = asyncio.run(runner.run("Answer the question."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert steps[0].action_type == "output_rejected"
    assert "missing required fields" in steps[0].metadata["rejection_reason"]
    correction = backend.requests[1]["messages"][-1]["content"]
    assert "missing required fields" in correction
    assert steps[0].metadata["interaction_lifecycle_kind"] == (
        "standalone_request"
    )
    assert steps[1].metadata["interaction_lifecycle_kind"] == (
        "protocol_correction"
    )
    assert steps[1].metadata["parent_context_request_id"] == steps[0].metadata[
        "context_request_id"
    ]
    correction_manifest = json.loads(
        (
            runner.runtime_store.root
            / "context"
            / f"{steps[1].metadata['context_request_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert correction_manifest["parent_request_id"] == steps[0].metadata[
        "context_request_id"
    ]
    assert correction_manifest["lifecycle_kind"] == "protocol_correction"


def test_qwen_forced_structured_retry_does_not_force_terminal_answer(
    tmp_path: Path,
) -> None:
    backend = QwenFakeBackend([_output_response(), _output_response()])
    validations = 0

    def validator(_parsed: BaseModel, _steps: List[StageStep]) -> tuple[bool, str]:
        nonlocal validations
        validations += 1
        if validations == 1:
            return False, "continue is required while a high-salience route is open"
        return True, ""

    runner = StageRunner(
        llm=backend,
        system_prompt="Return one answer.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=1,
        attach_image=False,
        output_validator=validator,
        runtime_store=CaseRuntimeStore(
            tmp_path,
            case_id="forced-correction",
            attempt_id="attempt",
        ),
    )

    parsed, steps = asyncio.run(runner.run("Answer the question."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    forced_prompt = backend.requests[1]["messages"][-1]["content"]
    assert "last validation attempt" in forced_prompt
    assert "continue is required while a high-salience route is open" in forced_prompt
    assert "no more tool turns" not in forced_prompt.lower()
    assert "tools" not in backend.requests[1]
    assert backend.requests[1]["response_format"]["type"] == "json_schema"
    assert steps[-1].metadata["interaction_lifecycle_kind"] == (
        "protocol_correction"
    )
    assert steps[-1].metadata["parent_context_request_id"] == steps[0].metadata[
        "context_request_id"
    ]


def test_qwen_forced_output_preserves_final_budget_and_thinking_policy() -> None:
    empty_raw = {
        "choices": [
            {"message": {"role": "assistant", "content": ""}}
        ]
    }
    backend = QwenFakeBackend(
        [
            LLMResponse(
                text="",
                prompt_tokens=10,
                completion_tokens=1,
                raw=empty_raw,
            ),
            _output_response(),
        ]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Return one answer.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=1,
        attach_image=False,
        max_output_tokens=128,
        generation_config={"enable_thinking": False},
        final_output_max_tokens=512,
        final_output_generation_config={"enable_thinking": True},
    )

    parsed, steps = asyncio.run(runner.run("Answer the question."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert backend.requests[0]["max_tokens"] == 128
    assert backend.requests[0]["generation_config"] == {
        "enable_thinking": False
    }
    assert backend.requests[1]["max_tokens"] == 512
    assert backend.requests[1]["generation_config"] == {
        "enable_thinking": True
    }
    assert backend.requests[1]["response_format"]["type"] == "json_schema"
    assert steps[-1].metadata["forced_output"] is True


def test_local_qwen_forwards_stage_thinking_switch_to_lmdeploy() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    async def run() -> None:
        backend = APIBackend(
            provider="qwen_local",
            model_name="ifv-qwen3-vl-8b-thinking-smoke",
            max_retries=0,
        )
        backend._shared_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await backend.get_response(
                [{"role": "user", "content": "Return JSON."}],
                response_format={"type": "json_object"},
                generation_config={"enable_thinking": False},
            )
        finally:
            await backend.aclose()
        assert response.text == '{"answer":"ok"}'

    asyncio.run(run())

    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen35_forwards_budget_and_sampling_controls() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    async def run() -> None:
        backend = APIBackend(
            provider="qwen_local",
            model_name="ifv-qwen3.5-9b-vllm",
            max_retries=0,
        )
        backend._shared_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await backend.get_response(
                [{"role": "user", "content": "Return JSON."}],
                response_format={"type": "json_object"},
                generation_config={
                    "enable_thinking": True,
                    "thinking_token_budget": 1024,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                },
            )
        finally:
            await backend.aclose()
        assert response.text == '{"answer":"ok"}'

    asyncio.run(run())

    assert captured["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["thinking_token_budget"] == 1024
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 20
    assert captured["presence_penalty"] == 1.5


def test_qwen_reasoning_is_archived_but_not_reintroduced(
    tmp_path: Path,
) -> None:
    output = {"answer": "ceremonial coach"}
    raw = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning": "private reasoning that must not become context",
                    "content": json.dumps(output),
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 9},
    }
    backend = QwenFakeBackend(
        [
            LLMResponse(
                text=json.dumps(output),
                prompt_tokens=11,
                completion_tokens=9,
                raw=raw,
            )
        ]
    )
    store = CaseRuntimeStore(tmp_path, case_id="reasoning-case", attempt_id="attempt")
    runner = StageRunner(
        llm=backend,
        system_prompt="Return the structured result.",
        tools=[],
        output_schema=AnswerOutput,
        max_rounds=1,
        stage_name="image_account_planning",
        runtime_store=store,
        generation_config={"enable_thinking": True, "thinking_token_budget": 1024},
    )

    parsed, steps = asyncio.run(runner.run("Find the actual transport."))

    assert parsed == AnswerOutput(answer="ceremonial coach")
    assert steps[0].metadata["response_reasoning_chars"] > 0
    descriptor = steps[0].metadata["reasoning_artifact"]
    assert descriptor["media_type"].startswith("text/plain")
    assert store.artifacts.read_bytes(descriptor).decode("utf-8") == raw[
        "choices"
    ][0]["message"]["reasoning"]
    context = json.loads(
        (store.root / "context" / "req-000001.json").read_text(encoding="utf-8")
    )
    assert context["response_reasoning_chars"] == steps[0].metadata[
        "response_reasoning_chars"
    ]
    assert context["reasoning_artifact"]["sha256"] == descriptor["sha256"]
    assert all(
        "private reasoning" not in json.dumps(request["messages"], ensure_ascii=False)
        for request in backend.requests
    )


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


def test_qwen_http_error_preserves_bounded_provider_detail() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                request=request,
                json={"error": {"message": "invalid generated tool call"}},
            )

        backend = APIBackend(
            provider="qwen_local",
            model_name="ifv-qwen3-vl-8b-thinking-smoke",
            max_retries=0,
        )
        backend._shared_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(RuntimeError, match="invalid generated tool call"):
                await backend.get_response(
                    [{"role": "user", "content": "Use one tool."}],
                    tools=[LookupTool().schema],
                )
        finally:
            await backend.aclose()

    import pytest

    asyncio.run(run())
