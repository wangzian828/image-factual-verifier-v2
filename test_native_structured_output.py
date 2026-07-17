from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from src.orchestrator.stage_runner import InteractionSession, StageRunner
from test_support_models import StructuredJudgmentOutput


class StructuredFakeBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = list(responses)
        self.requests = []

    async def create_interaction(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)

    async def get_response(self, messages, **kwargs):
        raise AssertionError("Gemini no-tool stages must use native structured output")


def response(interaction_id: str, output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": interaction_id,
        "status": "completed",
        "usage": {"total_input_tokens": 10, "total_output_tokens": 20},
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(output)}],
            }
        ],
    }


def valid_judgment() -> Dict[str, Any]:
    return {
        "verdict": "unverifiable",
        "confidence": 0.5,
        "reasoning_chain": "Insufficient grounded evidence.",
        "key_evidence": [],
        "anomalies": [],
        "overall_assessment": "The image cannot be verified.",
    }


def test_no_tool_stage_uses_native_json_schema() -> None:
    backend = StructuredFakeBackend([response("i1", valid_judgment())])
    runner = StageRunner(
        llm=backend,
        system_prompt="Judge the evidence.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        stage_name="judgment",
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("Evidence context"))
    assert parsed is not None
    assert parsed.verdict == "unverifiable"
    assert steps[0].metadata["structured_output"] is True
    request = backend.requests[0]
    assert request["system_instruction"] == "Judge the evidence."
    assert request["response_format"]["type"] == "text"
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["response_format"]["schema"]["additionalProperties"] is False
    assert "verdict" in request["response_format"]["schema"]["properties"]
    assert set(request["response_format"]["schema"]["required"]) == set(
        request["response_format"]["schema"]["properties"]
    )
    assert "$defs" not in json.dumps(request["response_format"]["schema"])
    assert '"default"' not in json.dumps(request["response_format"]["schema"])


def test_stage_generation_config_is_forwarded() -> None:
    backend = StructuredFakeBackend([response("i1", valid_judgment())])
    runner = StageRunner(
        llm=backend,
        system_prompt="Judge the evidence.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        stage_name="judgment",
        attach_image=False,
        max_output_tokens=8192,
        generation_config={"thinking_level": "minimal"},
    )

    parsed, _steps = asyncio.run(runner.run("Evidence context"))

    assert parsed is not None
    assert backend.requests[0]["max_tokens"] == 8192
    assert backend.requests[0]["generation_config"] == {
        "thinking_level": "minimal"
    }


def test_invalid_structured_output_is_corrected_in_same_chain() -> None:
    invalid = valid_judgment()
    invalid["verdict"] = "probably_real"
    backend = StructuredFakeBackend([response("i1", invalid), response("i2", valid_judgment())])
    runner = StageRunner(
        llm=backend,
        system_prompt="Judge the evidence.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        stage_name="judgment",
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("Evidence context"))
    assert parsed is not None
    assert len(steps) == 2
    assert steps[0].action_type == "output_rejected"
    assert backend.requests[1]["previous_interaction_id"] == "i1"
    assert backend.requests[1]["response_format"] == backend.requests[0]["response_format"]
    assert backend.requests[1]["system_instruction"] == "Judge the evidence."


def test_empty_object_cannot_become_default_judgment() -> None:
    backend = StructuredFakeBackend(
        [response("i1", {}), response("i2", valid_judgment())]
    )
    runner = StageRunner(
        llm=backend,
        system_prompt="Judge the evidence.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        stage_name="judgment",
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("Evidence context"))
    assert parsed is not None
    assert parsed.verdict == "unverifiable"
    assert steps[0].action_type == "output_rejected"
    assert len(backend.requests) == 2


def test_shared_interaction_session_attaches_image_once_then_reuses_parent(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
        )
    )
    backend = StructuredFakeBackend(
        [
            response("planning-interaction", valid_judgment()),
            response("checkpoint-interaction", valid_judgment()),
        ]
    )
    session = InteractionSession()
    planning = StageRunner(
        llm=backend,
        system_prompt="Plan from the original image.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        image_path=str(image_path),
        stage_name="planning",
        attach_image=True,
        interaction_session=session,
    )
    checkpoint = StageRunner(
        llm=backend,
        system_prompt="Review the accumulated evidence.",
        tools=[],
        output_schema=StructuredJudgmentOutput,
        max_rounds=1,
        stage_name="checkpoint",
        attach_image=False,
        interaction_session=session,
    )

    _, planning_steps = asyncio.run(
        planning.run("Initial image-grounded planning context")
    )
    asyncio.run(checkpoint.run("Evidence checkpoint context"))

    planning_request, checkpoint_request = backend.requests
    assert planning_request["previous_interaction_id"] is None
    assert isinstance(planning_request["input_payload"], list)
    assert [item["type"] for item in planning_request["input_payload"]] == [
        "text",
        "image",
    ]
    assert checkpoint_request["previous_interaction_id"] == (
        "planning-interaction"
    )
    assert checkpoint_request["input_payload"] == "Evidence checkpoint context"
    snapshot = planning_steps[0].metadata["policy_input"]["input_payload"]
    assert snapshot == [
        {"type": "text", "text": "Initial image-grounded planning context"},
        {
            "type": "image",
            "runtime_image": True,
            "mime_type": "image/png",
        },
    ]
    assert "data" not in json.dumps(snapshot)
    assert session.previous_interaction_id == "checkpoint-interaction"
    assert session.pending_input == []


if __name__ == "__main__":
    test_no_tool_stage_uses_native_json_schema()
    test_invalid_structured_output_is_corrected_in_same_chain()
    print("Native structured output tests passed.")
