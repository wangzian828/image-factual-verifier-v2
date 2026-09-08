from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from src.integrations.gemini import RUNTIME_METRICS_KEY
from src.tools.visual_anomaly import (
    VISUAL_ANOMALY_MAX_OUTPUT_TOKENS,
    VISUAL_ANOMALY_RESPONSE_SCHEMA,
    VISUAL_ANOMALY_SYSTEM_INSTRUCTION,
    VisualAnomalyTool,
)


VALID_RESPONSE = {
    "anomalies": [
        {
            "name": "Broken hand geometry",
            "region": "lower-left hand",
            "phenomenon": "Two fingers merge into one continuous shape.",
            "reasoning": "The visible finger topology is anatomically implausible.",
            "severity": 82,
            "type": "relation_mismatch",
            "entities_involved": ["left hand", "fingers"],
        }
    ],
    "target_relation_status": "not_observed",
    "confidence": 0.91,
    "notes": "The anomaly is localized and clearly visible.",
}
RUNTIME_METRICS = {
    "llm_api_calls": 1,
    "tokens": {"prompt": 91, "completion": 23, "thought": 0},
}


class FakeVisionClient:
    def __init__(
        self,
        payload: Any = None,
        error: Exception | None = None,
        *,
        provider: str = "qwen_local",
    ) -> None:
        self.payload = payload
        self.error = error
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def create_image_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.payload


def _payload(output: dict[str, Any]) -> dict[str, Any]:
    return {**output, RUNTIME_METRICS_KEY: RUNTIME_METRICS}


def _run_tool(
    tmp_path,
    client: Any,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)
    tool = VisualAnomalyTool(
        client=client,
        provider=getattr(client, "provider", "qwen_local"),
        model_name="served-teacher-model",
        image_path=str(image_path),
    )
    return asyncio.run(tool.call_async(params or {}))


@pytest.mark.parametrize("provider", ["gemini", "qwen_local"])
def test_visual_anomaly_uses_provider_neutral_structured_vision(
    tmp_path,
    provider: str,
) -> None:
    client = FakeVisionClient(_payload(VALID_RESPONSE), provider=provider)

    result = _run_tool(
        tmp_path,
        client,
        {
            "focus_areas": [" inspect the left hand "],
            "context": "The hand is central to the image claim.",
            "check_type": "ai_generation",
        },
    )

    assert result == {
        "status": "success",
        "focus_areas": ["inspect the left hand"],
        **VALID_RESPONSE,
        RUNTIME_METRICS_KEY: RUNTIME_METRICS,
    }
    assert len(client.calls) == 1

    request = client.calls[0]
    assert request["system_prompt"] == VISUAL_ANOMALY_SYSTEM_INSTRUCTION
    assert "inspect the left hand" in request["user_text"]
    assert request["image_input"].endswith("input.png")
    assert request["max_tokens"] == VISUAL_ANOMALY_MAX_OUTPUT_TOKENS
    assert request["model_name"] == "served-teacher-model"
    assert request["temperature"] == 0.0
    assert request["response_schema"] == VISUAL_ANOMALY_RESPONSE_SCHEMA


def _without_required_field(value: dict[str, Any]) -> None:
    del value["anomalies"][0]["reasoning"]


def _with_invalid_anomaly_type(value: dict[str, Any]) -> None:
    value["anomalies"][0]["type"] = "unknown"


def _with_out_of_range_severity(value: dict[str, Any]) -> None:
    value["anomalies"][0]["severity"] = 101


def _with_out_of_range_confidence(value: dict[str, Any]) -> None:
    value["confidence"] = -0.01


def _with_internal_field_name(value: dict[str, Any]) -> None:
    value["anomalies"][0]["anomaly_type"] = value["anomalies"][0].pop("type")


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (_without_required_field, "anomalies.0.reasoning"),
        (_with_invalid_anomaly_type, "anomalies.0.type"),
        (_with_out_of_range_severity, "anomalies.0.severity"),
        (_with_out_of_range_confidence, "confidence"),
        (_with_internal_field_name, "anomalies.0.type"),
    ],
)
def test_visual_anomaly_rejects_invalid_structured_output(
    tmp_path,
    mutate,
    expected_error: str,
) -> None:
    invalid = copy.deepcopy(VALID_RESPONSE)
    mutate(invalid)
    client = FakeVisionClient(_payload(invalid))

    result = _run_tool(tmp_path, client)

    assert result["status"] == "error"
    assert "failed schema validation" in result["error"]
    assert expected_error in result["error"]
    assert result[RUNTIME_METRICS_KEY]["llm_api_calls"] == 1
    assert len(client.calls) == 1


def test_visual_anomaly_rejects_non_object_client_response(tmp_path) -> None:
    client = FakeVisionClient("not-an-object")

    result = _run_tool(tmp_path, client)

    assert result == {
        "status": "error",
        "error": (
            "Visual anomaly request failed: "
            "TypeError: VLM response must be a JSON object."
        ),
    }
    assert len(client.calls) == 1


def test_visual_anomaly_request_failure_has_no_fallback(tmp_path) -> None:
    client = FakeVisionClient(error=RuntimeError("vision endpoint unavailable"))

    result = _run_tool(tmp_path, client)

    assert result == {
        "status": "error",
        "error": (
            "Visual anomaly request failed: "
            "RuntimeError: vision endpoint unavailable"
        ),
    }
    assert len(client.calls) == 1


def test_visual_anomaly_requires_structured_single_image_client(tmp_path) -> None:
    class UnsupportedClient:
        provider = "qwen_local"

    result = _run_tool(tmp_path, UnsupportedClient())

    assert result == {
        "status": "error",
        "error": "VLM client does not support structured single-image analysis.",
    }
