from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from src.tools.visual_anomaly import VisualAnomalyTool


VALID_RESPONSE = {
    "anomalies": [
        {
            "name": "Broken hand geometry",
            "region": "lower-left hand",
            "phenomenon": "Two fingers merge into one continuous shape.",
            "reasoning": "The visible finger topology is anatomically implausible.",
            "severity": 82,
            "type": "ai_generation",
            "entities_involved": ["left hand", "fingers"],
        }
    ],
    "overall_authenticity": "likely_ai",
    "confidence": 0.91,
    "notes": "The anomaly is localized and clearly visible.",
}


def _interaction_payload(output: Any) -> dict[str, Any]:
    text = output if isinstance(output, str) else json.dumps(output)
    return {
        "id": "interaction-visual-anomaly",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


class FakeInteractionsBackend:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.legacy_calls = 0

    async def create_interaction(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return self.payload

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        self.legacy_calls += 1
        raise AssertionError("visual anomaly analysis must not call get_response")


def _run_tool(
    tmp_path,
    backend: FakeInteractionsBackend,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"test-image-bytes")
    tool = VisualAnomalyTool(vlm_backend=backend, image_path=str(image_path))
    return asyncio.run(tool.call_async(params or {}))


def test_visual_anomaly_uses_interactions_with_exact_structured_schema(tmp_path) -> None:
    backend = FakeInteractionsBackend(_interaction_payload(VALID_RESPONSE))

    result = _run_tool(
        tmp_path,
        backend,
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
    }
    assert backend.legacy_calls == 0
    assert len(backend.calls) == 1

    request = backend.calls[0]
    assert request["store"] is True
    assert request["background"] is False
    assert request["max_tokens"] == 8192
    assert request["temperature"] == 0.0
    assert request["generation_config"] == {"thinking_level": "minimal"}
    assert request["system_instruction"].endswith(
        "return only one JSON object without markdown or commentary."
    )
    assert request["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "anomalies": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 200,
                            },
                            "region": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 200,
                            },
                            "phenomenon": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1200,
                            },
                            "reasoning": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1200,
                            },
                            "severity": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "type": {
                                "type": "string",
                                "enum": [
                                    "ai_generation",
                                    "manipulation",
                                    "physical_inconsistency",
                                    "logical_inconsistency",
                                ],
                            },
                            "entities_involved": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 12,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 160,
                                },
                            },
                        },
                        "required": [
                            "name",
                            "region",
                            "phenomenon",
                            "reasoning",
                            "severity",
                            "type",
                            "entities_involved",
                        ],
                    },
                },
                "overall_authenticity": {
                    "type": "string",
                    "enum": [
                        "authentic",
                        "likely_ai",
                        "likely_manipulated",
                        "uncertain",
                    ],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "notes": {"type": "string", "maxLength": 1200},
            },
            "required": [
                "anomalies",
                "overall_authenticity",
                "confidence",
                "notes",
            ],
        },
    }

    interaction_input = request["input_payload"]
    assert [item["type"] for item in interaction_input] == ["text", "image"]
    assert "inspect the left hand" in interaction_input[0]["text"]
    assert interaction_input[1]["mime_type"] == "image/png"
    assert interaction_input[1]["data"]
    assert "image_url" not in interaction_input[1]


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
    backend = FakeInteractionsBackend(_interaction_payload(invalid))

    result = _run_tool(tmp_path, backend)

    assert result["status"] == "error"
    assert result["error"]
    assert "failed schema validation" in result["error"]
    assert expected_error in result["error"]
    assert len(backend.calls) == 1
    assert backend.legacy_calls == 0


def test_visual_anomaly_does_not_extract_json_from_markdown(tmp_path) -> None:
    fenced = f"```json\n{json.dumps(VALID_RESPONSE)}\n```"
    backend = FakeInteractionsBackend(_interaction_payload(fenced))

    result = _run_tool(tmp_path, backend)

    assert result["status"] == "error"
    assert result["error"]
    assert "failed schema validation" in result["error"]
    assert len(backend.calls) == 1
    assert backend.legacy_calls == 0


def test_visual_anomaly_interactions_failure_has_no_legacy_fallback(tmp_path) -> None:
    backend = FakeInteractionsBackend(error=RuntimeError("interaction unavailable"))

    result = _run_tool(tmp_path, backend)

    assert result == {
        "status": "error",
        "error": (
            "Visual anomaly Interactions request failed: "
            "RuntimeError: interaction unavailable"
        ),
    }
    assert len(backend.calls) == 1
    assert backend.legacy_calls == 0
