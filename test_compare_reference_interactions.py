from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.integrations.gemini import RUNTIME_METRICS_KEY
from src.orchestrator.evidence_semantics import derive_edit_evidence_summary
from src.tools.compare_reference import (
    COMPARE_RESPONSE_SCHEMA,
    DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS,
    SYSTEM_INSTRUCTION,
    CompareWithReferenceTool,
)


RUNTIME_METRICS = {
    "llm_api_calls": 1,
    "tokens": {"prompt": 101, "completion": 17, "thought": 0},
}


def valid_comparison() -> dict[str, Any]:
    return {
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": True,
        "likely_different_original_capture": False,
        "differences": [
            {
                "region": "outer edges",
                "description": "The current image has a tighter crop.",
                "type": "crop",
                "significance": "low",
            }
        ],
        "overall_observation": "The images are near-duplicates with a benign crop.",
        "confidence": 0.94,
    }


def _reference_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color="gray").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _payload(output: dict[str, Any]) -> dict[str, Any]:
    return {**output, RUNTIME_METRICS_KEY: RUNTIME_METRICS}


class FakeVisionClient:
    def __init__(
        self,
        payload: Any,
        *,
        provider: str = "qwen_local",
    ) -> None:
        self.payload = payload
        self.provider = provider
        self.requests: list[dict[str, Any]] = []

    def create_images_json(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def make_tool(tmp_path: Path, client: Any) -> CompareWithReferenceTool:
    image_path = tmp_path / "current.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(
        client=client,
        provider=getattr(client, "provider", "qwen_local"),
        model_name="served-teacher-model",
        image_path=str(image_path),
    )

    async def fake_download(_url: str) -> str:
        return _reference_data_url()

    tool._download_reference = fake_download
    return tool


@pytest.mark.parametrize("provider", ["gemini", "qwen_local"])
def test_compare_uses_provider_neutral_multi_image_contract(
    tmp_path: Path,
    provider: str,
) -> None:
    client = FakeVisionClient(_payload(valid_comparison()), provider=provider)
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async(
            {
                "reference_url": "https://example.test/reference.jpg",
                "focus": "banner text",
            }
        )
    )

    assert result["status"] == "success"
    assert result["confidence"] == 0.94
    assert result[RUNTIME_METRICS_KEY] == RUNTIME_METRICS
    assert len(client.requests) == 1

    request = client.requests[0]
    assert request["system_prompt"] == SYSTEM_INSTRUCTION
    assert "banner text" in request["user_text"]
    assert len(request["image_inputs"]) == 2
    assert all(
        image_input.startswith("data:image/jpeg;base64,")
        for image_input in request["image_inputs"]
    )
    assert request["max_tokens"] == DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS
    assert request["model_name"] == "served-teacher-model"
    assert request["temperature"] == 0.0
    assert request["response_schema"] == COMPARE_RESPONSE_SCHEMA
    assert "edit_evidence_present" not in request["response_schema"]["properties"]
    assert "edit_evidence_strength" not in request["response_schema"]["properties"]


def test_compare_short_circuits_pixel_identical_images_without_vlm(
    tmp_path: Path,
) -> None:
    client = FakeVisionClient(
        AssertionError("pixel-identical images must not call the VLM")
    )
    image_path = tmp_path / "current.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    reference_data = image_path.read_bytes()

    tool = CompareWithReferenceTool(
        client=client,
        image_path=str(image_path),
    )

    async def same_download(_url: str) -> dict[str, Any]:
        return {
            "data_url": (
                "data:image/png;base64,"
                + base64.b64encode(reference_data).decode("ascii")
            ),
            "resolved_url": "https://example.test/reference.png",
            "download_method": "direct",
            "attempted_urls": ["https://example.test/reference.png"],
        }

    tool._download_reference = same_download
    result = asyncio.run(
        tool.call_async(
            {"reference_url": "https://example.test/reference.png"}
        )
    )

    assert result["status"] == "success"
    assert result["comparison_method"] == "deterministic_exact_pixels"
    assert result["same_capture_or_near_duplicate"] is True
    assert result["confidence"] == 1.0
    assert result[RUNTIME_METRICS_KEY] == {}
    assert client.requests == []


def test_compare_propagates_vlm_failure_as_tool_error(tmp_path: Path) -> None:
    client = FakeVisionClient(RuntimeError("endpoint unavailable"))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "endpoint unavailable" in result["error"]
    assert len(client.requests) == 1


def test_compare_rejects_invalid_output_contract(tmp_path: Path) -> None:
    invalid = deepcopy(valid_comparison())
    invalid["confidence"] = "0.94"
    client = FakeVisionClient(_payload(invalid))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "confidence' must be a number" in result["error"]
    assert result[RUNTIME_METRICS_KEY]["llm_api_calls"] == 1


def test_compare_repairs_edit_present_without_typed_difference(tmp_path: Path) -> None:
    invalid = deepcopy(valid_comparison())
    invalid["edit_evidence_present"] = True
    invalid["edit_evidence_strength"] = "strong"
    client = FakeVisionClient(_payload(invalid))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "success"
    assert result["edit_evidence_present"] is False
    assert result["edit_evidence_strength"] == "none"
    assert result["contract_repairs"] == [
        "edit_evidence_present_derived_from_difference_types",
        "edit_evidence_strength_derived_from_edit_difference_significance",
    ]


def test_compare_repairs_redundant_edit_summary_without_failing(tmp_path: Path) -> None:
    invalid = deepcopy(valid_comparison())
    invalid["edit_evidence_present"] = False
    invalid["edit_evidence_strength"] = "moderate"
    client = FakeVisionClient(_payload(invalid))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "success"
    assert result["edit_evidence_present"] is False
    assert result["edit_evidence_strength"] == "none"
    assert result["contract_repairs"] == [
        "legacy_edit_evidence_present_ignored",
        "edit_evidence_strength_derived_from_edit_difference_significance",
    ]
    assert result["raw_edit_evidence_summary"] == {
        "edit_evidence_present": False,
        "edit_evidence_strength": "moderate",
    }


def test_compare_derives_edit_flag_from_difference_type(tmp_path: Path) -> None:
    output = valid_comparison()
    output["differences"] = [
        {
            "region": "top banner",
            "description": "A new banner appears only in the current image.",
            "type": "addition",
            "significance": "high",
        }
    ]
    client = FakeVisionClient(_payload(output))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "success"
    assert result["edit_evidence_present"] is True
    assert result["edit_evidence_strength"] == "strong"
    assert result["differences"][0]["is_edit_evidence"] is True


def test_compare_rejects_malformed_legacy_edit_summary_without_keyerror(
    tmp_path: Path,
) -> None:
    output = deepcopy(valid_comparison())
    output["edit_evidence_present"] = "false"
    client = FakeVisionClient(_payload(output))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "legacy comparison output field 'edit_evidence_present'" in result["error"]
    assert "KeyError" not in result["error"]


def test_edit_summary_is_derived_for_replay_even_when_legacy_fields_are_absent() -> None:
    assert derive_edit_evidence_summary(
        [
            {
                "type": "addition",
                "significance": "high",
            }
        ]
    ) == (True, "strong")
    assert derive_edit_evidence_summary(
        [
            {
                "type": "crop",
                "significance": "high",
            }
        ]
    ) == (False, "none")
    assert derive_edit_evidence_summary(
        [{"type": "modification"}]
    ) == (True, "none")


def test_compare_accepts_an_unrelated_reference_image(tmp_path: Path) -> None:
    output = valid_comparison()
    output.update(
        {
            "same_subject_or_scene": False,
            "same_capture_or_near_duplicate": False,
            "likely_different_original_capture": True,
            "differences": [
                {
                    "region": "entire image",
                    "description": (
                        "The reference depicts a different subject and scene."
                    ),
                    "type": "unrelated_content",
                    "significance": "high",
                }
            ],
            "overall_observation": (
                "The reference image is unrelated to the current image."
            ),
        }
    )
    client = FakeVisionClient(_payload(output))
    tool = make_tool(tmp_path, client)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/unrelated.jpg"})
    )

    assert result["status"] == "success"
    assert result["same_subject_or_scene"] is False
    assert result["likely_different_original_capture"] is True


def test_compare_requires_structured_multi_image_client(tmp_path: Path) -> None:
    class UnsupportedClient:
        provider = "qwen_local"

    image_path = tmp_path / "current.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(
        client=UnsupportedClient(),
        image_path=str(image_path),
    )

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result == {
        "status": "error",
        "error": "VLM client does not support structured multi-image comparison.",
    }


def test_reference_download_builds_transform_and_page_image_fallbacks() -> None:
    variants = CompareWithReferenceTool._reference_url_variants(
        "https://example.org/photo.jpg?width=1600&quality=70"
    )
    extracted = CompareWithReferenceTool._extract_page_image_urls(
        (
            '<html><head><meta property="og:image" '
            'content="/media/original.jpg"></head></html>'
        ),
        base_url="https://example.org/article",
    )

    assert variants == (
        "https://example.org/photo.jpg?width=1600&quality=70",
        "https://example.org/photo.jpg",
    )
    assert extracted == ("https://example.org/media/original.jpg",)


def test_reference_download_cache_reuses_success_without_network() -> None:
    tool = CompareWithReferenceTool()
    tool._put_reference_cache(
        "https://example.org/reference.jpg\n",
        {
            "data_url": "data:image/jpeg;base64,cmVm",
            "resolved_url": "https://example.org/reference.jpg",
            "download_method": "direct",
            "attempted_urls": ["https://example.org/reference.jpg"],
        },
    )

    result = asyncio.run(
        tool._download_reference("https://example.org/reference.jpg")
    )

    assert result["cache_hit"] is True
    assert result["data_url"] == "data:image/jpeg;base64,cmVm"
    assert tool._download_subcalls(result) == []
