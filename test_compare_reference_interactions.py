from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from src.integrations.gemini import normalize_json_schema
from src.tools.compare_reference import (
    COMPARE_RESPONSE_SCHEMA,
    CompareWithReferenceTool,
)


def valid_comparison() -> dict:
    return {
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": True,
        "likely_different_original_capture": False,
        "edit_evidence_present": False,
        "edit_evidence_strength": "none",
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


def interaction(output: object, *, status: str = "completed") -> dict:
    return {
        "id": "compare-interaction-1",
        "status": status,
        "usage": {
            "total_input_tokens": 101,
            "total_output_tokens": 17,
            "total_thought_tokens": 0,
        },
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(output)}],
            }
        ],
    }


class FakeBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, payload: dict | Exception) -> None:
        self.payload = payload
        self.requests = []
        self.legacy_called = False

    async def create_interaction(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def get_response(self, *_args, **_kwargs):
        self.legacy_called = True
        raise AssertionError("get_response must not be used")


def make_tool(tmp_path: Path, backend: FakeBackend) -> CompareWithReferenceTool:
    image_path = tmp_path / "current.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(vlm_backend=backend, image_path=str(image_path))

    async def fake_download(_url: str) -> str:
        return "data:image/jpeg;base64,cmVmZXJlbmNl"

    tool._download_reference = fake_download
    return tool


def test_compare_uses_two_interactions_content_images_and_exact_schema(tmp_path: Path) -> None:
    backend = FakeBackend(interaction(valid_comparison()))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async(
            {"reference_url": "https://example.test/reference.jpg", "focus": "banner text"}
        )
    )

    assert result["status"] == "success"
    assert result["confidence"] == 0.94
    assert result["__runtime_metrics__"] == {
        "llm_api_calls": 1,
        "tokens": {"prompt": 101, "completion": 17, "thought": 0},
    }
    assert backend.legacy_called is False
    assert len(backend.requests) == 1

    request = backend.requests[0]
    assert request["store"] is True
    assert request["max_tokens"] == 8192
    assert request["temperature"] == 0.0
    assert request["generation_config"] == {"thinking_level": "low"}
    assert request["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": normalize_json_schema(
            COMPARE_RESPONSE_SCHEMA,
            require_all_properties=True,
        ),
    }
    assert [item["type"] for item in request["input_payload"]] == [
        "text",
        "image",
        "image",
    ]
    reference_image, current_image = request["input_payload"][1:]
    assert reference_image == {
        "type": "image",
        "mime_type": "image/jpeg",
        "data": "cmVmZXJlbmNl",
    }
    assert current_image["type"] == "image"
    assert current_image["mime_type"] == "image/jpeg"
    assert current_image["data"]
    assert "image_url" not in json.dumps(request["input_payload"])


def test_compare_short_circuits_pixel_identical_images_without_vlm(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        AssertionError("pixel-identical images must not call the VLM")
    )
    image_path = tmp_path / "current.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    reference_data = image_path.read_bytes()
    import base64

    tool = CompareWithReferenceTool(
        vlm_backend=backend,
        image_path=str(image_path),
    )

    async def same_download(_url: str) -> dict:
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
    assert result["__runtime_metrics__"] == {}
    assert backend.requests == []


def test_compare_propagates_interactions_failure_as_tool_error(tmp_path: Path) -> None:
    backend = FakeBackend(RuntimeError("endpoint unavailable"))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "endpoint unavailable" in result["error"]
    assert backend.legacy_called is False
    assert len(backend.requests) == 1


def test_compare_rejects_invalid_output_contract(tmp_path: Path) -> None:
    invalid = deepcopy(valid_comparison())
    invalid["confidence"] = "0.94"
    backend = FakeBackend(interaction(invalid))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "confidence' must be a number" in result["error"]
    assert result["__runtime_metrics__"]["llm_api_calls"] == 1


def test_compare_rejects_inconsistent_edit_summary(tmp_path: Path) -> None:
    invalid = deepcopy(valid_comparison())
    invalid["edit_evidence_present"] = True
    invalid["edit_evidence_strength"] = "strong"
    backend = FakeBackend(interaction(invalid))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "error"
    assert "must match differences marked as edit evidence" in result["error"]


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
    output["edit_evidence_present"] = True
    output["edit_evidence_strength"] = "strong"
    backend = FakeBackend(interaction(output))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result["status"] == "success"
    assert result["differences"][0]["is_edit_evidence"] is True


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
    backend = FakeBackend(interaction(output))
    tool = make_tool(tmp_path, backend)

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/unrelated.jpg"})
    )

    assert result["status"] == "success"
    assert result["same_subject_or_scene"] is False
    assert result["likely_different_original_capture"] is True


def test_compare_requires_create_interaction_without_legacy_fallback(tmp_path: Path) -> None:
    class LegacyOnlyBackend:
        async def get_response(self, *_args, **_kwargs):
            raise AssertionError("legacy backend must not be called")

    image_path = tmp_path / "current.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(
        vlm_backend=LegacyOnlyBackend(),
        image_path=str(image_path),
    )

    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/reference.jpg"})
    )

    assert result == {
        "status": "error",
        "error": "Gemini Interactions backend not configured for image comparison.",
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
