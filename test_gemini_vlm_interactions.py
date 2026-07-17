from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import httpx
import pytest
from PIL import Image

from src.integrations.vlm.factory import build_vlm_client


class FakeAsyncClient:
    def __init__(self, requests, output=None):
        self.requests = requests
        self.output = output or {"scene": "test", "objects": ["square"]}

    async def post(self, url, *, headers, json):
        self.requests.append({"url": url, "headers": headers, "body": json})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "interaction-vlm",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 83,
                    "total_output_tokens": 11,
                    "total_thought_tokens": 0,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": json_module(self.output)}],
                    }
                ],
            },
        )

    async def aclose(self):
        return None


def json_module(value):
    return json.dumps(value)


def test_gemini_vlm_uses_interactions(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    requests = []
    fake_client = FakeAsyncClient(requests)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)

    tmp_dir = tempfile.mkdtemp(prefix="gemini-vlm-test-")
    image_path = Path(tmp_dir) / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    try:
        client = build_vlm_client(provider="gemini", model_name="gemini-test")
        result = client.create_image_json(
            system_prompt="Describe the image as JSON.",
            user_text="Inspect it.",
            image_input=str(image_path),
            max_tokens=128,
        )
    finally:
        image_path.unlink(missing_ok=True)
        os.rmdir(tmp_dir)

    assert result == {
        "scene": "test",
        "objects": ["square"],
        "__runtime_metrics__": {
            "llm_api_calls": 1,
            "tokens": {"prompt": 83, "completion": 11, "thought": 0},
        },
    }
    assert len(requests) == 1
    request = requests[0]
    assert request["url"].endswith("/v1beta/interactions")
    assert request["headers"]["x-goog-api-key"] == "test-key"
    assert request["body"]["model"] == "gemini-test"
    assert request["body"]["store"] is True
    assert request["body"]["stream"] is False
    assert request["body"]["generation_config"]["max_output_tokens"] == 8192
    assert request["body"]["generation_config"]["thinking_level"] == "minimal"
    assert request["body"]["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "object"},
    }
    assert request["body"]["input"][0] == {"type": "text", "text": "Inspect it."}
    image_item = request["body"]["input"][1]
    assert image_item["type"] == "image"
    assert image_item["mime_type"] == "image/png"
    assert image_item["data"]


def test_gemini_vlm_uses_custom_schema_and_uri(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    requests = []
    fake_client = FakeAsyncClient(requests)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
    schema = {
        "type": "object",
        "properties": {"scene": {"type": "string"}},
        "required": ["scene"],
        "additionalProperties": False,
    }

    client = build_vlm_client(provider="gemini", model_name="gemini-test")
    result = client.create_image_json(
        system_prompt="Describe the image as JSON.",
        user_text="Inspect it.",
        image_input="https://example.test/input.png",
        max_tokens=128,
        response_schema=schema,
    )

    assert result["scene"] == "test"
    assert result["objects"] == ["square"]
    request = requests[0]["body"]
    assert request["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }
    assert request["input"][1] == {
        "type": "image",
        "uri": "https://example.test/input.png",
    }
    assert "image_url" not in request["input"][1]


def test_gemini_vlm_supports_ordered_multi_view_input(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    requests = []
    fake_client = FakeAsyncClient(requests)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)

    client = build_vlm_client(provider="gemini", model_name="gemini-test")
    result = client.create_images_json(
        system_prompt="Compare the supplied views as JSON.",
        user_text="The first image is the original.",
        image_inputs=[
            "https://example.test/original.png",
            "https://example.test/detail.png",
        ],
        max_tokens=128,
    )

    assert result["scene"] == "test"
    request = requests[0]["body"]
    assert request["input"] == [
        {"type": "text", "text": "The first image is the original."},
        {
            "type": "image",
            "uri": "https://example.test/original.png",
        },
        {
            "type": "image",
            "uri": "https://example.test/detail.png",
        },
    ]


def test_gemini_vlm_rejects_missing_required_schema_paths(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    requests = []
    fake_client = FakeAsyncClient(requests, output={"scene": "test"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
    schema = {
        "type": "object",
        "properties": {
            "scene": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
        },
    }

    client = build_vlm_client(provider="gemini", model_name="gemini-test")
    with pytest.raises(RuntimeError, match=r"\$\.details") as error:
        client.create_image_json(
            system_prompt="Describe the image as JSON.",
            user_text="Inspect it.",
            image_input="https://example.test/input.png",
            max_tokens=128,
            response_schema=schema,
        )
    assert error.value._gemini_runtime_metrics["llm_api_calls"] == 1


def test_gemini_vlm_refuses_legacy_wire_protocol(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="requires wire_api='interactions'"):
        build_vlm_client(
            provider="gemini",
            model_name="gemini-test",
            wire_api="chat_completions",
        )


def test_gemini_vlm_refuses_constructor_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    with pytest.raises(ValueError, match="credentials must come from"):
        build_vlm_client(
            provider="gemini",
            model_name="gemini-test",
            api_key="constructor-key",
        )


def test_gemini_vlm_uses_production_timeout_floor(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    monkeypatch.setenv("GEMINI_VISION_TIMEOUT_SECONDS", "240")
    client = build_vlm_client(
        provider="gemini",
        model_name="gemini-test",
        timeout=60.0,
    )
    assert client.timeout == 240.0


def test_gemini_vlm_uses_bounded_default_timeout(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    monkeypatch.delenv("GEMINI_VISION_TIMEOUT_SECONDS", raising=False)

    client = build_vlm_client(
        provider="gemini",
        model_name="gemini-test",
        timeout=60.0,
    )

    assert client.timeout == 90.0


if __name__ == "__main__":
    from pytest import MonkeyPatch

    patch = MonkeyPatch()
    try:
        test_gemini_vlm_uses_interactions(patch)
        patch.undo()
        patch = MonkeyPatch()
        test_gemini_vlm_refuses_legacy_wire_protocol(patch)
    finally:
        patch.undo()
    print("Gemini VLM Interactions tests passed.")
