from __future__ import annotations

import base64
import os

import pytest
import requests

import scripts.generate_gpt_image_samples as generator


class FakeInteractionsClient:
    requests = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def generate_image(self, **kwargs):
        self.requests.append({"client": self.kwargs, "request": kwargs})
        return {"id": "image-interaction", "steps": []}


def test_dataset_generation_defaults_to_gemini_interactions() -> None:
    assert generator.DEFAULT_PROVIDER == "gemini"


def test_dataset_gemini_generation_uses_shared_interactions_client(monkeypatch) -> None:
    monkeypatch.setattr(generator, "GeminiInteractionsClient", FakeInteractionsClient)
    monkeypatch.setenv("GEMINI_IMAGE_API_URL", "https://example.test/v1beta/interactions")
    FakeInteractionsClient.requests.clear()

    payload = generator._request_gemini_image(
        model="gemini-image-test",
        prompt="Generate a test source image",
        size="2048x1152",
    )

    assert payload["id"] == "image-interaction"
    call = FakeInteractionsClient.requests[0]
    assert call["client"]["base_url"] == "https://example.test/v1beta/interactions"
    assert call["client"]["max_retries"] == 0
    assert call["request"] == {
        "model": "gemini-image-test",
        "prompt": "Generate a test source image",
        "aspect_ratio": "16:9",
        "image_size": "2K",
        "background": False,
        "store": True,
    }


def test_dataset_gemini_image_request_omits_unsupported_delivery(monkeypatch) -> None:
    monkeypatch.setattr(generator, "GeminiInteractionsClient", FakeInteractionsClient)
    FakeInteractionsClient.requests.clear()

    generator._request_gemini_image(
        model="gemini-image-test",
        prompt="Generate a test source image",
        size="1024x1024",
    )

    assert "delivery" not in FakeInteractionsClient.requests[0]["request"]


@pytest.mark.parametrize("field", ["data", "base64"])
def test_extract_gemini_image_supports_inline_base64(field) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\ninline"
    payload = {
        "steps": [
            {
                "content": [
                    {
                        "type": "image",
                        "mime_type": "image/png",
                        field: base64.b64encode(image_bytes).decode("ascii"),
                    }
                ]
            }
        ]
    }

    assert generator._extract_gemini_image(payload) == (image_bytes, "image/png")


class FakeDownloadResponse:
    def __init__(self, *, content=b"downloaded", content_type="image/webp", error=None):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.error = error
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True
        if self.error is not None:
            raise self.error


def test_extract_gemini_image_downloads_uri_and_checks_http(monkeypatch) -> None:
    response = FakeDownloadResponse()
    calls = []

    def fake_get(uri, *, timeout):
        calls.append({"uri": uri, "timeout": timeout})
        return response

    monkeypatch.setattr(generator.requests, "get", fake_get)
    payload = {
        "output": [
            {
                "content": [
                    {"type": "image", "uri": "https://example.test/generated.webp"}
                ]
            }
        ]
    }

    assert generator._extract_gemini_image(payload) == (b"downloaded", "image/webp")
    assert calls == [{"uri": "https://example.test/generated.webp", "timeout": 180}]
    assert response.raise_for_status_called is True


def test_extract_gemini_image_propagates_uri_http_error(monkeypatch) -> None:
    error = requests.HTTPError("404 image not found")
    response = FakeDownloadResponse(error=error)
    monkeypatch.setattr(generator.requests, "get", lambda *args, **kwargs: response)
    payload = {
        "steps": [
            {
                "content": [
                    {"type": "image", "uri": "https://example.test/missing.png"}
                ]
            }
        ]
    }

    with pytest.raises(requests.HTTPError, match="404 image not found"):
        generator._extract_gemini_image(payload)
    assert response.raise_for_status_called is True


if __name__ == "__main__":
    from pytest import MonkeyPatch

    patch = MonkeyPatch()
    try:
        test_dataset_gemini_generation_uses_shared_interactions_client(patch)
    finally:
        patch.undo()
        os.environ.pop("GEMINI_IMAGE_API_URL", None)
    print("Image generation Interactions test passed.")
