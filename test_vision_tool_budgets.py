from __future__ import annotations

from src.integrations.vlm.openai_vlm import (
    DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS,
    OpenAIVisionClient,
)
from src.tools.compare_reference import (
    DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS,
    CompareWithReferenceTool,
)


def test_gemini_vision_uses_the_established_default_output_floor(monkeypatch) -> None:
    client = OpenAIVisionClient(provider="gemini")
    client.api_key = "test-key"
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        client,
        "_create_gemini_interactions_images_json",
        fake_request,
    )
    monkeypatch.delenv("GEMINI_VISION_MIN_OUTPUT_TOKENS", raising=False)

    result = client.create_image_json(
        system_prompt="Return JSON.",
        user_text="Inspect.",
        image_input="image.jpg",
        max_tokens=300,
        response_schema={"type": "object"},
    )

    assert result == {"ok": True}
    assert captured["max_tokens"] == DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS


def test_gemini_vision_output_floor_can_be_overridden(monkeypatch) -> None:
    client = OpenAIVisionClient(provider="gemini")
    client.api_key = "test-key"
    captured = {}
    monkeypatch.setattr(
        client,
        "_create_gemini_interactions_images_json",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    monkeypatch.setenv("GEMINI_VISION_MIN_OUTPUT_TOKENS", "512")

    client.create_image_json(
        system_prompt="Return JSON.",
        user_text="Inspect.",
        image_input="image.jpg",
        max_tokens=300,
        response_schema={"type": "object"},
    )

    assert captured["max_tokens"] == 512


def test_reference_compare_output_budget_is_bounded_and_configurable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS", raising=False)
    assert (
        CompareWithReferenceTool._configured_max_output_tokens()
        == DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS
    )

    monkeypatch.setenv("GEMINI_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS", "8192")
    assert CompareWithReferenceTool._configured_max_output_tokens() == 8192

    monkeypatch.setenv("GEMINI_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS", "bad")
    assert (
        CompareWithReferenceTool._configured_max_output_tokens()
        == DEFAULT_REFERENCE_COMPARE_MAX_OUTPUT_TOKENS
    )
