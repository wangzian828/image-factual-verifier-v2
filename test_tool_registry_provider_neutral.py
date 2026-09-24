from __future__ import annotations

from src.orchestrator.tool_registry import build_all_tools_with_health


class FakeVisionClient:
    provider = "qwen_local"
    api_key = "test-key"

    def create_image_json(
        self,
        *,
        system_prompt,
        user_text,
        image_input,
        max_tokens,
        model_name=None,
        temperature=0.0,
        response_schema=None,
    ):
        raise AssertionError("runtime call is outside this registry test")

    def create_images_json(
        self,
        *,
        system_prompt,
        user_text,
        image_inputs,
        max_tokens,
        model_name=None,
        temperature=0.0,
        response_schema=None,
    ):
        raise AssertionError("runtime call is outside this registry test")


def test_qwen_teacher_visual_tools_share_provider_neutral_client(
    monkeypatch,
) -> None:
    client = FakeVisionClient()
    monkeypatch.setattr(
        "src.orchestrator.tool_registry.build_vlm_client",
        lambda **_kwargs: client,
    )

    tools, health = build_all_tools_with_health(
        vlm_provider="qwen_local",
        vlm_model="served-teacher-model",
        vlm_wire_api="chat_completions",
        vlm_base_url="http://teacher.test/v1",
    )

    anomaly = tools["analyze_visual_anomalies"]
    comparison = tools["compare_with_reference"]
    assert health["analyze_visual_anomalies"].available is True
    assert health["compare_with_reference"].available is True
    assert anomaly.client is client
    assert comparison.client is client
    assert anomaly.model_name == "served-teacher-model"
    assert comparison.model_name == "served-teacher-model"


def test_visual_tool_registry_passes_separate_private_scope(monkeypatch) -> None:
    client = FakeVisionClient()
    captured = {}

    def fake_build_vlm_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(
        "src.orchestrator.tool_registry.build_vlm_client", fake_build_vlm_client
    )
    vision_salt = "ifv-case-v1-" + "V" * 43
    browse_salt = "ifv-case-v1-" + "B" * 43
    build_all_tools_with_health(
        vlm_provider="qwen_local", vlm_model="served-teacher-model",
        prefix_cache_salt=browse_salt, vision_prefix_cache_salt=vision_salt,
    )
    assert captured["cache_salt"] == vision_salt
    assert vision_salt != browse_salt
