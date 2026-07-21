from __future__ import annotations

import json

from src.integrations.vlm.qwen_vl import QwenVLClient
from src.integrations.vlm.openai_vlm import OpenAIVisionClient


class RecordingChatClient:
    requests = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_json_completion(self, **kwargs):
        self.requests.append(kwargs)
        return json.dumps(
            {
                "answer_status": "ambiguous",
                "summary": "The detail view is too small.",
            }
        )


def test_qwen_vlm_supports_ordered_multi_view_input(monkeypatch) -> None:
    RecordingChatClient.requests = []
    monkeypatch.setattr(
        "src.integrations.vlm.qwen_vl.OpenAICompatibleChatClient",
        RecordingChatClient,
    )
    encoded = {
        "original.png": "data:image/png;base64,b3JpZ2luYWw=",
        "detail.png": "data:image/png;base64,ZGV0YWls",
    }
    monkeypatch.setattr(
        "src.integrations.vlm.qwen_vl.image_to_data_url",
        lambda value: encoded[value],
    )
    client = QwenVLClient(
        api_key="test-key",
        base_url="http://127.0.0.1:8899/v1",
        model_name="Qwen3-VL-8B",
    )
    result = client.create_images_json(
        system_prompt="Inspect all views.",
        user_text="The first image is the original.",
        image_inputs=[
            "original.png",
            "detail.png",
        ],
        max_tokens=256,
    )

    assert result["answer_status"] == "ambiguous"
    request = RecordingChatClient.requests[0]
    content = request["messages"][1]["content"]
    assert content[0] == {
        "type": "text",
        "text": "The first image is the original.",
    }
    assert content[1]["image_url"]["url"] == encoded["original.png"]
    assert content[2]["image_url"]["url"] == encoded["detail.png"]


def test_local_qwen_vision_uses_direct_mode_and_native_schema(
    monkeypatch,
) -> None:
    RecordingChatClient.requests = []
    monkeypatch.delenv("QWEN_LOCAL_VISION_MIN_OUTPUT_TOKENS", raising=False)
    monkeypatch.setattr(
        "src.integrations.vlm.openai_vlm.OpenAICompatibleChatClient",
        RecordingChatClient,
    )
    monkeypatch.setattr(
        "src.integrations.vlm.openai_vlm.image_to_data_url",
        lambda _value: "data:image/png;base64,AA==",
    )
    client = OpenAIVisionClient(
        api_key="none",
        provider="qwen_local",
        base_url="http://127.0.0.1:8901/v1",
        model_name="ifv-qwen3-vl-8b-thinking-vllm",
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    result = client.create_image_json(
        system_prompt="Inspect the image.",
        user_text="Return the observation.",
        image_input="image.png",
        max_tokens=2000,
        response_schema=schema,
    )

    assert result["answer_status"] == "ambiguous"
    request = RecordingChatClient.requests[0]
    assert request["max_tokens"] == 8192
    assert request["response_schema"]["required"] == ["answer"]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
