from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.integrations.llm.openai_compatible import OpenAICompatibleChatClient, resolve_qwen_api_key
from src.tools.vision_utils import image_to_data_url


@dataclass
class QwenVLClient:
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3.6-plus"
    timeout: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.api_key = resolve_qwen_api_key(self.api_key)

    def create_image_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_input: str,
        max_tokens: int,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not set. Add it to the environment before using Qwen VL tools.")

        image_url = image_to_data_url(image_input)
        chat = OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        content = chat.create_json_completion(
            model_name=model_name or self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        return parse_json_object(content)


def parse_json_object(content: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback = fallback or {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        left = content.find("{")
        right = content.rfind("}")
        if left == -1 or right == -1 or left > right:
            return dict(fallback)
        try:
            return json.loads(content[left : right + 1])
        except json.JSONDecodeError:
            return dict(fallback)
