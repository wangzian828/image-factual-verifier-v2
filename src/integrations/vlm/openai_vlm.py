from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_api_key,
    resolve_model_base_url,
    resolve_model_wire_api,
)
from src.integrations.vlm.qwen_vl import parse_json_object
from src.tools.vision_utils import image_to_data_url


@dataclass
class OpenAIVisionClient:
    api_key: Optional[str] = None
    provider: str = "openai"
    base_url: Optional[str] = None
    wire_api: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    timeout: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.provider = self.provider.lower().strip()
        self.api_key = resolve_model_api_key(self.provider, self.api_key)
        self.base_url = resolve_model_base_url(self.provider, self.base_url)
        self.wire_api = resolve_model_wire_api(self.provider, self.wire_api)

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
            if self.provider == "lmdeploy":
                self.api_key = "none"
            else:
                env_name = "NECODEX_API_KEY" if self.provider == "necodex" else "OPENAI_API_KEY"
                raise RuntimeError(f"{env_name} is not set. Add it to the environment before using vision tools.")

        image_url = image_to_data_url(image_input)
        chat = OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            wire_api=self.wire_api or "chat_completions",
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
