from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from src.integrations.gemini import (
    GeminiInteractionsClient,
    RUNTIME_METRICS_KEY,
    attach_runtime_metrics,
    interaction_runtime_metrics,
    missing_required_paths,
    normalize_json_schema,
    require_minimal_thinking,
)
from src.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_api_key,
    resolve_model_base_url,
    resolve_model_wire_api,
)
from src.integrations.vlm.qwen_vl import parse_json_object
from src.tools.vision_utils import image_to_data_url


DEFAULT_JSON_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
}


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
        if self.provider == "gemini" and self.api_key is not None:
            raise ValueError(
                "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self.api_key = resolve_model_api_key(self.provider, self.api_key)
        self.wire_api = resolve_model_wire_api(self.provider, self.wire_api)
        self.base_url = resolve_model_base_url(self.provider, self.base_url, self.wire_api)

    def create_image_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_input: str,
        max_tokens: int,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            if self.provider == "lmdeploy":
                self.api_key = "none"
            else:
                if self.provider == "gemini":
                    env_name = "GEMINI_API_KEY"
                elif self.provider == "necodex":
                    env_name = "NECODEX_API_KEY"
                elif self.provider == "qwen":
                    env_name = "QWEN_API_KEY"
                else:
                    env_name = "OPENAI_API_KEY"
                raise RuntimeError(f"{env_name} is not set. Add it to the environment before using vision tools.")

        if self.provider == "gemini":
            minimum_tokens = max(
                1,
                int(os.getenv("GEMINI_VISION_MIN_OUTPUT_TOKENS", "8192")),
            )
            return self._create_gemini_interactions_image_json(
                system_prompt=system_prompt,
                user_text=user_text,
                image_input=image_input,
                max_tokens=max(max_tokens, minimum_tokens),
                model_name=model_name or self.model_name,
                temperature=temperature,
                response_schema=response_schema,
            )

        image_url = image_to_data_url(image_input)
        chat = OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            wire_api=self.wire_api or "chat_completions",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        content = chat.create_json_completion(
            model_name=model_name or self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        parsed = parse_json_object(content)
        if not parsed:
            raise RuntimeError("Vision model did not return a valid JSON object.")
        return parsed

    def _create_gemini_interactions_image_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_input: str,
        max_tokens: int,
        model_name: str,
        temperature: float,
        response_schema: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if self.wire_api != "interactions":
            raise RuntimeError(
                "Gemini vision requires wire_api='interactions'; protocol fallback is disabled."
            )

        image_item = self._to_interactions_image(image_input)
        schema = normalize_json_schema(
            DEFAULT_JSON_OBJECT_SCHEMA if response_schema is None else response_schema,
            require_all_properties=response_schema is not None,
        )

        async def _request() -> Dict[str, Any]:
            async with GeminiInteractionsClient(
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ) as client:
                payload = await client.create(
                    model=model_name,
                    input=[
                        {"type": "text", "text": user_text},
                        image_item,
                    ],
                    system_instruction=(
                        f"{system_prompt}\n\n"
                        "Return exactly one JSON object without markdown or commentary."
                    ),
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": schema,
                    },
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "thinking_level": require_minimal_thinking(
                            os.getenv("GEMINI_VISION_THINKING_LEVEL", "minimal"),
                            env_name="GEMINI_VISION_THINKING_LEVEL",
                        ),
                    },
                    background=False,
                    store=True,
                )
                runtime_metrics = interaction_runtime_metrics(payload)
                try:
                    content = client.extract_text(payload)
                    if not content.strip():
                        raise RuntimeError("Gemini Interactions vision response was empty.")
                    parsed = parse_json_object(content)
                    if not parsed:
                        raise RuntimeError(
                            "Gemini Interactions vision response was not a valid JSON object: "
                            + content[:500]
                        )
                    missing = missing_required_paths(parsed, schema)
                    if missing:
                        raise RuntimeError(
                            "Gemini Interactions vision response is missing required fields: "
                            + ", ".join(missing[:20])
                        )
                except Exception as exc:
                    raise attach_runtime_metrics(exc, runtime_metrics)
                parsed[RUNTIME_METRICS_KEY] = runtime_metrics
                return parsed

        return self._run_async(_request())

    @staticmethod
    def _to_interactions_image(image_input: str) -> Dict[str, Any]:
        if image_input.startswith(("http://", "https://")):
            return OpenAIVisionClient._data_url_to_interactions_image(image_input)

        image_url = image_to_data_url(image_input)
        return OpenAIVisionClient._data_url_to_interactions_image(image_url)

    @staticmethod
    def _data_url_to_interactions_image(image_url: str) -> Dict[str, Any]:
        if not (image_url.startswith("data:") and ";base64," in image_url):
            return {"type": "image", "uri": image_url}
        header, data = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return {"type": "image", "mime_type": mime_type, "data": data}

    @staticmethod
    def _run_async(coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coroutine).result()
