from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

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
DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS = 8192


class _PersistentAsyncRuntime:
    """Run synchronous vision calls on one reusable async transport thread."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ifv-gemini-vision",
        )
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._started = False
        self._closed = False

    def run(self, coroutine: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("The reusable vision runtime is closed.")
            self._started = True
        return self._executor.submit(self._run_on_loop, coroutine).result()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _run_on_loop(self, coroutine: Any) -> Any:
        runner = getattr(self._thread_local, "runner", None)
        if hasattr(asyncio, "Runner"):
            if runner is None:
                runner = asyncio.Runner()
                self._thread_local.runner = runner
            return runner.run(coroutine)

        loop = getattr(self._thread_local, "loop", None)
        if loop is None:
            loop = asyncio.new_event_loop()
            self._thread_local.loop = loop
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)

    def close(self, cleanup: Any = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started

        if started:
            if cleanup is not None:
                self._executor.submit(self._run_on_loop, cleanup).result(
                    timeout=10
                )
            self._executor.submit(self._close_runner).result(timeout=10)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _close_runner(self) -> None:
        runner = getattr(self._thread_local, "runner", None)
        if runner is not None:
            runner.close()
            return
        loop = getattr(self._thread_local, "loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()


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
        self._async_runtime: Optional[_PersistentAsyncRuntime] = (
            _PersistentAsyncRuntime()
            if self.provider == "gemini"
            else None
        )
        self._gemini_interactions_client: Optional[GeminiInteractionsClient] = None

    def close(self) -> None:
        """Close the reusable vision transport if it has been started."""

        if self._async_runtime is None:
            return
        if self._async_runtime.closed:
            return
        client = self._gemini_interactions_client
        cleanup = client.aclose() if client is not None else None
        self._async_runtime.close(cleanup)

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
        return self.create_images_json(
            system_prompt=system_prompt,
            user_text=user_text,
            image_inputs=[image_input],
            max_tokens=max_tokens,
            model_name=model_name,
            temperature=temperature,
            response_schema=response_schema,
        )

    def create_images_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_inputs: Sequence[str],
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
        images = [str(item).strip() for item in image_inputs if str(item).strip()]
        if not images:
            raise ValueError("create_images_json requires at least one image input.")

        if self.provider == "gemini":
            try:
                minimum_tokens = max(
                    1,
                    int(
                        os.getenv(
                            "GEMINI_VISION_MIN_OUTPUT_TOKENS",
                            str(DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS),
                        )
                    ),
                )
            except ValueError:
                minimum_tokens = DEFAULT_GEMINI_VISION_MIN_OUTPUT_TOKENS
            return self._create_gemini_interactions_images_json(
                system_prompt=system_prompt,
                user_text=user_text,
                image_inputs=images,
                max_tokens=max(max_tokens, minimum_tokens),
                model_name=model_name or self.model_name,
                temperature=temperature,
                response_schema=response_schema,
            )

        schema = None
        effective_max_tokens = max_tokens
        if self.provider == "qwen_local":
            effective_max_tokens = max(
                max_tokens,
                max(
                    1,
                    int(
                        os.getenv(
                            "QWEN_LOCAL_VISION_MIN_OUTPUT_TOKENS",
                            "8192",
                        )
                    ),
                ),
            )
            if response_schema is not None:
                schema = normalize_json_schema(
                    response_schema,
                    require_all_properties=True,
                )

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
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image_input)},
                        }
                        for image_input in images
                    ],
                ],
            },
        ]
        content = chat.create_json_completion(
            model_name=model_name or self.model_name,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            messages=messages,
            response_schema=schema,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if self.provider == "qwen_local"
                else None
            ),
        )
        parsed = parse_json_object(content)
        if not parsed:
            raise RuntimeError("Vision model did not return a valid JSON object.")
        return parsed

    def _create_gemini_interactions_images_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_inputs: Sequence[str],
        max_tokens: int,
        model_name: str,
        temperature: float,
        response_schema: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if self.wire_api != "interactions":
            raise RuntimeError(
                "Gemini vision requires wire_api='interactions'; protocol fallback is disabled."
            )

        schema = normalize_json_schema(
            DEFAULT_JSON_OBJECT_SCHEMA if response_schema is None else response_schema,
            require_all_properties=response_schema is not None,
        )

        async def _request() -> Dict[str, Any]:
            client = self._gemini_interactions_client
            if client is None:
                client = GeminiInteractionsClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
                self._gemini_interactions_client = client
            payload = await client.create(
                model=model_name,
                input=[
                    {"type": "text", "text": user_text},
                    *[
                        self._to_interactions_image(image_input)
                        for image_input in image_inputs
                    ],
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
                        os.getenv("GEMINI_VISION_THINKING_LEVEL", "low"),
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

    def _run_async(self, coroutine: Any) -> Any:
        if self._async_runtime is None:
            raise RuntimeError("Reusable async vision runtime is unavailable.")
        return self._async_runtime.run(coroutine)
