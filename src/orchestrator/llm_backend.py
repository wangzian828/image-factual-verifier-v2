# -*- coding: utf-8 -*-
"""LLM Backend abstraction layer.

Provides a unified async interface for API-based LLM inference.
"""
from __future__ import annotations

import json
import os
import atexit
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from src.integrations.gemini import (
    GeminiInteractionsClient,
    extract_text,
    messages_to_input,
    validate_interaction_response,
)
from src.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_api_key,
    resolve_model_base_url,
    resolve_model_wire_api,
)


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate a response given a conversation history."""
        ...


class APIBackend(LLMBackend):
    """OpenAI-compatible API backend using async httpx.

    Supports qwen (DashScope), qwen_local (OpenAI-compatible local serving),
    necodex, and standard OpenAI endpoints.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        wire_api: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 1,
        proxy: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider.lower().strip()
        if self.provider == "gemini" and api_key is not None:
            raise ValueError(
                "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self.model_name = model_name
        self.wire_api = resolve_model_wire_api(self.provider, wire_api)
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = base_url or self._resolve_base_url()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.proxy = proxy or self._resolve_proxy()
        self.extra_body = extra_body or self._resolve_extra_body()
        self._shared_client: Optional[httpx.AsyncClient] = None
        self._client_registered_for_cleanup = False

    def _resolve_api_key(self) -> str:
        if self.provider == "necodex":
            return os.getenv("NECODEX_API_KEY", "")
        elif self.provider == "gpustack":
            return os.getenv("GPUSTACK_API_KEY", "")
        if self.provider == "lmdeploy":
            return os.getenv("LMDEPLOY_API_KEY", "none")
        if self.provider == "qwen_local":
            return os.getenv("QWEN_LOCAL_API_KEY", "none")
        return resolve_model_api_key(self.provider, None) or ""

    def _resolve_base_url(self) -> str:
        if self.provider == "gemini" and self.wire_api == "interactions":
            return os.getenv("GEMINI_INTERACTIONS_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
        if self.provider == "necodex":
            return os.getenv("NECODEX_BASE_URL", "https://api.sbbbbbbbbb.xyz/v1")
        elif self.provider == "gpustack":
            return "http://10.254.47.36/v1"
        elif self.provider == "lmdeploy":
            return os.getenv("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1")
        elif self.provider == "qwen_local":
            return os.getenv("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8899/v1")
        return resolve_model_base_url(self.provider, None, self.wire_api) or "https://api.openai.com/v1"

    def _resolve_proxy(self) -> Optional[str]:
        if self.provider == "gpustack":
            return os.getenv("GPUSTACK_PROXY", "http://100.10.1.210:80")
        if self.provider in {"lmdeploy", "qwen_local"}:
            return None  # Local service, no proxy needed
        # Use HTTP proxy for external API calls
        return os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", None))

    def _resolve_extra_body(self) -> Dict[str, Any]:
        if self.provider == "gpustack":
            # Keep this externally hosted Qwen-family endpoint deterministic.
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _client_kwargs(self) -> Dict[str, Any]:
        client_kwargs: Dict[str, Any] = {"timeout": self.timeout}
        if self.proxy:
            client_kwargs["proxy"] = self.proxy
        else:
            client_kwargs["trust_env"] = False
        return client_kwargs

    def _register_client_cleanup(self) -> None:
        if self._client_registered_for_cleanup:
            return
        self._client_registered_for_cleanup = True
        atexit.register(self._close_shared_client_sync)

    def _close_shared_client_sync(self) -> None:
        client = self._shared_client
        self._shared_client = None
        if client is None:
            return
        try:
            import asyncio
            asyncio.run(client.aclose())
        except Exception:
            pass

    def _get_shared_client(self) -> httpx.AsyncClient:
        if self._shared_client is None:
            self._shared_client = httpx.AsyncClient(**self._client_kwargs())
            self._register_client_cleanup()
        return self._shared_client

    async def aclose(self) -> None:
        """Close the shared HTTP client used by this backend."""
        client = self._shared_client
        self._shared_client = None
        if client is not None:
            await client.aclose()

    def _interactions_base_url(self) -> str:
        if self.base_url.rstrip("/").endswith("/interactions"):
            return self.base_url.rstrip("/")
        return os.getenv(
            "GEMINI_INTERACTIONS_URL",
            "https://generativelanguage.googleapis.com/v1beta/interactions",
        )

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Call the API asynchronously with retry."""
        if self.wire_api == "interactions":
            return await self._get_interactions_response(messages, **kwargs)
        if self.wire_api == "responses":
            return await self._get_responses_response(messages, **kwargs)
        if self.wire_api == "chat_completions":
            return await self._get_chat_completions_response(messages, **kwargs)
        raise RuntimeError(f"Unsupported wire API at runtime: {self.wire_api}")

    async def _get_chat_completions_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request body
        body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        tools = kwargs.get("tools")
        if tools:
            body["tools"] = self._openai_tool_schemas(tools)
            body["tool_choice"] = kwargs.get("tool_choice", "auto")
            body["parallel_tool_calls"] = False
        response_format = kwargs.get("response_format")
        if response_format:
            body["response_format"] = response_format
        generation_config = kwargs.get("generation_config")

        # For models with internal reasoning (gpt-5.5, o1), give enough space
        # for both reasoning and content output.
        if self.provider == "necodex":
            body["max_tokens"] = 32768

        # Merge extra_body (e.g., chat_template_kwargs for GPUStack)
        if self.extra_body:
            body.update(self.extra_body)

        # LMDeploy exposes Qwen's thinking switch through the chat template,
        # not as an OpenAI top-level generation field. Keep this local so other
        # OpenAI-compatible providers never receive an unsupported parameter.
        if (
            self.provider in {"qwen_local", "lmdeploy"}
            and isinstance(generation_config, dict)
            and "enable_thinking" in generation_config
        ):
            template_kwargs = body.get("chat_template_kwargs", {})
            template_kwargs = (
                dict(template_kwargs)
                if isinstance(template_kwargs, dict)
                else {}
            )
            template_kwargs["enable_thinking"] = bool(
                generation_config["enable_thinking"]
            )
            body["chat_template_kwargs"] = template_kwargs

        # vLLM exposes Qwen's reasoning wall and sampling controls as top-level
        # request fields, separately from chat-template switches.  Forward only
        # the explicit allowlist so internal stage metadata never leaks onto the
        # provider wire.
        if self.provider in {"qwen_local", "lmdeploy"} and isinstance(
            generation_config, dict
        ):
            for key in (
                "thinking_token_budget",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "seed",
            ):
                if key in generation_config:
                    body[key] = generation_config[key]

        last_error: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                client = self._get_shared_client()
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()

                # Handle empty response body
                if not response.content or not response.content.strip():
                    if attempt + 1 < attempts:
                        import asyncio
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    raise RuntimeError("Chat Completions returned an empty response body.")

                data = response.json()

                choice = data["choices"][0]
                text = self._extract_chat_completion_text(
                    choice,
                    allow_reasoning_fallback=bool(response_format),
                )
                if not text.strip():
                    # Qwen/vLLM can occasionally return a successful HTTP
                    # response containing only hidden reasoning or an empty
                    # choice. Treat that as a bounded transient at the wire
                    # boundary, just like an empty body; do not turn it into an
                    # Agent engineering error before StageRunner can continue.
                    if attempt + 1 < attempts:
                        last_error = RuntimeError(
                            "Chat Completions returned an empty model response."
                        )
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError("Chat Completions returned an empty model response.")

                usage = data.get("usage", {})

                return LLMResponse(
                    text=text,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    raw=data,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                    httpx.LocalProtocolError, httpx.ReadError) as e:
                last_error = e
                if attempt + 1 < attempts:
                    import asyncio
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt + 1 < attempts:
                    last_error = e
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(self._http_status_error_detail(e)) from e
            except Exception as e:
                # Catch SSL errors and other transient network issues
                if "SSL" in str(type(e).__name__) or "ssl" in str(e).lower():
                    last_error = e
                    if attempt + 1 < attempts:
                        import asyncio
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                raise

        raise last_error or RuntimeError("All retries exhausted")

    @staticmethod
    def _http_status_error_detail(error: httpx.HTTPStatusError) -> str:
        """Preserve a bounded provider error body for durable diagnostics."""

        response = error.response
        try:
            detail = response.text.strip()
        except Exception:
            detail = ""
        if len(detail) > 4000:
            detail = detail[:4000] + "... [truncated]"
        suffix = f": {detail}" if detail else ""
        return (
            f"HTTP {response.status_code} {response.reason_phrase} for "
            f"{response.request.url}{suffix}"
        )

    @staticmethod
    def _openai_tool_schemas(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in tools:
            if isinstance(item.get("function"), dict):
                normalized.append(item)
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "description": item.get("description", ""),
                        "parameters": item.get("parameters", {}),
                    },
                }
            )
        return normalized

    @staticmethod
    def _extract_chat_completion_text(
        choice: Dict[str, Any],
        *,
        allow_reasoning_fallback: bool = False,
    ) -> str:
        """Extract assistant text from OpenAI-compatible chat completion payloads.

        Different providers are slightly inconsistent here:
        - `message.content` may be a string
        - `message.content` may be a list of content parts
        - `message.content` may be omitted while provider-specific fields exist
        """
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            if len(tool_calls) != 1:
                return "<parallel_tool_calls>" + json.dumps(
                    tool_calls,
                    ensure_ascii=False,
                ) + "</parallel_tool_calls>"
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            serialized = json.dumps(
                {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
                ensure_ascii=False,
            )
            return f"<tool_call>{serialized}</tool_call>"

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return "\n".join(parts)

        # Some reasoning-aware servers place constrained JSON in this field
        # because the checkpoint's generation prompt begins inside <think>.
        # Only schema-bound requests may consume it; their normal parser still
        # validates the candidate before it can become a canonical action.
        reasoning = message.get("reasoning_content")
        if (
            allow_reasoning_fallback
            and isinstance(reasoning, str)
            and reasoning.strip()
        ):
            return reasoning

        return ""

    async def _get_responses_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "input": OpenAICompatibleChatClient._to_responses_input(messages),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            "text": {"format": {"type": "text"}},
        }
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            body["temperature"] = temperature

        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                client = self._get_shared_client()
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                text = OpenAICompatibleChatClient._extract_responses_text(data)
                usage = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
                    raw=data,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                    httpx.LocalProtocolError, httpx.ReadError) as e:
                last_error = e
                if attempt < 4:
                    import asyncio
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < 4:
                    last_error = e
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_error or RuntimeError("All retries exhausted")

    async def _get_interactions_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        data = await self.create_interaction(
            input_payload=messages_to_input(messages),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            store=True,
        )
        _, status = validate_interaction_response(data)
        if status != "completed":
            raise RuntimeError(
                f"Gemini text completion requires status=completed, received status={status}."
            )
        text = extract_text(data)
        if not text.strip():
            raise RuntimeError("Gemini Interactions returned an empty model response.")
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("total_input_tokens", usage.get("input_tokens", 0)),
            completion_tokens=usage.get("total_output_tokens", usage.get("output_tokens", 0)),
            raw=data,
        )

    async def create_interaction(
        self,
        *,
        input_payload: Any,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        previous_interaction_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        store: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        background: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self.provider != "gemini":
            raise RuntimeError("create_interaction is only supported for Gemini.")
        if self.wire_api != "interactions":
            raise RuntimeError("Gemini requires wire_api='interactions'.")

        effective_generation_config: Dict[str, Any] = dict(generation_config or {})
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens:
            effective_generation_config["max_output_tokens"] = effective_max_tokens
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            effective_generation_config["temperature"] = effective_temperature
        client = GeminiInteractionsClient(
            base_url=self._interactions_base_url(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=float(
                os.getenv("GEMINI_RETRY_BASE_DELAY_SECONDS", "3")
            ),
            retry_jitter=float(
                os.getenv("GEMINI_RETRY_JITTER_SECONDS", "2")
            ),
            retry_max_delay=float(
                os.getenv("GEMINI_RETRY_MAX_DELAY_SECONDS", "60")
            ),
            client=self._get_shared_client(),
        )
        return await client.create(
            model=self.model_name,
            input=input_payload,
            system_instruction=system_instruction,
            tools=tools,
            previous_interaction_id=previous_interaction_id,
            response_format=response_format,
            generation_config=effective_generation_config or None,
            background=background,
            store=store,
        )
