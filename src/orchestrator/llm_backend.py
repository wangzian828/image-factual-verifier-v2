# -*- coding: utf-8 -*-
"""LLM Backend abstraction layer.

Provides a unified async interface for LLM inference.
- APIBackend: for OpenAI-compatible APIs (qwen, necodex, openai)
- RolloutEngineBackend: stub for rLLM training integration
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


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

    Supports qwen (dashscope), necodex, and standard OpenAI endpoints.
    """

    def __init__(
        self,
        provider: str = "qwen",
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        proxy: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = base_url or self._resolve_base_url()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.proxy = proxy or self._resolve_proxy()
        self.extra_body = extra_body or self._resolve_extra_body()

    def _resolve_api_key(self) -> str:
        if self.provider == "qwen":
            return os.getenv("QWEN_API_KEY", "")
        elif self.provider == "necodex":
            return os.getenv("NECODEX_API_KEY", "")
        elif self.provider == "gpustack":
            return os.getenv("GPUSTACK_API_KEY", "")
        elif self.provider == "lmdeploy":
            return os.getenv("LMDEPLOY_API_KEY", "none")
        else:
            return os.getenv("OPENAI_API_KEY", "")

    def _resolve_base_url(self) -> str:
        if self.provider == "qwen":
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif self.provider == "necodex":
            return os.getenv("NECODEX_BASE_URL", "https://api.sbbbbbbbbb.xyz/v1")
        elif self.provider == "gpustack":
            return "http://10.254.47.36/v1"
        elif self.provider == "lmdeploy":
            return os.getenv("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1")
        else:
            return "https://api.openai.com/v1"

    def _resolve_proxy(self) -> Optional[str]:
        if self.provider == "gpustack":
            return os.getenv("GPUSTACK_PROXY", "http://100.10.1.210:80")
        if self.provider == "lmdeploy":
            return None  # Local service, no proxy needed
        # Use HTTP proxy for external API calls
        return os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", None))

    def _resolve_extra_body(self) -> Dict[str, Any]:
        if self.provider == "gpustack":
            # Must disable thinking mode for Qwen3.5
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Call the API asynchronously with retry."""
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

        # For models with internal reasoning (gpt-5.5, o1), give enough space
        # for both reasoning and content output.
        if self.provider == "necodex":
            body["max_tokens"] = 32768

        # Merge extra_body (e.g., chat_template_kwargs for GPUStack)
        if self.extra_body:
            body.update(self.extra_body)

        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                client_kwargs: Dict[str, Any] = {"timeout": self.timeout}
                if self.proxy:
                    client_kwargs["proxy"] = self.proxy
                else:
                    # Disable env proxy for local services
                    client_kwargs["trust_env"] = False
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()

                    # Handle empty response body
                    if not response.content or not response.content.strip():
                        if attempt < 4:
                            import asyncio
                            await asyncio.sleep(3 * (attempt + 1))
                            continue
                        return LLMResponse(text="", prompt_tokens=0, completion_tokens=0)

                    data = response.json()

                choice = data["choices"][0]
                text = choice["message"]["content"] or ""

                # gpt-5.5 / o1 style: if content is empty but reasoning was done,
                # check for reasoning_content field
                if not text and choice["message"].get("reasoning_content"):
                    text = choice["message"]["reasoning_content"]

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
            except Exception as e:
                # Catch SSL errors and other transient network issues
                if "SSL" in str(type(e).__name__) or "ssl" in str(e).lower():
                    last_error = e
                    if attempt < 4:
                        import asyncio
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                raise

        raise last_error or RuntimeError("All retries exhausted")


class RolloutEngineBackend(LLMBackend):
    """rLLM RolloutEngine backend (for RL training).

    Stub implementation - to be filled when integrating with rLLM.
    """

    def __init__(self, rollout_engine: Any):
        self.engine = rollout_engine

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Delegate to rLLM's rollout engine."""
        response = await self.engine.get_model_response(messages=messages, **kwargs)
        return LLMResponse(
            text=response.text if hasattr(response, "text") else str(response),
            prompt_tokens=getattr(response, "prompt_length", 0),
            completion_tokens=getattr(response, "completion_length", 0),
        )
