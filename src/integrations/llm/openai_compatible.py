from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openai import APITimeoutError


@dataclass
class OpenAICompatibleChatClient:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    wire_api: str = "chat_completions"
    timeout: float = 60.0
    max_retries: int = 2

    def build_client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("LLM API key is not set.")
        # Use proxy from environment if available
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
        http_client = httpx.Client(
            proxy=proxy,
            timeout=self.timeout,
        )
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )

    def create_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.wire_api == "responses":
                    return self._create_responses_json_completion(
                        model_name=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                return self._create_chat_json_completion(
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except (APITimeoutError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("create_json_completion failed without a captured exception.")

    def _create_chat_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        client = self.build_client()
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return response.choices[0].message.content or "{}"

    def _create_responses_json_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM API key is not set.")
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": model_name,
            "input": self._to_responses_input(messages),
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
        with httpx.Client(proxy=proxy, timeout=self.timeout) as client:
            response = client.post(
                f"{base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return self._extract_responses_text(response.json())

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
                continue

            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"type": "input_text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    parts.append({"type": "input_image", "image_url": image_url})
            converted.append({"role": role, "content": parts})
        return converted

    @staticmethod
    def _extract_responses_text(payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        chunks: List[str] = []
        for output_item in payload.get("output", []) or []:
            for content_item in output_item.get("content", []) or []:
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks) or "{}"


def resolve_qwen_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def resolve_openai_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("OPENAI_API_KEY")


def resolve_necodex_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("NECODEX_API_KEY")


def resolve_model_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    provider = provider.lower().strip()
    if provider == "qwen":
        return resolve_qwen_api_key(api_key)
    if provider == "openai":
        return resolve_openai_api_key(api_key)
    if provider == "necodex":
        return resolve_necodex_api_key(api_key)
    return api_key


def resolve_model_base_url(provider: str, base_url: Optional[str]) -> Optional[str]:
    provider = provider.lower().strip()
    if base_url:
        return base_url
    if provider == "qwen":
        return os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if provider == "necodex":
        return os.getenv("NECODEX_BASE_URL") or "https://api.sbbbbbbbbb.xyz/v1"
    return base_url


def resolve_model_wire_api(provider: str, wire_api: Optional[str]) -> str:
    if wire_api:
        return wire_api
    provider = provider.lower().strip()
    if provider == "necodex":
        return os.getenv("NECODEX_WIRE_API") or "responses"
    return "chat_completions"
