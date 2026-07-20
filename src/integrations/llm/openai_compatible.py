from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class OpenAICompatibleChatClient:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    wire_api: str = "chat_completions"
    timeout: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.wire_api = validate_wire_api(self.wire_api)
        if self.wire_api == "openai_compat":
            self.wire_api = "chat_completions"
        if self.wire_api == "interactions":
            raise ValueError(
                "OpenAICompatibleChatClient does not implement Gemini Interactions; "
                "use GeminiInteractionsClient."
            )
        self._thread_local = threading.local()

    def _client_kwargs(self) -> Dict[str, Any]:
        is_local = self.base_url and ("127.0.0.1" in self.base_url or "localhost" in self.base_url)
        if is_local:
            return {"timeout": self.timeout, "trust_env": False}
        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("http_proxy")
        )
        kwargs: Dict[str, Any] = {"timeout": self.timeout}
        if proxy:
            kwargs["proxy"] = proxy
            kwargs["trust_env"] = True
        return kwargs

    def _get_client(self) -> httpx.Client:
        client = getattr(self._thread_local, "client", None)
        if client is None or client.is_closed:
            client = httpx.Client(**self._client_kwargs())
            self._thread_local.client = client
        return client

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
                if self.wire_api == "chat_completions":
                    return self._create_chat_json_completion(
                        model_name=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                raise RuntimeError(f"Unsupported wire API at runtime: {self.wire_api}")
            except httpx.HTTPStatusError as exc:
                last_error = exc
                raise
            except httpx.TimeoutException as exc:
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
        if not self.api_key:
            raise RuntimeError("LLM API key is not set.")
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        payload: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        response = self._get_client().post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Chat Completions returned an empty JSON response.")
        return content

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

        is_local = base_url and ("127.0.0.1" in base_url or "localhost" in base_url)
        if is_local:
            proxy = None
        else:
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
        response = self._get_client().post(
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
        text = "\n".join(chunks).strip()
        if not text:
            raise RuntimeError("Responses API returned an empty response.")
        return text

def resolve_qwen_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def resolve_gemini_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is not None:
        raise ValueError(
            "Gemini credentials must come from GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def resolve_openai_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("OPENAI_API_KEY")


def resolve_necodex_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("NECODEX_API_KEY")


def resolve_model_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    provider = provider.lower().strip()
    if provider == "gemini":
        return resolve_gemini_api_key(api_key)
    if provider == "qwen":
        return resolve_qwen_api_key(api_key)
    if provider == "openai":
        return resolve_openai_api_key(api_key)
    if provider == "necodex":
        return resolve_necodex_api_key(api_key)
    if provider == "lmdeploy":
        return api_key or os.getenv("LMDEPLOY_API_KEY", "none")
    if provider == "qwen_local":
        return api_key or os.getenv("QWEN_LOCAL_API_KEY", "none")
    return api_key


def resolve_model_base_url(
    provider: str,
    base_url: Optional[str],
    wire_api: Optional[str] = None,
) -> Optional[str]:
    provider = provider.lower().strip()
    if base_url:
        return base_url
    if provider == "gemini":
        resolve_model_wire_api(provider, wire_api)
        return os.getenv("GEMINI_INTERACTIONS_URL") or "https://generativelanguage.googleapis.com/v1beta/interactions"
    if provider == "qwen":
        return os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if provider == "necodex":
        return os.getenv("NECODEX_BASE_URL") or "https://api.sbbbbbbbbb.xyz/v1"
    if provider == "lmdeploy":
        return os.getenv("LMDEPLOY_BASE_URL", "http://127.0.0.1:8899/v1")
    if provider == "qwen_local":
        return os.getenv("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8899/v1")
    return base_url


def resolve_model_wire_api(provider: str, wire_api: Optional[str]) -> str:
    provider = provider.lower().strip()
    if wire_api:
        value = validate_wire_api(wire_api)
        if value == "openai_compat":
            value = "chat_completions"
        if provider == "gemini" and value != "interactions":
            raise ValueError("Gemini requires wire_api='interactions'; protocol fallback is disabled.")
        return value
    if provider == "gemini":
        value = validate_wire_api(os.getenv("GEMINI_WIRE_API", "interactions"))
        if value != "interactions":
            raise ValueError("Gemini requires wire_api='interactions'; protocol fallback is disabled.")
        return "interactions"
    if provider == "necodex":
        return validate_wire_api(os.getenv("NECODEX_WIRE_API") or "responses")
    return "chat_completions"


def validate_wire_api(wire_api: str) -> str:
    value = str(wire_api or "").strip().lower()
    allowed = {"interactions", "responses", "chat_completions", "openai_compat"}
    if value not in allowed:
        raise ValueError(
            f"Unsupported wire_api {wire_api!r}; expected one of {', '.join(sorted(allowed))}."
        )
    return value
