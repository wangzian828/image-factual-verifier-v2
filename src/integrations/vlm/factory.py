from __future__ import annotations

from typing import Any, Optional

from src.integrations.vlm.openai_vlm import OpenAIVisionClient
from src.integrations.vlm.qwen_vl import QwenVLClient


def build_vlm_client(
    *,
    provider: str = "qwen",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Any:
    provider = provider.lower().strip()
    if provider == "qwen":
        return QwenVLClient(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=model_name or "qwen3.6-plus",
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider in {"openai", "necodex"}:
        return OpenAIVisionClient(
            api_key=api_key,
            provider=provider,
            base_url=base_url,
            model_name=model_name or ("gpt-5.5" if provider == "necodex" else "gpt-4o-mini"),
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported VLM provider: {provider}")
