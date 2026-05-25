from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import requests


JINA_READER_PREFIX = "https://r.jina.ai/http://"
DEFAULT_MAX_CHARS = 12000
DEFAULT_SNIPPET_CHARS = 2000


def normalize_reader_url(url: str) -> str:
    if url.startswith("https://"):
        return f"https://r.jina.ai/http://{url.removeprefix('https://')}"
    if url.startswith("http://"):
        return f"https://r.jina.ai/http://{url.removeprefix('http://')}"
    return f"{JINA_READER_PREFIX}{url}"


def compress_whitespace(text: str) -> str:
    return " ".join(text.split())


@dataclass
class JinaReaderClient:
    api_key: Optional[str] = None
    timeout: int = 30
    max_chars: int = DEFAULT_MAX_CHARS
    snippet_chars: int = DEFAULT_SNIPPET_CHARS

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS")

    def visit(self, url: str, goal: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(normalize_reader_url(url), headers=headers, timeout=self.timeout)
        response.raise_for_status()
        content = response.text[: self.max_chars]
        evidence = self.extract_goal_snippet(content, goal)
        return {
            "url": url,
            "goal": goal,
            "provider": "jina_reader",
            "content": content,
            "evidence": evidence,
            "summary": evidence,
        }

    def extract_goal_snippet(self, content: str, goal: str) -> str:
        compact_goal = [token.lower() for token in goal.split() if token.strip()]
        compact_content = compress_whitespace(content)
        lower_content = compact_content.lower()

        best_index = -1
        for token in compact_goal:
            idx = lower_content.find(token)
            if idx != -1:
                best_index = idx
                break

        if best_index == -1:
            return compact_content[: self.snippet_chars]

        start = max(0, best_index - self.snippet_chars // 3)
        end = min(len(compact_content), best_index + self.snippet_chars)
        return compact_content[start:end]
