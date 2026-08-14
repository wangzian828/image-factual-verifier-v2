from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests


JINA_SEARCH_ENDPOINT = "https://s.jina.ai/"
JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"


def _api_key(value: Optional[str]) -> str:
    return str(
        value
        or os.getenv("JINA_API_KEY")
        or os.getenv("JINA_API_KEYS")
        or ""
    ).strip()


@dataclass
class JinaSearchClient:
    api_key: Optional[str] = None
    endpoint: str = JINA_SEARCH_ENDPOINT
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.api_key = _api_key(self.api_key)

    def search(self, query: str, *, top_k: int = 10) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is not set for jina_search.")
        response = requests.get(
            self.endpoint,
            params={"q": query},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", payload.get("results", payload))
        if not isinstance(rows, list):
            raise RuntimeError("Jina Search returned an invalid result list.")
        return {
            "query": query,
            "provider": "jina_search",
            "results": [
                self._normalize_result(query, row, index + 1)
                for index, row in enumerate(rows[:top_k])
                if isinstance(row, Mapping)
            ],
        }

    @staticmethod
    def _normalize_result(
        query: str,
        row: Mapping[str, Any],
        rank: int,
    ) -> Dict[str, Any]:
        content = str(
            row.get("content")
            or row.get("text")
            or row.get("description")
            or ""
        ).strip()
        snippet = str(
            row.get("description")
            or row.get("snippet")
            or content[:2000]
        ).strip()
        return {
            "query": query,
            "rank": rank,
            "title": str(row.get("title") or "").strip(),
            "url": str(row.get("url") or row.get("link") or "").strip(),
            "snippet": snippet[:2000],
            "content_preview": content[:8000],
            "source": str(row.get("source") or "").strip(),
            "date": row.get("publishedTime") or row.get("date"),
        }


@dataclass
class JinaRerankerClient:
    api_key: Optional[str] = None
    endpoint: str = JINA_RERANK_ENDPOINT
    model: Optional[str] = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.api_key = _api_key(self.api_key)
        self.model = (
            self.model
            or os.getenv("JINA_RERANKER_MODEL")
            or "jina-reranker-v3"
        ).strip()

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        *,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is not set for jina_rerank.")
        rows = [str(item).strip() for item in documents]
        if not rows:
            return []
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": rows,
        }
        if top_n is not None:
            payload["top_n"] = max(1, int(top_n))
        response = requests.post(
            self.endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        results = body.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Jina Reranker returned an invalid result list.")
        return [dict(item) for item in results if isinstance(item, Mapping)]
