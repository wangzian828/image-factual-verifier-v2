from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests


JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"


def _api_key(value: Optional[str]) -> str:
    return str(
        value
        or os.getenv("JINA_API_KEY")
        or os.getenv("JINA_API_KEYS")
        or ""
    ).strip()


@dataclass
class JinaRerankerClient:
    """Jina reranking only; it never performs search or page fetching."""

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
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        *,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is not set for candidate reranking.")
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
        response = self._get_session().post(
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
