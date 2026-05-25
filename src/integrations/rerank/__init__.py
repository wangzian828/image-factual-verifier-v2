# -*- coding: utf-8 -*-
"""Rerank integration: re-score search results by relevance to a query.

Supports multiple backends:
- DashScope (Alibaba Cloud / Qwen) — gte-rerank model
- Cohere — rerank-v3.5
- Local fallback — simple BM25-style keyword overlap scoring
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


class RerankClient(ABC):
    """Base class for rerank clients."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: The query to rank against.
            documents: List of document texts to rank.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'index', 'score', 'text' sorted by relevance.
        """
        ...


@dataclass
class DashScopeRerankClient(RerankClient):
    """Alibaba DashScope rerank API (gte-rerank model).

    Uses the DashScope text-reranking API.
    Requires DASHSCOPE_API_KEY environment variable.
    """

    api_key: Optional[str] = None
    model: str = "gte-rerank"
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-reranking/text-reranking"
    timeout: int = 15

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_KEY")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": min(top_k, len(documents)),
                "return_documents": True,
            },
        }

        response = requests.post(
            self.endpoint, json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        # DashScope returns: {"output": {"results": [{"index": 0, "relevance_score": 0.9, "document": {"text": "..."}}]}}
        results = data.get("output", {}).get("results", [])
        return [
            {
                "index": r["index"],
                "score": r.get("relevance_score", 0.0),
                "text": r.get("document", {}).get("text", documents[r["index"]]),
            }
            for r in results[:top_k]
        ]


@dataclass
class CohereRerankClient(RerankClient):
    """Cohere rerank API.

    Requires COHERE_API_KEY environment variable.
    """

    api_key: Optional[str] = None
    model: str = "rerank-v3.5"
    endpoint: str = "https://api.cohere.com/v2/rerank"
    timeout: int = 15

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("COHERE_API_KEY")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set.")
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": True,
        }

        response = requests.post(
            self.endpoint, json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        return [
            {
                "index": r["index"],
                "score": r.get("relevance_score", 0.0),
                "text": r.get("document", {}).get("text", documents[r["index"]]),
            }
            for r in results[:top_k]
        ]


@dataclass
class KeywordRerankClient(RerankClient):
    """Simple keyword-overlap reranker (no API needed).

    Uses character n-gram overlap as a proxy for relevance.
    Useful as a fallback when no rerank API is available.
    """

    ngram_size: int = 2

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        query_ngrams = self._get_ngrams(query.lower())
        scored = []
        for i, doc in enumerate(documents):
            doc_ngrams = self._get_ngrams(doc.lower())
            if not query_ngrams or not doc_ngrams:
                score = 0.0
            else:
                overlap = len(query_ngrams & doc_ngrams)
                score = overlap / len(query_ngrams)
            scored.append({"index": i, "score": score, "text": doc})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _get_ngrams(self, text: str) -> set:
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}


def build_rerank_client(provider: str = "auto") -> RerankClient:
    """Build a rerank client based on available API keys.

    Args:
        provider: "dashscope", "cohere", "keyword", or "auto" (try in order).

    Returns:
        A RerankClient instance.
    """
    if provider == "dashscope":
        return DashScopeRerankClient()
    elif provider == "cohere":
        return CohereRerankClient()
    elif provider == "keyword":
        return KeywordRerankClient()
    elif provider == "auto":
        # Try DashScope first, then Cohere, then keyword fallback
        if os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_KEY"):
            return DashScopeRerankClient()
        elif os.getenv("COHERE_API_KEY"):
            return CohereRerankClient()
        else:
            return KeywordRerankClient()
    else:
        raise ValueError(f"Unknown rerank provider: {provider}")
