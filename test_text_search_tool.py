from __future__ import annotations

import asyncio
from typing import Any

from src.tools.text_search import TextSearchTool


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, str | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        gl: str | None = None,
        hl: str | None = None,
        time_range: str | None = None,
    ) -> dict:
        self.calls.append((query, top_k, gl, hl))
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result for {query}",
                    "url": f"https://example.org/{query}",
                    "snippet": "Candidate snippet only.",
                }
            ],
            "answer_box": {"answer": "provider aggregate"},
            "knowledge_graph": {"title": "provider aggregate"},
        }


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, documents, top_n))
        return [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.25},
        ]


class FailingReranker:
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("jina reranker unavailable")


def _assert_discovery_only(result: dict) -> None:
    forbidden = {
        "visited_pages",
        "selected_url",
        "evidence",
        "summary",
        "stance",
        "directness",
        "evidence_eligible",
        "artifact_sha256",
        "evidence_span",
        "retrieved_at",
        "_ifv_runtime_metrics",
    }
    assert result["status"] == "success"
    assert set(result).isdisjoint(forbidden)
    assert len(result["queries"]) == 1
    assert len(result["subcalls"]) == 1
    assert result["subcalls"][0]["kind"] == "search_query"
    assert result["subcalls"][0]["request_count"] == 1
    assert result["subcalls"][0]["result_count"] == 1
    for response in result["queries"]:
        assert response["results"]
        assert set(response).isdisjoint(forbidden)
        assert set(response["timings"]) == {"search_ms"}


def test_text_search_returns_discovery_without_hidden_visits_or_llm() -> None:
    client = FakeSearchClient()
    tool = TextSearchTool(client=client, top_k=5)

    result = tool.search(
        ["monarch Antarctica"],
        gl="us",
        hl="en",
        goal="Do monarch butterflies migrate to Antarctica?",
    )

    _assert_discovery_only(result)
    assert client.calls == [
        ("monarch Antarctica", 5, "us", "en"),
    ]


def test_text_search_rejects_multiple_queries_without_provider_calls() -> None:
    client = FakeSearchClient()
    tool = TextSearchTool(client=client, top_k=4)

    result = asyncio.run(
        tool.call_async(
            {
                "queries": ["one", "two"],
                "goal": "Find candidate pages.",
            }
        )
    )

    assert result["status"] == "error"
    assert "exactly one query" in result["error"]
    assert client.calls == []


def test_async_text_search_has_the_same_single_query_contract() -> None:
    client = FakeSearchClient()
    tool = TextSearchTool(client=client, top_k=4)

    result = asyncio.run(
        tool.call_async(
            {
                "queries": ["one"],
                "goal": "Find candidate pages.",
            }
        )
    )

    _assert_discovery_only(result)


def test_text_search_uses_jina_only_to_rerank_serper_candidates() -> None:
    client = FakeSearchClient()
    client.search = lambda query, **kwargs: {
        "query": query,
        "results": [
            {
                "title": "Low relevance",
                "url": "https://example.org/low",
                "snippet": "unrelated",
            },
            {
                "title": "High relevance",
                "url": "https://example.org/high",
                "snippet": "newborn sloth size",
            },
        ],
    }
    reranker = FakeReranker()
    tool = TextSearchTool(
        client=client,
        candidate_reranker=reranker,
        top_k=2,
    )

    result = tool.search(
        ["newborn sloth size"],
        goal="Find reliable newborn sloth size information.",
    )

    assert result["status"] == "success"
    assert [
        item["url"]
        for item in result["queries"][0]["results"]
    ] == [
        "https://example.org/high",
        "https://example.org/low",
    ]
    assert result["queries"][0]["candidate_selection"]["status"] == "success"
    assert result["subcalls"][-1]["kind"] == "candidate_rerank"
    assert reranker.calls[0][0] == "Find reliable newborn sloth size information."


def test_text_search_records_jina_reranker_failure_without_hiding_it() -> None:
    tool = TextSearchTool(
        client=FakeSearchClient(),
        candidate_reranker=FailingReranker(),
        top_k=2,
    )

    result = tool.search(["monarch Antarctica"])

    assert result["status"] == "success"
    selection = result["queries"][0]["candidate_selection"]
    assert selection["status"] == "error"
    assert "jina reranker unavailable" in selection["error"]
    assert result["subcalls"][-1]["status"] == "error"
