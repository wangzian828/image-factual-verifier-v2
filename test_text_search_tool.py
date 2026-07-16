from __future__ import annotations

import asyncio

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
    assert len(result["queries"]) == 2
    for response in result["queries"]:
        assert response["results"]
        assert set(response).isdisjoint(forbidden)
        assert set(response["timings"]) == {"search_ms"}


def test_text_search_returns_discovery_without_hidden_visits_or_llm() -> None:
    client = FakeSearchClient()
    tool = TextSearchTool(client=client, top_k=5)

    result = tool.search(
        ["monarch", "antarctica"],
        gl="us",
        hl="en",
        goal="Do monarch butterflies migrate to Antarctica?",
    )

    _assert_discovery_only(result)
    assert client.calls == [
        ("monarch", 5, "us", "en"),
        ("antarctica", 5, "us", "en"),
    ]


def test_async_text_search_has_the_same_discovery_contract() -> None:
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

    _assert_discovery_only(result)
