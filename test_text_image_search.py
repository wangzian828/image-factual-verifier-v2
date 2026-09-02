from __future__ import annotations

import asyncio

from src.tools.text_image_search import TextImageSearchTool


class FakeImageSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, str | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        gl: str | None = None,
        hl: str | None = None,
    ) -> list[dict]:
        self.calls.append((query, top_k, gl, hl))
        return [
            {
                "title": "Candidate image",
                "url": "https://example.org/page",
                "image_url": "https://cdn.example.org/image.jpg",
                "source": "example.org",
            }
        ]


def test_text_image_search_returns_unverified_image_discovery() -> None:
    client = FakeImageSearchClient()
    tool = TextImageSearchTool(client=client, top_k=3)

    result = tool.call(
        {
            "query": "NOAA Ship Henry B. Bigelow",
            "gl": "us",
            "hl": "en",
        }
    )

    assert result["status"] == "success"
    assert result["observation_status"] == "has_results"
    assert result["results"][0]["image_url"].endswith("image.jpg")
    assert result["reference_image_candidates"] == [
        "https://cdn.example.org/image.jpg"
    ]
    assert result["candidate_page_urls"] == ["https://example.org/page"]
    assert client.calls == [
        ("NOAA Ship Henry B. Bigelow", 3, "us", "en")
    ]


def test_text_image_search_rejects_empty_query_without_provider_call() -> None:
    client = FakeImageSearchClient()
    tool = TextImageSearchTool(client=client)

    result = asyncio.run(tool.call_async({"query": "  "}))

    assert result["status"] == "error"
    assert "non-empty" in result["error"]
    assert client.calls == []
