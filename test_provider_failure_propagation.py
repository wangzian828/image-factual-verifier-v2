from __future__ import annotations

import asyncio
import hashlib
import json
import re

import pytest

from src.integrations.browse.jina_reader import JinaReaderClient
from src.tools.reverse_image_search import ReverseImageSearchTool
from src.tools.text_search import TextSearchTool
from src.tools.visit import VisitTool


class StubTextSearchClient:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def search(self, query: str, **_kwargs) -> dict:
        return {"query": query, "provider": "serper", "results": self.results}


class StubVlmClient:
    def create_image_json(self, **_kwargs) -> dict:
        return {"query": "semantic query", "keywords": []}


class StubImageSearchClient:
    def search(self, **_kwargs) -> list[dict]:
        return [
            {
                "url": "https://semantic.example/page",
                "image_url": "https://semantic.example/image.jpg",
            }
        ]


def _successful_visit(url: str) -> dict:
    return {
        "status": "success",
        "url": url,
        "provider": "jina_reader",
        "evidence": "evidence",
        "summary": "summary",
        "rationale": "relevant",
        "relevance": "high",
        "stance": "support",
        "blocked": False,
    }


def test_extractor_rejects_unknown_passage_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaReaderClient(fetch_provider="jina")
    monkeypatch.setattr(
        client,
        "_extract_with_llm",
        lambda _content, _goal: {
            "rationale": "claimed match",
            "passage_id": 999,
            "summary": "Fabricated summary.",
            "relevance": "high",
            "stance": "support",
        },
    )

    with pytest.raises(RuntimeError, match="unknown passage_id"):
        client.extract_goal_evidence("The source page contains a different statement.", "verify")


def test_extractor_rejects_invalid_stance(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaReaderClient(fetch_provider="jina")
    monkeypatch.setattr(
        client,
        "_extract_with_llm",
        lambda _content, _goal: {
            "rationale": "relevant",
            "passage_id": 0,
            "summary": "Relevant statement.",
            "relevance": "high",
            "stance": "probably_supports",
        },
    )

    with pytest.raises(RuntimeError, match="invalid stance"):
        client.extract_goal_evidence("The source page contains this statement.", "verify")


def test_jina_navigation_is_excluded_from_exact_passage_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = JinaReaderClient(fetch_provider="jina", max_chars=4000)
    page = """Title: Artemis II Mission - NASA

URL Source: https://www.nasa.gov/mission/artemis-ii/

Published Time: 2026-04-10T00:00:00Z

Markdown Content:

[![Image 1: NASA Logo](https://images.nasa.gov/logo.png)](https://www.nasa.gov/)

* [News](https://www.nasa.gov/news/)
* [Images](https://www.nasa.gov/images/)
* [About](https://www.nasa.gov/about/)

## Mission

NASA describes Artemis II as a crewed lunar flyby around the Moon.
"""

    def select_fact_passage(formatted: str, _goal: str) -> dict:
        assert "URL Source" not in formatted
        assert "NASA Logo" not in formatted
        assert "https://" not in formatted
        assert "[News]" not in formatted
        match = re.search(
            r"\[PASSAGE (\d+)\] (NASA describes Artemis II[^\n]+)",
            formatted,
        )
        assert match is not None
        return {
            "rationale": "The passage directly answers the goal.",
            "passage_id": int(match.group(1)),
            "summary": "NASA describes a crewed lunar flyby.",
            "relevance": "high",
            "stance": "support",
            "directness": "direct",
        }

    monkeypatch.setattr(client, "_extract_with_llm", select_fact_passage)
    result = client.extract_goal_evidence(page, "Is Artemis II a crewed lunar flyby?")
    document = client._prepare_evidence_document(page)[: client.max_chars]

    assert result["evidence"] == (
        "NASA describes Artemis II as a crewed lunar flyby around the Moon."
    )
    span = result["evidence_span"]
    assert document[span["start"] : span["end"]] == result["evidence"]
    assert result["artifact_sha256"] == hashlib.sha256(document.encode("utf-8")).hexdigest()


def test_jina_visit_many_marks_all_provider_failures_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaReaderClient(fetch_provider="jina")

    def fail(_url: str, _goal: str) -> dict:
        raise RuntimeError("selected Jina provider unavailable")

    monkeypatch.setattr(client, "visit", fail)
    result = client.visit_many(["https://one.example", "https://two.example"], "goal")

    assert result["status"] == "error"
    assert "selected Jina provider unavailable" in result["error"]
    assert len(result["visits"]) == 2
    assert all(visit["status"] == "error" for visit in result["visits"])


def test_jina_visit_many_allows_partial_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaReaderClient(fetch_provider="jina")

    def visit(url: str, _goal: str) -> dict:
        if "one.example" in url:
            raise RuntimeError("first page failed")
        return _successful_visit(url)

    monkeypatch.setattr(client, "visit", visit)
    result = client.visit_many(["https://one.example", "https://two.example"], "goal")

    assert result["status"] == "success"
    assert "error" not in result
    assert result["selected_url"] == "https://two.example"


def test_jina_visit_many_treats_all_blocked_pages_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaReaderClient(fetch_provider="jina")

    def blocked(url: str, _goal: str) -> dict:
        return {
            "status": "error",
            "url": url,
            "provider": "jina_reader",
            "blocked": True,
            "blocked_reason": "cloudflare",
            "error": "Page visit was blocked by security verification: cloudflare",
            "relevance": "low",
        }

    monkeypatch.setattr(client, "visit", blocked)
    result = client.visit_many(["https://one.example", "https://two.example"], "goal")

    assert result["status"] == "error"
    assert "security verification" in result["error"]


def test_visit_tool_converts_single_provider_exception_to_error() -> None:
    class FailingBrowseClient:
        def visit(self, _url: str, _goal: str) -> dict:
            raise RuntimeError("direct provider failed")

    result = VisitTool(client=FailingBrowseClient()).visit("https://example.test", "goal")

    assert result == {
        "status": "error",
        "error": "RuntimeError: direct provider failed",
    }


def test_visit_tool_rejects_unaggregated_all_url_failures() -> None:
    class BrowseClient:
        def visit_many(self, _urls: list[str], _goal: str) -> dict:
            return {
                "visits": [
                    {"url": "https://one.example", "status": "error", "error": "one failed"},
                    {"url": "https://two.example", "status": "error", "error": "two failed"},
                ]
            }

    result = VisitTool(client=BrowseClient()).visit(
        ["https://one.example", "https://two.example"],
        "goal",
    )

    assert result["status"] == "error"
    assert result["error"] == "one failed; two failed"


@pytest.mark.parametrize("async_call", [False, True])
def test_text_search_propagates_search_provider_error(async_call: bool) -> None:
    class FailingSearchClient:
        def search(self, *_args, **_kwargs) -> dict:
            raise RuntimeError("Serper search failed")

    tool = TextSearchTool(client=FailingSearchClient())

    if async_call:
        result = asyncio.run(tool.call_async({"queries": ["query"]}))
    else:
        result = tool.search(["query"])

    assert result["status"] == "error"
    assert result["error"] == "RuntimeError: Serper search failed"
    assert result["queries"][0]["search_error"] == (
        "RuntimeError: Serper search failed"
    )


def test_text_search_empty_provider_results_are_success() -> None:
    tool = TextSearchTool(client=StubTextSearchClient([]))
    result = tool.search("query")

    assert result["status"] == "success"
    assert result["queries"][0]["results"] == []
    assert "error" not in result


def test_text_search_does_not_emit_browse_or_evidence_fields() -> None:
    tool = TextSearchTool(
        client=StubTextSearchClient([{"url": "https://result.example"}]),
    )
    result = tool.search("query", goal="Check a factual proposition")

    serialized = json.dumps(result)
    assert result["status"] == "success"
    assert "visited_pages" not in serialized
    assert "evidence_eligible" not in serialized
    assert "selected_url" not in serialized
    assert "artifact_sha256" not in serialized


def test_reverse_image_search_visual_failure_is_not_masked_by_semantic_success() -> None:
    class FailingVisualSearchClient:
        def search(self, _image_input: str, **_kwargs) -> dict:
            raise RuntimeError("selected visual provider failed")

    tool = ReverseImageSearchTool(
        vlm_client=StubVlmClient(),
        image_search_client=StubImageSearchClient(),
        lens_client=object(),
        visual_search_client=FailingVisualSearchClient(),
    )
    result = tool.search("image.png")

    assert result["status"] == "error"
    assert result["semantic_results"]
    assert "selected visual provider failed" in result["error"]


def test_reverse_image_search_explicit_visual_error_is_not_masked() -> None:
    class FailingVisualSearchClient:
        def search(self, _image_input: str, **_kwargs) -> dict:
            return {
                "status": "error",
                "error": "Lens provider rejected the request",
                "provider": "serper_lens",
                "results": [],
            }

    tool = ReverseImageSearchTool(
        vlm_client=StubVlmClient(),
        image_search_client=StubImageSearchClient(),
        lens_client=object(),
        visual_search_client=FailingVisualSearchClient(),
    )
    result = tool.search("image.png")

    assert result["status"] == "error"
    assert result["semantic_results"]
    assert result["error"] == "Lens provider rejected the request"


def test_reverse_image_search_visual_zero_results_are_success() -> None:
    class EmptyVisualSearchClient:
        def search(self, _image_input: str, **_kwargs) -> dict:
            return {
                "provider": "serper_lens",
                "image_url": "https://images.example/input.png",
                "results": [],
                "timings": {},
            }

    class EmptyImageSearchClient:
        def search(self, **_kwargs) -> list[dict]:
            return []

    tool = ReverseImageSearchTool(
        vlm_client=StubVlmClient(),
        image_search_client=EmptyImageSearchClient(),
        lens_client=object(),
        visual_search_client=EmptyVisualSearchClient(),
    )
    result = tool.search("image.png")

    assert result["status"] == "success"
    assert result["lens_results"] == []
    assert result["semantic_results"] == []
    assert "error" not in result
