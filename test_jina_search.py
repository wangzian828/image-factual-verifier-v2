from __future__ import annotations

from typing import Any

import pytest

from src.integrations.search.jina_search import JinaRerankerClient, JinaSearchClient
from src.tools.jina_search import JinaSearchTool


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def test_jina_search_normalizes_content_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.integrations.search.jina_search.requests.get",
        lambda *args, **kwargs: _Response(
            {
                "data": [
                    {
                        "title": "Zoo",
                        "url": "https://example.test/zoo",
                        "content": "Newborn sloths weigh several hundred grams.",
                        "publishedTime": "2026-01-01",
                    }
                ]
            }
        ),
    )

    result = JinaSearchClient(api_key="test").search("newborn sloth size")

    assert result["provider"] == "jina_search"
    assert result["results"][0]["url"] == "https://example.test/zoo"
    assert "several hundred grams" in result["results"][0]["content_preview"]
    assert result["results"][0]["snippet"]


def test_jina_search_tool_reports_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("JINA_API_KEYS", raising=False)
    tool = JinaSearchTool(client=JinaSearchClient(api_key=""))

    result = tool.call({"queries": ["newborn sloth size"]})

    assert result["status"] == "error"
    assert result["provider"] == "jina_search"
    assert "JINA_API_KEY" in result["error"]


def test_jina_reranker_preserves_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(
        "src.integrations.search.jina_search.requests.post",
        fail,
    )

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        JinaRerankerClient(api_key="test").rerank(
            "query",
            ["document"],
        )
