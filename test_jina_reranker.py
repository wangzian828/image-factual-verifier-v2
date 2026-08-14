from __future__ import annotations

from typing import Any

import pytest

from src.integrations.search.jina_reranker import JinaRerankerClient


def test_jina_reranker_preserves_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(
        "src.integrations.search.jina_reranker.requests.Session.post",
        fail,
    )

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        JinaRerankerClient(api_key="test").rerank(
            "newborn sloth size",
            ["candidate document"],
        )
