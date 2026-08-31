from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.integrations.browse import jina_reader


class FakeGeminiInteractionsClient:
    instances = 0
    requests = 0
    closes = 0
    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, **_: Any) -> None:
        type(self).instances += 1

    async def create(self, **_: Any) -> dict[str, Any]:
        cls = type(self)
        with cls.lock:
            cls.requests += 1
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        await asyncio.sleep(0.01)
        with cls.lock:
            cls.active -= 1
        return {
            "id": f"interaction-{cls.requests}",
            "status": "completed",
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_thought_tokens": 0,
            },
            "output_text": json.dumps(
                {
                    "rationale": "matched",
                    "passage_id": 0,
                    "supporting_passage_ids": [],
                    "summary": "The passage supports the relation.",
                    "relevance": "high",
                    "relation_scope": "same_relation",
                    "relation_stance": "supports",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                }
            ),
        }

    async def aclose(self) -> None:
        type(self).closes += 1


def _reset_fake() -> None:
    FakeGeminiInteractionsClient.instances = 0
    FakeGeminiInteractionsClient.requests = 0
    FakeGeminiInteractionsClient.closes = 0
    FakeGeminiInteractionsClient.active = 0
    FakeGeminiInteractionsClient.max_active = 0


def _extract(reader: jina_reader.JinaReaderClient) -> dict[str, Any]:
    return reader._extract_with_llm(
        "[PASSAGE 0] The source states the claimed relation.",
        image_claim="The image depicts the claimed relation.",
        retrieval_goal="Find the source passage for the relation.",
    )


def test_jina_reuses_one_gemini_transport_and_allows_parallel_extracts(
    monkeypatch,
) -> None:
    _reset_fake()
    monkeypatch.setattr(
        jina_reader,
        "GeminiInteractionsClient",
        FakeGeminiInteractionsClient,
    )
    reader = jina_reader.JinaReaderClient(
        extract_provider="gemini",
        extract_model="fake-model",
    )
    try:
        assert _extract(reader)["stance"] == "support"
        assert _extract(reader)["stance"] == "support"

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: _extract(reader), range(3)))

        assert all(item["stance"] == "support" for item in results)
        assert FakeGeminiInteractionsClient.instances == 1
        assert FakeGeminiInteractionsClient.requests == 5
        assert FakeGeminiInteractionsClient.max_active >= 2
    finally:
        reader.close()

    assert FakeGeminiInteractionsClient.closes == 1


def test_jina_reuses_page_visit_executor_across_batches() -> None:
    reader = jina_reader.JinaReaderClient(max_workers=3)

    def fake_visit(
        url: str,
        *,
        image_claim: str,
        retrieval_goal: str,
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "url": url,
            "image_claim": image_claim,
            "retrieval_goal": retrieval_goal,
            "evidence": f"Evidence from {url}",
            "summary": "ok",
            "relevance": "medium",
        }

    reader.visit = fake_visit  # type: ignore[method-assign]
    try:
        reader.visit_many(
            ["https://one.example", "https://two.example"],
            image_claim="claim",
            retrieval_goal="goal",
        )
        executor = reader._visit_executor
        assert executor is not None

        reader.visit_many(
            ["https://three.example", "https://four.example"],
            image_claim="claim",
            retrieval_goal="goal",
        )
        assert reader._visit_executor is executor
    finally:
        reader.close()

    assert executor._shutdown is True


def test_jina_summary_context_has_a_hard_character_bound() -> None:
    passages = [
        {
            "passage_id": index,
            "start": index * 1000,
            "end": (index + 1) * 1000,
            "text": (
                f"Relation evidence paragraph {index}. "
                + ("relevant relation text " * 80)
            ),
        }
        for index in range(80)
    ]

    selected = jina_reader.JinaReaderClient._select_goal_passages(
        passages,
        "relation evidence",
        max_chars=999999,
        max_passages=999,
    )

    formatted = jina_reader.JinaReaderClient._format_evidence_passages(selected)
    assert len(formatted) <= jina_reader.MAX_EXTRACT_INPUT_CHARS
    assert 0 < len(selected) < len(passages)
