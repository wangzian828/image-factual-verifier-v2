from __future__ import annotations

from typing import Any

from src.tools.visit import MAX_VISIT_URLS_PER_ACTION, VisitTool


class BatchClient:
    def __init__(self) -> None:
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def visit(self, url: str, **_: Any) -> dict:
        self.single_calls.append(url)
        return {
            "status": "success",
            "url": url,
            "evidence": f"Evidence from {url}",
            "summary": "single page",
            "relevance": "medium",
        }

    def visit_many(self, urls: list[str], **_: Any) -> dict:
        self.batch_calls.append(urls)
        visits = [
            {
                "status": "success",
                "url": url,
                "evidence": f"Evidence from {url}",
                "summary": "independent page",
                "relevance": "medium",
            }
            for url in urls
        ]
        return {
            "status": "success",
            "visits": visits,
            "evidence": visits[0]["evidence"],
            "summary": visits[0]["summary"],
            "relevance": visits[0]["relevance"],
        }


def test_visit_batches_up_to_three_pages_without_merging_page_records() -> None:
    client = BatchClient()
    tool = VisitTool(client=client)

    result = tool.call(
        {
            "url": [
                "https://one.example",
                "https://two.example",
                "https://three.example",
            ],
            "image_claim": "claim",
            "retrieval_goal": "goal",
        }
    )

    assert result["status"] == "success"
    assert client.single_calls == []
    assert client.batch_calls == [[
        "https://one.example",
        "https://two.example",
        "https://three.example",
    ]]
    assert [item["url"] for item in result["visits"]] == [
        "https://one.example",
        "https://two.example",
        "https://three.example",
    ]


def test_visit_rejects_more_than_three_pages_before_provider_call() -> None:
    client = BatchClient()
    tool = VisitTool(client=client)

    result = tool.call(
        {
            "url": [
                f"https://{index}.example"
                for index in range(MAX_VISIT_URLS_PER_ACTION + 1)
            ],
            "image_claim": "claim",
            "retrieval_goal": "goal",
        }
    )

    assert result["status"] == "error"
    assert client.single_calls == []
    assert client.batch_calls == []
