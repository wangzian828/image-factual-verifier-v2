from __future__ import annotations

from src.tools.count_objects import COUNT_RESPONSE_SCHEMA, CountObjectsTool


class FakeVisionClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create_image_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_count_objects_uses_exact_response_schema() -> None:
    client = FakeVisionClient(
        {
            "target": "windows",
            "count": 3,
            "confidence": 0.9,
            "details": "Three distinct windows are visible.",
            "locations": ["left", "center", "right"],
        }
    )
    tool = CountObjectsTool(client=client, model_name="gemini-test")

    result = tool.call({"image_input": "input.png", "target_object": "windows"})

    assert result == {
        "status": "success",
        "target": "windows",
        "count": 3,
        "confidence": 0.9,
        "details": "Three distinct windows are visible.",
        "locations": ["left", "center", "right"],
    }
    assert len(client.calls) == 1
    assert client.calls[0]["response_schema"] == COUNT_RESPONSE_SCHEMA
    assert client.calls[0]["model_name"] == "gemini-test"


def test_count_objects_rejects_invalid_structured_output() -> None:
    client = FakeVisionClient(
        {
            "target": "windows",
            "count": 2.5,
            "confidence": 0.9,
            "details": "Ambiguous count.",
            "locations": [],
        }
    )
    tool = CountObjectsTool(client=client)

    result = tool.call({"image_input": "input.png", "target_object": "windows"})

    assert result["status"] == "error"
    assert "non-negative integer" in result["error"]


def test_count_objects_rejects_invalid_bbox_without_full_image_fallback() -> None:
    client = FakeVisionClient({})
    tool = CountObjectsTool(client=client)

    result = tool.call(
        {
            "image_input": "input.png",
            "target_object": "windows",
            "bbox": [0.8, 0.1, 0.2, 0.9],
        }
    )

    assert result["status"] == "error"
    assert "x1 < x2" in result["error"]
    assert client.calls == []
