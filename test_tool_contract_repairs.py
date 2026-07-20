from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from src.integrations.vlm.qwen_vl import QwenVLClient
from src.orchestrator.stage_runner import StageRunner
from src.orchestrator.task_store import record_tool_observation
from src.orchestrator.tool_registry import build_all_tools_with_health
from src.tools.crop_and_inspect import CropAndInspectTool
from src.tools.ocr_with_position import OCRWithPositionTool
from test_image_only_state_machine import _runtime_state


def test_qwen_structured_vision_accepts_and_validates_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeChat:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def create_json_completion(self, **kwargs):
            calls.append(kwargs)
            return '{"name":"visible text","count":2}'

    monkeypatch.setattr(
        "src.integrations.vlm.qwen_vl.OpenAICompatibleChatClient",
        FakeChat,
    )
    monkeypatch.setattr(
        "src.integrations.vlm.qwen_vl.image_to_data_url",
        lambda _value: "data:image/png;base64,AA==",
    )
    client = QwenVLClient(api_key="test-key")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }

    result = client.create_image_json(
        system_prompt="Return JSON.",
        user_text="Inspect.",
        image_input="input.png",
        max_tokens=100,
        response_schema=schema,
    )

    assert result == {"name": "visible text", "count": 2}
    assert "response_schema" in inspect.signature(
        client.create_image_json
    ).parameters


def test_qwen_health_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    _, health = build_all_tools_with_health(
        vlm_provider="qwen",
        vlm_model="qwen-test",
    )

    structured = {
        "perceive_scene",
        "crop_and_inspect",
        "crop_and_search",
        "count_objects",
        "check_consistency",
        "reverse_image_search",
    }
    assert all(not health[name].available for name in structured)
    assert all("QWEN_API_KEY" in health[name].error for name in structured)


def test_error_search_payload_cannot_create_discovery() -> None:
    case, state = _runtime_state()
    task = state.tasks[0]
    step = SimpleNamespace(
        action_type="tool_call",
        tool_name="text_search",
        tool_args={"task_id": task.task_id},
        tool_result=json.dumps(
            {
                "status": "error",
                "error": "provider failed after returning partial rows",
                "queries": [
                    {
                        "query": "partial",
                        "results": [
                            {
                                "title": "must not land",
                                "url": "https://example.test/partial",
                                "snippet": "partial row",
                            }
                        ],
                    }
                ],
            }
        ),
        metadata={"function_call_id": "call-error-partial"},
    )

    update = record_tool_observation(
        state,
        step,
        image_sha256=case.image_sha256,
    )

    assert update["created_discovery_ids"] == []
    assert update["created_failure_ids"]
    assert state.discoveries == []


def test_search_runtime_payload_is_the_same_payload_shown_to_model() -> None:
    runner = object.__new__(StageRunner)
    runner.tool_response_max_chars = 6000
    raw = {
        "status": "success",
        "queries": [
            {
                "query": "audit",
                "provider": "serper",
                "results": [
                    {
                        "title": f"result {index}",
                        "url": f"https://example.test/{index}",
                        "snippet": "candidate",
                    }
                    for index in range(8)
                ],
            }
        ],
    }

    canonical = runner._canonical_tool_result("text_search", raw)
    serialized = json.dumps(canonical)

    assert runner._compact_tool_result_for_context(
        "text_search",
        serialized,
    ) == canonical
    assert canonical["queries"][0]["results"]
    assert "top_results" not in canonical["queries"][0]


def test_visit_canonical_payload_preserves_exact_passage_and_span() -> None:
    runner = object.__new__(StageRunner)
    runner.tool_response_max_chars = 6000
    runner.prior_steps = []
    evidence = "A" * 500
    raw = {
        "status": "success",
        "url": "https://example.test/evidence",
        "selected_url": "https://example.test/evidence",
        "provider": "direct_reader",
        "image_claim": "The image claims the subject used a bus.",
        "retrieval_goal": "Identify the transport actually used.",
        "summary": "Direct source passage.",
        "evidence": evidence,
        "relevance": "high",
        "stance": "support",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "a" * 64,
        "evidence_span": {"start": 1200, "end": 1700},
        "retrieved_at": "2026-07-16T00:00:00+00:00",
        "injection_flags": [],
        "evidence_eligible": True,
        "evidence_records": [
            {
                "url": "https://example.test/evidence",
                "selected_url": "https://example.test/evidence",
                "evidence": "Supporting exact span.",
                "image_claim": "The image claims the subject used a bus.",
                "retrieval_goal": "Identify the transport actually used.",
                "relevance": "medium",
                "stance": "unclear",
                "directness": "indirect",
                "context_only": True,
                "temporal_alignment": "not_applicable",
                "artifact_sha256": "a" * 64,
                "evidence_span": {"start": 1800, "end": 1822},
                "retrieved_at": "2026-07-16T00:00:00+00:00",
                "injection_flags": [],
                "evidence_eligible": True,
            }
        ],
        "subcalls": [
            {
                "kind": "page_fetch",
                "provider": "direct_reader",
                "status": "success",
                "request_count": 1,
            }
        ],
    }

    canonical = runner._canonical_tool_result("visit", raw)

    assert canonical["evidence"] == evidence
    assert len(canonical["evidence"]) == (
        canonical["evidence_span"]["end"]
        - canonical["evidence_span"]["start"]
    )
    assert canonical["evidence_records"][0]["evidence"] == (
        "Supporting exact span."
    )
    assert canonical["evidence_records"][0]["evidence_span"] == {
        "start": 1800,
        "end": 1822,
    }
    assert canonical["image_claim"] == (
        "The image claims the subject used a bus."
    )
    assert canonical["retrieval_goal"] == (
        "Identify the transport actually used."
    )
    assert canonical["evidence_records"][0]["image_claim"] == (
        canonical["image_claim"]
    )


def test_crop_and_inspect_rejects_reversed_or_negative_bbox() -> None:
    tool = CropAndInspectTool(client=object())

    result = tool.call(
        {
            "image_input": "unused.png",
            "bbox": [0.8, 0.8, -0.2, -0.2],
            "focus_question": "What is visible?",
        }
    )

    assert result["status"] == "error"
    assert "bbox" in result["error"]


def test_ocr_excludes_low_confidence_regions_from_canonical_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    class Reader:
        def readtext(self, _value):
            return [
                (
                    [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "garbage",
                    0.08,
                ),
                (
                    [[20, 0], [90, 0], [90, 10], [20, 10]],
                    "Tysons Corner",
                    0.92,
                ),
            ]

    tool = OCRWithPositionTool(_reader=Reader(), min_confidence=0.5)
    result = tool.call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["full_text"] == "Tysons Corner"
    assert [item["text"] for item in result["text_regions"]] == [
        "Tysons Corner"
    ]
    assert [item["text"] for item in result["rejected_text_regions"]] == [
        "garbage"
    ]
    assert result["ocr_backend"] == "easyocr"
    assert result["artifact_sha256"]
    assert tool._reader is not None


def test_ocr_prefers_configured_ppocr_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "sign.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "text_regions": [
                    {
                        "text": "Tysons Corner",
                        "confidence": 0.97,
                        "bbox": [10, 5, 90, 25],
                    }
                ]
            }

    calls = []
    monkeypatch.setattr(
        "src.tools.ocr_with_position.requests.post",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or Response()
        ),
    )
    tool = OCRWithPositionTool(
        ppocr_service_url="http://127.0.0.1:9999/ocr",
    )

    result = tool.call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["ocr_backend"] == "ppocr_service"
    assert result["full_text"] == "Tysons Corner"
    assert result["backend_attempts"] == [
        {"backend": "ppocr_service", "status": "success"}
    ]
    assert len(calls) == 1


def test_ocr_service_failure_falls_back_to_easyocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "fallback.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    class Reader:
        def readtext(self, _value):
            return [
                (
                    [[10, 5], [90, 5], [90, 25], [10, 25]],
                    "fallback text",
                    0.9,
                )
            ]

    monkeypatch.setattr(
        "src.tools.ocr_with_position.requests.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("service unavailable")
        ),
    )
    tool = OCRWithPositionTool(
        _reader=Reader(),
        ppocr_service_url="http://127.0.0.1:9999/ocr",
    )

    result = tool.call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["ocr_backend"] == "easyocr"
    assert result["full_text"] == "fallback text"
    assert result["backend_attempts"][0]["status"] == "error"
    assert result["backend_attempts"][1] == {
        "backend": "easyocr",
        "status": "success",
    }


def test_ocr_all_backend_failures_preserve_subcall_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "all-fail.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    class Reader:
        def readtext(self, _value):
            raise RuntimeError("easyocr unavailable")

    monkeypatch.setattr(
        "src.tools.ocr_with_position.requests.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("ppocr unavailable")
        ),
    )
    tool = OCRWithPositionTool(
        _reader=Reader(),
        ppocr_service_url="http://127.0.0.1:9999/ocr",
    )

    result = tool.call({"image_input": str(image_path)})

    assert result["status"] == "error"
    assert [item["provider"] for item in result["subcalls"]] == [
        "ppocr_service",
        "easyocr",
    ]
    assert all(item["status"] == "error" for item in result["subcalls"])


def test_ocr_rejects_reversed_bbox_and_hashes_actual_crop(
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "crop.png"
    image = Image.new("RGB", (100, 100), "white")
    image.paste("black", (0, 0, 50, 50))
    image.save(image_path)

    class Reader:
        def readtext(self, _value):
            return []

    tool = OCRWithPositionTool(_reader=Reader())
    invalid = tool.call(
        {
            "image_input": str(image_path),
            "bbox": [0.8, 0.8, 0.2, 0.2],
        }
    )
    cropped = tool.call(
        {
            "image_input": str(image_path),
            "bbox": [0.0, 0.0, 0.5, 0.5],
        }
    )
    full = tool.call({"image_input": str(image_path)})

    assert invalid["status"] == "error"
    assert "ordered" in invalid["error"]
    assert cropped["artifact_sha256"] != full["artifact_sha256"]
