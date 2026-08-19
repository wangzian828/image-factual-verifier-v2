from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from src.integrations.vlm.qwen_vl import QwenVLClient
from src.orchestrator.stage_runner import StageRunner
from src.orchestrator.task_store import record_tool_observation
from src.orchestrator.tool_registry import build_all_tools_with_health
from src.tools.crop_and_inspect import CropAndInspectTool, INSPECT_SCHEMA
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
                    for index in range(10)
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
    assert len(canonical["queries"][0]["results"]) == 10
    assert canonical["queries"][0]["results"][-1]["url"] == (
        "https://example.test/9"
    )
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
        "relation_scope": "same_relation",
        "relation_stance": "supports",
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
                "relation_scope": "partial_relation",
                "relation_stance": "background",
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


def test_crop_and_inspect_requests_observations_without_direct_answer_field() -> None:
    assert "observations" in INSPECT_SCHEMA["properties"]
    assert "answer" not in INSPECT_SCHEMA["properties"]


def test_crop_and_inspect_keeps_legacy_findings_as_observations(
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "legacy-crop.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    class FakeClient:
        def create_image_json(self, **kwargs):
            return {
                "description": "A small cropped region.",
                "findings": ["A visible horizontal edge crosses the region."],
                "answer": "This is definitely fake.",
            }

    result = CropAndInspectTool(client=FakeClient()).call(
        {
            "image_input": str(image_path),
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "focus_question": "What is visibly present?",
        }
    )

    assert result["status"] == "success"
    assert result["observations"] == [
        "A visible horizontal edge crosses the region."
    ]
    assert "answer" not in result


class _FakeOCRResponse:
    def __init__(self, payload=None, *, text="", status_code=200):
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self.ok = 200 <= status_code < 400

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class _FakeOCRHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.responses.pop(0)


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_filters_low_confidence_regions_and_parses_jsonl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    result_jsonl = "\n".join(
        [
            json.dumps(
                {
                    "result": {
                        "layoutParsingResults": [
                            {
                                "prunedResult": json.dumps(
                                    {
                                        "rec_polys": [
                                            [
                                                [0, 0],
                                                [20, 0],
                                                [20, 10],
                                                [0, 10],
                                            ],
                                            [
                                                [20, 0],
                                                [90, 0],
                                                [90, 10],
                                                [20, 10],
                                            ],
                                        ],
                                        "rec_texts": [
                                            "garbage",
                                            "Tysons Corner",
                                        ],
                                        "rec_scores": [0.08, 0.92],
                                    }
                                )
                            }
                        ]
                    }
                }
            )
        ]
    )
    http = _FakeOCRHTTP(
        [
            _FakeOCRResponse({"data": {"jobId": "job-1"}}),
            _FakeOCRResponse({"data": {"state": "pending"}}),
            _FakeOCRResponse(
                {
                    "data": {
                        "state": "done",
                        "resultUrl": {"jsonUrl": "https://result.test/job-1.jsonl"},
                    }
                }
            ),
            _FakeOCRResponse(text=result_jsonl),
        ]
    )
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "test-token")
    tool = OCRWithPositionTool(
        http_client=http,
        poll_seconds=0.2,
        min_confidence=0.5,
    )
    result = tool.call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["full_text"] == "Tysons Corner"
    assert [item["text"] for item in result["text_regions"]] == [
        "Tysons Corner"
    ]
    assert [item["text"] for item in result["rejected_text_regions"]] == [
        "garbage"
    ]
    assert result["ocr_backend"] == "paddleocr_api"
    assert result["artifact_sha256"]
    assert len(http.calls) == 4
    assert http.calls[0][0] == "post"
    assert http.calls[0][2]["data"]["model"] == "PaddleOCR-VL-1.6"
    assert http.calls[0][2]["headers"]["Authorization"] == "bearer test-token"


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_retries_transient_queue_full_without_local_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "queue-retry.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    http = _FakeOCRHTTP(
        [
            _FakeOCRResponse(
                {"code": 10010, "msg": "任务提交队列已满，请稍后重试"},
                status_code=400,
            ),
            _FakeOCRResponse({"data": {"jobId": "job-retry"}}),
            _FakeOCRResponse(
                {
                    "data": {
                        "state": "done",
                        "resultUrl": {
                            "jsonUrl": "https://result.test/job-retry.jsonl"
                        },
                    }
                }
            ),
            _FakeOCRResponse(
                text=json.dumps({"result": {"layoutParsingResults": []}})
            ),
        ]
    )
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "test-token")
    monkeypatch.setenv("PADDLEOCR_API_SUBMIT_RETRIES", "1")
    monkeypatch.setenv("PADDLEOCR_API_RETRY_BACKOFF_SECONDS", "0")

    result = OCRWithPositionTool(http_client=http, poll_seconds=0.2).call(
        {"image_input": str(image_path)}
    )

    assert result["status"] == "success"
    assert result["ocr_backend"] == "paddleocr_api"
    assert [call[0] for call in http.calls] == ["post", "post", "get", "get"]
    assert result["backend_attempts"][0]["status"] == "retry"
    assert result["backend_attempts"][1]["status"] == "success"


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_accepts_axis_aligned_boxes_and_maps_crop_coordinates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "crop.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    http = _FakeOCRHTTP(
        [
            _FakeOCRResponse({"data": {"jobId": "job-2"}}),
            _FakeOCRResponse(
                {
                    "data": {
                        "state": "done",
                        "resultUrl": {"jsonUrl": "https://result.test/job-2.jsonl"},
                    }
                }
            ),
            _FakeOCRResponse(
                text=json.dumps(
                    {
                        "result": {
                            "layoutParsingResults": [
                                {
                                    "prunedResult": json.dumps(
                                        {
                                            "rec_boxes": [[6, 5, 54, 20]],
                                            "rec_texts": ["Tysons Corner"],
                                            "rec_scores": [0.92],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                )
            ),
        ]
    )
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "test-token")
    result = OCRWithPositionTool(
        http_client=http,
        poll_seconds=0.2,
    ).call(
        {
            "image_input": str(image_path),
            "bbox": [0.2, 0.2, 0.8, 0.8],
        }
    )

    assert result["status"] == "success"
    assert result["ocr_backend"] == "paddleocr_api"
    assert result["full_text"] == "Tysons Corner"
    assert result["text_regions"][0]["bbox"] == [0.26, 0.25, 0.74, 0.4]
    assert result["requested_bbox"] == [0.2, 0.2, 0.8, 0.8]


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_failure_is_explicit_without_local_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "failure.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    http = _FakeOCRHTTP(
        [
            _FakeOCRResponse(
                {"error": "quota exceeded"},
                text="quota exceeded",
                status_code=429,
            )
        ]
    )
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "test-token")
    result = OCRWithPositionTool(http_client=http).call(
        {"image_input": str(image_path)}
    )

    assert result["status"] == "error"
    assert "HTTP 429" in result["error"]
    assert result["backend_attempts"][0]["backend"] == "paddleocr_api"
    assert len(result["subcalls"]) == 1
    assert result["subcalls"][0]["provider"] == "paddleocr_api"


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_requires_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "missing-token.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    monkeypatch.delenv("PADDLEOCR_API_TOKEN", raising=False)
    result = OCRWithPositionTool().call({"image_input": str(image_path)})

    assert result["status"] == "error"
    assert "PADDLEOCR_API_TOKEN is required" in result["error"]


@pytest.mark.skip(reason="PaddleOCR API backend was removed; runtime is EasyOCR-only")
def test_ocr_api_does_not_treat_image_only_markdown_as_text(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "image-only.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    http = _FakeOCRHTTP(
        [
            _FakeOCRResponse({"data": {"jobId": "job-3"}}),
            _FakeOCRResponse(
                {
                    "data": {
                        "state": "done",
                        "resultUrl": {"jsonUrl": "https://result.test/job-3.jsonl"},
                    }
                }
            ),
            _FakeOCRResponse(
                text=json.dumps(
                    {
                        "result": {
                            "layoutParsingResults": [
                                {
                                    "markdown": {
                                        "text": (
                                            '<div><img src="imgs/image.jpg" '
                                            'alt="Image" /></div>'
                                        )
                                    },
                                    "prunedResult": {
                                        "parsing_res_list": [
                                            {
                                                "block_content": "",
                                                "block_bbox": [0, 0, 100, 40],
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    }
                )
            ),
        ]
    )
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "test-token")

    result = OCRWithPositionTool(
        http_client=http,
        poll_seconds=0.2,
    ).call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["text_regions"] == []
    assert result["total_regions"] == 0
    assert result["full_text"] == ""


def test_ocr_rejects_reversed_bbox_and_hashes_actual_crop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    monkeypatch.setenv("OCR_BACKEND", "easyocr")

    class EmptyReader:
        def readtext(self, _image, **_kwargs):
            return []

    monkeypatch.setattr(
        OCRWithPositionTool,
        "_get_reader",
        lambda _self: EmptyReader(),
    )
    monkeypatch.setattr(
        OCRWithPositionTool,
        "_ocr_model",
        lambda _self: "easyocr-test",
    )
    image_path = tmp_path / "crop.png"
    image = Image.new("RGB", (100, 100), "white")
    image.paste("black", (0, 0, 50, 50))
    image.save(image_path)

    tool = OCRWithPositionTool()
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
    assert cropped["status"] == "success"
    assert full["status"] == "success"
    assert cropped["artifact_sha256"]
    assert full["artifact_sha256"]
    assert cropped["artifact_sha256"] != full["artifact_sha256"]
