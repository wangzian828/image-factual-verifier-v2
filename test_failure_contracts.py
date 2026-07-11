"""Failure-path contracts for the active multi-stage agent."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

import src.orchestrator.pipeline as pipeline_module
from src.integrations.browse.jina_reader import JinaReaderClient
from src.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_wire_api,
)
from src.integrations.search.visual_search import (
    ImageUploadClient,
    VisualReverseSearchClient,
)
from src.integrations.vlm.qwen_vl import parse_json_object
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import (
    InvestigationQuestion,
    PerceptionReport,
    VerificationPlan,
    VerificationState,
)
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_health import ToolHealth
from src.orchestrator.tool_registry import REQUIRED_TOOLS
from src.orchestrator.tool_result import ToolResultContractError, serialize_tool_result
from src.redaction import REDACTED, sanitize_for_persistence
from src.tools.compare_reference import CompareWithReferenceTool
from test_unit import FakeLLM, FakeOrchestrator, FakeTool, make_test_image


def test_malformed_json_and_statusless_tool_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_json_object("prefix {not-json} suffix")
    with pytest.raises(ToolResultContractError, match="missing a valid status"):
        serialize_tool_result({"results": []})

    step = StageStep(
        action_type="tool_call",
        tool_name="text_search",
        tool_result='{"results": []}',
    )
    assert Orchestrator._tool_step_succeeded(step) is False


def test_verification_raises_when_every_tool_fails() -> None:
    image_path = make_test_image()
    try:
        orchestrator = FakeOrchestrator(image_path)
        orchestrator.max_rounds_verification = 1
        orchestrator.max_verification_iterations = 1
        orchestrator.llm = FakeLLM(
            [
                '<tool_call>{"name":"reverse_image_search","arguments":{"question_id":"q0"}}</tool_call>',
                '<output>{"evidence":[],"visual_anomalies":[],"authenticity_assessment":"uncertain","key_findings":[]}</output>',
            ]
        )
        orchestrator.all_tools["reverse_image_search"] = FakeTool(
            "reverse_image_search",
            {"status": "error", "error": "search backend unavailable"},
            parameters={
                "type": "object",
                "properties": {"image_input": {"type": "string"}},
                "required": ["image_input"],
            },
        )
        state = VerificationState(
            image_path=image_path,
            perception=PerceptionReport(scene_description="test image"),
            plan=VerificationPlan(
                questions=[
                    InvestigationQuestion(
                        question_id="q0",
                        question="Find the image source.",
                        priority=1,
                    )
                ]
            ),
        )

        with pytest.raises(RuntimeError, match="every attempted tool call failed"):
            asyncio.run(orchestrator._run_verification(state, image_path))
    finally:
        Path(image_path).unlink(missing_ok=True)
        Path(image_path).parent.rmdir()


def test_required_tool_failure_aborts_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    health = {
        name: ToolHealth(available=name != "visit", error="dependency missing" if name == "visit" else "")
        for name in ("perceive_scene", "text_search", "visit", "reverse_image_search")
    }
    monkeypatch.setattr(
        pipeline_module,
        "build_all_tools_with_health",
        lambda **_kwargs: ({}, health),
    )

    with pytest.raises(RuntimeError, match="visit: dependency missing"):
        Orchestrator(
            provider="lmdeploy",
            model_name="test-model",
            validate_startup=True,
        )


def test_positioned_ocr_is_a_required_runtime_tool() -> None:
    assert "ocr_with_position" in REQUIRED_TOOLS


def test_perception_aborts_when_positioned_ocr_fails() -> None:
    image_path = make_test_image()
    try:
        orchestrator = FakeOrchestrator(image_path)
        orchestrator.all_tools["ocr_with_position"] = FakeTool(
            "ocr_with_position",
            {"status": "error", "error": "OCR model unavailable"},
            parameters={
                "type": "object",
                "properties": {"image_input": {"type": "string"}},
                "required": ["image_input"],
            },
        )
        state = VerificationState(image_path=image_path)

        with pytest.raises(RuntimeError, match="ocr_with_position failed"):
            asyncio.run(orchestrator._run_perception(state, image_path))

        assert [step.tool_name for step in state.all_steps] == [
            "perceive_scene",
            "ocr_with_position",
        ]
    finally:
        Path(image_path).unlink(missing_ok=True)
        Path(image_path).parent.rmdir()


@pytest.mark.parametrize("wire_api", ["chat_completions", "responses", "openai_compat"])
def test_gemini_rejects_non_interactions_protocol(wire_api: str) -> None:
    with pytest.raises(ValueError, match="requires wire_api='interactions'"):
        resolve_model_wire_api("gemini", wire_api)


def test_unknown_wire_protocol_and_empty_responses_fail() -> None:
    with pytest.raises(ValueError, match="Unsupported wire_api"):
        resolve_model_wire_api("openai", "interactionz")
    with pytest.raises(RuntimeError, match="empty response"):
        OpenAICompatibleChatClient._extract_responses_text({"output": []})


def test_compatible_client_rejects_interactions_protocol() -> None:
    with pytest.raises(ValueError, match="use GeminiInteractionsClient"):
        OpenAICompatibleChatClient(
            api_key="not-used",
            base_url="https://example.test",
            wire_api="interactions",
        )


def test_gemini_browse_key_cannot_substitute_for_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROWSE_EXTRACT_PROVIDER", "gemini")
    monkeypatch.setenv("BROWSE_EXTRACT_API_KEY", "wrong-channel")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="cannot authenticate Gemini"):
        JinaReaderClient()


def test_upload_provider_does_not_fall_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    client = ImageUploadClient(provider="custom", upload_api_url="https://upload.test")
    called = {"oss": False}

    def fail_custom(_path: Path, _url: str) -> str:
        raise RuntimeError("custom upload failed")

    def unexpected_oss(_path: Path) -> str:
        called["oss"] = True
        return "https://unexpected.test/image.png"

    monkeypatch.setattr(client, "_upload_via_http", fail_custom)
    monkeypatch.setattr(client, "_upload_to_oss", unexpected_oss)
    with pytest.raises(RuntimeError, match="custom upload failed"):
        client.upload(str(image_path))
    assert called["oss"] is False


def test_visual_search_provider_does_not_fall_through() -> None:
    class Upload:
        last_upload_meta = {}

        @staticmethod
        def upload(_path: str) -> str:
            return "https://images.test/input.png"

    class Zhipu:
        api_key = "test-key"

        @staticmethod
        def search(_url: str, *, top_k: int):
            raise RuntimeError(f"zhipu failed at {top_k}")

    class Serper:
        called = False

        def search(self, **_kwargs):
            self.called = True
            return []

    serper = Serper()
    client = VisualReverseSearchClient(
        upload_client=Upload(),
        zhipu_client=Zhipu(),
        serper_lens_client=serper,
        provider="zhipu_image_search",
    )
    with pytest.raises(RuntimeError, match="zhipu failed"):
        client.search("local.png")
    assert serper.called is False


def test_browse_fetch_provider_does_not_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSE_FETCH_PROVIDER", "jina")
    client = JinaReaderClient()
    called = {"direct": False}

    def fail_jina(_url: str) -> str:
        raise RuntimeError("jina failed")

    def unexpected_direct(_url: str) -> str:
        called["direct"] = True
        return "unexpected"

    monkeypatch.setattr(client, "_fetch_with_jina", fail_jina)
    monkeypatch.setattr(client, "_fetch_direct", unexpected_direct)
    with pytest.raises(RuntimeError, match="jina failed"):
        client.fetch_page_content("https://example.test")
    assert called["direct"] is False


def test_cache_ttl_and_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timestamps = iter([1000.0, 1001.0, 2000.0])
    monkeypatch.setattr("src.orchestrator.tool_cache.time.time", lambda: next(timestamps))
    cache = ToolResultCache(
        cache_dir=str(tmp_path),
        enabled=True,
        ttl_seconds=60.0,
        namespace="provider-a",
    )
    result = json.dumps({"status": "success", "value": 1})
    cache.put("tool", {"query": "x"}, result)
    assert json.loads(cache.get("tool", {"query": "x"}) or "{}") == json.loads(result)
    assert cache.get("tool", {"query": "x"}) is None

    other = ToolResultCache(
        cache_dir=str(tmp_path),
        enabled=True,
        ttl_seconds=60.0,
        namespace="provider-b",
    )
    assert other.get("tool", {"query": "x"}) is None


def test_compare_reference_rejects_legacy_backend_without_calling_it(tmp_path: Path) -> None:
    class Backend:
        called = False

        async def get_response(self, _messages, **_kwargs):
            self.called = True
            raise AssertionError("legacy backend must not be called")

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(vlm_backend=Backend(), image_path=str(image_path))

    result = asyncio.run(tool.call_async({"reference_url": "https://example.test/ref.png"}))
    assert result == {
        "status": "error",
        "error": "Gemini Interactions backend not configured for image comparison.",
    }
    assert tool.vlm_backend.called is False


def test_face_functionality_is_absent_from_active_source() -> None:
    root = Path(__file__).parent
    forbidden_files = {"face_detect.py", "verify_face_identity.py"}
    assert not any(path.name in forbidden_files for path in (root / "src").rglob("*.py"))

    forbidden_symbols = ("FaceDetection", "face_detect", "verify_face_identity")
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (root / "src").rglob("*.py")
    )
    assert not any(symbol in source_text for symbol in forbidden_symbols)


def test_persistence_redacts_signed_urls_and_secret_fields() -> None:
    signed_url = (
        "https://bucket.example/image.jpg?OSSAccessKeyId=key-id&Expires=123"
        "&Signature=secret-signature&keep=value"
    )
    payload = {
        "api_key": "top-secret",
        "url": signed_url,
        "tool_result": json.dumps({"status": "success", "image_url": signed_url}),
    }

    sanitized = sanitize_for_persistence(payload)
    assert sanitized["api_key"] == REDACTED
    assert "key-id" not in sanitized["url"]
    assert "secret-signature" not in sanitized["url"]
    assert "keep=value" in sanitized["url"]
    assert "key-id" not in sanitized["tool_result"]
    assert "REDACTED" in sanitized["tool_result"]
