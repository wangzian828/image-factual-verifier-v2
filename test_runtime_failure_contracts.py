"""Focused engineering-failure contracts for the v3 image-only runtime."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

import src.orchestrator.pipeline as pipeline_module
from src.orchestrator.investigation_models import (
    FactOrigin,
    ImageOnlyInvestigationState,
    InvestigationEvidence,
    InvestigationBrief,
    ResearchTask,
    VisualFact,
    VisualReinspectionRecord,
    VisualReinspectionRequest,
)
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.integrations.browse.jina_reader import JinaReaderClient
from src.integrations.llm.openai_compatible import (
    OpenAICompatibleChatClient,
    resolve_model_wire_api,
)
from src.integrations.search.visual_search import (
    ImageUploadClient,
    VisualReverseSearchClient,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.stage_runner import StageRunner, StageStep
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    VerificationState,
)
import src.orchestrator.tool_execution as tool_execution_module
from src.orchestrator.tool_execution import ToolActionRecord
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_health import ToolHealth
from src.orchestrator.tool_registry import REQUIRED_TOOLS
from src.orchestrator.tool_result import (
    ToolResultContractError,
    serialize_tool_result,
)
from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.task_store import (
    _unique_visual_view_artifacts,
    record_tool_observation,
)
from src.orchestrator.task_store import state_from_bootstrap
from src.redaction import REDACTED, sanitize_for_persistence
from src.tools.base import BaseTool
from src.tools.compare_reference import CompareWithReferenceTool


class StaticTool(BaseTool):
    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self.description = name
        self.parameters = {
            "type": "object",
            "properties": {"image_input": {"type": "string"}},
            "required": ["image_input"],
        }
        self.result = result

    def call(self, _params: dict[str, Any]) -> dict[str, Any]:
        return dict(self.result)


class HangingAsyncTool(BaseTool):
    name = "hanging_tool"
    description = "Never returns without an outer action deadline."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("async path must be used")

    async def call_async(self, _params: dict[str, Any]) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {"status": "success"}


class HangingSyncTool(BaseTool):
    name = "hanging_sync_tool"
    description = "Blocks a worker thread past the action deadline."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, _params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.2)
        return {"status": "success"}


def _bare_perception_orchestrator(
    tools: dict[str, BaseTool],
) -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.llm = type("LLM", (), {"wire_api": "interactions"})()
    orchestrator.all_tools = tools
    orchestrator.tool_health_summary = {}
    orchestrator.cacheable_tools = set()
    orchestrator.tool_cache = ToolResultCache(enabled=False)
    return orchestrator


def test_malformed_tool_result_is_not_success() -> None:
    with pytest.raises(ToolResultContractError, match="missing a valid status"):
        serialize_tool_result({"results": []})
    step = StageStep(
        action_type="tool_call",
        tool_name="text_search",
        tool_result='{"results":[]}',
    )
    assert Orchestrator._tool_step_succeeded(step) is False


def test_image_only_investigation_requires_a_successful_tool_result() -> None:
    state = VerificationState(
        all_steps=[
            StageStep(
                stage_name="image_only_investigation",
                action_type="tool_call",
                tool_name="reverse_image_search",
                tool_result='{"status":"error","error":"provider unavailable"}',
            )
        ]
    )
    with pytest.raises(
        RuntimeError,
        match="every attempted image-only investigation tool call failed",
    ):
        Orchestrator._require_successful_image_only_investigation(state)

    state.all_steps.append(
        StageStep(
            stage_name="image_only_investigation",
            action_type="tool_call",
            tool_name="visit",
            tool_result='{"status":"success","evidence":"qualified result"}',
        )
    )
    Orchestrator._require_successful_image_only_investigation(state)


def test_required_reflection_fails_closed_on_invalid_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidReflectionRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(self, _context: str) -> tuple[None, list[StageStep]]:
            return None, [
                StageStep(
                    stage_name="image_only_reflection",
                    action_type="output_rejected",
                    metadata={"rejection_reason": "invalid structured output"},
                )
            ]

    case = ImageOnlyRuntimeCase(
        case_id="case-reflection-failure",
        image_path="fixture.jpg",
        image_sha256="a" * 64,
    )
    investigation = state_from_bootstrap(
        build_bootstrap_investigation(
            case,
            PerceptionReport(
                scene_description="A person holds a packaged product.",
                entities=[
                    Entity(
                        name="person",
                        entity_type="person",
                        bbox=[0.1, 0.1, 0.6, 0.9],
                        confidence=0.9,
                    )
                ],
            ),
        )
    )
    state = VerificationState(
        image_path=case.image_path,
        image_id=case.case_id,
        runtime_case=case,
    )
    orchestrator = object.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.llm = type("LLM", (), {"wire_api": "interactions"})()
    orchestrator.date_prefix = ""
    monkeypatch.setattr(pipeline_module, "StageRunner", InvalidReflectionRunner)

    with pytest.raises(
        RuntimeError,
        match="mandatory image-only Reflection did not produce valid structured output",
    ):
        asyncio.run(
            orchestrator._run_image_only_reflection(
                state,
                investigation,
                evidence_gain=False,
                decision_gain=False,
            )
        )

    assert investigation.reflections == []
    assert any(
        step.stage_name == "image_only_reflection"
        and step.action_type == "output_rejected"
        for step in state.all_steps
    )


def test_required_tool_failure_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = {
        name: ToolHealth(
            available=name != "visit",
            error="dependency missing" if name == "visit" else "",
        )
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


def test_positioned_ocr_is_required() -> None:
    assert "ocr_with_position" in REQUIRED_TOOLS


def test_perception_aborts_when_positioned_ocr_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    orchestrator = _bare_perception_orchestrator(
        {
            "perceive_scene": StaticTool(
                "perceive_scene",
                {
                    "status": "success",
                    "entities": [],
                    "scene_description": "A literal scene.",
                    "image_type": "photo",
                },
            ),
            "ocr_with_position": StaticTool(
                "ocr_with_position",
                {"status": "error", "error": "OCR model unavailable"},
            ),
        }
    )
    state = VerificationState(image_path=str(image_path))
    with pytest.raises(RuntimeError, match="ocr_with_position failed"):
        asyncio.run(orchestrator._run_perception(state, str(image_path)))
    assert [step.tool_name for step in state.all_steps] == [
        "perceive_scene",
        "ocr_with_position",
    ]


def test_perception_exception_is_persisted_as_failed_step(
    tmp_path: Path,
) -> None:
    class RaisingTool(StaticTool):
        def call(self, _params: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("vision endpoint failed")

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    orchestrator = _bare_perception_orchestrator(
        {"perceive_scene": RaisingTool("perceive_scene", {})}
    )
    state = VerificationState(image_path=str(image_path))
    with pytest.raises(RuntimeError, match="perceive_scene failed"):
        asyncio.run(orchestrator._run_perception(state, str(image_path)))
    assert json.loads(state.all_steps[0].tool_result) == {
        "status": "error",
        "error": "RuntimeError: vision endpoint failed",
    }
    assert state.all_steps[0].metadata["tool_exception"] == "RuntimeError"


def test_perception_tool_action_deadline_returns_structured_failure() -> None:
    orchestrator = _bare_perception_orchestrator(
        {"hanging_tool": HangingAsyncTool()}
    )
    orchestrator.tool_action_timeout_seconds = 0.01

    serialized, metadata = asyncio.run(
        orchestrator._execute_tool(
            "hanging_tool",
            {},
            "",
        )
    )

    assert json.loads(serialized)["status"] == "error"
    assert "ToolActionTimeout" in json.loads(serialized)["error"]
    assert metadata["tool_exception"] == "ToolActionTimeout"


def test_stage_runner_deadlines_cover_tools_and_native_requests() -> None:
    class HangingLLM:
        async def create_interaction(self, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.Event().wait()
            return {}

    runner = StageRunner(
        llm=HangingLLM(),
        system_prompt="test",
        tools=[HangingAsyncTool()],
        request_timeout_seconds=5.0,
        tool_timeout_seconds=5.0,
    )
    runner.request_timeout_seconds = 0.01
    runner.tool_timeout_seconds = 0.01

    serialized, metadata = asyncio.run(
        runner._execute_tool("hanging_tool", {})
    )
    assert "ToolActionTimeout" in json.loads(serialized)["error"]
    assert metadata["tool_exception"] == "ToolActionTimeout"
    assert metadata["tool_execution_status"] == "timed_out"
    assert metadata["tool_action_id"].startswith("tool-")
    assert metadata["completed_at"]
    with pytest.raises(TimeoutError, match="Gemini request exceeded"):
        asyncio.run(runner._create_interaction())


def test_stage_runner_does_not_wait_for_blocking_sync_tool_after_deadline() -> None:
    runner = StageRunner(
        llm=SimpleNamespace(provider="", wire_api=""),
        system_prompt="test",
        tools=[HangingSyncTool()],
    )
    runner.tool_timeout_seconds = 0.01

    loop = asyncio.new_event_loop()
    try:
        started = time.perf_counter()
        serialized, metadata = loop.run_until_complete(
            runner._execute_tool("hanging_sync_tool", {})
        )
        elapsed = time.perf_counter() - started
    finally:
        loop.close()

    assert elapsed < 0.15
    assert "ToolActionTimeout" in json.loads(serialized)["error"]
    assert metadata["tool_execution_status"] == "timed_out"


def test_stage_runner_records_successful_tool_lifecycle() -> None:
    tool = StaticTool("static_tool", {"status": "success", "value": 3})
    runner = StageRunner(
        llm=SimpleNamespace(provider="", wire_api=""),
        system_prompt="test",
        tools=[tool],
    )

    serialized, metadata = asyncio.run(
        runner._execute_tool("static_tool", {})
    )

    assert json.loads(serialized) == {"status": "success", "value": 3}
    assert metadata["tool_success"] is True
    assert metadata["tool_execution_status"] == "completed"
    assert metadata["tool_action_id"].startswith("tool-")
    assert metadata["requested_at"]
    assert metadata["started_at"]
    assert metadata["completed_at"]


def test_tool_action_record_supports_pending_resume_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter(
        [
            "2026-08-09T00:00:00+00:00",
            "2026-08-09T00:00:01+00:00",
            "2026-08-09T00:00:02+00:00",
            "2026-08-09T00:00:03+00:00",
        ]
    )
    monkeypatch.setattr(
        tool_execution_module,
        "_utc_now",
        lambda: next(clock_values),
    )
    record = ToolActionRecord.start(
        "reverse_image_search",
        timeout_seconds=12.5,
        continuation_id="interaction-1",
    )
    started_at = record.started_at

    pending = record.mark_pending()
    assert pending["tool_execution_status"] == "pending"
    assert pending["completed_at"] is None
    assert pending["retry_count"] == 0

    resumed = record.resume(continuation_id="interaction-2")
    assert resumed["tool_execution_status"] == "running"
    assert resumed["continuation_id"] == "interaction-2"
    assert resumed["retry_count"] == 1

    finished = record.finish("completed")
    assert finished["tool_execution_status"] == "completed"
    assert finished["completed_at"]


def test_visual_reinspection_failure_does_not_exhaust_main_task() -> None:
    state = ImageOnlyInvestigationState(
        brief=InvestigationBrief(
            brief_id="brief-1",
            case_id="case-1",
        ),
        facts=[
            VisualFact(
                fact_id="fact-1",
                kind="attribute",
                statement="A visible cue appears in the scene.",
                subject_entity_id="entity-1",
                predicate="appears",
                basis_ids=["basis-1"],
                origin=FactOrigin(
                    type="input_image",
                    origin_ids=["origin-1"],
                ),
            )
        ],
        tasks=[
            ResearchTask(
                task_id="task-1",
                fact_ids=["fact-1"],
                question="What is visible?",
                purpose="Inspect the scene.",
                origin_ids=["origin-1"],
            )
        ],
        evidence=[
            InvestigationEvidence(
                evidence_id="evidence-1",
                task_id="task-1",
                fact_ids=["fact-1"],
                function_call_id="call-evidence-1",
                tool_name="focused_visual_inspection",
                evidence_kind="image_region",
                source_url="",
                source_family="image:fake",
                source_class="visual",
                exact_text="Visible cue.",
                image_claim="",
                retrieval_goal="",
                image_region=[0.1, 0.1, 0.2, 0.2],
                artifact_sha256="a" * 64,
                retrieved_at="2026-08-09T00:00:00+00:00",
                stance="neutral",
                quality="weak",
                directness="direct",
                claim_binding="pixel_observation",
                visual_question_id="vq-1",
                visual_scope="scene",
                visual_answer_status="ambiguous",
                visual_observations=[],
            )
        ],
        visual_reinspections=[
            VisualReinspectionRecord(
                visual_question_id="vq-1",
                task_id="task-1",
                fact_id="fact-1",
                created_action_count=0,
                request=VisualReinspectionRequest(
                    reason="identity",
                    scope="scene",
                    question="What is visible?",
                    expected_property="A clear scene-level cue.",
                    anchor_fact_ids=["fact-1"],
                    grounding_evidence_ids=["evidence-1"],
                ),
                anchor_regions=[],
            )
        ],
    )
    step = StageStep(
        action_type="tool_call",
        tool_name="focused_visual_inspection",
        tool_args={"__question_id": "task-1"},
        tool_result=json.dumps(
            {"status": "error", "error": "provider unavailable"}
        ),
        metadata={"function_call_id": "call-1"},
    )

    update = record_tool_observation(state, step, image_sha256="a" * 64)

    assert update["task_status"] == "active"
    assert update["created_evidence_ids"] == []
    assert update["created_failure_ids"]
    assert update["visual_view_artifacts"] == []
    assert state.visual_reinspections[0].status == "failed"
    assert state.tasks[0].status != "exhausted"


def test_native_function_result_reinjects_visual_view_artifacts(
    tmp_path: Path,
) -> None:
    runtime_store = CaseRuntimeStore(tmp_path, case_id="case-reinject")
    runner = StageRunner(
        llm=SimpleNamespace(provider="", wire_api=""),
        system_prompt="test",
        tools=[],
        runtime_store=runtime_store,
    )
    crop_1 = tmp_path / "crop-1.png"
    crop_2 = tmp_path / "crop-2.png"
    Image.new("RGB", (4, 4), "red").save(crop_1)
    Image.new("RGB", (4, 4), "blue").save(crop_2)
    artifact_1 = runtime_store.artifacts.put_bytes(
        crop_1.read_bytes(),
        media_type="image/png",
        suffix=".png",
    )
    artifact_2 = runtime_store.artifacts.put_bytes(
        crop_2.read_bytes(),
        media_type="image/png",
        suffix=".png",
    )
    state_update = {
        "visual_view_artifacts": [
            {
                "view_index": 1,
                "kind": "anchor_detail",
                "region": [0.1, 0.1, 0.4, 0.4],
                "artifact": artifact_1,
            },
            {
                "view_index": 2,
                "kind": "anchor_detail",
                "region": [0.5, 0.5, 0.8, 0.8],
                "artifact": artifact_2,
            },
        ]
    }
    result = runner._build_native_function_result(
        call_id="call-visual",
        tool_name="focused_visual_inspection",
        tool_args={"__question_id": "task-1"},
        result=json.dumps(
            {
                "status": "success",
                "answer_status": "observed",
                "summary": "Visible cues are present.",
                "observations": [],
                "limitations": [],
                "view_artifacts": state_update["visual_view_artifacts"],
            }
        ),
        state_update=state_update,
    )

    image_items = [
        item for item in result["result"] if item["type"] == "image"
    ]
    assert len(image_items) == 2
    assert all(item["data"] for item in image_items)


def test_visual_view_artifact_deduplication_accepts_mapping_records() -> None:
    first = {
        "view_index": 1,
        "kind": "anchor_detail",
        "artifact": {"sha256": "a" * 64},
    }
    duplicate = dict(first)
    second = {
        "view_index": 2,
        "kind": "anchor_detail",
        "artifact": {"sha256": "b" * 64},
    }

    assert _unique_visual_view_artifacts(
        [first, duplicate, second]
    ) == [first, second]


def test_tool_internal_usage_is_counted_and_hidden(tmp_path: Path) -> None:
    class PerceptionTool(StaticTool):
        def call(self, _params: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "success",
                "scene_description": "A literal scene.",
                "image_type": "photo",
                "entities": [],
                "__runtime_metrics__": {
                    "llm_api_calls": 1,
                    "tokens": {"prompt": 45, "completion": 12, "thought": 0},
                },
            }

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    orchestrator = _bare_perception_orchestrator(
        {
            "perceive_scene": PerceptionTool("perceive_scene", {}),
            "ocr_with_position": StaticTool(
                "ocr_with_position",
                {"status": "success", "text_regions": []},
            ),
        }
    )
    state = VerificationState(image_path=str(image_path))
    report = asyncio.run(orchestrator._run_perception(state, str(image_path)))
    assert report.scene_description == "A literal scene."
    assert state.llm_api_calls == 1
    assert state.token_usage == {"prompt": 45, "completion": 12, "thought": 0}
    assert "__runtime_metrics__" not in state.all_steps[0].tool_result


def test_nonzero_thought_tokens_are_recorded_without_failure() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.llm = type("LLM", (), {"wire_api": "interactions"})()
    state = VerificationState()
    step = StageStep(
        stage_name="image_only_investigation",
        action_type="tool_call",
        tool_name="visit",
        metadata={
            "tool_llm_api_calls": 1,
            "tool_tokens": {"prompt": 20, "completion": 4, "thought": 2},
        },
    )
    orchestrator._record_stage_steps(state, [step])

    assert state.token_usage == {"prompt": 20, "completion": 4, "thought": 2}


def test_image_account_planning_records_reasoning_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_PLANNING_THINKING_LEVEL", raising=False)
    orchestrator = object.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.llm = type("LLM", (), {"wire_api": "interactions"})()
    state = VerificationState()
    step = StageStep(
        stage_name="image_account_planning",
        action_type="output",
        tokens={"prompt": 20, "completion": 4, "thought": 12},
    )

    orchestrator._record_stage_steps(state, [step])

    assert Orchestrator._stage_thinking_level("PLANNING") == "high"
    assert state.token_usage == {"prompt": 20, "completion": 4, "thought": 12}


def test_non_planning_stage_cannot_enable_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_VERIFICATION_THINKING_LEVEL", "high")

    with pytest.raises(ValueError, match="must be one of low"):
        Orchestrator._stage_thinking_level("VERIFICATION")


@pytest.mark.parametrize("wire_api", ["chat_completions", "responses", "openai_compat"])
def test_gemini_rejects_non_interactions_protocol(wire_api: str) -> None:
    with pytest.raises(ValueError, match="requires wire_api='interactions'"):
        resolve_model_wire_api("gemini", wire_api)


def test_compatible_client_rejects_interactions_protocol() -> None:
    with pytest.raises(ValueError, match="use GeminiInteractionsClient"):
        OpenAICompatibleChatClient(
            api_key="not-used",
            base_url="https://example.test",
            wire_api="interactions",
        )


def test_compatible_json_client_accepts_valid_reasoning_object_only() -> None:
    choice = {
        "message": {
            "content": None,
            "reasoning_content": '{"entities": [], "image_type": "photo"}',
        }
    }

    assert OpenAICompatibleChatClient._extract_chat_json_text(choice) == (
        '{"entities": [], "image_type": "photo"}'
    )
    choice["message"]["reasoning_content"] = "I think this is a photo."
    assert OpenAICompatibleChatClient._extract_chat_json_text(choice) == ""
    choice["message"]["reasoning_content"] = '[{"image_type": "photo"}]'
    assert OpenAICompatibleChatClient._extract_chat_json_text(choice) == ""
    choice["message"]["reasoning_content"] = None
    choice["message"]["reasoning"] = '{"entities": [], "image_type": "illustration"}'
    assert OpenAICompatibleChatClient._extract_chat_json_text(choice) == (
        '{"entities": [], "image_type": "illustration"}'
    )


def test_compatible_json_client_forwards_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAICompatibleChatClient(
        api_key="none",
        base_url="http://127.0.0.1:8901/v1",
        wire_api="chat_completions",
    )
    captured = {}

    def fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(client, "_create_chat_json_completion", fake_chat_completion)

    assert client.create_json_completion(
        model_name="ifv-qwen3.5-9b",
        messages=[{"role": "user", "content": "inspect"}],
        max_tokens=128,
        chat_template_kwargs={"enable_thinking": False},
    ) == "{}"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_compatible_responses_client_does_not_send_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAICompatibleChatClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        wire_api="responses",
    )
    captured = {}

    def fake_responses_completion(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(
        client,
        "_create_responses_json_completion",
        fake_responses_completion,
    )

    assert client.create_json_completion(
        model_name="example",
        messages=[{"role": "user", "content": "inspect"}],
        max_tokens=128,
        chat_template_kwargs={"enable_thinking": False},
    ) == "{}"
    assert "chat_template_kwargs" not in captured


def test_upload_provider_does_not_fall_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    client = ImageUploadClient(
        provider="custom",
        upload_api_url="https://upload.test",
    )
    called = {"oss": False}

    monkeypatch.setattr(
        client,
        "_upload_via_http",
        lambda _path, _url: (_ for _ in ()).throw(
            RuntimeError("custom upload failed")
        ),
    )

    def unexpected_oss(_path: Path) -> str:
        called["oss"] = True
        return "https://unexpected.test/image.png"

    monkeypatch.setattr(client, "_upload_to_oss", unexpected_oss)
    with pytest.raises(RuntimeError, match="custom upload failed"):
        client.upload(str(image_path))
    assert called["oss"] is False


def test_visual_search_provider_does_not_fall_through() -> None:
    class Upload:
        last_upload_meta: dict[str, Any] = {}

        @staticmethod
        def upload(_path: str) -> str:
            return "https://images.test/input.png"

    class Zhipu:
        api_key = "test-key"

        @staticmethod
        def search(_url: str, *, top_k: int) -> Any:
            raise RuntimeError(f"zhipu failed at {top_k}")

    class Serper:
        called = False

        def search(self, **_kwargs: Any) -> list[Any]:
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


def test_browse_provider_falls_back_to_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROWSE_FETCH_PROVIDER", "jina")
    client = JinaReaderClient()
    called = {"direct": False}
    monkeypatch.setattr(
        client,
        "_fetch_with_jina",
        lambda _url: (_ for _ in ()).throw(RuntimeError("jina failed")),
    )

    def direct(_url: str) -> str:
        called["direct"] = True
        return "direct fallback"

    monkeypatch.setattr(client, "_fetch_direct", direct)
    content, provider = client.fetch_page_content("https://example.test")
    assert called["direct"] is True
    assert content == "direct fallback"
    assert provider == "direct_reader"


def test_cache_ttl_and_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timestamps = iter([1000.0, 1001.0, 2000.0])
    monkeypatch.setattr(
        "src.orchestrator.tool_cache.time.time",
        lambda: next(timestamps),
    )
    cache = ToolResultCache(
        cache_dir=str(tmp_path),
        enabled=True,
        ttl_seconds=60.0,
        namespace="provider-a",
    )
    result = '{"status":"success","value":1}'
    cache.put("tool", {"query": "x"}, result)
    assert json.loads(cache.get("tool", {"query": "x"}) or "{}") == json.loads(
        result
    )
    assert cache.get("tool", {"query": "x"}) is None
    other = ToolResultCache(
        cache_dir=str(tmp_path),
        enabled=True,
        ttl_seconds=60.0,
        namespace="provider-b",
    )
    assert other.get("tool", {"query": "x"}) is None


def test_compare_reference_rejects_legacy_backend(tmp_path: Path) -> None:
    class Backend:
        called = False

        async def get_response(self, _messages: Any, **_kwargs: Any) -> Any:
            self.called = True
            raise AssertionError("legacy backend must not be called")

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    tool = CompareWithReferenceTool(
        vlm_backend=Backend(),
        image_path=str(image_path),
    )
    result = asyncio.run(
        tool.call_async({"reference_url": "https://example.test/ref.png"})
    )
    assert result["status"] == "error"
    assert "Interactions backend not configured" in result["error"]
    assert tool.vlm_backend.called is False


def test_persistence_redacts_credentials_and_signed_urls() -> None:
    signed_url = (
        "https://bucket.example/image.jpg?OSSAccessKeyId=key-id&Expires=123"
        "&Signature=secret-signature&keep=value"
    )
    sanitized = sanitize_for_persistence(
        {
            "api_key": "top-secret",
            "url": signed_url,
            "tool_result": json.dumps(
                {"status": "success", "image_url": signed_url}
            ),
        }
    )
    assert sanitized["api_key"] == REDACTED
    assert "key-id" not in sanitized["url"]
    assert "secret-signature" not in sanitized["tool_result"]
    assert "keep=value" in sanitized["url"]


def test_face_recognition_modules_are_absent() -> None:
    source_root = Path(__file__).parent / "src"
    forbidden_files = {"face_detect.py", "verify_face_identity.py"}
    assert not any(path.name in forbidden_files for path in source_root.rglob("*.py"))
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in source_root.rglob("*.py")
    )
    assert "FaceDetection" not in source_text
    assert "verify_face_identity" not in source_text
