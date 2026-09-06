from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

from src.orchestrator.investigation_models import DiscrepancyJudgment
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.react_runtime import UnifiedReactState
from src.orchestrator.state import ImageOnlyRuntimeCase
from src.workflow import VerificationWorkflow


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class _AsyncResource:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _SyncResource:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_orchestrator_aclose_releases_shared_resources_once() -> None:
    llm = _AsyncResource()
    shared_vision = _SyncResource()
    shared_reader = _SyncResource()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.llm = llm
    orchestrator.all_tools = {
        "perceive_scene": SimpleNamespace(client=shared_vision),
        "crop_and_inspect": SimpleNamespace(client=shared_vision),
        "visit": SimpleNamespace(browse_client=shared_reader),
        "crop_and_search": SimpleNamespace(
            vlm_client=shared_vision,
            browse_client=shared_reader,
        ),
    }

    run(orchestrator.aclose())

    assert llm.close_calls == 1
    assert shared_vision.close_calls == 1
    assert shared_reader.close_calls == 1


def test_orchestrator_aclose_releases_nested_http_clients_once() -> None:
    upload = _SyncResource()
    lens = _SyncResource()
    ocr = _SyncResource()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.llm = _AsyncResource()
    orchestrator.all_tools = {
        "reverse_image_search": SimpleNamespace(
            visual_search_client=SimpleNamespace(
                upload_client=upload,
                serper_lens_client=lens,
            )
        ),
        "ocr_with_position": SimpleNamespace(baidu_client=ocr),
    }

    run(orchestrator.aclose())

    assert upload.close_calls == 1
    assert lens.close_calls == 1
    assert ocr.close_calls == 1


def test_run_batch_closes_every_isolated_child() -> None:
    children: list[Any] = []

    class Child:
        def __init__(self, result: dict[str, Any]) -> None:
            self.result = result
            self.close_calls = 0

        async def run_single(self, *_: Any, **__: Any) -> dict[str, Any]:
            return self.result

        async def aclose(self) -> None:
            self.close_calls += 1

    workflow = VerificationWorkflow.__new__(VerificationWorkflow)
    workflow.config = SimpleNamespace(sampling_seed=123)
    workflow._orchestrator = None

    def make_child(_config: Any) -> Child:
        child = Child({"verdict": "real"})
        children.append(child)
        return child

    workflow._new_batch_child = make_child  # type: ignore[method-assign]
    results = run(
        workflow.run_batch(
            ["first.jpg", "second.jpg", "third.jpg"],
            concurrency=2,
        )
    )

    assert [item["verdict"] for item in results] == ["real", "real", "real"]
    assert len(children) == 3
    assert [child.close_calls for child in children] == [1, 1, 1]


def test_orchestrator_run_returns_raw_history_terminal_fields(tmp_path: Any) -> None:
    image_path = tmp_path / "case.jpg"
    image_path.write_bytes(b"raw-history-terminal-fixture")
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    runtime_case = ImageOnlyRuntimeCase(
        case_id="case-terminal",
        image_path=str(image_path),
        image_sha256=image_sha256,
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tool_health_summary = {}

    async def fake_runtime(
        state: Any,
        *,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
    ) -> tuple[UnifiedReactState, DiscrepancyJudgment, dict[str, Any], None]:
        investigation = UnifiedReactState(
            case_id=runtime_case.case_id,
            image_sha256=runtime_case.image_sha256,
            action_count=1,
            stop_reason="model_finished",
            finish_rationale="The retained observation is sufficient.",
        )
        judgment = DiscrepancyJudgment(
            verdict="real",
            confidence=0.9,
            overall_assessment="The retained observation supports the verdict.",
            fact_check_report={
                "headline": "Fixture verdict",
                "claim_under_review": "The image is factually accurate.",
                "verdict_summary": "The fixture supports the claim.",
                "key_findings": ["A retained observation is available."],
                "evidence_summary": "The retained observation supports the verdict.",
                "remaining_uncertainties": [],
            },
        )
        return investigation, judgment, {"observation_ids": []}, None

    orchestrator._run_react_runtime_policy = fake_runtime  # type: ignore[method-assign]

    result = run(
        orchestrator.run(
            str(image_path),
            runtime_case,
            episode_id="episode-terminal",
        )
    )

    assert result["termination"] == "success"
    assert result["stop_reason"] == "model_finished"
    assert result["action_count"] == 1
    assert "investigation_status" not in result
    assert "verification_layers" not in result
