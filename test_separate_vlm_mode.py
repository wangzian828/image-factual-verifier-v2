from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import src.orchestrator.pipeline as pipeline_module
from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.investigation_models import (
    DiscrepancyJudgmentOutput,
    DiscrepancyVerdictBasis,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.runtime_case import image_sha256
from src.orchestrator.stage_runner import InteractionSession
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    VerificationState,
)
from src.orchestrator.task_store import state_from_bootstrap
from src.orchestrator.tool_cache import ToolResultCache
from src.tools.base import BaseTool
from src.workflow import WorkflowConfig


def _fixture(tmp_path: Path) -> tuple[Path, ImageOnlyRuntimeCase, Any]:
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (32, 24), color="white").save(image_path)
    case = ImageOnlyRuntimeCase(
        case_id="separate-vlm-case",
        image_path=str(image_path),
        image_sha256=image_sha256(str(image_path)),
    )
    perception = PerceptionReport(
        scene_description="A person is visible beside an object.",
        entities=[
            Entity(
                name="person",
                entity_type="person",
                bbox=[0.1, 0.1, 0.7, 0.9],
                confidence=0.9,
            )
        ],
    )
    investigation = state_from_bootstrap(
        build_bootstrap_investigation(case, perception)
    )
    return image_path, case, investigation


def test_separate_vlm_is_explicit_and_default_stays_legacy() -> None:
    assert WorkflowConfig().image_access_mode == "direct_multimodal"
    assert WorkflowConfig(image_access_mode="separate_vlm").image_access_mode == (
        "separate_vlm"
    )
    with pytest.raises(ValueError, match="image_access_mode"):
        WorkflowConfig(image_access_mode="unknown")  # type: ignore[arg-type]


def test_separate_vlm_planning_does_not_attach_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path, _case, investigation = _fixture(tmp_path)
    captured: dict[str, Any] = {}

    class CapturingRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def run(self, _context: str) -> tuple[None, list[Any]]:
            return None, []

    monkeypatch.setattr(pipeline_module, "StageRunner", CapturingRunner)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.model_name = "gemini-3.7-flash"
    orchestrator.llm = object()
    orchestrator.image_access_mode = "separate_vlm"
    orchestrator.date_prefix = ""
    orchestrator.source_access_policy = None
    orchestrator.stage_request_timeout_seconds = 30.0
    orchestrator.runtime_store = None

    state = VerificationState(image_path=str(image_path))
    with pytest.raises(RuntimeError, match="Image Account Planning"):
        import asyncio

        asyncio.run(
            orchestrator._run_image_account_planning(
                state,
                investigation,
                image_path=str(image_path),
            )
        )

    assert captured["attach_image"] is False


def test_separate_vlm_final_judgment_does_not_attach_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path, _case, investigation = _fixture(tmp_path)
    captured: dict[str, Any] = {}

    class CapturingRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def run(self, _context: str) -> tuple[Any, list[Any]]:
            captured["context"] = _context
            return (
                DiscrepancyJudgmentOutput(
                    verdict="real",
                    confidence=0.7,
                    overall_assessment="The supplied evidence is consistent.",
                ),
                [],
            )

    monkeypatch.setattr(pipeline_module, "StageRunner", CapturingRunner)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.model_name = "gemini-3.7-flash"
    orchestrator.llm = object()
    orchestrator.image_access_mode = "separate_vlm"
    orchestrator.date_prefix = ""
    orchestrator.stage_request_timeout_seconds = 30.0

    state = VerificationState(image_path=str(image_path))
    basis = DiscrepancyVerdictBasis(verdict_target="The target relation holds.")
    import asyncio

    asyncio.run(
        orchestrator._run_discrepancy_judgment(
            state,
            investigation,
            "",
            basis,
            image_path=str(image_path),
            final_visual_audit={
                "summary": "The relation is visibly present.",
                "answer_status": "observed",
            },
            interaction_session=InteractionSession(),
        )
    )

    assert captured["attach_image"] is False
    assert '"final_visual_audit"' in captured["context"]
    assert "does not include image pixels" in captured["context"]


def test_final_visual_audit_is_not_investigation_budget(
    tmp_path: Path,
) -> None:
    image_path, case, investigation = _fixture(tmp_path)

    class StaticVisualTool(BaseTool):
        name = "focused_visual_inspection"
        description = "Controlled VLM audit."
        parameters = {
            "type": "object",
            "properties": {"image_input": {"type": "string"}},
            "required": ["image_input"],
        }

        def call(self, _params: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "success",
                "answer_status": "observed",
                "summary": "A concrete visible relation is present.",
                "observations": [],
                "limitations": [],
            }

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "gemini"
    orchestrator.model_name = "gemini-3.7-flash"
    orchestrator.image_access_mode = "separate_vlm"
    orchestrator.all_tools = {
        "focused_visual_inspection": StaticVisualTool(),
    }
    orchestrator.cacheable_tools = set()
    orchestrator.tool_cache = ToolResultCache(enabled=False)
    orchestrator.tool_action_timeout_seconds = 30.0

    state = VerificationState(
        image_path=str(image_path),
        image_id=case.case_id,
        runtime_case=case,
        perception=PerceptionReport(
            scene_description="A person is beside an object.",
            entities=[],
        ),
    )
    basis = DiscrepancyVerdictBasis(verdict_target="The target relation holds.")

    import asyncio

    audit = asyncio.run(
        orchestrator._run_final_visual_audit(
            state,
            investigation,
            compiled_verdict="",
            basis=basis,
            image_path=str(image_path),
        )
    )

    assert audit["stage"] == "image_only_final_visual_audit"
    assert audit["main_llm_received_image"] is False
    assert state.total_tool_calls == 1
    assert investigation.action_count == 0
    assert state.all_steps[0].stage_name == "image_only_final_visual_audit"
