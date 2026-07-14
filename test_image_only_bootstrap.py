from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.runtime_case import image_sha256
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.tools.base import BaseTool
from src.workflow import VerificationWorkflow, WorkflowConfig


class StaticTool(BaseTool):
    def __init__(self, name: str, result: Dict[str, Any]) -> None:
        self.name = name
        self.description = f"Controlled image-only fixture for {name}."
        self.parameters = {
            "type": "object",
            "properties": {"image_input": {"type": "string"}},
            "required": ["image_input"],
        }
        self.result = result

    def call(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.result)


def _case(image_path: Path) -> ImageOnlyRuntimeCase:
    return ImageOnlyRuntimeCase(
        case_id="case_image_only_bootstrap",
        image_path=str(image_path.resolve()),
        image_sha256=image_sha256(str(image_path)),
    )


def _perception() -> PerceptionReport:
    return PerceptionReport(
        scene_description="A marked research vessel is visible on the water.",
        entities=[
            Entity(
                name="research vessel",
                entity_type="object",
                bbox=[0.1, 0.2, 0.9, 0.9],
                confidence=0.94,
            )
        ],
        text_regions=[
            TextRegion(
                text="HENRY B. BIGELOW",
                bbox_quad=[[0.2, 0.4], [0.7, 0.4], [0.7, 0.5], [0.2, 0.5]],
                confidence=0.98,
                language="en",
            ),
            TextRegion(
                text="R 225",
                bbox_quad=[[0.7, 0.5], [0.82, 0.5], [0.82, 0.58], [0.7, 0.58]],
                confidence=0.96,
                language="en",
            ),
            TextRegion(
                text="OOUOHUEUIRUU",
                bbox_quad=[[0.1, 0.8], [0.3, 0.8], [0.3, 0.85], [0.1, 0.85]],
                confidence=0.05,
                language="en",
            ),
        ],
    )


def test_image_only_bootstrap_is_deterministic_grounded_and_bounded(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"image-only-bootstrap-fixture")
    case = _case(image_path)

    first = build_bootstrap_investigation(case, _perception())
    second = build_bootstrap_investigation(case, _perception())

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.brief.input_mode == "image_only"
    assert "query" not in first.brief.model_dump_json().lower()
    assert 1 <= len(first.tasks) <= 4
    assert any(
        fact.kind == "text_claim" and "HENRY B. BIGELOW" in fact.statement
        for fact in first.facts
    )
    assert any(
        fact.kind == "relation"
        and fact.predicate == "context_suggested_by_text"
        and "HENRY B. BIGELOW" in fact.statement
        for fact in first.facts
    )
    serialized = first.model_dump_json()
    assert "OOUOHUEUIRUU" not in serialized
    fact_ids = {fact.fact_id for fact in first.facts}
    assert all(set(task.fact_ids) <= fact_ids for task in first.tasks)
    joint_text_task = next(
        task
        for task in first.tasks
        if "jointly indicated" in task.question
    )
    assert len(joint_text_task.fact_ids) == 4
    assert "HENRY B. BIGELOW" in joint_text_task.question
    assert "R 225" in joint_text_task.question
    assert all(task.origin_ids for task in first.tasks)
    assert not first.findings


def test_image_only_workflow_persists_bootstrap_before_required_search_boundary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"image-only-bootstrap-fixture")
    case = _case(image_path)
    workflow = VerificationWorkflow(
        WorkflowConfig(
            output_dir=str(tmp_path / "traces"),
            save_traces=True,
            decision_policy_version="reinspect-v2",
        )
    )
    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled-image-only-bootstrap",
        validate_startup=False,
    )
    orchestrator.vlm_provider = "controlled"
    orchestrator.all_tools = {
        "perceive_scene": StaticTool(
            "perceive_scene",
            {
                "status": "success",
                "entities": [
                    {
                        "name": "research vessel",
                        "entity_type": "object",
                        "bbox": [0.1, 0.2, 0.9, 0.9],
                        "confidence": 0.94,
                    }
                ],
                "scene_description": (
                    "A marked research vessel is visible on the water."
                ),
                "image_type": "photo",
            },
        ),
        "ocr_with_position": StaticTool(
            "ocr_with_position",
            {
                "status": "success",
                "text_regions": [
                    {
                        "text": "HENRY B. BIGELOW",
                        "bbox": [0.2, 0.4, 0.7, 0.5],
                        "confidence": 0.98,
                        "language": "en",
                    },
                    {
                        "text": "R 225",
                        "bbox": [0.7, 0.5, 0.82, 0.58],
                        "confidence": 0.96,
                        "language": "en",
                    },
                ],
            },
        ),
    }
    orchestrator.tool_health_summary = {
        name: {"available": True, "error": ""}
        for name in orchestrator.all_tools
    }
    workflow._orchestrator = orchestrator

    with pytest.raises(
        RuntimeError,
        match="reverse_image_search.*unavailable",
    ):
        asyncio.run(
            workflow.run_single(
                str(image_path),
                case.case_id,
                runtime_case=case,
            )
        )

    trace_path = tmp_path / "traces" / f"{case.case_id}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    state = trace["state"]
    assert trace["verdict"] == "error"
    assert trace["input_mode"] == "image_only"
    assert trace["decision_policy_version"] == "reinspect-v2"
    assert set(state["runtime_case"]) == {
        "case_id",
        "image_path",
        "image_sha256",
    }
    assert state["investigation_state"]["brief"]["case_id"] == case.case_id
    assert state["investigation_state"]["facts"]
    assert state["investigation_state"]["tasks"]
    assert state["investigation_brief"]["case_id"] == case.case_id
    assert state["visual_entities"]
    assert state["visual_facts"]
    assert state["research_tasks"]
    assert state["retrieval_anchors"]
    assert state["findings"] == []
    assert state["total_tool_calls"] == 2
    assert [step["tool_name"] for step in state["all_steps"]] == [
        "perceive_scene",
        "ocr_with_position",
    ]


def test_batch_preserves_each_case_error_artifacts() -> None:
    workflow = VerificationWorkflow(WorkflowConfig(save_traces=False))

    async def fail_with_bound_result(
        path: str,
        image_id: str = "",
        *,
        runtime_case: Any = None,
    ) -> Dict[str, Any]:
        _ = runtime_case
        error = RuntimeError(f"failed {image_id}")
        error._ifv_result = {  # type: ignore[attr-defined]
            "image_id": image_id,
            "image_path": path,
            "verdict": "error",
            "termination": "error",
            "time_taken": 1.0 if image_id == "first" else 2.0,
            "total_tool_calls": 2,
            "llm_api_calls": 1,
            "error": str(error),
        }
        raise error

    workflow.run_single = fail_with_bound_result  # type: ignore[method-assign]
    results = asyncio.run(
        workflow.run_batch(
            ["first.jpg", "second.jpg"],
            image_ids=["first", "second"],
        )
    )

    assert [result["image_id"] for result in results] == ["first", "second"]
    assert [result["time_taken"] for result in results] == [1.0, 2.0]
    assert all(result["total_tool_calls"] == 2 for result in results)


def test_workflow_rejects_non_image_only_runtime_case(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"image-only-only")
    workflow = VerificationWorkflow(WorkflowConfig(save_traces=False))
    with pytest.raises(
        TypeError,
        match="ImageOnlyRuntimeCase only",
    ):
        asyncio.run(
            workflow.run_single(
                str(image_path),
                runtime_case=object(),  # type: ignore[arg-type]
            )
        )
