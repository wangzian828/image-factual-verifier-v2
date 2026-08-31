from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import activate_initial_decisive_facts
from src.orchestrator.investigation_models import (
    TargetFactProposal,
    TargetPlanningOutput,
)
from src.orchestrator.runtime_case import image_sha256
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
from src.tools.base import BaseTool
from src.workflow import VerificationWorkflow, WorkflowConfig
from src.orchestrator.task_store import (
    apply_target_planning,
    state_from_bootstrap,
)
from test_frozen_v3_runtime import install_frozen_v3_runtime


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


class PlanningBoundaryBackend:
    provider = "gemini"
    wire_api = "interactions"

    async def create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        system = str(kwargs.get("system_instruction", ""))
        if "initial target-planning step" in system:
            raw = kwargs.get("input_payload", "")
            if isinstance(raw, list):
                text = next(
                    (
                        str(item.get("text", ""))
                        for item in raw
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                    ),
                    json.dumps(raw),
                )
            else:
                text = raw
            context = json.loads(text)
            facts = context["pixel_grounded_facts"]
            vessel_fact = next(
                fact
                for fact in facts
                if fact["predicate"] == "visible_in"
                and "research vessel" in fact["statement"]
            )
            name_fact = next(
                fact
                for fact in facts
                if fact["predicate"] == "reads"
                and "HENRY B. BIGELOW" in fact["statement"]
            )
            return {
                "id": "planning-boundary",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_thought_tokens": 0,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "proposals": [
                                            {
                                                "statement": (
                                                    "The visible research vessel "
                                                    "is identified as HENRY B. "
                                                    "BIGELOW."
                                                ),
                                                "kind": "relation",
                                                "predicate": "identified_as",
                                                "parent_fact_ids": [
                                                    vessel_fact["fact_id"],
                                                    name_fact["fact_id"],
                                                ],
                                                "question": (
                                                    "Does reliable evidence "
                                                    "identify the visible research "
                                                    "vessel as HENRY B. BIGELOW?"
                                                ),
                                                "purpose": (
                                                    "Verify the visible vessel "
                                                    "identity."
                                                ),
                                                "suggested_tools": [
                                                    "reverse_image_search",
                                                    "text_search",
                                                    "visit",
                                                ],
                                                "suggested_queries": [
                                                    "HENRY B. BIGELOW R 225"
                                                ],
                                                "decision_relevance": "decisive",
                                            }
                                        ],
                                        "remaining_target_gaps": [],
                                    }
                                ),
                            }
                        ],
                    }
                ],
            }
        raise RuntimeError("controlled investigation boundary")


class InvalidPlanningFallbackBackend:
    provider = "gemini"
    wire_api = "interactions"

    async def create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        system = str(kwargs.get("system_instruction", ""))
        if "initial target-planning step" in system:
            return {
                "id": "invalid-planning-fallback",
                "status": "completed",
                "usage": {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_thought_tokens": 0,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "proposals": [
                                            {
                                                "statement": (
                                                    "Butterflies, trees, penguins, "
                                                    "and icebergs coexist in one "
                                                    "real-world location."
                                                ),
                                                "kind": "relation",
                                                "predicate": "located_at",
                                                "parent_fact_ids": ["unknown"],
                                                "question": "Is this coexistence real?",
                                                "purpose": "Test the full scene.",
                                                "suggested_tools": ["text_search"],
                                                "suggested_queries": [],
                                                "decision_relevance": "decisive",
                                            }
                                        ],
                                        "remaining_target_gaps": [],
                                    }
                                ),
                            }
                        ],
                    }
                ],
            }
        raise RuntimeError("controlled investigation boundary")


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


def test_normalize_text_region_does_not_require_instance_state() -> None:
    region = Orchestrator._normalize_text_region(
        {
            "text": "VISIBLE",
            "bbox": [0.1, 0.2, 0.4, 0.5],
            "confidence": 0.9,
        }
    )

    assert region is not None
    assert region.text == "VISIBLE"
    assert region.text_role == "unknown"
    assert region.bbox_quad == [
        [0.1, 0.2],
        [0.4, 0.2],
        [0.4, 0.5],
        [0.1, 0.5],
    ]


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
    scene_task = next(
        task
        for task in first.tasks
        if "image-grounded scene proposition" in task.question
    )
    assert "HENRY B. BIGELOW R 225" in scene_task.suggested_queries
    assert '"HENRY B. BIGELOW"' in scene_task.suggested_queries
    assert not any(
        "jointly indicated" in task.question
        for task in first.tasks
    )
    assert all(task.origin_ids for task in first.tasks)
    assert not first.findings


def test_duplicate_entity_names_preserve_every_fact_anchor(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"duplicate-entity-bootstrap-fixture")
    perception = PerceptionReport(
        scene_description="Several employees welcome a visitor.",
        entities=[
            Entity(
                name="employee in green shirt",
                entity_type="person",
                bbox=[0.05, 0.1, 0.25, 0.9],
                confidence=0.9,
            ),
            Entity(
                name="employee in green shirt",
                entity_type="person",
                bbox=[0.3, 0.1, 0.5, 0.9],
                confidence=0.88,
            ),
            Entity(
                name="employee in green shirt",
                entity_type="person",
                bbox=[0.55, 0.1, 0.75, 0.9],
                confidence=0.86,
            ),
        ],
    )

    bootstrap = build_bootstrap_investigation(
        _case(image_path),
        perception,
    )

    known_basis_ids = {
        *(item.entity_id for item in bootstrap.entities),
        *(item.anchor_id for item in bootstrap.retrieval_anchors),
    }
    assert all(
        set(fact.basis_ids) <= known_basis_ids
        for fact in bootstrap.facts
    )
    duplicate_anchors = [
        item
        for item in bootstrap.retrieval_anchors
        if item.value == "employee in green shirt"
    ]
    assert len(duplicate_anchors) == 3
    duplicate_tasks = [
        task
        for task in bootstrap.tasks
        if task.suggested_queries == ["employee in green shirt"]
    ]
    assert len(duplicate_tasks) == 1


def test_target_planning_dynamically_separates_source_binding_and_integrity(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "screenshot.jpg"
    image_path.write_bytes(b"screenshot-bootstrap-fixture")
    perception = PerceptionReport(
        scene_description="A screenshot of an X post with a reply below it.",
        image_type="screenshot",
        text_regions=[
            TextRegion(
                text="Major Tom @dingzhen47",
                bbox_quad=[
                    [0.1, 0.1],
                    [0.5, 0.1],
                    [0.5, 0.2],
                    [0.1, 0.2],
                ],
                confidence=0.99,
                language="en",
            ),
            TextRegion(
                text="学生用AI写，学校用AI查",
                bbox_quad=[
                    [0.1, 0.25],
                    [0.9, 0.25],
                    [0.9, 0.4],
                    [0.1, 0.4],
                ],
                confidence=0.99,
                language="zh",
            ),
            TextRegion(
                text="5/18/25",
                bbox_quad=[
                    [0.1, 0.45],
                    [0.3, 0.45],
                    [0.3, 0.5],
                    [0.1, 0.5],
                ],
                confidence=0.98,
                language="en",
            ),
        ],
    )

    bootstrap = build_bootstrap_investigation(
        _case(image_path),
        perception,
    )
    assert bootstrap.brief.media_type == "screenshot"
    assert not any(
        fact.predicate in {"source_record_matches", "visual_integrity"}
        for fact in bootstrap.facts
    )
    state = state_from_bootstrap(bootstrap)
    scene_fact = next(
        fact
        for fact in state.facts
        if fact.predicate == "appears_to_depict"
    )
    text_facts = [
        fact for fact in state.facts if fact.predicate == "reads"
    ]
    planned = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The visible Major Tom @dingzhen47 post text "
                        "学生用AI写，学校用AI查 dated 5/18/25 matches an "
                        "original public source record."
                    ),
                    predicate="source_record_matches",
                    parent_fact_ids=[
                        fact.fact_id for fact in text_facts[:3]
                    ],
                    question=(
                        "Does an original public record match the visible account, "
                        "text, date, and reply relation?"
                    ),
                    purpose="Resolve the visible public-record attribution.",
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=[
                        '"学生用AI写，学校用AI查" @dingzhen47'
                    ],
                ),
                TargetFactProposal(
                    statement=(
                        "The visible post layout has no material manipulation "
                        "affecting its displayed account, text, date, or reply."
                    ),
                    kind="internal_consistency",
                    predicate="visual_integrity",
                    parent_fact_ids=[scene_fact.fact_id],
                    question=(
                        "Are visible account, text, date, or reply elements "
                        "materially manipulated?"
                    ),
                    purpose="Inspect the salient visual integrity property.",
                    suggested_tools=[
                        "analyze_visual_anomalies",
                        "check_consistency",
                    ],
                ),
            ]
        ),
    )
    assert len(planned["accepted_fact_ids"]) == 2
    assert len(planned["accepted_task_ids"]) == 2
    activate_initial_decisive_facts(state)
    decisive_predicates = {
        fact.predicate
        for fact in state.facts
        if fact.fact_id in state.decisive_fact_ids
    }
    assert decisive_predicates == {"source_record_matches"}


def test_source_record_target_cannot_absorb_pixel_integrity_scope(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "screenshot.jpg"
    image_path.write_bytes(b"target-scope-fixture")
    perception = PerceptionReport(
        scene_description="A screenshot of a public post.",
        image_type="screenshot",
        text_regions=[
            TextRegion(
                text="@visible_account",
                bbox_quad=[
                    [0.1, 0.1],
                    [0.5, 0.1],
                    [0.5, 0.2],
                    [0.1, 0.2],
                ],
                confidence=0.99,
            ),
            TextRegion(
                text="Distinctive visible post text",
                bbox_quad=[
                    [0.1, 0.25],
                    [0.9, 0.25],
                    [0.9, 0.4],
                    [0.1, 0.4],
                ],
                confidence=0.99,
            ),
        ],
    )
    state = state_from_bootstrap(
        build_bootstrap_investigation(_case(image_path), perception)
    )
    text_facts = [
        fact for fact in state.facts if fact.predicate == "reads"
    ]

    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The visible @visible_account post matches a public "
                        "source record."
                    ),
                    predicate="source_record_matches",
                    parent_fact_ids=[
                        fact.fact_id for fact in text_facts
                    ],
                    question=(
                        "Does the source record match, and is the image unmodified?"
                    ),
                    purpose="Detect whether the image was digitally fabricated.",
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=['"Distinctive visible post text"'],
                )
            ]
        ),
    )

    assert not update["accepted_fact_ids"]
    assert "propose visual_integrity separately" in update[
        "rejected_reasons"
    ][0]


def test_image_only_workflow_persists_bootstrap_before_investigation_boundary(
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
    orchestrator.llm = PlanningBoundaryBackend()
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
    install_frozen_v3_runtime(orchestrator)
    workflow._orchestrator = orchestrator

    with pytest.raises(RuntimeError, match="controlled investigation boundary"):
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
    assert [
        step["tool_name"]
        for step in state["all_steps"]
        if step["tool_name"]
    ] == [
        "perceive_scene",
        "ocr_with_position",
    ]
    assert any(
        step["stage"] == "image_only_planning"
        and step["action_type"] == "output"
        for step in state["all_steps"]
    )


def test_invalid_target_planning_fails_without_broad_scene_fallback(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "planning-fallback.jpg"
    image_path.write_bytes(b"planning-fallback-fixture")
    case = _case(image_path)
    bootstrap = build_bootstrap_investigation(case, _perception())
    investigation = state_from_bootstrap(bootstrap)
    state = VerificationState(
        image_path=str(image_path),
        image_id=case.case_id,
        runtime_case=case,
        input_mode="image_only",
        decision_policy_version="reinspect-v2",
    )
    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled-invalid-planning",
        validate_startup=False,
    )
    orchestrator.llm = InvalidPlanningFallbackBackend()

    with pytest.raises(
        RuntimeError,
        match="did not establish an atomic core fact",
    ):
        asyncio.run(
            orchestrator._run_image_only_target_planning(
                state,
                investigation,
            )
        )

    assert investigation.tasks == bootstrap.tasks
    assert investigation.core_verdict_fact_id is None
    assert any(
        step.action_type == "planning_revision"
        for step in state.all_steps
    )


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
