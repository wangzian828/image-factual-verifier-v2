from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import activate_initial_decisive_facts
from src.orchestrator.runtime_case import image_sha256
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.task_store import state_from_bootstrap
from src.tools.base import BaseTool
from src.workflow import VerificationWorkflow, WorkflowConfig


def _completed(interaction_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": interaction_id,
        "status": "completed",
        "usage": {
            "total_input_tokens": 20,
            "total_output_tokens": 10,
            "total_thought_tokens": 0,
        },
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(payload)}],
            }
        ],
    }


def _call(
    interaction_id: str,
    call_id: str,
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": interaction_id,
        "status": "requires_action",
        "usage": {
            "total_input_tokens": 20,
            "total_output_tokens": 10,
            "total_thought_tokens": 0,
        },
        "steps": [
            {
                "id": call_id,
                "type": "function_call",
                "name": name,
                "arguments": arguments,
            }
        ],
    }


class AdaptiveImageOnlyBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, task_ids: List[str]) -> None:
        self.pending_task_ids = list(task_ids)
        self.pending_phase = "compare"
        self.requests: List[Dict[str, Any]] = []
        self.counter = 0

    async def create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        self.requests.append(kwargs)
        self.counter += 1
        system = str(kwargs.get("system_instruction", ""))
        interaction_id = f"interaction-{self.counter}"
        if "structured Reflection step" in system:
            return _completed(
                interaction_id,
                {
                    "task_updates": [],
                    "new_tasks": [],
                    "proposed_decisive_fact_ids": [],
                    "recommended_next_task_ids": [],
                    "remaining_gaps": [],
                    "ready_to_finish": True,
                },
            )
        if "constrained final synthesizer" in system:
            raw = kwargs.get("input_payload", "")
            text = raw if isinstance(raw, str) else json.dumps(raw)
            context = json.loads(text)
            basis = context["compiled_basis"]
            return _completed(
                interaction_id,
                {
                    "verdict": context["compiled_verdict"],
                    "confidence": 0.95,
                    "policy_rule_id": "reinspect-v2",
                    "selected_fact_ids": basis["fact_ids"],
                    "selected_finding_ids": basis["finding_ids"],
                    "selected_evidence_ids": basis["evidence_ids"],
                    "overall_assessment": (
                        "The controlled official evidence supports every decisive "
                        "fact in the image-only investigation."
                    ),
                    "unresolved_gaps": basis["unresolved_gaps"],
                },
            )
        if "No more tool turns remain" in system:
            return _completed(
                interaction_id,
                {
                    "segment_summary": "Controlled investigation segment completed.",
                    "ready_for_reflection": True,
                },
            )
        if self.pending_task_ids:
            task_id = self.pending_task_ids[0]
            if self.pending_phase == "visit":
                self.pending_task_ids.pop(0)
                self.pending_phase = "compare"
                return _call(
                    interaction_id,
                    f"call-{self.counter}",
                    "visit",
                    {
                        "question_id": task_id,
                        "url": [
                            "https://www.noaa.gov/controlled-reference"
                        ],
                        "goal": (
                            f"Resolve the controlled scene fact for {task_id}."
                        ),
                    },
                )
            self.pending_phase = "visit"
            return _call(
                interaction_id,
                f"call-{self.counter}",
                "compare_with_reference",
                {
                    "question_id": task_id,
                    "reference_url": (
                        "https://www.noaa.gov/controlled-reference.jpg"
                    ),
                    "source_page_url": (
                        "https://www.noaa.gov/controlled-reference"
                    ),
                    "focus": f"Resolve the controlled scene fact for {task_id}.",
                },
            )
        return _completed(
            interaction_id,
            {
                "segment_summary": "No additional action is required.",
                "ready_for_reflection": True,
            },
        )

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("image-only v3 must use Gemini Interactions")


class StaticTool(BaseTool):
    def __init__(
        self,
        name: str,
        result: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> None:
        self.name = name
        self.description = f"Controlled {name} tool."
        self.result = result
        self.parameters = parameters
        self.calls: List[Dict[str, Any]] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        return dict(self.result)


class ControlledVisitTool(StaticTool):
    def __init__(self) -> None:
        super().__init__(
            "visit",
            {},
            {
                "type": "object",
                "properties": {
                    "url": {"type": ["string", "array"]},
                    "goal": {"type": "string"},
                },
                "required": ["url", "goal"],
            },
        )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        statement = str(params["goal"])
        excerpt = f"NOAA official record confirms: {statement}"
        return {
            "status": "success",
            "selected_url": "https://www.noaa.gov/controlled-reference",
            "url": "https://www.noaa.gov/controlled-reference",
            "goal": statement,
            "evidence": excerpt,
            "summary": excerpt,
            "relevance": "high",
            "stance": "support",
            "directness": "direct",
            "temporal_alignment": "not_applicable",
            "artifact_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
            "evidence_span": {"start": 0, "end": len(excerpt)},
            "retrieved_at": "2026-07-15T00:00:00+00:00",
            "injection_flags": [],
            "evidence_eligible": True,
            "visits": [],
        }


class ControlledCompareTool(StaticTool):
    def __init__(self) -> None:
        super().__init__(
            "compare_with_reference",
            {
                "status": "success",
                "reference_url": (
                    "https://www.noaa.gov/controlled-reference.jpg"
                ),
                "resolved_reference_url": (
                    "https://www.noaa.gov/controlled-reference.jpg"
                ),
                "source_page_url": (
                    "https://www.noaa.gov/controlled-reference"
                ),
                "download_method": "direct",
                "attempted_urls": [
                    "https://www.noaa.gov/controlled-reference.jpg"
                ],
                "same_subject_or_scene": True,
                "same_capture_or_near_duplicate": True,
                "likely_different_original_capture": False,
                "edit_evidence_present": False,
                "edit_evidence_strength": "none",
                "differences": [],
                "overall_observation": (
                    "The reference and current image are the same original capture."
                ),
                "confidence": 0.99,
            },
            {
                "type": "object",
                "properties": {
                    "reference_url": {"type": "string"},
                    "source_page_url": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["reference_url"],
            },
        )


def _perception() -> PerceptionReport:
    return PerceptionReport(
        scene_description="A marked research vessel is visible on the water.",
        entities=[
            Entity(
                name="NOAA Ship Henry B. Bigelow",
                entity_type="object",
                bbox=[0.1, 0.2, 0.9, 0.9],
                confidence=0.98,
            ),
            Entity(
                name="NOAA logo",
                entity_type="logo",
                bbox=[0.7, 0.4, 0.8, 0.5],
                confidence=0.95,
            ),
        ],
        text_regions=[
            TextRegion(
                text="HENRY B. BIGELOW",
                bbox_quad=[[0.2, 0.4], [0.7, 0.4], [0.7, 0.5], [0.2, 0.5]],
                confidence=0.98,
            ),
            TextRegion(
                text="R 225",
                bbox_quad=[[0.7, 0.5], [0.82, 0.5], [0.82, 0.58], [0.7, 0.58]],
                confidence=0.96,
            ),
        ],
    )


def test_scripted_image_only_complete_trajectory(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"controlled-image-only-v3")
    case = ImageOnlyRuntimeCase(
        case_id="case_scripted_v3",
        image_path=str(image_path.resolve()),
        image_sha256=image_sha256(str(image_path)),
    )
    bootstrap = build_bootstrap_investigation(case, _perception())
    investigation = state_from_bootstrap(bootstrap)
    activate_initial_decisive_facts(investigation)
    react_task_ids = [
        task.task_id
        for task in bootstrap.tasks
        if any(
            fact_id in investigation.decisive_fact_ids
            for fact_id in task.fact_ids
        )
    ]
    assert len(react_task_ids) == 1

    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled-image-only-v3",
        validate_startup=False,
    )
    orchestrator.vlm_provider = "controlled"
    orchestrator.llm = AdaptiveImageOnlyBackend(react_task_ids)
    orchestrator.all_tools = {
        "perceive_scene": StaticTool(
            "perceive_scene",
            {
                "status": "success",
                "entities": [
                    item.model_dump()
                    for item in _perception().entities
                ],
                "scene_description": _perception().scene_description,
                "image_type": "photo",
            },
            {
                "type": "object",
                "properties": {"image_input": {"type": "string"}},
                "required": ["image_input"],
            },
        ),
        "ocr_with_position": StaticTool(
            "ocr_with_position",
            {
                "status": "success",
                "text_regions": [
                    {
                        "text": item.text,
                        "bbox_quad": item.bbox_quad,
                        "confidence": item.confidence,
                        "language": item.language,
                    }
                    for item in _perception().text_regions
                ],
            },
            {
                "type": "object",
                "properties": {"image_input": {"type": "string"}},
                "required": ["image_input"],
            },
        ),
        "reverse_image_search": StaticTool(
            "reverse_image_search",
            {
                "status": "success",
                "candidate_page_urls": [
                    "https://www.noaa.gov/controlled-reference"
                ],
                "reference_image_candidates": [
                    "https://www.noaa.gov/controlled-reference.jpg"
                ],
                "lens_results": [
                    {
                        "title": "NOAA controlled reference",
                        "url": "https://www.noaa.gov/controlled-reference",
                        "snippet": "Discovery only.",
                        "image_url": (
                            "https://www.noaa.gov/controlled-reference.jpg"
                        ),
                    }
                ],
                "semantic_results": [],
            },
            {
                "type": "object",
                "properties": {"image_input": {"type": "string"}},
                "required": ["image_input"],
            },
        ),
        "compare_with_reference": ControlledCompareTool(),
        "visit": ControlledVisitTool(),
    }
    orchestrator.tool_health_summary = {
        name: {"available": True, "error": ""}
        for name in orchestrator.all_tools
    }
    workflow = VerificationWorkflow(
        WorkflowConfig(
            output_dir=str(tmp_path / "traces"),
            save_traces=True,
        )
    )
    workflow._orchestrator = orchestrator

    result = asyncio.run(
        workflow.run_single(
            str(image_path),
            case.case_id,
            runtime_case=case,
        )
    )

    assert result["termination"] == "success"
    assert result["verdict"] == "real"
    assert result["verdict_basis"]["policy_rule_id"] == "reinspect-v2"
    state = result["state"]["investigation_state"]
    assert state["action_count"] == 2
    assert len(state["reflections"]) == 0
    assert state["coverage_audits"][-1]["stop_reason"] == "coverage_complete"
    assert state["discoveries"]
    assert state["evidence"]
    assert state["findings"]
    assert all(
        discovery["promoted_evidence_id"] is None
        for discovery in state["discoveries"]
    )
    trace_path = tmp_path / "traces" / f"{case.case_id}.json"
    assert trace_path.is_file()
