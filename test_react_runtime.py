from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.orchestrator.react_runtime import (
    FinishInvestigationTool,
    RuntimeToolAdapter,
    available_unified_react_runtime_tools,
    compile_react_judgment_basis,
    new_unified_react_runtime_state,
    reduce_react_action,
    render_react_judgment_context,
    render_react_runtime_context,
    validate_react_action,
)
from src.orchestrator.investigation_models import (
    DiscrepancyJudgmentOutput,
    FactCheckReport,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.state import ImageOnlyRuntimeCase
from src.orchestrator.stage_runner import StageRunner
from src.tools.base import BaseTool
from scripts.audit_real_trace import audit_trace


def _case() -> ImageOnlyRuntimeCase:
    return ImageOnlyRuntimeCase(
        case_id="react-runtime-fixture",
        image_path="fixture.jpg",
        image_sha256="a" * 64,
    )


class _VisitTool(BaseTool):
    name = "visit"
    description = "Visit a page."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
            },
            "image_claim": {"type": "string"},
            "retrieval_goal": {"type": "string"},
        },
        "required": ["url", "image_claim", "retrieval_goal"],
    }

    def __init__(self) -> None:
        self.received: dict = {}

    def call(self, params: dict) -> dict:
        self.received = dict(params)
        return {"status": "success"}


def _bootstrap(state) -> None:
    state.visual_memory = {
        "scene_description": "A bridge over a river.",
        "entities": [],
        "relations": [],
        "text_regions": [],
    }


def _progress(
    status: str = "investigating",
    basis: str = "The factual question remains unresolved.",
) -> dict:
    return {"status": status, "basis": basis}


def test_runtime_state_has_no_preconstructed_target_graph() -> None:
    state = new_unified_react_runtime_state(_case())
    assert "target_facts" not in state.model_dump()
    assert "search_hypotheses" not in state.model_dump()
    assert "crop_and_search" not in available_unified_react_runtime_tools(state)
    assert "perceive_scene" in available_unified_react_runtime_tools(state)
    assert "ocr_with_position" in available_unified_react_runtime_tools(state)
    assert "text_search" in available_unified_react_runtime_tools(state)


def test_public_tool_schema_hides_mature_internal_fields() -> None:
    state = new_unified_react_runtime_state(_case())
    _bootstrap(state)
    delegate = _VisitTool()
    adapter = RuntimeToolAdapter(delegate=delegate, state=state)

    assert "image_claim" not in adapter.parameters["properties"]
    assert "retrieval_goal" not in adapter.parameters["properties"]
    assert "investigation_progress" in adapter.parameters["properties"]
    assert "investigation_progress" in adapter.parameters["required"]
    adapter.call(
        {
            "url": ["https://example.test/page"],
            "question": "Check the event.",
            "investigation_progress": _progress(),
        }
    )
    assert delegate.received["image_claim"] == "Check the event."
    assert delegate.received["retrieval_goal"] == "Check the event."
    assert "investigation_progress" not in delegate.received


def test_action_memory_is_bounded_and_react_context_has_current_image_marker() -> None:
    state = new_unified_react_runtime_state(_case())
    _bootstrap(state)
    assert not validate_react_action(
        state,
        tool_name="text_search",
        tool_args={
            "queries": "bridge river",
            "investigation_progress": _progress(),
        },
    )
    update = reduce_react_action(
        state,
        tool_name="text_search",
        tool_args={
            "queries": "bridge river",
            "investigation_progress": _progress(),
        },
        call_id="search-1",
        serialized_result=json.dumps(
            {
                "status": "success",
                "queries": [
                    {
                        "query": "bridge river",
                        "results": [
                            {
                                "url": "https://example.test/page",
                                "title": "Bridge event",
                                "snippet": "A bridge event.",
                            }
                        ],
                    }
                ],
            }
        ),
    )
    assert update["created_discovery_ids"]
    context = json.loads(render_react_runtime_context(state))
    assert context["original_image"]["attached_to_this_request"] is True
    assert "target_facts" not in context
    assert "search_hypotheses" not in context
    assert context["budget"]["actions_used"] == 1
    assert context["budget"]["tool_budgets"]["text_search"] == {
        "used": 1,
        "limit": 16,
        "remaining": 15,
    }
    assert context["investigation_progress"]["status"] == "investigating"


def test_tool_availability_does_not_depend_on_model_progress_status() -> None:
    state = new_unified_react_runtime_state(_case())
    expected = available_unified_react_runtime_tools(state)

    for status in (
        "investigating",
        "decision_capable_support",
        "decision_capable_refute",
    ):
        state.investigation_progress = _progress(status)
        assert available_unified_react_runtime_tools(state) == expected


def test_finish_requires_model_declared_directional_progress() -> None:
    state = new_unified_react_runtime_state(_case())
    _bootstrap(state)
    tool = FinishInvestigationTool()
    assert "investigation_progress" in tool.parameters["required"]
    assert (
        validate_react_action(
            state,
            tool_name="finish_investigation",
            tool_args={
                "rationale": "I have finished.",
                "investigation_progress": _progress(),
            },
        )
        != ""
    )
    assert not validate_react_action(
        state,
        tool_name="finish_investigation",
        tool_args={
            "rationale": "A source directly resolves the event shown.",
            "investigation_progress": _progress(
                "decision_capable_support",
                "The official source directly identifies the pictured event.",
            ),
        },
    )


def test_judgment_basis_contains_compact_investigation_not_repeated_workspace() -> None:
    state = new_unified_react_runtime_state(_case())
    _bootstrap(state)
    basis = compile_react_judgment_basis(state)
    assert basis["schema_version"] == "ifv-unified-judgment-basis-v1"
    assert "target_facts" not in basis
    assert "workspace" not in basis


def test_visit_evidence_records_enter_react_ledger_and_final_context() -> None:
    state = new_unified_react_runtime_state(_case())
    _bootstrap(state)
    update = reduce_react_action(
        state,
        tool_name="visit",
        tool_args={
            "url": ["https://example.test/official"],
            "investigation_progress": _progress(),
        },
        call_id="visit-1",
        serialized_result=json.dumps(
            {
                "status": "success",
                "summary": "A convenience page summary that must not replace passages.",
                "evidence_records": [
                    {
                        "url": "https://example.test/official",
                        "evidence": "The official release confirms the pictured bridge opened in 2026.",
                        "stance": "support",
                        "directness": "direct",
                        "relevance": "high",
                    },
                    {
                        "url": "https://example.test/official",
                        "evidence": "The structure spans the river beside Highway 8.",
                        "stance": "unclear",
                        "directness": "indirect",
                        "relevance": "medium",
                    },
                ],
            }
        ),
    )

    assert len(update["created_evidence_ids"]) == 2
    assert len(state.evidence) == 2
    assert state.evidence[0]["excerpt"] == (
        "The official release confirms the pictured bridge opened in 2026."
    )
    assert state.evidence[0]["evidence_class"] == "decision_capable_support"

    context = json.loads(
        render_react_judgment_context(state, compile_react_judgment_basis(state))
    )
    assert context["investigation"]["evidence"][0]["excerpt"] == (
        "The official release confirms the pictured bridge opened in 2026."
    )
    assert "verdict_evidence_ids" in context["output_requirements"]


def test_final_judgment_cannot_cite_missing_runtime_evidence() -> None:
    valid = DiscrepancyJudgmentOutput(
        verdict="real",
        confidence=0.7,
        verdict_evidence_ids=["evidence-1"],
        overall_assessment="The recorded official release supports the event.",
        fact_check_report=FactCheckReport(
            headline="Recorded release supports the event",
            claim_under_review="The image depicts the documented event.",
            verdict_summary="The available ledger supports real.",
            key_findings=["The official release names the event."],
            evidence_summary="The report uses the recorded release.",
        ),
    )
    assert Orchestrator._validate_react_judgment(
        valid,
        basis={"evidence_ids": ["evidence-1"]},
    ) == (True, "")

    invalid = valid.model_copy(
        update={"verdict_evidence_ids": ["invented-evidence-id"]}
    )
    accepted, reason = Orchestrator._validate_react_judgment(
        invalid,
        basis={"evidence_ids": ["evidence-1"]},
    )
    assert not accepted
    assert "absent from the final investigation evidence ledger" in reason


def test_current_runtime_trace_passes_current_audit_without_target_graph(
    tmp_path: Path,
) -> None:
    state = new_unified_react_runtime_state(_case())
    update = reduce_react_action(
        state,
        tool_name="perceive_scene",
        tool_args={"investigation_progress": _progress()},
        call_id="call-scene",
        serialized_result=json.dumps(
            {
                "status": "success",
                "scene_description": "A bridge crosses a river.",
            }
        ),
    )
    basis = compile_react_judgment_basis(state)
    judgment = {
        "policy_rule_id": "unified-react-v1",
        "verdict": "real",
        "confidence": 0.6,
        "verdict_evidence_ids": basis["evidence_ids"],
        "selected_evidence_ids": basis["evidence_ids"],
        "overall_assessment": "The final binary label is a bounded judgment.",
        "fact_check_report": {
            "headline": "Bounded image fact check",
            "claim_under_review": "The image depicts a bridge crossing a river.",
            "verdict_summary": "The image is labeled real.",
            "key_findings": ["The visible scene is a bridge over water."],
            "evidence_summary": "The report uses the recorded image observation.",
            "remaining_uncertainties": [],
        },
        "evidence_citations": [],
        "unresolved_gaps": [],
    }
    action = {
        "stage": "unified_react",
        "action_type": "tool_call",
        "tool_name": "perceive_scene",
        "tool_args": {"investigation_progress": _progress()},
        "tool_result": json.dumps(
            {"status": "success", "scene_description": "A bridge crosses a river."}
        ),
        "tokens": {"thought": 12},
        "metadata": {
            "native_interactions": True,
            "interaction_id": "interaction-scene",
            "previous_interaction_id": "",
            "interaction_lifecycle_kind": "tool_roundtrip",
            "function_call_id": "call-scene",
            "tool_success": True,
            "policy_action": {
                "type": "tool_call",
                "name": "perceive_scene",
                "arguments": {"investigation_progress": _progress()},
            },
            "react_state_delta": update,
        },
    }
    final = {
        "stage": "unified_judgment",
        "action_type": "output",
        "tokens": {"thought": 8},
        "metadata": {
            "native_interactions": True,
            "interaction_id": "interaction-judgment",
            "previous_interaction_id": "",
            "interaction_lifecycle_kind": "standalone_request",
            "policy_action": {"type": "output", "value": judgment},
        },
        "output": judgment,
    }
    trace = {
        "image_id": "react-runtime-fixture",
        "image_path": "fixture.jpg",
        "input_mode": "image_only",
        "decision_policy_version": "unified-react-v1",
        "verdict": "real",
        "judgment": judgment,
        "verdict_basis": basis,
        "termination": "success",
        "token_usage": {"thought": 20},
        "state": {
            "image_id": "react-runtime-fixture",
            "input_mode": "image_only",
            "decision_policy_version": "unified-react-v1",
            "termination": "success",
            "token_usage": {"thought": 20},
            "runtime_case": {
                "case_id": "react-runtime-fixture",
                "image_path": "fixture.jpg",
                "image_sha256": "a" * 64,
            },
            "all_steps": [action, final],
            "investigation_state": state.model_dump(mode="json"),
            "judgment": judgment,
        },
    }
    path = tmp_path / "react-runtime-trace.json"
    path.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")

    report = audit_trace(path)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["unified_react_actions"] == 1
    assert report.stats["unified_react_target_facts"] == 0


def test_candidate_reference_images_are_reinjected_as_bounded_multimodal_items(
    monkeypatch,
) -> None:
    runner = StageRunner(
        llm=object(),
        system_prompt="",
        tools=[],
        attach_image=False,
    )
    result = json.dumps(
        {
            "status": "success",
            "reference_image_candidates": [
                "https://cdn.example.test/one",
                "https://cdn.example.test/two",
                "https://cdn.example.test/three",
                "https://cdn.example.test/four",
                "not-a-url",
            ],
        }
    )

    monkeypatch.setattr(
        StageRunner,
        "_native_candidate_image_item",
        staticmethod(
            lambda url: {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": url.rsplit("/", 1)[-1],
            }
        ),
    )

    items = asyncio.run(
        runner._visual_reinjection_items(
            "reverse_image_search",
            result=result,
        )
    )

    assert items == [
        {"type": "image", "mime_type": "image/jpeg", "data": "one"},
        {"type": "image", "mime_type": "image/jpeg", "data": "two"},
        {"type": "image", "mime_type": "image/jpeg", "data": "three"},
    ]
