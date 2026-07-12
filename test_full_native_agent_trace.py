from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from src.orchestrator.investigation_state import _id as investigation_id
from src.orchestrator.ledger import VerificationLedger, image_sha256
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.source_provenance import classify_source
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_health import ToolHealth
from src.orchestrator.state import VerificationCase, ClaimMode
from src.tools.base import BaseTool
from src.workflow import VerificationWorkflow, WorkflowConfig
from src.trace_viewer import render_trace_file


REFERENCE_PAGE = "https://www.nasa.gov/reference-launch"
REFERENCE_IMAGE = "https://images.nasa.gov/reference-launch.jpg"
LOCATION_PAGE = "https://www.nasa.gov/location-record"
LOCATION_EXCERPT = "NASA identifies the reference launch scene as Cape Canaveral in Florida."
LOCATION_HASH = hashlib.sha256(LOCATION_EXCERPT.encode("utf-8")).hexdigest()


def interaction_output(interaction_id: str, value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": interaction_id,
        "status": "completed",
        "usage": {"total_input_tokens": 20, "total_output_tokens": 10},
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(value)}],
            }
        ],
    }


def interaction_call(
    interaction_id: str,
    call_id: str,
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": interaction_id,
        "status": "requires_action",
        "usage": {"total_input_tokens": 20, "total_output_tokens": 10},
        "steps": [
            {
                "id": call_id,
                "type": "function_call",
                "name": name,
                "arguments": arguments,
            }
        ],
    }


class ScriptedInteractionsBackend:
    provider = "gemini"
    wire_api = "interactions"

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Gemini Interactions request")
        return self.responses.pop(0)

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("The reference agent must use Gemini Interactions only")


class StaticTool(BaseTool):
    def __init__(self, name: str, result: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        self.name = name
        self.description = f"Controlled integration fixture for {name}."
        self.parameters = parameters
        self.result = result
        self.calls: List[Dict[str, Any]] = []

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(dict(params))
        return dict(self.result)


def _verification_output(evidence: List[Dict[str, Any]], *, assessment: str) -> Dict[str, Any]:
    return {
        "evidence": evidence,
        "visual_anomalies": [],
        "authenticity_assessment": assessment,
        "key_findings": ["Controlled reference investigation updated its evidence state."],
        "source_findings": [],
        "visual_evidence": [],
        "world_model": {},
        "question_resolutions": [],
        "coverage_complete": False,
        "unresolved_priority_questions": [],
        "exhausted_priority_questions": [],
        "iteration_count": 0,
    }


def _build_responses(case: VerificationCase) -> List[Dict[str, Any]]:
    discovery_id = VerificationLedger._id(
        "discovery", "call-reverse-q0", REFERENCE_PAGE, "NASA reference launch"
    )
    visual_question_id = investigation_id(
        "vq",
        "claim-q0",
        investigation_id("reference", classify_source(REFERENCE_IMAGE).canonical_url),
    )
    visual_excerpt = "Near-duplicate NASA launch scene with no factual edits visible."
    image_source_id = VerificationLedger._id("source", "image", case.image_sha256)
    q0_evidence_id = VerificationLedger._id(
        "evidence",
        "call-compare-q0",
        "claim-q0",
        image_source_id,
        visual_excerpt,
    )
    location_identity = classify_source(LOCATION_PAGE)
    q1_source_id = VerificationLedger._id(
        "source", location_identity.canonical_url, LOCATION_HASH
    )
    q1_evidence_id = VerificationLedger._id(
        "evidence",
        "call-visit-q1",
        "claim-q1",
        q1_source_id,
        LOCATION_EXCERPT,
    )

    q0_evidence = {
        "function_call_id": "call-compare-q0",
        "source": REFERENCE_IMAGE,
        "summary": visual_excerpt,
        "raw_excerpt": visual_excerpt,
        "direction": "supports",
        "quality": "moderate",
        "tool_used": "compare_with_reference",
        "related_question": "q0",
    }
    q1_evidence = {
        "function_call_id": "call-visit-q1",
        "source": LOCATION_PAGE,
        "summary": LOCATION_EXCERPT,
        "raw_excerpt": LOCATION_EXCERPT,
        "direction": "supports",
        "quality": "strong",
        "tool_used": "visit",
        "related_question": "q1",
    }
    plan = {
        "questions": [
            {
                "question_id": "q0",
                "question": "Does the image match NASA's reference launch image without factual edits?",
                "claim_text": "The image matches NASA's reference launch image without factual edits.",
                "why": "Visual provenance is decisive.",
                "suggested_tools": ["reverse_image_search", "compare_with_reference"],
                "suggested_queries": ["NASA reference launch image"],
                "related_entities": ["NASA", "launch"],
                "priority": 1,
            },
            {
                "question_id": "q1",
                "question": "Was the launch scene captured at Cape Canaveral in Florida?",
                "claim_text": "The launch scene was captured at Cape Canaveral in Florida.",
                "why": "The location claim remains independently checkable.",
                "suggested_tools": ["visit"],
                "suggested_queries": ["NASA Cape Canaveral reference launch"],
                "related_entities": ["Cape Canaveral", "Florida", "launch"],
                "priority": 1,
            },
        ],
        "image_intent": "Controlled NASA launch reference fixture.",
        "is_trying_to_be_real": True,
        "risk_assessment": "controlled_integration_fixture",
        "revision": 0,
        "revision_reason": "",
    }
    revision = {
        "question_updates": [
            {
                "question_id": "q1",
                "question": "Does NASA's official location record identify this launch scene as Cape Canaveral in Florida?",
                "claim_text": "The launch scene was captured at Cape Canaveral in Florida.",
                "why": "The first iteration left the location claim unresolved.",
                "suggested_tools": ["visit"],
                "suggested_queries": ["site:nasa.gov Cape Canaveral Florida reference launch"],
                "related_entities": ["NASA", "Cape Canaveral", "Florida", "launch"],
                "priority": 1,
            }
        ],
        "revision_reason": "Use the official location record for the remaining claim.",
    }
    judgment = {
        "verdict": "real",
        "confidence": 0.9,
        "claim_decisions": [
            {
                "claim_id": "claim-q0",
                "decision": "support",
                "evidence_ids": [q0_evidence_id],
                "reason": None,
            },
            {
                "claim_id": "claim-q1",
                "decision": "support",
                "evidence_ids": [q1_evidence_id],
                "reason": None,
            },
        ],
        "selected_evidence_ids": [q0_evidence_id, q1_evidence_id],
        "policy_rule_id": "reinspect-v1",
        "unverifiable_reasons": [],
    }
    first_output = _verification_output([q0_evidence], assessment="uncertain")
    second_output = _verification_output([q0_evidence, q1_evidence], assessment="authentic")
    return [
        interaction_output("interaction-planning", plan),
        interaction_call(
            "interaction-v1-root",
            "call-reverse-q0",
            "reverse_image_search",
            {"question_id": "q0"},
        ),
        interaction_call(
            "interaction-v1-touch-q1",
            "call-touch-q1",
            "current_time",
            {"question_id": "q1"},
        ),
        interaction_call(
            "interaction-v1-compare",
            "call-compare-q0",
            "compare_with_reference",
            {
                "question_id": "q0",
                "reference_url": REFERENCE_IMAGE,
                "focus": "same capture and factual edits",
                "visual_question_id": visual_question_id,
                "source_discovery_id": discovery_id,
                "expected_property": (
                    "Whether the discovered reference is the same visual and contains factual edits."
                ),
            },
        ),
        interaction_output("interaction-v1-output", first_output),
        interaction_output("interaction-v1-forced", first_output),
        interaction_output("interaction-replanning", revision),
        interaction_call(
            "interaction-v2-root",
            "call-visit-q1",
            "visit",
            {
                "question_id": "q1",
                "url": [LOCATION_PAGE],
                "goal": "Verify the Cape Canaveral location in NASA's official record.",
            },
        ),
        interaction_output("interaction-v2-output", second_output),
        interaction_output("interaction-judgment", judgment),
    ]


class FullNativeReferenceOrchestrator(Orchestrator):
    def __init__(self, image_path: str, case: VerificationCase) -> None:
        super().__init__(
            provider="gemini",
            model_name="scripted-gemini-interactions",
            llm_wire_api="interactions",
            vlm_wire_api="interactions",
            max_rounds_verification=3,
            validate_startup=False,
        )
        self.max_verification_iterations = 2
        self.tool_cache = ToolResultCache(enabled=False)
        self.date_prefix = "Current date: 2026-07-11 (controlled integration fixture).\n\n"
        self.llm = ScriptedInteractionsBackend(_build_responses(case))
        self.all_tools = self._tools(image_path)
        self.tool_health = {
            name: ToolHealth(available=True) for name in self.all_tools
        }
        self.tool_health_summary = {
            name: {"available": True, "error": ""} for name in self.all_tools
        }

    @staticmethod
    def _tools(image_path: str) -> Dict[str, BaseTool]:
        _ = image_path
        return {
            "perceive_scene": StaticTool(
                "perceive_scene",
                {
                    "status": "success",
                    "entities": [
                        {
                            "name": "NASA launch fixture",
                            "entity_type": "scene_element",
                            "bbox": [0.1, 0.1, 0.9, 0.9],
                            "confidence": 0.99,
                            "attributes": {"fixture": "controlled"},
                        }
                    ],
                    "scene_description": "A controlled launch-pad integration fixture.",
                    "image_type": "illustration",
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
                            "text": "CONTROLLED NATIVE AGENT TRACE",
                            "bbox": [0.1, 0.1, 0.9, 0.3],
                            "confidence": 0.99,
                            "language": "en",
                        }
                    ],
                    "full_text": "CONTROLLED NATIVE AGENT TRACE",
                    "total_regions": 1,
                },
                {
                    "type": "object",
                    "properties": {
                        "image_input": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["image_input"],
                },
            ),
            "current_time": StaticTool(
                "current_time",
                {
                    "status": "success",
                    "datetime": "2026-07-11T00:00:00+00:00",
                    "timezone": "UTC",
                },
                {"type": "object", "properties": {}, "required": []},
            ),
            "reverse_image_search": StaticTool(
                "reverse_image_search",
                {
                    "status": "success",
                    "candidate_page_urls": [REFERENCE_PAGE],
                    "reference_image_url": REFERENCE_IMAGE,
                    "lens_results": [
                        {
                            "title": "NASA reference launch",
                            "url": REFERENCE_PAGE,
                            "snippet": "Controlled discovery result.",
                            "image_url": REFERENCE_IMAGE,
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
            "compare_with_reference": StaticTool(
                "compare_with_reference",
                {
                    "status": "success",
                    "reference_url": REFERENCE_IMAGE,
                    "same_subject_or_scene": True,
                    "same_capture_or_near_duplicate": True,
                    "likely_different_original_capture": False,
                    "edit_evidence_present": False,
                    "edit_evidence_strength": "none",
                    "differences": [],
                    "overall_observation": (
                        "Near-duplicate NASA launch scene with no factual edits visible."
                    ),
                    "confidence": 0.98,
                },
                {
                    "type": "object",
                    "properties": {
                        "reference_url": {"type": "string"},
                        "focus": {"type": "string"},
                        "visual_question_id": {"type": "string"},
                        "source_evidence_id": {"type": "string"},
                        "source_discovery_id": {"type": "string"},
                        "expected_property": {"type": "string"},
                    },
                    "required": ["reference_url"],
                },
            ),
            "visit": StaticTool(
                "visit",
                {
                    "status": "success",
                    "selected_url": LOCATION_PAGE,
                    "goal": "The launch scene was captured at Cape Canaveral in Florida.",
                    "summary": LOCATION_EXCERPT,
                    "evidence": LOCATION_EXCERPT,
                    "relevance": "high",
                    "stance": "support",
                    "artifact_sha256": LOCATION_HASH,
                    "evidence_span": {"start": 0, "end": len(LOCATION_EXCERPT)},
                    "retrieved_at": "2026-07-11T00:00:00+00:00",
                    "injection_flags": [],
                    "directness": "direct",
                    "evidence_eligible": True,
                    "visits": [],
                },
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "array", "items": {"type": "string"}},
                        "goal": {"type": "string"},
                    },
                    "required": ["url", "goal"],
                },
            ),
        }


def make_reference_image(path: Path) -> None:
    image = Image.new("RGB", (960, 540), "#eef1f3")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 880, 450), outline="#334155", width=6)
    draw.line((480, 140, 480, 390), fill="#c2410c", width=18)
    draw.polygon([(430, 390), (530, 390), (480, 455)], fill="#c2410c")
    draw.text((170, 25), "CONTROLLED NATIVE AGENT TRACE", fill="#111827")
    draw.text((250, 475), "NOT A REAL FACT-CHECK SAMPLE", fill="#7f1d1d")
    image.save(path)


def run_reference_trace(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "controlled-native-agent-fixture.png"
    make_reference_image(image_path)
    case = VerificationCase(
        case_id="multistage-iterative-agent-reference",
        image_path=str(image_path.resolve()),
        image_sha256=image_sha256(str(image_path)),
        claim_mode=ClaimMode.EXTERNAL,
        user_claim="Controlled fixture: verify visual provenance and launch location.",
        decision_policy_version="reinspect-v1",
    )
    workflow = VerificationWorkflow(
        WorkflowConfig(
            provider="gemini",
            model_name="scripted-gemini-interactions",
            llm_wire_api="interactions",
            vlm_wire_api="interactions",
            output_dir=str(output_dir),
            save_traces=True,
        )
    )
    orchestrator = FullNativeReferenceOrchestrator(str(image_path), case)
    workflow._orchestrator = orchestrator
    result = asyncio.run(
        workflow.run_single(
            str(image_path),
            case.case_id,
            verification_case=case,
        )
    )
    assert not orchestrator.llm.responses
    return result


def assert_full_reference_trace(result: Dict[str, Any]) -> None:
    state = result["state"]
    steps = state["all_steps"]
    assert result["termination"] == "success"
    assert result["verdict"] == "real"
    assert len(state["coverage_audits"]) == 2
    assert state["coverage_audits"][0]["complete"] is False
    assert state["coverage_audits"][1]["complete"] is True
    assert len(state["plan_history"]) == 2
    assert state["plan_history"][1]["revision"] == 1
    assert {step["metadata"]["verification_iteration"] for step in steps if step.get("stage") == "verification"} == {1, 2}
    assert any(step["stage"] == "replanning" for step in steps)
    native = [step for step in steps if step.get("metadata", {}).get("native_interactions")]
    assert native
    assert all(step["metadata"].get("interaction_id") for step in native)
    tool_steps = [step for step in native if step["action_type"] == "tool_call"]
    assert all(step["metadata"].get("function_call_id") for step in tool_steps)
    assert any(step["metadata"].get("previous_interaction_id") for step in native)
    assert any(step["action_type"] == "output_rejected" for step in steps)
    ledgers = state["ledgers"]
    assert len(ledgers["claims"]) == 2
    assert len(ledgers["evidence"]) == 2
    assert ledgers["discoveries"]
    assert not ledgers["failures"]
    investigation = state["investigation_state"]
    assert investigation["visual_questions"]
    assert investigation["visual_questions"][0]["status"] == "resolved"
    assert investigation["region_observations"]
    assert investigation["stopping_assessments"][-1]["can_stop"] is True


def test_full_native_multistage_reference_trace(tmp_path: Path) -> None:
    result = run_reference_trace(tmp_path)
    assert_full_reference_trace(result)
    trace = tmp_path / "multistage-iterative-agent-reference.json"
    html = tmp_path / "multistage-iterative-agent-reference.html"
    assert trace.is_file() and not html.exists()
    render_trace_file(str(trace), str(html))
    rendered = html.read_text(encoding="utf-8")
    for marker in (
        "Coverage Audits",
        "Plan revision 1",
        "Gemini Interactions",
        "Verification Ledgers",
        "Visual Questions",
        "previous_interaction_id",
    ):
        assert marker in rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the full native-agent reference trace.")
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.export_dir.exists():
        shutil.rmtree(args.export_dir)
    result = run_reference_trace(args.export_dir)
    assert_full_reference_trace(result)
    trace = args.export_dir / "multistage-iterative-agent-reference.json"
    html = args.export_dir / "multistage-iterative-agent-reference.html"
    render_trace_file(str(trace), str(html))
    print(trace)
    print(html)


if __name__ == "__main__":
    main()
