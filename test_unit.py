# -*- coding: utf-8 -*-
"""Focused tests for the active orchestrator pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from src.orchestrator.context import ContextRenderer
from src.orchestrator.llm_backend import LLMBackend, LLMResponse
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.stage_runner import StageRunner
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import (
    CoverageAudit,
    Entity,
    FinalJudgment,
    InvestigationQuestion,
    PlanRevision,
    PerceptionReport,
    TextRegion,
    VerificationPlan,
    VerificationResult,
    QuestionResolution,
)
from src.orchestrator.tool_cache import ToolResultCache
from src.tools.base import BaseTool
from src.trace_viewer import save_trace_html


class FakeLLM(LLMBackend):
    """Deterministic scripted LLM for tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def get_response(self, messages, **kwargs) -> LLMResponse:
        if self.calls >= len(self.responses):
            raise RuntimeError("No more scripted LLM responses.")
        text = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=5)


class FakeTool(BaseTool):
    def __init__(self, name: str, result, parameters=None):
        self.name = name
        self.description = name
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}
        self._result = result
        self.calls = []

    def call(self, params):
        self.calls.append(dict(params))
        return self._result


class FakeOrchestrator(Orchestrator):
    """Orchestrator wired with fake LLM/tool behavior for deterministic end-to-end tests."""

    def __init__(self, image_path: str):
        super().__init__(
            provider="lmdeploy",
            model_name="fake-model",
            timeout=60.0,
            validate_startup=False,
        )
        self.tool_cache = ToolResultCache(enabled=False)
        self.date_prefix = "Current date: 2026-07-10 (UTC+08:00).\n\n"
        self.llm = FakeLLM(
            [
                self._planning_response(),
                self._verification_round_1(),
                self._verification_round_2(),
                self._verification_output(),
                self._judgment_response(),
            ]
        )
        self.all_tools = self._build_fake_tools(image_path)
        self.tool_health = {name: type("Health", (), {"available": True, "error": ""})() for name in self.all_tools}
        self.tool_health_summary = {name: {"available": True, "error": ""} for name in self.all_tools}

    def _build_fake_tools(self, image_path: str):
        return {
            "perceive_scene": FakeTool(
                "perceive_scene",
                {
                    "status": "success",
                    "entities": [
                        {"name": "NASA logo", "entity_type": "logo", "bbox": [0.1, 0.1, 0.3, 0.2], "confidence": 0.95, "attributes": {}},
                        {"name": "Launch pad", "entity_type": "scene_element", "bbox": [0.2, 0.2, 0.9, 0.9], "confidence": 0.88, "attributes": {}},
                    ],
                    "scene_description": "A launch pad scene with a visible NASA logo.",
                    "image_type": "photo",
                },
                parameters={"type": "object", "properties": {"image_input": {"type": "string"}}, "required": ["image_input"]},
            ),
            "ocr_with_position": FakeTool(
                "ocr_with_position",
                {
                    "status": "success",
                    "text_regions": [
                        {
                            "text": "NASA",
                            "bbox": [0.1, 0.1, 0.3, 0.18],
                            "confidence": 0.99,
                            "language": "en",
                        }
                    ],
                    "total_regions": 1,
                    "full_text": "NASA",
                },
                parameters={"type": "object", "properties": {"image_input": {"type": "string"}}, "required": ["image_input"]},
            ),
            "reverse_image_search": FakeTool(
                "reverse_image_search",
                {
                    "status": "success",
                    "candidate_page_urls": ["https://www.nasa.gov/example-launch"],
                    "reference_image_url": "https://images.nasa.gov/launch.jpg",
                    "lens_results": [
                        {
                            "title": "NASA Launch Photo",
                            "url": "https://www.nasa.gov/example-launch",
                            "snippet": "Official NASA launch imagery.",
                            "image_url": "https://images.nasa.gov/launch.jpg",
                        }
                    ],
                    "semantic_results": [],
                },
                parameters={"type": "object", "properties": {"image_input": {"type": "string"}}, "required": ["image_input"]},
            ),
            "visit": FakeTool(
                "visit",
                {
                    "status": "success",
                    "selected_url": "https://www.nasa.gov/example-launch",
                    "summary": "NASA published the same launch-pad image on its official site.",
                    "evidence": "NASA published the same launch-pad image on its official site.",
                    "relevance": "high",
                    "stance": "support",
                    "artifact_sha256": "a" * 64,
                    "evidence_span": {"start": 0, "end": 62},
                    "retrieved_at": "2026-07-11T00:00:00+00:00",
                    "injection_flags": [],
                    "directness": "direct",
                    "evidence_eligible": True,
                    "visits": [
                        {
                            "url": "https://www.nasa.gov/example-launch",
                            "summary": "NASA published the same image.",
                            "evidence": "NASA published the same image.",
                            "relevance": "high",
                            "stance": "support",
                            "artifact_sha256": "b" * 64,
                            "evidence_span": {"start": 0, "end": 30},
                            "retrieved_at": "2026-07-11T00:00:00+00:00",
                            "injection_flags": [],
                            "directness": "direct",
                            "evidence_eligible": True,
                        }
                    ],
                },
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "array"}, "goal": {"type": "string"}},
                    "required": ["url", "goal"],
                },
            ),
        }

    @staticmethod
    def _planning_response() -> str:
        return """<think>Use the visible brand and scene to investigate provenance.</think>
<output>{
  "questions": [
    {
      "question_id": "q0",
      "question": "Did NASA publish this launch-pad image?",
      "why": "Official provenance is the strongest authenticity signal.",
      "suggested_tools": ["reverse_image_search", "visit"],
      "suggested_queries": ["NASA launch pad image"],
      "related_entities": ["NASA"],
      "priority": 1
    }
  ],
  "image_intent": "Show a real NASA launch-pad scene.",
  "is_trying_to_be_real": true,
  "risk_assessment": "official_photo_claim",
  "revision": 0,
  "revision_reason": ""
}</output>"""

    @staticmethod
    def _verification_round_1() -> str:
        return """<think>Find an origin page for the image first.</think>
<tool_call>{"name":"reverse_image_search","arguments":{"question_id":"q0","image_input":"ignored.jpg"}}</tool_call>"""

    @staticmethod
    def _verification_round_2() -> str:
        return """<think>The reverse match points to an official NASA page. Read that source next.</think>
<tool_call>{"name":"visit","arguments":{"question_id":"q0","url":["https://www.nasa.gov/example-launch"],"goal":"verify publication provenance"}}</tool_call>"""

    @staticmethod
    def _verification_output() -> str:
        return """<think>Official provenance has been established.</think>
<output>{
  "evidence": [
    {
      "function_call_id": "legacy-verification-2-2",
      "source": "https://www.nasa.gov/example-launch",
      "summary": "NASA published the same launch-pad image on its official site.",
      "raw_excerpt": "NASA published the same launch-pad image on its official site.",
      "direction": "supports",
      "quality": "strong",
      "tool_used": "visit",
      "related_question": "q0"
    }
  ],
  "visual_anomalies": [],
  "authenticity_assessment": "authentic",
  "key_findings": ["Official NASA provenance found for the image."],
  "source_findings": [],
  "visual_evidence": [],
  "world_model": {},
  "question_resolutions": [],
  "coverage_complete": false,
  "unresolved_priority_questions": [],
  "exhausted_priority_questions": [],
  "iteration_count": 1
}</output>"""

    @staticmethod
    def _judgment_response() -> str:
        return """<think>Strong official provenance supports authenticity.</think>
<output>{
  "verdict": "real",
  "confidence": 0.92,
  "reasoning_chain": "The image matches an official NASA publication, and no contradictory visual evidence was found.",
  "key_evidence": ["NASA published the same launch-pad image on its official site."],
  "anomalies": [],
  "overall_assessment": "The image is supported by direct official source evidence and appears real.",
  "claim_decisions": [{
    "claim_id": "claim-q0",
    "decision": "support",
    "evidence_ids": ["evidence-31240b4e2c484290a8c4"],
    "reason": null
  }],
  "selected_evidence_ids": ["evidence-31240b4e2c484290a8c4"],
  "policy_rule_id": "reinspect-v1",
  "unverifiable_reasons": []
}</output>"""


def make_test_image() -> str:
    tmp_dir = tempfile.mkdtemp(prefix="verifier-test-")
    image_path = os.path.join(tmp_dir, "sample.png")
    Image.new("RGB", (32, 32), color=(220, 220, 220)).save(image_path)
    return image_path


def test_context_rendering():
    perception = PerceptionReport(
        entities=[Entity(name="NASA logo", entity_type="logo", confidence=0.9)],
        text_regions=[TextRegion(text="NASA", bbox_quad=[[0, 0], [1, 0], [1, 1], [0, 1]], confidence=0.99, language="en")],
        scene_description="A launch pad scene.",
        image_type="photo",
    )
    plan = VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did NASA publish this image?",
                why="Source provenance matters.",
                suggested_tools=["reverse_image_search", "visit"],
                suggested_queries=["NASA launch pad image"],
                related_entities=["NASA"],
                priority=1,
            )
        ],
        image_intent="Show a real NASA launch scene.",
        is_trying_to_be_real=True,
        risk_assessment="official_photo_claim",
    )
    verification = VerificationResult(
        authenticity_assessment="authentic",
        key_findings=["Official NASA provenance found."],
    )

    planning_text = ContextRenderer.render_for_planning(perception)
    verification_text = ContextRenderer.render_for_verification(perception, plan)
    judgment_text = ContextRenderer.render_for_judgment(perception, plan, verification)

    assert "NASA logo" in planning_text
    assert "Did NASA publish this image?" in verification_text
    assert "Official NASA provenance found." in judgment_text


def test_stage_runner_multi_round_tool_use():
    llm = FakeLLM(
        [
            """<think>Search first.</think><tool_call>{"name":"text_search","arguments":{"question_id":"q0","queries":["nasa launch"]}}</tool_call>""",
            """<think>Now output.</think><output>{"verdict":"real","confidence":0.8,"reasoning_chain":"ok","key_evidence":[],"anomalies":[],"overall_assessment":"done"}</output>""",
        ]
    )
    tool = FakeTool(
        "text_search",
        {
            "status": "success",
            "queries": [
                {"query": "nasa launch", "results": [{"title": "NASA Launch", "url": "https://www.nasa.gov", "snippet": "Official NASA launch page"}]}
            ],
        },
        parameters={"type": "object", "properties": {"queries": {"type": "array"}}, "required": ["queries"]},
    )
    runner = StageRunner(
        llm=llm,
        system_prompt="You are testing.",
        tools=[tool],
        output_schema=FinalJudgment,
        max_rounds=3,
        image_path="",
        stage_name="verification",
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("test context"))
    assert parsed is not None
    assert parsed.verdict == "real"
    assert [step.action_type for step in steps[:2]] == ["tool_call", "output"]
    assert tool.calls[0]["queries"] == ["nasa launch"]


def test_verification_rejects_zero_tool_output():
    llm = FakeLLM(
        [
            '<output>{"evidence":[],"visual_anomalies":[],"authenticity_assessment":"authentic","key_findings":["memory only"],"source_findings":[],"visual_evidence":[],"world_model":{},"question_resolutions":[],"coverage_complete":false,"unresolved_priority_questions":[],"exhausted_priority_questions":[],"iteration_count":1}</output>',
            '<output>{"evidence":[],"visual_anomalies":[],"authenticity_assessment":"authentic","key_findings":["still memory only"],"source_findings":[],"visual_evidence":[],"world_model":{},"question_resolutions":[],"coverage_complete":false,"unresolved_priority_questions":[],"exhausted_priority_questions":[],"iteration_count":1}</output>',
        ]
    )
    tool = FakeTool("text_search", {}, parameters={"type": "object", "properties": {}, "required": []})
    runner = StageRunner(
        llm=llm,
        system_prompt="Investigate before answering.",
        tools=[tool],
        output_schema=VerificationResult,
        max_rounds=1,
        stage_name="verification",
        min_tool_calls=1,
        attach_image=False,
    )
    parsed, steps = asyncio.run(runner.run("[q0] verify this claim"))
    assert parsed is None
    assert steps[0].action_type == "output_rejected"
    assert "at least 1 tool calls" in steps[0].metadata["rejection_reason"]


def test_tool_error_is_not_supporting_evidence():
    orchestrator = object.__new__(Orchestrator)
    state = type("State", (), {})()
    state.plan = VerificationPlan(
        questions=[InvestigationQuestion(question_id="q0", question="Is it consistent?", priority=1)]
    )
    state.perception = PerceptionReport()
    steps = [
        StageStep(
            round=1,
            stage_name="verification",
            action_type="tool_call",
            tool_name="check_consistency",
            tool_args={"__question_id": "q0"},
            tool_result='{"status":"error","error":"backend unavailable"}',
        )
    ]
    result = orchestrator._build_verification_result_from_steps(steps, state)
    assert not result.evidence
    assert result.authenticity_assessment == "uncertain"
    assert any("excluded from evidence" in finding for finding in result.key_findings)


class ReplanningOrchestrator(FakeOrchestrator):
    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.max_rounds_verification = 3
        self.max_verification_iterations = 2
        self.llm = FakeLLM(
            [
                self._two_question_plan(),
                self._q0_tool_call(),
                self._q0_visit_call(),
                self._q0_only_output(),
                self._q0_only_output(),
                self._revised_plan(),
                self._q1_tool_call(),
                self._both_questions_output(),
                self._judgment_response(),
            ]
        )
        self.all_tools["text_search"] = FakeTool(
            "text_search",
            {
                "status": "success",
                "queries": [{
                    "query": "NASA launch location",
                    "selected_url": "https://example.edu/nasa-launch-location",
                    "summary": "A university archive identifies the launch location shown in the image.",
                    "evidence": "A university archive identifies the launch location shown in the image.",
                    "relevance": "high",
                    "stance": "support",
                    "artifact_sha256": "c" * 64,
                    "evidence_span": {"start": 0, "end": 71},
                    "retrieved_at": "2026-07-11T00:00:00+00:00",
                    "injection_flags": [],
                    "directness": "direct",
                    "evidence_eligible": True,
                    "results": [],
                }],
            },
            parameters={
                "type": "object",
                "properties": {"queries": {"type": "array"}},
                "required": ["queries"],
            },
        )

    @staticmethod
    def _two_question_plan() -> str:
        return """<output>{
          "questions": [
            {"question_id":"q0","question":"Did NASA publish this image?","why":"provenance","suggested_tools":["reverse_image_search"],"suggested_queries":["NASA launch image"],"related_entities":["NASA"],"priority":1},
            {"question_id":"q1","question":"Where was the launch image captured?","why":"location context","suggested_tools":["text_search"],"suggested_queries":["NASA launch location"],"related_entities":["launch"],"priority":1}
          ],
          "image_intent":"Show a real NASA launch.","is_trying_to_be_real":true,"risk_assessment":"event_photo","revision":0,"revision_reason":""
        }</output>"""

    @staticmethod
    def _q0_tool_call() -> str:
        return '<tool_call>{"name":"reverse_image_search","arguments":{"question_id":"q0"}}</tool_call>'

    @staticmethod
    def _q1_tool_call() -> str:
        return '<tool_call>{"name":"text_search","arguments":{"question_id":"q1","queries":["NASA launch location"]}}</tool_call>'

    @staticmethod
    def _q0_visit_call() -> str:
        return '<tool_call>{"name":"visit","arguments":{"question_id":"q0","url":["https://www.nasa.gov/example-launch"],"goal":"Did NASA publish this image?"}}</tool_call>'

    @staticmethod
    def _q0_only_output() -> str:
        return """<output>{"evidence":[{"function_call_id":"legacy-verification-2-2","source":"https://www.nasa.gov/example-launch","summary":"NASA published the same image.","raw_excerpt":"NASA published the same image.","direction":"supports","quality":"strong","tool_used":"visit","related_question":"q0"}],"visual_anomalies":[],"authenticity_assessment":"uncertain","key_findings":["q0 resolved"],"source_findings":[],"visual_evidence":[],"world_model":{},"question_resolutions":[],"coverage_complete":false,"unresolved_priority_questions":["q1"],"exhausted_priority_questions":[],"iteration_count":1}</output>"""

    @staticmethod
    def _revised_plan() -> str:
        return """<output>{
          "question_updates": [
            {"question_id":"q1","question":"Where was the launch image captured?","why":"location context","suggested_tools":["text_search"],"suggested_queries":["NASA launch location official archive"],"related_entities":["launch"],"priority":1}
          ],
          "revision_reason":"Refine the unresolved location query."
        }</output>"""

    @staticmethod
    def _both_questions_output() -> str:
        return """<output>{"evidence":[
          {"function_call_id":"legacy-verification-2-2","source":"https://www.nasa.gov/example-launch","summary":"NASA published the same image.","raw_excerpt":"NASA published the same image.","direction":"supports","quality":"strong","tool_used":"visit","related_question":"q0"},
          {"function_call_id":"legacy-verification-1-5","source":"https://example.edu/nasa-launch-location","summary":"A university archive identifies the launch location shown in the image.","raw_excerpt":"A university archive identifies the launch location shown in the image.","direction":"supports","quality":"moderate","tool_used":"text_search","related_question":"q1"}
        ],"visual_anomalies":[],"authenticity_assessment":"authentic","key_findings":["Both priority questions resolved"],"source_findings":[],"visual_evidence":[],"world_model":{},"question_resolutions":[],"coverage_complete":false,"unresolved_priority_questions":[],"exhausted_priority_questions":[],"iteration_count":2}</output>"""

    @staticmethod
    def _judgment_response() -> str:
        return """<output>{
          "verdict":"real",
          "confidence":0.92,
          "reasoning_chain":"Both claims are supported.",
          "key_evidence":[],
          "anomalies":[],
          "overall_assessment":"Supported.",
          "claim_decisions":[
            {"claim_id":"claim-q0","decision":"support","evidence_ids":["evidence-31240b4e2c484290a8c4"],"reason":null},
            {"claim_id":"claim-q1","decision":"support","evidence_ids":["evidence-c3e4da86360f861d07a5"],"reason":null}
          ],
          "selected_evidence_ids":["evidence-31240b4e2c484290a8c4","evidence-c3e4da86360f861d07a5"],
          "policy_rule_id":"reinspect-v1",
          "unverifiable_reasons":[]
        }</output>"""


def test_uncovered_priority_question_triggers_replanning():
    image_path = make_test_image()
    try:
        orchestrator = ReplanningOrchestrator(image_path)
        result = asyncio.run(orchestrator.run(image_path, "replanning-sample"))
        state = result["state"]
        assert result["verdict"] == "real"
        assert len(state["plan_history"]) == 2
        assert state["plan_history"][1]["revision"] == 1
        assert len(state["coverage_audits"]) == 2
        assert state["coverage_audits"][0]["complete"] is False
        assert state["coverage_audits"][0]["unresolved_priority_questions"] == ["q1"]
        assert state["coverage_audits"][1]["complete"] is True
        assert state["verification"]["iteration_count"] == 2
        assert any(step["stage"] == "replanning" for step in state["all_steps"])
    finally:
        shutil.rmtree(os.path.dirname(image_path), ignore_errors=True)


def test_plan_revision_only_updates_unresolved_question():
    current = VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Resolved provenance question",
                suggested_tools=["reverse_image_search"],
                priority=1,
            ),
            InvestigationQuestion(
                question_id="q1",
                question="Original unresolved location question",
                suggested_tools=["text_search"],
                priority=1,
            ),
        ],
        image_intent="Show a launch.",
        risk_assessment="event photo",
    )
    audit = CoverageAudit(
        unresolved_priority_questions=["q1"],
        question_resolutions=[
            QuestionResolution(question_id="q0", status="resolved"),
            QuestionResolution(question_id="q1", status="unanswered"),
        ],
        reason="q1 remains unresolved",
    )
    revision = PlanRevision(
        question_updates=[
            InvestigationQuestion(
                question_id="q1",
                question="Refined unresolved location question",
                suggested_tools=["visit"],
                suggested_queries=["official launch location"],
                priority=1,
            )
        ],
        revision_reason="Use an official location source.",
    )

    accepted, reason = Orchestrator._validate_plan_revision(revision, current, audit)
    assert accepted, reason
    revised = Orchestrator._apply_plan_revision(current, revision, audit)
    assert revised.questions[0].question == "Resolved provenance question"
    assert revised.questions[1].question == "Refined unresolved location question"
    assert revised.revision == 1
    assert revised.revision_reason == "Use an official location source."


def test_plan_revision_rejects_resolved_question_update():
    current = VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Resolved question",
                suggested_tools=["text_search"],
                priority=1,
            ),
            InvestigationQuestion(
                question_id="q1",
                question="Unresolved question",
                suggested_tools=["text_search"],
                priority=1,
            ),
        ]
    )
    audit = CoverageAudit(unresolved_priority_questions=["q1"])
    invalid = PlanRevision(
        question_updates=[
            InvestigationQuestion(
                question_id="q0",
                question="Rewrite resolved question",
                suggested_tools=["visit"],
                priority=1,
            )
        ]
    )
    accepted, reason = Orchestrator._validate_plan_revision(invalid, current, audit)
    assert not accepted
    assert "resolved or exhausted" in reason


def test_orchestrator_end_to_end_and_trace_export():
    image_path = make_test_image()
    try:
        orchestrator = FakeOrchestrator(image_path)
        result = asyncio.run(orchestrator.run(image_path, "sample-image"))
        assert result["verdict"] == "real"
        assert result["termination"] == "success"
        state = result["state"]
        steps = state["all_steps"]
        assert any(step["stage"] == "verification" and step["tool_name"] == "reverse_image_search" for step in steps)
        assert any(step["stage"] == "verification" and step["tool_name"] == "visit" for step in steps)
        assert state["verification"]["authenticity_assessment"] == "authentic"
        assert state["judgment"]["confidence"] > 0.9

        out_dir = tempfile.mkdtemp(prefix="verifier-trace-")
        try:
            html_path = os.path.join(out_dir, "trace.html")
            save_trace_html(result, html_path)
            html = Path(html_path).read_text(encoding="utf-8")
            assert "verification" in html
            assert "reverse_image_search" in html
            assert "NASA published the same launch-pad image" in html
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        shutil.rmtree(os.path.dirname(image_path), ignore_errors=True)


if __name__ == "__main__":
    test_context_rendering()
    test_stage_runner_multi_round_tool_use()
    test_verification_rejects_zero_tool_output()
    test_tool_error_is_not_supporting_evidence()
    test_uncovered_priority_question_triggers_replanning()
    test_orchestrator_end_to_end_and_trace_export()
    print("All unit tests passed.")
