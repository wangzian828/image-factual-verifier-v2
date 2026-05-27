# -*- coding: utf-8 -*-
"""Main Orchestrator: 4-stage pipeline for image factual verification.

Runs: Perception → Planning → Verification → Judgment
Each stage is either a StageRunner (with tools) or a single LLM call (no tools).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.orchestrator.context import ContextRenderer
from src.orchestrator.llm_backend import APIBackend, LLMBackend
from src.orchestrator.stage_runner import StageRunner, StageStep
from src.orchestrator.stages import perception, planning, verification, judgment
from src.orchestrator.state import (
    Entity,
    FaceDetection,
    FinalJudgment,
    PerceptionReport,
    TextRegion,
    VerificationPlan,
    VerificationResult,
    VerificationState,
)
from src.orchestrator.tool_registry import build_all_tools, build_stage_tools


class Orchestrator:
    """4-stage pipeline orchestrator for image factual verification.

    Usage:
        orchestrator = Orchestrator(provider="lmdeploy", model_name="/gsdata/home/wza/models/Qwen3-VL-8B-Thinking")
        result = await orchestrator.run("path/to/image.jpg")
    """

    def __init__(
        self,
        provider: str = "lmdeploy",
        model_name: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking",
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        max_rounds_perception: int = 5,
        max_rounds_verification: int = 8,
        timeout: float = 300.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.provider = provider
        self.model_name = model_name
        self.vlm_provider = vlm_provider or provider
        self.vlm_model = vlm_model or model_name
        self.max_rounds_perception = max_rounds_perception
        self.max_rounds_verification = max_rounds_verification
        self.timeout = timeout

        # Build LLM backend
        self.llm = APIBackend(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Build all tools (lazy init internally)
        self.all_tools = build_all_tools(
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
        )

    async def run(self, image_path: str, image_id: str = "") -> Dict[str, Any]:
        """Run the full 4-stage verification pipeline.

        Args:
            image_path: Path to the image file.
            image_id: Optional identifier for tracking.

        Returns:
            Dict with judgment, state, and metadata.
        """
        state = VerificationState(
            image_path=image_path,
            image_id=image_id or image_path,
        )

        start_time = time.time()

        # === Stage 1: PERCEPTION ===
        state.perception = await self._run_perception(state, image_path)

        # === Stage 2: PLANNING ===
        state.plan = await self._run_planning(state, image_path)

        # === Stage 3: VERIFICATION ===
        # Check timeout
        if time.time() - start_time > self.timeout * 0.8:
            state.termination = "timeout"
        else:
            state.verification = await self._run_verification(state, image_path)

        # === Stage 4: JUDGMENT ===
        state.judgment = await self._run_judgment(state, image_path)

        # Finalize
        if not state.termination:
            state.termination = "success"

        total_time = time.time() - start_time
        state.stage_timings["total"] = round(total_time, 2)

        return {
            "image_id": state.image_id,
            "image_path": state.image_path,
            "judgment": state.judgment.model_dump() if state.judgment else None,
            "verdict": state.judgment.verdict if state.judgment else "unverifiable",
            "confidence": state.judgment.confidence if state.judgment else 0.0,
            "overall_assessment": state.judgment.overall_assessment if state.judgment else "",
            "state": state.to_dict(),
            "termination": state.termination,
            "time_taken": total_time,
            "token_usage": state.token_usage,
            "total_tool_calls": state.total_tool_calls,
            "llm_api_calls": state.llm_api_calls,
        }

    # =========================================================================
    # Stage implementations
    # =========================================================================

    async def _run_perception(
        self, state: VerificationState, image_path: str
    ) -> PerceptionReport:
        """Stage 1: Extract all observable content from the image.

        Two-step approach:
        1. Model directly observes the image and outputs PerceptionReport (no tools)
        2. Pipeline automatically calls OCR/face_detect based on what was observed
        """
        t0 = time.time()

        # Step 1: Model observes directly (no tools, 1 LLM call)
        runner = StageRunner(
            llm=self.llm,
            system_prompt=perception.SYSTEM_PROMPT,
            tools=[],
            output_schema=PerceptionReport,
            max_rounds=2,
            image_path=image_path,
            stage_name="perception",
        )

        result, steps = await runner.run(
            "请仔细观察这张图片，提取所有可见的实体、文字和人脸信息。直接输出 <output>...</output>。"
        )

        state.all_steps.extend(steps)
        state.llm_api_calls += runner.llm_api_calls
        self._accumulate_tokens(state, steps)

        if not result or not isinstance(result, PerceptionReport):
            result = PerceptionReport(scene_description="(perception failed)")

        # Step 2: Automatically supplement with tools based on observations
        result = await self._supplement_perception(result, image_path, state)

        state.stage_timings["perception"] = round(time.time() - t0, 2)
        return result

    async def _supplement_perception(
        self, report: PerceptionReport, image_path: str, state: VerificationState
    ) -> PerceptionReport:
        """Automatically call OCR and face_detect based on perception results."""
        import asyncio
        import json as _json

        has_text = (
            bool(report.text_regions)
            or any(e.entity_type in ("text", "logo") for e in report.entities)
            or any("text" in e.name.lower() or "文字" in e.name for e in report.entities)
        )
        has_person = any(
            e.entity_type == "person" or "person" in e.name.lower() or "人" in e.name
            for e in report.entities
        )

        # Always call OCR (most images have text; cheap CPU operation)
        if "ocr_with_position" in self.all_tools:
            try:
                ocr_tool = self.all_tools["ocr_with_position"]
                loop = asyncio.get_event_loop()
                ocr_result = await loop.run_in_executor(
                    None, ocr_tool.call, {"image_input": image_path}
                )
                state.total_tool_calls += 1
                if isinstance(ocr_result, dict) and ocr_result.get("status") == "success":
                    for t in ocr_result.get("text_regions", []):
                        report.text_regions.append(TextRegion(
                            text=t.get("text", ""),
                            bbox_quad=t.get("bbox_quad", []),
                            confidence=t.get("confidence", 0.0),
                            language=t.get("language", "unknown"),
                        ))
            except Exception:
                pass

        # Call face_detect if people detected (or if perception failed — be safe)
        perception_failed = not report.entities and not report.scene_description
        if (has_person or perception_failed) and "face_detect" in self.all_tools:
            try:
                face_tool = self.all_tools["face_detect"]
                loop = asyncio.get_event_loop()
                face_result = await loop.run_in_executor(
                    None, face_tool.call, {"image_input": image_path}
                )
                state.total_tool_calls += 1
                if isinstance(face_result, dict) and face_result.get("status") == "success":
                    for f in face_result.get("faces", []):
                        report.faces.append(FaceDetection(
                            bbox=f.get("bbox", []),
                            confidence=f.get("confidence", 0.0),
                        ))
            except Exception:
                pass

        return report

    def _reconstruct_perception_from_steps(self, steps: List) -> PerceptionReport:
        """Reconstruct PerceptionReport from tool call results when model output fails."""
        import json as _json

        entities = []
        text_regions = []
        faces = []
        scene_description = ""
        image_type = "photo"

        for step in steps:
            if step.action_type != "tool_call" or not step.tool_result:
                continue
            try:
                data = _json.loads(step.tool_result) if isinstance(step.tool_result, str) else step.tool_result
            except (_json.JSONDecodeError, TypeError):
                continue

            if not isinstance(data, dict) or data.get("status") == "error":
                continue

            if step.tool_name == "perceive_scene":
                for e in data.get("entities", []):
                    entities.append(Entity(
                        name=e.get("name", ""),
                        entity_type=e.get("entity_type", ""),
                        bbox=e.get("bbox", []),
                        confidence=e.get("confidence", 1.0),
                        attributes=e.get("attributes", {}),
                    ))
                scene_description = data.get("scene_description", "")
                image_type = data.get("image_type", "photo")

            elif step.tool_name == "ocr_with_position":
                for t in data.get("text_regions", []):
                    text_regions.append(TextRegion(
                        text=t.get("text", ""),
                        bbox_quad=t.get("bbox_quad", []),
                        confidence=t.get("confidence", 0.0),
                        language=t.get("language", "unknown"),
                    ))

            elif step.tool_name == "face_detect":
                for f in data.get("faces", []):
                    faces.append(FaceDetection(
                        bbox=f.get("bbox", []),
                        confidence=f.get("confidence", 0.0),
                    ))

        if not entities and not text_regions and not faces and not scene_description:
            return PerceptionReport(scene_description="(perception failed)")

        return PerceptionReport(
            entities=entities,
            text_regions=text_regions,
            faces=faces,
            scene_description=scene_description,
            image_type=image_type,
        )

    async def _run_planning(
        self, state: VerificationState, image_path: str
    ) -> VerificationPlan:
        """Stage 2: Determine what to investigate (single LLM call, no tools)."""
        t0 = time.time()

        # Build input context from perception
        input_context = ContextRenderer.render_for_planning(state.perception)
        input_context += "\n\n请分析这张图片，输出验证计划。直接输出 <output>...</output>。"

        runner = StageRunner(
            llm=self.llm,
            system_prompt=planning.SYSTEM_PROMPT,
            tools=[],  # No tools for planning
            output_schema=VerificationPlan,
            max_rounds=2,
            image_path=image_path,
            stage_name="planning",
        )

        result, steps = await runner.run(input_context)

        state.stage_timings["planning"] = round(time.time() - t0, 2)
        state.all_steps.extend(steps)
        state.llm_api_calls += runner.llm_api_calls
        self._accumulate_tokens(state, steps)

        if result and isinstance(result, VerificationPlan):
            # Inject default question if empty
            if not result.questions:
                from src.orchestrator.state import InvestigationQuestion
                result.questions = [InvestigationQuestion(
                    question_id="q0",
                    question="这张图片是否是真实的？",
                    why="默认问题：无法从图片中识别出具体的事实声明",
                    suggested_tools=["reverse_image_search", "analyze_visual_anomalies"],
                    suggested_queries=[],
                    priority=1,
                )]
            return result

        # Fallback
        from src.orchestrator.state import InvestigationQuestion
        return VerificationPlan(
            questions=[InvestigationQuestion(
                question_id="q0",
                question="这张图片是否是真实的？",
                why="规划阶段失败，使用默认问题",
                suggested_tools=["reverse_image_search", "text_search"],
                suggested_queries=[],
                priority=1,
            )],
            image_intent="(planning failed)",
            risk_assessment="uncertain",
        )

    async def _run_verification(
        self, state: VerificationState, image_path: str
    ) -> VerificationResult:
        """Stage 3: Gather evidence using tools."""
        t0 = time.time()

        tools = build_stage_tools("verification", self.all_tools)
        input_context = ContextRenderer.render_for_verification(
            state.perception, state.plan
        )

        # Extract mandatory tools from high-priority plan questions
        mandatory_tools = []
        if state.plan and state.plan.questions:
            for q in state.plan.questions:
                if q.priority == 1 and q.suggested_tools:
                    for tool in q.suggested_tools:
                        if tool not in mandatory_tools:
                            mandatory_tools.append(tool)

        runner = StageRunner(
            llm=self.llm,
            system_prompt=verification.SYSTEM_PROMPT,
            tools=tools,
            output_schema=VerificationResult,
            max_rounds=self.max_rounds_verification,
            image_path=image_path,
            stage_name="verification",
            recent_rounds_to_keep=2,
            mandatory_tools=mandatory_tools,
        )

        result, steps = await runner.run(input_context)

        state.stage_timings["verification"] = round(time.time() - t0, 2)
        state.all_steps.extend(steps)
        state.total_tool_calls += sum(1 for s in steps if s.action_type == "tool_call")
        state.llm_api_calls += runner.llm_api_calls
        self._accumulate_tokens(state, steps)

        if result and isinstance(result, VerificationResult):
            return result

        # Fallback: reconstruct from tool results
        return self._reconstruct_verification_from_steps(steps)

    def _reconstruct_verification_from_steps(self, steps: List) -> VerificationResult:
        """Reconstruct VerificationResult from tool call results."""
        import json as _json

        evidence = []
        key_findings = []

        for step in steps:
            if step.action_type != "tool_call" or not step.tool_result:
                continue
            try:
                data = _json.loads(step.tool_result) if isinstance(step.tool_result, str) else step.tool_result
            except (_json.JSONDecodeError, TypeError):
                continue

            # Normalize: text_search/news_search return a list of query results
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = data[0]

            if not isinstance(data, dict):
                continue

            tool_name = step.tool_name or ""
            # Summarize tool result as evidence
            if data.get("status") == "error":
                continue

            summary = ""
            if tool_name == "text_search":
                results = data.get("results", [])
                if results:
                    summary = f"搜索到 {len(results)} 条结果"
                    for r in results[:2]:
                        summary += f"; {r.get('title', '')}"
                else:
                    summary = "搜索无结果"
            elif tool_name == "news_search":
                results = data.get("results", [])
                summary = f"新闻搜索: {len(results)} 条" if results else "新闻搜索无结果"
            elif tool_name == "reverse_image_search":
                results = data.get("lens_results") or data.get("semantic_results") or data.get("results") or []
                if results:
                    summary = f"反向搜图: {len(results)} 条匹配"
                    for r in results[:2]:
                        summary += f"; {r.get('title', '')[:50]}"
                else:
                    summary = "反向搜图无结果"
            elif tool_name == "check_consistency":
                consistent = data.get("consistent", True)
                aspect = data.get("aspect_checked", "all")
                details = data.get("details", "")[:80]
                status = "一致" if consistent else "不一致"
                summary = f"一致性检查({aspect}): {status}. {details}"
            elif tool_name == "crop_and_inspect":
                answer = data.get("answer", "")[:80]
                anomalies = data.get("anomalies", [])
                if anomalies:
                    summary = f"裁剪检查发现异常: {anomalies}"
                elif answer:
                    summary = f"裁剪检查: {answer}"
                else:
                    summary = "裁剪检查无异常"
            elif tool_name == "analyze_visual_anomalies":
                auth = data.get("overall_authenticity", "uncertain")
                anomalies = data.get("anomalies", [])
                n = len(anomalies) if isinstance(anomalies, list) else 0
                summary = f"视觉异常分析: {auth}, {n} 个异常"
            elif tool_name == "visit":
                evidence = data.get("evidence", data.get("summary", ""))[:100]
                summary = f"网页访问: {evidence}" if evidence else "网页访问无结果"
            else:
                # Generic summary
                summary = f"{tool_name}: {str(data)[:100]}"

            if summary:
                key_findings.append(f"[{tool_name}] {summary}")

        if not key_findings:
            key_findings = ["Verification stage did not produce structured output"]

        return VerificationResult(
            key_findings=key_findings,
            authenticity_assessment="uncertain",
        )

    async def _run_judgment(
        self, state: VerificationState, image_path: str
    ) -> FinalJudgment:
        """Stage 4: Produce final verdict (single LLM call, no tools)."""
        t0 = time.time()

        # Build comprehensive context for judgment
        verification_result = state.verification or VerificationResult()
        input_context = ContextRenderer.render_for_judgment(
            state.perception, state.plan, verification_result
        )
        input_context += "\n\n请综合以上所有信息，给出最终判定。直接输出 <output>...</output>。"

        runner = StageRunner(
            llm=self.llm,
            system_prompt=judgment.SYSTEM_PROMPT,
            tools=[],  # No tools for judgment
            output_schema=FinalJudgment,
            max_rounds=1,
            image_path=image_path,
            stage_name="judgment",
        )

        result, steps = await runner.run(input_context)

        state.stage_timings["judgment"] = round(time.time() - t0, 2)
        state.all_steps.extend(steps)
        state.llm_api_calls += runner.llm_api_calls
        self._accumulate_tokens(state, steps)

        if result and isinstance(result, FinalJudgment):
            return result

        # Fallback judgment from whatever we have
        return self._build_fallback_judgment(state)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _accumulate_tokens(state: VerificationState, steps: list) -> None:
        """Sum up token usage from steps."""
        for step in steps:
            state.token_usage["prompt"] += step.tokens.get("prompt", 0)
            state.token_usage["completion"] += step.tokens.get("completion", 0)

    @staticmethod
    def _build_fallback_judgment(state: VerificationState) -> FinalJudgment:
        """Build a minimal judgment when the judgment stage fails."""
        # Try to infer from verification result
        assessment = "uncertain"
        if state.verification:
            assessment = state.verification.authenticity_assessment

        verdict_map = {
            "authentic": "real",
            "likely_ai": "fake",
            "likely_manipulated": "fake",
            "uncertain": "unverifiable",
        }

        return FinalJudgment(
            verdict=verdict_map.get(assessment, "unverifiable"),
            confidence=0.3,
            reasoning_chain="Judgment stage failed. Verdict inferred from verification assessment.",
            key_evidence=state.verification.key_findings if state.verification else [],
            overall_assessment=f"Judgment stage failed. Authenticity assessment: {assessment}",
        )
