# -*- coding: utf-8 -*-
"""Main 4-stage orchestrator for image factual verification."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.orchestrator.context import ContextRenderer
from src.orchestrator.ledger import (
    build_verification_case,
    compile_runtime_ledgers,
    derive_unverifiable_reasons,
    verify_case_image,
)
from src.orchestrator.investigation_state import (
    InvestigationReducer,
    InvestigationState,
    pending_visual_question_ids,
)
from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.stage_runner import StageRunner, StageStep
from src.orchestrator.source_provenance import canonicalize_url, classify_source
from src.orchestrator.stages import judgment, planning, verification
from src.orchestrator.state import (
    CoverageAudit,
    Entity,
    EvidenceItem,
    FinalJudgment,
    LedgerJudgment,
    InvestigationQuestion,
    PlanRevision,
    PerceptionReport,
    QuestionResolution,
    TextRegion,
    VerificationPlan,
    VerificationResult,
    VerificationState,
    VerificationCase,
    VerificationLedgers,
    VisualAnomaly,
)
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_health import require_tools, summarize_health
from src.orchestrator.tool_registry import (
    REQUIRED_TOOLS,
    build_all_tools_with_health,
    build_stage_tools,
)
from src.orchestrator.tool_result import parse_tool_result, serialize_tool_result
from src.storage import default_tool_cache_dir


class Orchestrator:
    """4-stage orchestrator with a single multi-round verification path."""

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.5-flash",
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        llm_wire_api: Optional[str] = None,
        vlm_wire_api: Optional[str] = None,
        max_rounds_verification: int = 8,
        timeout: float = 900.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        validate_startup: bool = True,
    ):
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.vlm_provider = (vlm_provider or self.provider).lower().strip()
        self.vlm_model = vlm_model or model_name
        self.llm_wire_api = llm_wire_api or os.getenv("AGENT_LLM_WIRE_API")
        if self.llm_wire_api is None and self.provider == "gemini":
            self.llm_wire_api = os.getenv("GEMINI_WIRE_API")
        self.vlm_wire_api = vlm_wire_api or os.getenv("VISION_LLM_WIRE_API")
        if self.vlm_wire_api is None and self.vlm_provider == "gemini":
            self.vlm_wire_api = (
                os.getenv("GEMINI_VISION_WIRE_API")
                or os.getenv("GEMINI_WIRE_API")
            )
        self.max_rounds_verification = max_rounds_verification
        self.max_verification_iterations = max(1, int(os.getenv("MAX_VERIFICATION_ITERATIONS", "2")))
        self.timeout = timeout
        cache_namespace = os.getenv("TOOL_CACHE_NAMESPACE", "").strip() or "|".join(
            [
                "tool-contract-v1",
                self.provider,
                model_name,
                self.vlm_provider,
                self.vlm_model,
                os.getenv("VISUAL_SEARCH_PROVIDER", "serper_lens"),
                os.getenv("IMAGE_UPLOAD_PROVIDER", "oss"),
                os.getenv("BROWSE_FETCH_PROVIDER", "jina"),
            ]
        )
        self.tool_cache = ToolResultCache(
            cache_dir=os.getenv("TOOL_CACHE_DIR", default_tool_cache_dir()),
            enabled=os.getenv("TOOL_CACHE_ENABLED", "0").strip().lower() in {"1", "true", "yes"},
            ttl_seconds=float(os.getenv("TOOL_CACHE_TTL_SECONDS", "3600")),
            namespace=cache_namespace,
        )
        self.cacheable_tools = {
            "perceive_scene",
            "ocr_with_position",
            "text_search",
            "visit",
            "crop_and_inspect",
            "check_consistency",
            "analyze_visual_anomalies",
            "count_objects",
        }
        self.verification_tool_limits = {
            "current_time": 1,
            "ocr_with_position": 3,
            "reverse_image_search": 2,
            "text_search": 4,
            "visit": 4,
            "compare_with_reference": 2,
            "crop_and_search": 3,
            "crop_and_inspect": 3,
            "check_consistency": 2,
            "analyze_visual_anomalies": 2,
            "count_objects": 2,
        }

        self.llm = APIBackend(
            provider=self.provider,
            model_name=model_name,
            wire_api=self.llm_wire_api,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.all_tools, self.tool_health = build_all_tools_with_health(
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            vlm_wire_api=self.vlm_wire_api,
        )
        self.tool_health_summary = summarize_health(self.tool_health)
        if validate_startup:
            require_tools(self.tool_health, REQUIRED_TOOLS)
            self._validate_startup_configuration()
        now = datetime.now().astimezone()
        self.date_prefix = (
            f"Current date: {now.date().isoformat()} ({now.tzinfo}). "
            "Use this runtime date for time-sensitive judgments instead of model memory.\n\n"
        )

    def _validate_startup_configuration(self) -> None:
        if self.provider == "gemini" and not self.llm.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini agent."
            )
        if self.vlm_provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ):
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini perception."
            )
        if not (
            os.getenv("SERPER_API_KEY", "").strip()
            or os.getenv("SERPER_KEY_ID", "").strip()
        ):
            raise RuntimeError(
                "SERPER_API_KEY is required for text, image, and Lens search."
            )

        browse_extract_provider = os.getenv(
            "BROWSE_EXTRACT_PROVIDER", "gemini"
        ).strip().lower()
        if browse_extract_provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ):
            raise RuntimeError(
                "Gemini credentials are required for BROWSE_EXTRACT_PROVIDER=gemini."
            )

        reverse_tool = self.all_tools.get("reverse_image_search")
        visual_client = getattr(reverse_tool, "visual_search_client", None)
        upload_client = getattr(visual_client, "upload_client", None)
        if upload_client is not None:
            upload_client.validate_configuration()
        visual_provider = str(getattr(visual_client, "provider", "serper_lens"))
        if visual_provider == "zhipu_image_search" and not os.getenv(
            "ZHIPU_API_KEY", ""
        ).strip():
            raise RuntimeError(
                "ZHIPU_API_KEY is required for VISUAL_SEARCH_PROVIDER=zhipu_image_search."
            )

    def _sp(self, stage_system_prompt: str) -> str:
        return self.date_prefix + stage_system_prompt

    async def run(
        self,
        image_path: str,
        image_id: str = "",
        verification_case: Optional[VerificationCase] = None,
        user_claim: Optional[str] = None,
    ) -> Dict[str, Any]:
        runtime_case = verification_case or build_verification_case(
            image_path,
            case_id=image_id or os.path.basename(image_path) or image_path,
            user_claim=user_claim,
        )
        verify_case_image(runtime_case, image_path)
        state = VerificationState(
            image_path=image_path,
            image_id=image_id or os.path.basename(image_path) or image_path,
            verification_case=runtime_case,
            tool_health=self.tool_health_summary,
        )
        state.investigation_state = InvestigationState()
        self.last_state = state
        started = time.time()
        try:
            state.perception = await self._run_perception(state, image_path)
            if (
                runtime_case.claim_mode.value == "embedded_claim"
                and not runtime_case.claim_surface
            ):
                claim_surface = " ".join(
                    region.text.strip()
                    for region in state.perception.text_regions
                    if region.text.strip()
                )
                runtime_case = runtime_case.model_copy(
                    update={"claim_surface": claim_surface[:4000] or None}
                )
                state.verification_case = runtime_case
            self._check_timeout(started, state)
            state.plan = await self._run_planning(state, image_path)
            state.plan_history.append(state.plan.model_copy(deep=True))
            self._check_timeout(started, state)
            state.verification = await self._run_verification(state, image_path)
            state.ledgers = compile_runtime_ledgers(
                runtime_case,
                state.plan or VerificationPlan(),
                state.verification,
                state.all_steps,
            )
            self._check_timeout(started, state)
            state.judgment = await self._run_judgment(state, image_path)
            if state.judgment is None:
                raise RuntimeError("Judgment stage completed without a validated judgment.")
            state.termination = "success"
        except Exception as exc:
            state.termination = "error"
            state.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            state.stage_timings["total"] = round(time.time() - started, 2)

        return {
            "image_id": state.image_id,
            "image_path": state.image_path,
            "judgment": state.judgment.model_dump(),
            "verdict": state.judgment.verdict,
            "confidence": state.judgment.confidence,
            "overall_assessment": state.judgment.overall_assessment,
            "state": state.to_dict(),
            "termination": state.termination,
            "time_taken": state.stage_timings["total"],
            "token_usage": state.token_usage,
            "total_tool_calls": state.total_tool_calls,
            "llm_api_calls": state.llm_api_calls,
            "error": " | ".join(state.errors) if state.errors else None,
        }

    async def _run_perception(self, state: VerificationState, image_path: str) -> PerceptionReport:
        started = time.time()
        steps: List[StageStep] = []
        report = PerceptionReport(scene_description="")

        if "perceive_scene" not in self.all_tools:
            raise RuntimeError(
                "Required perception tool 'perceive_scene' is unavailable: "
                + str(self.tool_health_summary.get("perceive_scene", {}).get("error", "not registered"))
            )

        try:
            tool_result, metadata = await self._execute_tool(
                "perceive_scene",
                {"image_input": image_path},
                image_path,
            )
            step = StageStep(
                round=1,
                stage_name="perception",
                action_type="tool_call",
                tool_name="perceive_scene",
                tool_args={"image_input": image_path},
                tool_result=tool_result,
                metadata={"stage": "perception", **metadata},
            )
            steps.append(step)
            if not self._tool_step_succeeded(step):
                raise RuntimeError(f"perceive_scene failed: {tool_result[:1000]}")
            report = self._parse_perception_result(tool_result)
            if not report.scene_description and not report.entities:
                raise RuntimeError("perceive_scene returned no scene description or visible entities")

            if "ocr_with_position" not in self.all_tools:
                raise RuntimeError(
                    "Required perception tool 'ocr_with_position' is unavailable: "
                    + str(
                        self.tool_health_summary.get("ocr_with_position", {}).get(
                            "error", "not registered"
                        )
                    )
                )
            tool_result, metadata = await self._execute_tool(
                "ocr_with_position",
                {"image_input": image_path},
                image_path,
            )
            step = StageStep(
                round=2,
                stage_name="perception",
                action_type="tool_call",
                tool_name="ocr_with_position",
                tool_args={"image_input": image_path},
                tool_result=tool_result,
                metadata={"stage": "perception", **metadata},
            )
            steps.append(step)
            if not self._tool_step_succeeded(step):
                raise RuntimeError(f"ocr_with_position failed: {tool_result[:1000]}")
            report = self._merge_ocr(report, tool_result)
        finally:
            state.stage_timings["perception"] = round(time.time() - started, 2)
            state.all_steps.extend(steps)
            state.total_tool_calls += len(steps)
        return report

    async def _run_planning(self, state: VerificationState, image_path: str) -> VerificationPlan:
        started = time.time()
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(planning.SYSTEM_PROMPT),
            tools=[],
            output_schema=VerificationPlan,
            max_rounds=planning.MAX_ROUNDS,
            image_path=image_path,
            stage_name=planning.STAGE_NAME,
            recent_rounds_to_keep=1,
            output_validator=lambda parsed, steps: self._validate_plan_output(parsed),
            attach_image=False,
            max_output_tokens=self._stage_output_tokens("PLANNING", 8192),
        )
        context = ContextRenderer.render_for_planning(
            state.perception or PerceptionReport(),
            state.verification_case,
        )
        parsed, steps = await runner.run(context)
        self._record_stage_steps(state, steps)
        state.stage_timings["planning"] = round(time.time() - started, 2)
        if parsed is None:
            raise RuntimeError("Planning stage did not produce valid structured output.")
        return parsed

    async def _run_verification(self, state: VerificationState, image_path: str) -> VerificationResult:
        started = time.time()
        all_verification_steps: List[StageStep] = []
        parsed_results: List[VerificationResult] = []
        result = VerificationResult()

        for iteration in range(1, self.max_verification_iterations + 1):
            active_reinspect = self.provider == "gemini" and str(self.llm.wire_api).lower() == "interactions"

            def observation_callback(
                step: StageStep,
                cumulative_steps: List[StageStep],
            ) -> Optional[Dict[str, Any]]:
                previous_ledgers = state.ledgers.model_copy(deep=True)
                derived_observation = self._build_verification_result_from_steps(
                    cumulative_steps,
                    state,
                )
                current_ledgers = compile_runtime_ledgers(
                    state.verification_case or build_verification_case(image_path),
                    state.plan or VerificationPlan(),
                    derived_observation,
                    cumulative_steps,
                )
                update = InvestigationReducer.reduce(
                    state.investigation_state,
                    step=step,
                    previous_ledgers=previous_ledgers,
                    current_ledgers=current_ledgers,
                    plan=state.plan or VerificationPlan(),
                    perception=state.perception or PerceptionReport(),
                )
                self._gate_claims_on_pending_visual_questions(
                    current_ledgers,
                    state.investigation_state,
                )
                state.ledgers = current_ledgers
                return update

            runner = StageRunner(
                llm=self.llm,
                system_prompt=self._sp(verification.SYSTEM_PROMPT),
                tools=build_stage_tools("verification", self.all_tools),
                output_schema=VerificationResult,
                max_rounds=self.max_rounds_verification,
                image_path=image_path,
                stage_name=verification.STAGE_NAME,
                recent_rounds_to_keep=3,
                tool_cache=self.tool_cache,
                cacheable_tools=list(self.cacheable_tools),
                tool_call_limits=self.verification_tool_limits,
                should_stop=None,
                output_validator=lambda parsed, steps, plan=state.plan: self._validate_verification_output(
                    parsed,
                    list(all_verification_steps) + list(steps),
                    plan or VerificationPlan(),
                    investigation_state=(state.investigation_state if active_reinspect else None),
                    allow_incomplete_at_budget=(iteration >= self.max_verification_iterations),
                ),
                min_tool_calls=1,
                attach_image=False,
                prior_steps=list(all_verification_steps),
                observation_callback=(observation_callback if active_reinspect else None),
                visual_call_validator=(
                    lambda tool_name, args: InvestigationReducer.validate_visual_call(
                        state.investigation_state,
                        tool_name,
                        args,
                    )
                    if active_reinspect
                    else ""
                ),
                max_output_tokens=self._stage_output_tokens("VERIFICATION", 8192),
            )
            context = ContextRenderer.render_for_verification(
                state.perception or PerceptionReport(),
                state.plan or VerificationPlan(),
            )
            if all_verification_steps:
                context += "\n\n## Evidence retained from earlier iterations\n"
                context += ContextRenderer.render_for_judgment(
                    state.perception or PerceptionReport(),
                    state.plan or VerificationPlan(),
                    result,
                )
            parsed, iteration_steps = await runner.run(context)
            for step in iteration_steps:
                step.metadata["verification_iteration"] = iteration
            self._record_stage_steps(state, iteration_steps)
            all_verification_steps.extend(iteration_steps)
            if isinstance(parsed, VerificationResult):
                parsed_results.append(parsed)

            derived = self._build_verification_result_from_steps(all_verification_steps, state)
            result = self._merge_verification_results(
                derived,
                parsed_results,
                all_verification_steps,
                state.plan or VerificationPlan(),
            )
            state.ledgers = compile_runtime_ledgers(
                state.verification_case or build_verification_case(image_path),
                state.plan or VerificationPlan(),
                result,
                all_verification_steps,
            )
            self._gate_claims_on_pending_visual_questions(
                state.ledgers,
                state.investigation_state,
            )
            audit = self._audit_plan_coverage(
                state.plan or VerificationPlan(),
                all_verification_steps,
                result,
                iteration=iteration,
                ledgers=state.ledgers,
            )
            state.coverage_audits.append(audit)
            result.question_resolutions = audit.question_resolutions
            result.coverage_complete = audit.complete
            result.unresolved_priority_questions = audit.unresolved_priority_questions
            result.exhausted_priority_questions = audit.exhausted_priority_questions
            result.iteration_count = iteration
            if audit.complete:
                break
            if iteration < self.max_verification_iterations:
                state.plan = await self._replan_verification(state, result, audit, image_path)
                state.plan_history.append(state.plan.model_copy(deep=True))

        state.stage_timings["verification"] = round(time.time() - started, 2)
        successful_steps = [
            step for step in all_verification_steps if self._tool_step_succeeded(step)
        ]
        if not successful_steps:
            failures = self._summarize_tool_failures(all_verification_steps)
            raise RuntimeError(
                "Verification failed: every attempted tool call failed."
                + (f" Failures: {failures}" if failures else "")
            )
        if not parsed_results:
            raise RuntimeError(
                "Verification failed: the agent did not produce any accepted structured result."
            )
        if not state.coverage_audits or not state.coverage_audits[-1].investigation_complete:
            unresolved = (
                state.coverage_audits[-1].unresolved_priority_questions
                if state.coverage_audits
                else []
            )
            raise RuntimeError(
                "Verification failed: investigation did not reach a valid stopping state"
                + (f" ({', '.join(unresolved)})." if unresolved else ".")
            )
        return result

    @staticmethod
    def _gate_claims_on_pending_visual_questions(
        ledgers: VerificationLedgers,
        investigation_state: Optional[InvestigationState],
    ) -> None:
        if investigation_state is None:
            return
        pending_claim_ids = {
            item.claim_id
            for item in investigation_state.visual_questions
            if item.status == "pending"
        }
        for claim in ledgers.claims:
            if claim.claim_id in pending_claim_ids:
                claim.status = "open"
                claim.unresolved_distinction = (
                    "A search-conditioned visual question still requires a real image observation."
                )

    async def _run_judgment(self, state: VerificationState, image_path: str) -> FinalJudgment:
        started = time.time()
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(judgment.SYSTEM_PROMPT),
            tools=[],
            output_schema=LedgerJudgment,
            max_rounds=judgment.MAX_ROUNDS,
            image_path=image_path,
            stage_name=judgment.STAGE_NAME,
            recent_rounds_to_keep=1,
            output_validator=lambda parsed, steps: self._validate_judgment_output(
                parsed,
                state.verification or VerificationResult(),
                state.ledgers,
            ),
            attach_image=False,
            max_output_tokens=self._stage_output_tokens("JUDGMENT", 8192),
            generation_config={
                "thinking_level": os.getenv(
                    "GEMINI_JUDGMENT_THINKING_LEVEL", "minimal"
                ).strip().lower()
            },
        )
        context = ContextRenderer.render_for_judgment(
            state.perception or PerceptionReport(),
            state.plan or VerificationPlan(),
            state.verification or VerificationResult(),
            state.ledgers,
        )
        decisive_statuses = {
            item.status
            for item in state.ledgers.claims
            if item.criticality == "decisive"
        }
        expected_verdict = (
            "fake"
            if "refuted" in decisive_statuses
            else (
                "real"
                if decisive_statuses and decisive_statuses <= {"supported"}
                else "unverifiable"
            )
        )
        expected_reasons = [
            item.value for item in derive_unverifiable_reasons(state.ledgers)
        ]
        context += (
            "\n\nDeterministic policy output to copy exactly:\n"
            f"- verdict: {expected_verdict}\n"
            "- unverifiable_reasons: "
            + json.dumps(expected_reasons, ensure_ascii=False)
        )
        parsed, steps = await runner.run(context)
        self._record_stage_steps(state, steps)
        state.stage_timings["judgment"] = round(time.time() - started, 2)
        if parsed is None:
            raise RuntimeError("Judgment stage did not produce valid structured output.")
        return self._compile_judgment_from_ledgers(parsed, state.ledgers)

    async def _replan_verification(
        self,
        state: VerificationState,
        result: VerificationResult,
        audit: CoverageAudit,
        image_path: str,
    ) -> VerificationPlan:
        current_plan = state.plan or VerificationPlan()
        prompt = self._sp(planning.REPLANNING_SYSTEM_PROMPT)
        runner = StageRunner(
            llm=self.llm,
            system_prompt=prompt,
            tools=[],
            output_schema=PlanRevision,
            max_rounds=1,
            image_path=image_path,
            stage_name="replanning",
            recent_rounds_to_keep=1,
            output_validator=lambda parsed, steps: self._validate_plan_revision(
                parsed,
                current_plan,
                audit,
            ),
            attach_image=False,
            max_output_tokens=self._stage_output_tokens("REPLANNING", 8192),
        )
        context = ContextRenderer.render_for_replanning(
            state.perception or PerceptionReport(),
            current_plan,
            audit,
            result,
        )
        revised, steps = await runner.run(context)
        self._record_stage_steps(state, steps)
        if not isinstance(revised, PlanRevision):
            raise RuntimeError("Replanning did not produce a valid plan revision delta.")
        return self._apply_plan_revision(current_plan, revised, audit)

    def _validate_verification_output(
        self,
        parsed: VerificationResult,
        steps: List[StageStep],
        plan: VerificationPlan,
        investigation_state: Optional[InvestigationState] = None,
        allow_incomplete_at_budget: bool = False,
    ) -> tuple[bool, str]:
        successful_steps = [step for step in steps if self._tool_step_succeeded(step)]
        if not successful_steps:
            return False, "no successful tool result was collected"

        questions = {question.question_id: question for question in plan.questions}
        evidence = []
        for item in parsed.evidence:
            canonical = self._canonicalize_model_evidence(item, steps, plan)
            question = questions.get(item.related_question)
            if canonical is not None and question is not None and self._evidence_answers_question(
                canonical,
                question,
            ):
                evidence.append(canonical)
        ungrounded_anomalies = [
            anomaly
            for anomaly in parsed.visual_anomalies
            if not self._visual_anomaly_is_grounded(anomaly, steps)
        ]
        if ungrounded_anomalies:
            return False, "visual anomalies must copy one successful anomaly tool result and function_call_id"
        priority_ids = {q.question_id for q in plan.questions if q.priority == 1 and q.question_id}
        covered_ids = {item.related_question for item in evidence if item.related_question}
        exhausted_ids = {
            question_id for question_id in priority_ids
            if self._question_attempts_exhausted(question_id, steps)
        }
        missing = sorted(priority_ids - covered_ids)
        if missing and not allow_incomplete_at_budget:
            return False, f"priority questions lack grounded evidence: {', '.join(missing)}"
        if missing and parsed.authenticity_assessment != "uncertain":
            return False, "incomplete decisive coverage requires an uncertain verification assessment"
        if parsed.authenticity_assessment != "uncertain" and not evidence:
            return False, "a non-uncertain assessment requires grounded evidence"
        pending = pending_visual_question_ids(investigation_state) if investigation_state else []
        if pending and not allow_incomplete_at_budget:
            return False, (
                "pending search-driven visual questions require a real regional OCR, crop, "
                "count, or reference-comparison observation: " + ", ".join(pending)
            )
        return True, ""

    @staticmethod
    def _validate_plan_output(parsed: VerificationPlan) -> tuple[bool, str]:
        if not parsed.questions:
            return False, "the plan must contain at least one investigation question"
        if not any(question.priority == 1 for question in parsed.questions):
            return False, "the plan must contain at least one priority-1 question"
        ids = [question.question_id for question in parsed.questions]
        if any(not question_id for question_id in ids) or len(set(ids)) != len(ids):
            return False, "all investigation questions need unique non-empty question_id values"
        return True, ""

    @staticmethod
    def _validate_plan_revision(
        parsed: PlanRevision,
        current: VerificationPlan,
        audit: CoverageAudit,
    ) -> tuple[bool, str]:
        unresolved_ids = set(audit.unresolved_priority_questions)
        current_ids = {question.question_id for question in current.questions}
        update_ids = [question.question_id for question in parsed.question_updates]
        if not unresolved_ids:
            return False, "replanning requires at least one unresolved priority question"
        if any(not question_id for question_id in update_ids):
            return False, "every question update needs a non-empty question_id"
        if len(set(update_ids)) != len(update_ids):
            return False, "question update ids must be unique"
        unknown = sorted(set(update_ids) - current_ids)
        if unknown:
            return False, "question updates must use existing ids: " + ", ".join(unknown)
        resolved = sorted(set(update_ids) - unresolved_ids)
        if resolved:
            return False, "resolved or exhausted questions cannot be rewritten: " + ", ".join(resolved)
        missing = sorted(unresolved_ids - set(update_ids))
        if missing:
            return False, "all unresolved priority questions need an update: " + ", ".join(missing)
        if any(not question.question.strip() for question in parsed.question_updates):
            return False, "every question update needs a concrete question"
        if any(not question.suggested_tools for question in parsed.question_updates):
            return False, "every question update needs at least one suggested tool"
        return True, ""

    def _validate_judgment_output(
        self,
        parsed: FinalJudgment,
        verification_result: VerificationResult,
        ledgers: Optional[VerificationLedgers] = None,
    ) -> tuple[bool, str]:
        if ledgers is not None and ledgers.claims:
            return self._validate_ledger_judgment(parsed, ledgers)
        if parsed.verdict == "real" and not verification_result.coverage_complete:
            return False, "a real verdict is not allowed while priority questions remain unresolved"
        if parsed.verdict == "real" and any(
            item.status == "exhausted" for item in verification_result.question_resolutions
        ):
            return False, "a real verdict is not allowed when a priority question was exhausted without answer-bearing evidence"
        if parsed.verdict == "real" and not any(
            item.direction == "supports" and item.quality in {"strong", "moderate"}
            for item in verification_result.evidence
        ):
            return False, "a real verdict requires supporting grounded evidence"
        if parsed.verdict == "fake" and not any(
            item.direction == "refutes" for item in verification_result.evidence
        ) and not verification_result.visual_anomalies:
            return False, "a fake verdict requires refuting evidence or a concrete visual anomaly"
        return True, ""

    @staticmethod
    def _validate_ledger_judgment(
        parsed: LedgerJudgment,
        ledgers: VerificationLedgers,
    ) -> tuple[bool, str]:
        claims = {item.claim_id: item for item in ledgers.claims}
        evidence = {item.evidence_id: item for item in ledgers.evidence}
        decisive = {item.claim_id: item for item in ledgers.claims if item.criticality == "decisive"}
        if parsed.policy_rule_id != "reinspect-v1":
            return False, "judgment must use policy_rule_id=reinspect-v1"
        decision_ids = [item.claim_id for item in parsed.claim_decisions]
        if set(decision_ids) != set(decisive) or len(decision_ids) != len(set(decision_ids)):
            return False, "judgment needs exactly one decision for every decisive claim id"
        selected = set(parsed.selected_evidence_ids)
        if selected - set(evidence):
            return False, "judgment selected an unknown evidence id"
        decision_selected: set[str] = set()
        for decision in parsed.claim_decisions:
            claim = claims.get(decision.claim_id)
            if claim is None:
                return False, f"unknown claim id: {decision.claim_id}"
            for evidence_id in decision.evidence_ids:
                record = evidence.get(evidence_id)
                if record is None:
                    return False, f"unknown evidence id: {evidence_id}"
                if record.claim_id != decision.claim_id:
                    return False, "evidence id does not belong to the claim decision"
                if record.stance == "neutral":
                    return False, "neutral evidence cannot decide a claim"
                decision_selected.add(evidence_id)
            expected = {
                "supported": "support",
                "refuted": "refute",
            }.get(claim.status, "unresolved")
            if decision.decision != expected:
                return False, f"decision for {claim.claim_id} conflicts with ledger status {claim.status}"
            if expected in {"support", "refute"}:
                if not decision.evidence_ids:
                    return False, "a decisive support/refute decision requires evidence ids"
                if not all(evidence[item].stance == expected for item in decision.evidence_ids):
                    return False, "decision evidence stance conflicts with the claim decision"
            elif decision.evidence_ids:
                return False, "an unresolved claim cannot cite decisive evidence ids"
        if selected != decision_selected:
            return False, "selected_evidence_ids must equal the ids used by claim decisions"

        statuses = {item.status for item in decisive.values()}
        expected_verdict = (
            "fake"
            if "refuted" in statuses
            else ("real" if statuses and statuses <= {"supported"} else "unverifiable")
        )
        if parsed.verdict != expected_verdict:
            return False, f"verdict must be {expected_verdict} under the deterministic claim policy"
        if expected_verdict == "unverifiable" and not parsed.unverifiable_reasons:
            return False, "unverifiable verdict requires typed reasons"
        if expected_verdict != "unverifiable" and parsed.unverifiable_reasons:
            return False, "real/fake verdict cannot carry unverifiable reasons"
        if expected_verdict == "unverifiable":
            expected_reasons = set(derive_unverifiable_reasons(ledgers))
            if set(parsed.unverifiable_reasons) != expected_reasons:
                return False, "unverifiable reasons must exactly match the deterministic ledger policy"
        return True, ""

    @staticmethod
    def _compile_judgment_from_ledgers(
        parsed: LedgerJudgment,
        ledgers: VerificationLedgers,
    ) -> FinalJudgment:
        evidence = {item.evidence_id: item for item in ledgers.evidence}
        claims = {item.claim_id: item for item in ledgers.claims}
        selected = [evidence[item] for item in parsed.selected_evidence_ids if item in evidence]
        key_evidence = [item.exact_text for item in selected]
        decision_rows = []
        for item in parsed.claim_decisions:
            claim = claims[item.claim_id]
            decision_rows.append(f"{claim.text}: {item.decision}.")
        reasoning = " ".join(decision_rows)
        if parsed.verdict == "unverifiable":
            reasons = ", ".join(item.value for item in parsed.unverifiable_reasons)
            assessment = f"Unverifiable under policy reinspect-v1: {reasons}."
        else:
            assessment = (
                f"Verdict {parsed.verdict} under policy reinspect-v1 using "
                f"{len(selected)} validated evidence record(s)."
            )
        return FinalJudgment(
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            reasoning_chain=reasoning,
            key_evidence=key_evidence,
            anomalies=[],
            overall_assessment=assessment,
        )

    @staticmethod
    def _stage_output_tokens(stage_name: str, default: int) -> int:
        value = os.getenv(f"GEMINI_{stage_name}_MAX_OUTPUT_TOKENS", str(default)).strip()
        try:
            tokens = int(value)
        except ValueError as exc:
            raise ValueError(
                f"GEMINI_{stage_name}_MAX_OUTPUT_TOKENS must be an integer."
            ) from exc
        if tokens < 1:
            raise ValueError(
                f"GEMINI_{stage_name}_MAX_OUTPUT_TOKENS must be positive."
            )
        return tokens

    def _audit_plan_coverage(
        self,
        plan: VerificationPlan,
        steps: List[StageStep],
        result: VerificationResult,
        *,
        iteration: int,
        ledgers: Optional[VerificationLedgers] = None,
    ) -> CoverageAudit:
        successful_steps = [step for step in steps if self._tool_step_succeeded(step)]
        distinct_tools = sorted({step.tool_name for step in successful_steps if step.tool_name})
        resolutions: List[QuestionResolution] = []
        unresolved_priority: List[str] = []
        exhausted_priority: List[str] = []
        claim_statuses = {
            claim.question_id: claim.status
            for claim in (ledgers.claims if ledgers is not None else [])
            if claim.question_id
        }

        for question in plan.questions:
            question_steps = [
                step for step in steps
                if step.action_type == "tool_call"
                and str(step.tool_args.get("__question_id", "")) == question.question_id
            ]
            successful_question_steps = [step for step in question_steps if self._tool_step_succeeded(step)]
            evidence = [item for item in result.evidence if item.related_question == question.question_id]
            grounded = [
                item for item in evidence
                if self._evidence_item_is_grounded(item, steps)
                and self._evidence_answers_question(item, question)
            ]

            claim_status = claim_statuses.get(question.question_id, "open")
            if claim_status in {"supported", "refuted"}:
                status = "resolved"
                conclusion = f"Decisive claim was {claim_status} by validated ledger evidence."
                gap = ""
            elif claim_status == "conflicted":
                status = "in_progress"
                conclusion = ""
                gap = "Direct source evidence conflicts across the claim slot."
            elif question_steps and not successful_question_steps:
                status = "in_progress"
                conclusion = ""
                gap = "All attempted tools failed; use another query or tool."
            elif self._question_attempts_exhausted(question.question_id, steps):
                status = "exhausted"
                conclusion = "Reasonable investigation did not produce answer-bearing evidence."
                gap = ""
            elif successful_question_steps:
                status = "in_progress"
                conclusion = ""
                gap = "Tool results did not yield grounded evidence that answers the question."
            else:
                status = "unanswered"
                conclusion = ""
                gap = "No tool call has targeted this question."

            if question.priority == 1 and status != "resolved":
                unresolved_priority.append(question.question_id)
            if question.priority == 1 and status == "exhausted":
                exhausted_priority.append(question.question_id)
            resolutions.append(
                QuestionResolution(
                    question_id=question.question_id,
                    status=status,
                    conclusion=conclusion,
                    remaining_gap=gap,
                    tool_attempts=len(question_steps),
                    evidence_count=len(grounded),
                )
            )

        complete = bool(plan.questions) and not unresolved_priority
        investigation_complete = complete or iteration >= self.max_verification_iterations
        if not plan.questions:
            complete = False
        if complete:
            reason = "All priority-1 investigation questions have grounded evidence."
        elif investigation_complete:
            reason = (
                "Investigation budget completed with unresolved decisive claim slots: "
                + ", ".join(unresolved_priority or ["plan_has_no_questions"])
            )
        else:
            reason = "Unresolved priority questions: " + ", ".join(unresolved_priority or ["plan_has_no_questions"])
        return CoverageAudit(
            iteration=iteration,
            complete=complete,
            investigation_complete=investigation_complete,
            question_resolutions=resolutions,
            unresolved_priority_questions=unresolved_priority,
            exhausted_priority_questions=exhausted_priority,
            successful_tool_calls=len(successful_steps),
            distinct_tools=distinct_tools,
            evidence_count=sum(item.evidence_count for item in resolutions),
            reason=reason,
            claim_statuses=claim_statuses,
            unverifiable_reasons=(
                derive_unverifiable_reasons(ledgers)
                if ledgers is not None and investigation_complete and not complete
                else []
            ),
        )

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any], image_path: str) -> tuple[str, Dict[str, Any]]:
        tool = self.all_tools[tool_name]
        tool_args = dict(args)
        properties = tool.parameters.get("properties", {})
        if "image_input" in properties and not tool_args.get("image_input"):
            tool_args["image_input"] = image_path
        if hasattr(tool, "image_path") and image_path:
            tool.image_path = image_path

        cache_args = self._build_cache_args(tool_name, tool_args, image_path=image_path)
        started = time.perf_counter()
        if tool_name in self.cacheable_tools:
            cached = self.tool_cache.get(tool_name, cache_args)
            if cached is not None:
                _, succeeded = parse_tool_result(cached)
                return cached, {
                    "cache_hit": True,
                    "tool_success": succeeded,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "serialized_size": len(cached),
                }

        try:
            if hasattr(tool, "call_async"):
                result = await tool.call_async(tool_args)
            else:
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, tool_args)
        except Exception as exc:
            raise RuntimeError(f"{tool_name} failed: {type(exc).__name__}: {exc}") from exc

        serialized, succeeded = serialize_tool_result(result)
        if succeeded and tool_name in self.cacheable_tools:
            self.tool_cache.put(tool_name, cache_args, serialized)
        return serialized, {
            "cache_hit": False,
            "tool_success": succeeded,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "serialized_size": len(serialized),
        }

    @staticmethod
    def _build_cache_args(tool_name: str, tool_args: Dict[str, Any], *, image_path: str) -> Dict[str, Any]:
        cache_args = dict(tool_args)
        if tool_name in {"compare_with_reference", "analyze_visual_anomalies"} and image_path:
            cache_args["__image_input__"] = image_path
        return cache_args

    def _parse_perception_result(self, tool_result: str) -> PerceptionReport:
        data = self._safe_json_dict(tool_result)
        entities: List[Entity] = []
        for item in data.get("entities", []) or []:
            if not isinstance(item, dict):
                continue
            entities.append(
                Entity(
                    name=str(item.get("name", "")).strip(),
                    entity_type=str(item.get("entity_type", "")).strip(),
                    bbox=item.get("bbox", []) if isinstance(item.get("bbox"), list) else [],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    attributes=item.get("attributes", {}) if isinstance(item.get("attributes"), dict) else {},
                )
            )
        return PerceptionReport(
            entities=entities,
            text_regions=[],
            scene_description=str(data.get("scene_description", "")).strip(),
            image_type=str(data.get("image_type", "photo") or "photo").strip(),
        )

    def _merge_ocr(self, report: PerceptionReport, tool_result: str) -> PerceptionReport:
        data = self._safe_json_dict(tool_result)
        text_regions = list(report.text_regions)
        seen = {self._text_region_key(region) for region in text_regions}
        for item in data.get("text_regions", []) or []:
            normalized = self._normalize_text_region(item)
            if normalized is None:
                continue
            key = self._text_region_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            text_regions.append(normalized)
        return PerceptionReport(
            entities=report.entities,
            text_regions=text_regions,
            scene_description=report.scene_description,
            image_type=report.image_type,
        )

    @staticmethod
    def _text_region_key(region: TextRegion) -> str:
        return json.dumps({"text": region.text, "bbox_quad": region.bbox_quad}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _normalize_text_region(item: Any) -> Optional[TextRegion]:
        if not isinstance(item, dict):
            return None
        bbox_quad = item.get("bbox_quad", [])
        if not bbox_quad and isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4:
            x1, y1, x2, y2 = [float(value) for value in item["bbox"]]
            bbox_quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if not isinstance(bbox_quad, list):
            bbox_quad = []
        return TextRegion(
            text=str(item.get("text", "")).strip(),
            bbox_quad=bbox_quad,
            confidence=float(item.get("confidence", 0.0) or 0.0),
            language=str(item.get("language", "unknown") or "unknown"),
        )

    def _record_stage_steps(self, state: VerificationState, steps: Sequence[StageStep]) -> None:
        if not steps:
            return
        state.all_steps.extend(steps)
        state.total_tool_calls += sum(1 for step in steps if step.action_type == "tool_call")
        for step in steps:
            state.token_usage["prompt"] += step.tokens.get("prompt", 0)
            state.token_usage["completion"] += step.tokens.get("completion", 0)
        state.llm_api_calls += sum(
            1 for step in steps if step.metadata.get("llm_duration_ms") is not None
        )

    def _build_verification_result_from_steps(self, steps: List[StageStep], state: VerificationState) -> VerificationResult:
        evidence: List[EvidenceItem] = []
        key_findings: List[str] = []
        visual_anomalies: List[VisualAnomaly] = []
        visual_evidence: List[Dict[str, Any]] = []
        source_findings: List[Dict[str, Any]] = []
        assessment = "uncertain"
        current_date_anchor = ""

        for step in steps:
            if step.action_type != "tool_call":
                continue
            if not self._tool_step_succeeded(step):
                key_findings.append(f"[{step.tool_name}] Tool call failed and was excluded from evidence.")
                continue
            data = self._safe_json_dict(step.tool_result)

            if step.tool_name == "current_time":
                current_date_anchor = str(data.get("current_date", "")).strip()
                summary = f"Runtime date anchor: {current_date_anchor}"
                current_excerpt = str(data.get("current_datetime", "")).strip()
                if not current_excerpt:
                    key_findings.append("[current_time] Tool result lacked an exact datetime excerpt.")
                    continue
                evidence.append(
                    EvidenceItem(
                        function_call_id=self._step_function_call_id(step),
                        source="current_time",
                        summary=summary,
                        raw_excerpt=current_excerpt[:300],
                        direction="neutral",
                        quality="strong",
                        tool_used="current_time",
                        related_question=self._step_question_id(step),
                    )
                )
                key_findings.append(f"[current_time] {summary}")
                continue

            if step.tool_name in {"text_search", "visit", "reverse_image_search", "crop_and_search"}:
                summary, excerpt, url = self._summarize_external_evidence(step.tool_name, data, step.tool_result)
                evidence_eligible = step.tool_name in {"text_search", "visit", "crop_and_search"}
                if summary and evidence_eligible and excerpt and url:
                    provenance = self._find_browse_evidence_record(data, excerpt)
                    if self._browse_record_is_evidence_eligible(provenance, url):
                        stance = str(provenance.get("stance", "")).strip().lower()
                        direction = {"support": "supports", "refute": "refutes"}.get(
                            stance,
                            "neutral",
                        )
                        relevance = str(provenance.get("relevance", "")).strip().lower()
                        trusted = self._is_probably_trusted_url(url)
                        quality = (
                            "strong"
                            if trusted and relevance == "high"
                            else ("moderate" if relevance in {"high", "medium"} else "weak")
                        )
                        evidence.append(
                            EvidenceItem(
                                function_call_id=self._step_function_call_id(step),
                                source=url,
                                summary=excerpt[:300],
                                raw_excerpt=excerpt[:300],
                                direction=direction,
                                quality=quality,
                                tool_used=step.tool_name,
                                related_question=self._step_question_id(step),
                            )
                        )
                        key_findings.append(f"[{step.tool_name}] {summary}")
                        source_findings.append(
                            {
                                "round": step.round,
                                "function_call_id": self._step_function_call_id(step),
                                "tool_name": step.tool_name,
                                "finding_text": excerpt,
                                "source": url,
                                "artifact_sha256": provenance.get("artifact_sha256", ""),
                                "evidence_span": provenance.get("evidence_span", {}),
                                "retrieved_at": provenance.get("retrieved_at", ""),
                                "injection_flags": provenance.get("injection_flags", []),
                                "source_family": classify_source(
                                    url,
                                    injection_flags=provenance.get("injection_flags", []),
                                ).source_family,
                                "source_class": classify_source(
                                    url,
                                    injection_flags=provenance.get("injection_flags", []),
                                ).source_class,
                                "risk_flags": list(
                                    classify_source(
                                        url,
                                        injection_flags=provenance.get("injection_flags", []),
                                    ).risk_flags
                                ),
                            }
                        )
                    else:
                        key_findings.append(
                            f"[{step.tool_name}] Discovery result was not eligible as verdict evidence."
                        )
                if step.tool_name in {"reverse_image_search", "crop_and_search"}:
                    visual_evidence.append(self._visual_evidence_from_step(step, data))
                if step.tool_name == "reverse_image_search" and summary:
                    key_findings.append(
                        "[reverse_image_search] Candidate discovery only; visit or compare is required before verdict evidence."
                    )
                continue

            if step.tool_name == "compare_with_reference":
                signal = self._parse_compare_reference_signal(step.tool_result)
                if signal["details"]:
                    evidence.append(
                        EvidenceItem(
                            function_call_id=self._step_function_call_id(step),
                            source=str(step.tool_args.get("reference_url", "")),
                            summary=signal["summary"],
                            raw_excerpt=signal["details"][:300],
                            direction="refutes" if signal["refutes_authenticity"] else ("supports" if signal["supports_authenticity"] else "neutral"),
                            quality=signal["quality"],
                            tool_used="compare_with_reference",
                            related_question=self._step_question_id(step),
                        )
                    )
                key_findings.append(f"[compare_with_reference] {signal['summary']}")
                visual_evidence.append(self._visual_evidence_from_step(step, data))
                if signal["refutes_authenticity"]:
                    assessment = "likely_manipulated"
                elif signal["supports_authenticity"] and assessment == "uncertain":
                    assessment = "authentic"
                continue

            if step.tool_name == "check_consistency":
                consistent = bool(data.get("consistent", True))
                details = str(data.get("details", "")).strip()
                summary = "Visual consistency looks normal." if consistent else f"Visual consistency issues found: {details or 'see tool output'}"
                if details:
                    evidence.append(
                        EvidenceItem(
                            function_call_id=self._step_function_call_id(step),
                            source="check_consistency",
                            summary=summary,
                            raw_excerpt=details[:300],
                            direction="neutral" if consistent else "refutes",
                            quality="moderate",
                            tool_used="check_consistency",
                            related_question=self._step_question_id(step),
                        )
                    )
                key_findings.append(f"[check_consistency] {summary}")
                if not consistent:
                    assessment = "likely_manipulated"
                continue

            if step.tool_name == "analyze_visual_anomalies":
                raw = self._safe_json_dict(step.tool_result)
                if isinstance(raw, dict):
                    anomalies = raw.get("anomalies", []) or []
                    step_anomalies: List[VisualAnomaly] = []
                    for item in anomalies:
                        if not isinstance(item, dict):
                            continue
                        try:
                            step_anomalies.append(
                                VisualAnomaly(
                                    function_call_id=self._step_function_call_id(step),
                                    related_question=self._step_question_id(step),
                                    **item,
                                )
                            )
                        except Exception:
                            continue
                    anomaly_assessment = str(raw.get("overall_authenticity", assessment) or assessment)
                    notes = str(raw.get("notes", "")).strip()
                    summary = (
                        f"Visual anomaly analysis found {len(step_anomalies)} concrete anomalies."
                        if step_anomalies
                        else "Visual anomaly analysis did not identify a concrete anomaly."
                    )
                    visual_anomalies.extend(step_anomalies)
                    exact_excerpt = notes or (
                        step_anomalies[0].phenomenon if step_anomalies else ""
                    )
                    if exact_excerpt:
                        evidence.append(
                            EvidenceItem(
                                function_call_id=self._step_function_call_id(step),
                                source="analyze_visual_anomalies",
                                summary=summary,
                                raw_excerpt=exact_excerpt[:300],
                                direction="refutes" if anomaly_assessment in {"likely_ai", "likely_manipulated"} and step_anomalies else "neutral",
                                quality="moderate",
                                tool_used="analyze_visual_anomalies",
                                related_question=self._step_question_id(step),
                            )
                        )
                    if not self._should_downweight_date_only_anomaly(
                        [item.model_dump() for item in step_anomalies],
                        current_date_anchor,
                    ):
                        assessment = anomaly_assessment
                    else:
                        key_findings.append("[analyze_visual_anomalies] Date-only anomaly was downweighted using the runtime date anchor.")
                continue

            if step.tool_name == "crop_and_inspect":
                answer = str(data.get("answer", "") or data.get("description", "")).strip()
                findings = data.get("findings", []) or []
                anomalies = data.get("anomalies", []) or []
                if answer or findings or anomalies:
                    summary = answer or "; ".join(str(item) for item in findings[:2])
                    exact_excerpt = answer or (
                        str(findings[0]) if findings else (str(anomalies[0]) if anomalies else "")
                    )
                    if exact_excerpt:
                        evidence.append(
                            EvidenceItem(
                                function_call_id=self._step_function_call_id(step),
                                source="crop_and_inspect",
                                summary=summary,
                                raw_excerpt=exact_excerpt[:300],
                                direction="refutes" if anomalies else "neutral",
                                quality="moderate",
                                tool_used="crop_and_inspect",
                                related_question=self._step_question_id(step),
                            )
                        )
                    key_findings.append(f"[crop_and_inspect] {summary}")
                continue

            if step.tool_name == "count_objects":
                target = str(data.get("target", step.tool_args.get("target_object", "object"))).strip()
                count = data.get("count")
                details = str(data.get("details", "")).strip()
                if count is not None:
                    summary = f"Counted {count} instance(s) of {target}."
                    exact_excerpt = details or str(count)
                    evidence.append(
                        EvidenceItem(
                            function_call_id=self._step_function_call_id(step),
                            source="count_objects",
                            summary=summary,
                            raw_excerpt=exact_excerpt[:300],
                            direction="neutral",
                            quality="moderate",
                            tool_used="count_objects",
                            related_question=self._step_question_id(step),
                        )
                    )
                    key_findings.append(f"[count_objects] {summary}")
                continue

        if not key_findings:
            key_findings.append("Verification completed with limited usable evidence.")

        world_model = self._build_world_model(
            perception=state.perception,
            plan=state.plan,
            evidence=evidence,
            source_findings=source_findings,
            visual_evidence=visual_evidence,
        )

        return VerificationResult(
            evidence=evidence,
            visual_anomalies=visual_anomalies,
            authenticity_assessment=assessment,
            key_findings=key_findings,
            source_findings=source_findings,
            visual_evidence=visual_evidence,
            world_model=world_model,
        )

    def _merge_verification_results(
        self,
        derived: VerificationResult,
        parsed_results: Sequence[VerificationResult],
        steps: Optional[Sequence[StageStep]] = None,
        plan: Optional[VerificationPlan] = None,
    ) -> VerificationResult:
        merged_evidence: List[EvidenceItem] = list(derived.evidence)
        merged_anomalies: List[VisualAnomaly] = list(derived.visual_anomalies)
        for parsed in parsed_results:
            for item in parsed.evidence:
                if steps is None or plan is None:
                    continue
                canonical = self._canonicalize_model_evidence(item, steps, plan)
                question = next(
                    (
                        candidate
                        for candidate in plan.questions
                        if candidate.question_id == item.related_question
                    ),
                    None,
                )
                if canonical is not None and question is not None and self._evidence_answers_question(
                    canonical,
                    question,
                ):
                    merged_evidence.append(canonical)

        evidence: List[EvidenceItem] = []
        evidence_seen = set()
        for item in merged_evidence:
            key = (
                item.function_call_id,
                item.source,
                item.raw_excerpt,
                item.tool_used,
                item.related_question,
                item.direction,
            )
            if key in evidence_seen:
                continue
            evidence_seen.add(key)
            evidence.append(item)

        visual_anomalies: List[VisualAnomaly] = []
        anomaly_seen = set()
        for item in merged_anomalies:
            key = json.dumps(item.model_dump(), ensure_ascii=False, sort_keys=True)
            if key in anomaly_seen:
                continue
            anomaly_seen.add(key)
            visual_anomalies.append(item)
        if visual_anomalies or any(item.direction == "refutes" for item in evidence):
            assessment = "likely_manipulated"
        elif any(
            item.direction == "supports" and item.quality in {"strong", "moderate"}
            for item in evidence
        ):
            assessment = "authentic"
        else:
            assessment = "uncertain"
        key_findings: List[str] = []
        seen = set()
        for item in derived.key_findings:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            key_findings.append(key)
        return VerificationResult(
            evidence=evidence,
            visual_anomalies=visual_anomalies,
            authenticity_assessment=assessment,
            key_findings=key_findings[:12],
            source_findings=derived.source_findings,
            visual_evidence=derived.visual_evidence,
            world_model=derived.world_model,
            question_resolutions=derived.question_resolutions,
            coverage_complete=derived.coverage_complete,
            unresolved_priority_questions=derived.unresolved_priority_questions,
            exhausted_priority_questions=derived.exhausted_priority_questions,
            iteration_count=derived.iteration_count,
        )

    def _summarize_external_evidence(self, tool_name: str, data: Dict[str, Any], raw_result: str) -> tuple[str, str, str]:
        if tool_name == "text_search":
            queries = data.get("queries", [])
            if isinstance(queries, list) and queries and isinstance(queries[0], dict):
                data = queries[0]
            summary = str(data.get("summary", "") or data.get("evidence", "")).strip()
            excerpt = str(data.get("evidence", "") or data.get("summary", "")).strip()
            url = str(data.get("selected_url", "")).strip()
            return summary, excerpt, url

        if tool_name == "visit":
            summary = self._clean_source_summary(str(data.get("summary", "")).strip(), str(data.get("evidence", "")).strip())
            excerpt = str(data.get("evidence", "") or data.get("summary", "")).strip()
            url = str(data.get("selected_url", "") or data.get("url", "")).strip()
            visits = data.get("visits", []) or []
            if not url and isinstance(visits, list) and visits and isinstance(visits[0], dict):
                url = str(visits[0].get("url", "")).strip()
            return summary, excerpt, url

        if tool_name == "reverse_image_search":
            matches = data.get("lens_results") or data.get("semantic_results") or []
            if isinstance(matches, list) and matches:
                first = matches[0]
                title = str(first.get("title", "")).strip()
                snippet = str(first.get("snippet", "")).strip()
                url = str(first.get("url", "")).strip()
                summary = title or "Reverse image search found a likely match."
                return summary, snippet or title, url
            return "Reverse image search found no strong match.", "", ""

        if tool_name == "crop_and_search":
            summary = str(data.get("summary", "") or data.get("evidence", "")).strip()
            excerpt = str(data.get("evidence", "") or data.get("summary", "")).strip()
            url = str(data.get("selected_url", "")).strip()
            return summary, excerpt, url

        text = str(raw_result).strip()
        return text[:120], text[:300], ""

    def _visual_evidence_from_step(self, step: StageStep, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "round": step.round,
            "function_call_id": self._step_function_call_id(step),
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "selected_url": data.get("selected_url", ""),
            "candidate_page_urls": (data.get("candidate_page_urls", []) or [])[:5],
            "reference_image_url": data.get("reference_image_url", ""),
        }

    def _build_world_model(
        self,
        *,
        perception: Optional[PerceptionReport],
        plan: Optional[VerificationPlan],
        evidence: List[EvidenceItem],
        source_findings: List[Dict[str, Any]],
        visual_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        entities: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []

        if perception:
            for entity in perception.entities[:12]:
                if entity.name:
                    entities.append(
                        {
                            "entity_text": entity.name,
                            "entity_kind": entity.entity_type,
                            "bbox": entity.bbox,
                            "confidence": entity.confidence,
                        }
                    )
            for region in perception.text_regions[:8]:
                if region.text:
                    entities.append(
                        {
                            "entity_text": region.text,
                            "entity_kind": "text_region",
                            "bbox_quad": region.bbox_quad,
                            "confidence": region.confidence,
                        }
                    )

        if plan:
            if plan.image_intent:
                claims.append({"claim_text": plan.image_intent, "state": "candidate", "confidence": 0.5})
            for question in plan.questions[:6]:
                if question.question:
                    claims.append({"claim_text": question.question, "state": "investigation_target", "confidence": 0.4})

        for item in evidence[:12]:
            if item.summary:
                claims.append(
                    {
                        "claim_text": item.summary,
                        "state": "supported" if item.direction == "supports" else ("refuted" if item.direction == "refutes" else "observed"),
                        "confidence": 0.8 if item.quality == "strong" else 0.6,
                    }
                )

        return {
            "entities": self._dedupe_world_model_items(entities, "entity_text"),
            "claims": self._dedupe_world_model_items(claims, "claim_text"),
            "source_findings_count": len(source_findings),
            "visual_evidence_count": len(visual_evidence),
        }

    @staticmethod
    def _dedupe_world_model_items(items: List[Dict[str, Any]], text_key: str) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            value = str(item.get(text_key, "")).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            output.append(item)
        return output

    @staticmethod
    def _step_question_id(step: StageStep) -> str:
        return str(step.tool_args.get("__question_id", "")).strip()

    @staticmethod
    def _step_function_call_id(step: StageStep) -> str:
        return str(step.metadata.get("function_call_id", "")).strip()

    @staticmethod
    def _tool_step_succeeded(step: StageStep) -> bool:
        if step.action_type != "tool_call" or not step.tool_result:
            return False
        if step.metadata.get("tool_exception"):
            return False
        try:
            _, succeeded = parse_tool_result(step.tool_result)
        except Exception:
            return False
        explicit = step.metadata.get("tool_success")
        return succeeded and explicit is not False

    @staticmethod
    def _summarize_tool_failures(steps: Sequence[StageStep]) -> str:
        failures: List[str] = []
        for step in steps:
            if step.action_type != "tool_call":
                continue
            try:
                parsed, succeeded = parse_tool_result(step.tool_result)
            except Exception as exc:
                failures.append(f"{step.tool_name}: {exc}")
                continue
            if not succeeded:
                failures.append(f"{step.tool_name}: {parsed.get('error', 'unknown error')}")
        return "; ".join(failures[-8:])

    def _question_attempts_exhausted(
        self,
        question_id: str,
        steps: Sequence[StageStep],
    ) -> bool:
        attempts = [
            step for step in steps
            if step.action_type == "tool_call"
            and str(step.tool_args.get("__question_id", "")) == question_id
        ]
        successful = [step for step in attempts if self._tool_step_succeeded(step)]
        distinct_tools = {step.tool_name for step in successful}
        iterations = {
            int(step.metadata.get("verification_iteration", 1))
            for step in attempts
        }
        return len(successful) >= 2 and len(distinct_tools) >= 2 and len(iterations) >= 2

    @classmethod
    def _evidence_answers_question(
        cls,
        item: EvidenceItem,
        question: InvestigationQuestion,
    ) -> bool:
        evidence_text = str(item.raw_excerpt or "").strip()
        if not evidence_text or item.direction == "neutral":
            return False
        lowered = evidence_text.lower()
        non_answers = (
            "no strong match",
            "no result",
            "limited signal",
            "completed with limited",
            "tool call failed",
            "unavailable",
        )
        if any(token in lowered for token in non_answers):
            return False

        question_text = " ".join(
            [
                question.question,
                " ".join(question.related_entities),
                " ".join(question.suggested_queries),
            ]
        )
        return bool(cls._semantic_tokens(question_text) & cls._semantic_tokens(evidence_text))

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        text = str(value or "").lower()
        stopwords = {
            "about", "after", "before", "could", "does", "from", "have",
            "image", "into", "photo", "picture", "shown", "that", "this",
            "using", "what", "when", "where", "which", "whether", "with",
        }
        tokens: set[str] = set()
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text):
            if token in stopwords:
                continue
            normalized = token
            for suffix in ("ation", "ing", "ied", "ed", "es", "s"):
                if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
                    normalized = normalized[: -len(suffix)]
                    break
            tokens.add(normalized)
        for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(sequence) == 1:
                tokens.add(sequence)
            else:
                tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tokens

    @staticmethod
    def _apply_plan_revision(
        current: VerificationPlan,
        revision: PlanRevision,
        audit: CoverageAudit,
    ) -> VerificationPlan:
        updates = {item.question_id: item for item in revision.question_updates}
        revised = current.model_copy(deep=True)
        revised.questions = [
            updates.get(question.question_id, question)
            for question in current.questions
        ]
        revised.revision = current.revision + 1
        revised.revision_reason = (
            revision.revision_reason.strip()
            or audit.reason
            or "Priority questions remained unresolved."
        )
        return revised

    def _evidence_item_is_grounded(
        self,
        item: EvidenceItem,
        steps: Sequence[StageStep],
    ) -> bool:
        if not item.function_call_id or not item.tool_used or not item.related_question:
            return False
        candidates = [
            step for step in steps
            if self._step_function_call_id(step) == item.function_call_id
            if step.tool_name == item.tool_used
            and self._tool_step_succeeded(step)
            and str(step.tool_args.get("__question_id", "")) == item.related_question
        ]
        if len(candidates) != 1:
            return False

        source = str(item.source or "").strip().lower()
        excerpt = str(item.raw_excerpt or "").strip().lower()
        raw = str(candidates[0].tool_result or "").lower()
        if not excerpt or len(excerpt) < 8 or excerpt not in raw:
            return False
        if source and source not in {item.tool_used.lower(), "current_time"} and source not in raw:
            return False
        return True

    def _canonicalize_model_evidence(
        self,
        item: EvidenceItem,
        steps: Sequence[StageStep],
        plan: VerificationPlan,
    ) -> Optional[EvidenceItem]:
        if item.tool_used == "reverse_image_search":
            return None
        if not self._evidence_item_is_grounded(item, steps):
            return None
        if item.related_question not in {question.question_id for question in plan.questions}:
            return None
        step = next(
            candidate
            for candidate in steps
            if self._step_function_call_id(candidate) == item.function_call_id
        )
        data = self._safe_json_dict(step.tool_result)
        direction = item.direction
        quality = item.quality
        if item.tool_used == "current_time":
            direction = "neutral"
        elif item.tool_used == "check_consistency":
            direction = "neutral" if bool(data.get("consistent", True)) else "refutes"
            quality = "moderate"
        elif item.tool_used == "analyze_visual_anomalies":
            anomalies = data.get("anomalies", []) or []
            direction = "refutes" if anomalies else "neutral"
            quality = "moderate"
        elif item.tool_used == "count_objects":
            direction = "neutral"
            quality = "moderate"
        elif item.tool_used == "crop_and_inspect":
            direction = "refutes" if data.get("anomalies") else "neutral"
            quality = "moderate"
        elif item.tool_used == "compare_with_reference":
            signal = self._parse_compare_reference_signal(step.tool_result)
            direction = (
                "refutes"
                if signal["refutes_authenticity"]
                else ("supports" if signal["supports_authenticity"] else "neutral")
            )
            quality = signal["quality"]
        elif item.tool_used in {"visit", "text_search", "crop_and_search"}:
            provenance = self._find_browse_evidence_record(data, item.raw_excerpt)
            if not self._browse_record_is_evidence_eligible(provenance, item.source):
                return None
            stance = str(provenance.get("stance", "")).strip().lower()
            if stance not in {"support", "refute", "unclear"}:
                return None
            direction = {
                "support": "supports",
                "refute": "refutes",
                "unclear": "neutral",
            }[stance]
            relevance = str(provenance.get("relevance", "")).strip().lower()
            if relevance not in {"high", "medium", "low"}:
                return None
            quality = "strong" if relevance == "high" and self._is_probably_trusted_url(item.source) else (
                "moderate" if relevance in {"high", "medium"} else "weak"
            )
        return item.model_copy(
            update={
                "summary": item.raw_excerpt,
                "direction": direction,
                "quality": quality,
            }
        )

    @classmethod
    def _browse_record_is_evidence_eligible(
        cls,
        record: Optional[Dict[str, Any]],
        source: str,
    ) -> bool:
        if not record or not bool(record.get("evidence_eligible", False)):
            return False
        if record.get("injection_flags"):
            return False
        if str(record.get("directness", "")).strip().lower() != "direct":
            return False
        artifact_hash = str(record.get("artifact_sha256", "")).strip().lower()
        retrieved_at = str(record.get("retrieved_at", "")).strip()
        span = record.get("evidence_span")
        evidence = " ".join(str(record.get("evidence", "")).split())
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            return False
        if not retrieved_at or not isinstance(span, dict) or not evidence:
            return False
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            return False
        if end - start != len(evidence):
            return False
        try:
            datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        record_url = str(record.get("selected_url", "") or record.get("url", "")).strip()
        return bool(record_url) and cls._canonical_url(record_url) == cls._canonical_url(source)

    @classmethod
    def _find_browse_evidence_record(
        cls,
        data: Dict[str, Any],
        raw_excerpt: str,
    ) -> Optional[Dict[str, Any]]:
        excerpt = " ".join(str(raw_excerpt or "").split()).lower()
        if not excerpt:
            return None

        candidates: List[Dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                evidence = " ".join(str(value.get("evidence", "")).split()).lower()
                if (
                    evidence
                    and excerpt in evidence
                    and "stance" in value
                    and "relevance" in value
                ):
                    candidates.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(data)
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: len(str(candidate.get("evidence", ""))))

    def _visual_anomaly_is_grounded(
        self,
        anomaly: VisualAnomaly,
        steps: Sequence[StageStep],
    ) -> bool:
        if not anomaly.function_call_id or not anomaly.related_question:
            return False
        candidates = [
            step
            for step in steps
            if self._step_function_call_id(step) == anomaly.function_call_id
            and step.tool_name == "analyze_visual_anomalies"
            and self._tool_step_succeeded(step)
            and self._step_question_id(step) == anomaly.related_question
        ]
        if len(candidates) != 1:
            return False
        data = self._safe_json_dict(candidates[0].tool_result)
        for raw in data.get("anomalies", []) or []:
            if not isinstance(raw, dict):
                continue
            if all(
                raw.get(field) == getattr(anomaly, field)
                for field in (
                    "name",
                    "region",
                    "phenomenon",
                    "reasoning",
                    "severity",
                    "type",
                    "entities_involved",
                )
            ):
                return True
        return False

    @staticmethod
    def _safe_json_dict(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _is_probably_trusted_url(value: str) -> bool:
        return classify_source(value).source_class == "official"

    @staticmethod
    def _canonical_url(value: str) -> str:
        return canonicalize_url(value)

    @staticmethod
    def _clean_source_summary(summary: str, fallback: str = "") -> str:
        value = str(summary or "").strip()
        backup = str(fallback or "").strip()
        noisy_tokens = ("Markdown Content:", "Title:", "URL Source:", "![Image", "Suggested Searches")
        if value and not any(token in value for token in noisy_tokens):
            return value[:320]
        if backup and not any(token in backup for token in noisy_tokens):
            return backup[:260]
        return value[:260] if value else ""

    @staticmethod
    def _parse_compare_reference_signal(raw_result: str) -> Dict[str, Any]:
        parsed = Orchestrator._safe_json_dict(raw_result)
        details = str(parsed.get("overall_observation", parsed.get("manipulation_details", ""))).strip()
        same_subject = bool(parsed.get("same_subject_or_scene", parsed.get("is_same_scene", False)))
        same_capture = bool(parsed.get("same_capture_or_near_duplicate", False))
        different_capture = bool(parsed.get("likely_different_original_capture", False))
        edit_present = bool(parsed.get("edit_evidence_present", False))
        edit_strength = str(parsed.get("edit_evidence_strength", "none")).strip().lower()
        differences = parsed.get("differences", []) or []
        diff_types = [str(item.get("type", "")).strip().lower() for item in differences if isinstance(item, dict)]
        edit_diff_types = [
            str(item.get("type", "")).strip().lower()
            for item in differences
            if isinstance(item, dict) and bool(item.get("is_edit_evidence", str(item.get("type", "")).strip().lower() in {"addition", "removal", "modification"}))
        ]

        if edit_present and edit_diff_types:
            summary = "Reference comparison found direct edit evidence: " + ", ".join(sorted(set(edit_diff_types)))
        elif same_capture:
            summary = "Reference comparison indicates a near-duplicate or same capture."
        elif different_capture and same_subject:
            summary = "Reference comparison indicates different captures of the same subject or scene."
        elif same_subject:
            summary = "Reference comparison indicates the same subject or scene."
        else:
            summary = "Reference comparison produced limited signal."
        if details:
            summary = f"{summary} {details[:180]}".strip()

        quality = "strong" if edit_present and edit_strength in {"moderate", "strong"} else "moderate"
        return {
            "details": details,
            "summary": summary,
            "supports_authenticity": same_capture or (same_subject and different_capture and not edit_present),
            "refutes_authenticity": edit_present and bool(edit_diff_types),
            "quality": quality,
        }

    @staticmethod
    def _should_downweight_date_only_anomaly(anomalies: Any, current_date_anchor: str) -> bool:
        if not isinstance(anomalies, list) or len(anomalies) != 1 or not current_date_anchor:
            return False
        anomaly = anomalies[0]
        if not isinstance(anomaly, dict):
            return False
        text = " ".join(str(anomaly.get(field, "")).lower() for field in ("name", "phenomenon", "reasoning", "region", "type"))
        if not any(token in text for token in ("date", "year", "future", "timestamp")):
            return False
        return current_date_anchor[:4] in text

    def _check_timeout(self, started: float, state: VerificationState) -> None:
        if time.time() - started > self.timeout:
            state.termination = "timeout"
            raise TimeoutError(f"Pipeline exceeded timeout of {self.timeout} seconds.")
