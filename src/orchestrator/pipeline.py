# -*- coding: utf-8 -*-
"""VisualFact-driven image-only factual investigation orchestrator."""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.integrations.gemini import take_runtime_metrics
from src.integrations.clock.system_clock import SystemClockClient
from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import (
    audit_coverage,
    compile_verdict_basis,
)
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.image_only_prompts import (
    EVIDENCE_DECISION_SYSTEM_PROMPT as IMAGE_ONLY_EVIDENCE_DECISION_PROMPT,
    JUDGMENT_SYSTEM_PROMPT as IMAGE_ONLY_JUDGMENT_PROMPT,
    REACT_SYSTEM_PROMPT as IMAGE_ONLY_REACT_PROMPT,
    REFLECTION_SYSTEM_PROMPT as IMAGE_ONLY_REFLECTION_PROMPT,
    TARGET_PLANNING_SYSTEM_PROMPT as IMAGE_ONLY_TARGET_PLANNING_PROMPT,
    render_evidence_decision_context as render_image_only_evidence_decision_context,
    render_judgment_context as render_image_only_judgment_context,
    render_react_context as render_image_only_react_context,
    render_reflection_context as render_image_only_reflection_context,
    render_target_planning_context as render_image_only_target_planning_context,
    select_react_tasks as select_image_only_react_tasks,
)
from src.orchestrator.investigation_models import (
    EvidenceDecisionOutput,
    ImageOnlyInvestigationState,
    ImageOnlyJudgment,
    InvestigationSegmentOutput,
    ReflectionOutput,
    TargetPlanningOutput,
)
from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.stage_runner import StageRunner, StageStep
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.source_provenance import canonicalize_url
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_health import require_tools, summarize_health
from src.orchestrator.tool_registry import (
    REQUIRED_TOOLS,
    build_all_tools_with_health,
    build_stage_tools,
)
from src.orchestrator.tool_result import parse_tool_result, serialize_tool_result
from src.orchestrator.task_store import (
    MAX_TOOL_ACTIONS,
    REFLECTION_INTERVAL,
    apply_evidence_decision,
    apply_reflection,
    apply_target_planning,
    evidence_decision_checkpoint_reason,
    next_action_boundary,
    pending_evidence_decision_ids,
    remaining_material_routes,
    record_tool_observation,
    runtime_task_tool_names,
    state_from_bootstrap,
)
from src.storage import default_tool_cache_dir


class Orchestrator:
    """VisualFact-driven image-only v3 orchestrator."""

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.5-flash",
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        llm_wire_api: Optional[str] = None,
        vlm_wire_api: Optional[str] = None,
        timeout: float = 1800.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        validate_startup: bool = True,
        source_access_policy: Optional[SourceAccessPolicy] = None,
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
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.timeout = timeout
        self.tool_action_timeout_seconds = self._runtime_timeout(
            "AGENT_TOOL_ACTION_TIMEOUT_SECONDS",
            150.0,
        )
        self.stage_request_timeout_seconds = self._runtime_timeout(
            "AGENT_STAGE_REQUEST_TIMEOUT_SECONDS",
            120.0,
        )
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
                self.source_access_policy.cache_partition,
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
        }
        self.verification_tool_limits = {
            "current_time": 1,
            "ocr_with_position": 3,
            "reverse_image_search": 2,
            "text_search": 16,
            "visit": 16,
            "compare_with_reference": 6,
            "crop_and_inspect": 4,
            "check_consistency": 3,
            "analyze_visual_anomalies": 3,
        }

        self.llm = APIBackend(
            provider=self.provider,
            model_name=model_name,
            wire_api=self.llm_wire_api,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(
                os.getenv("AGENT_LLM_REQUEST_TIMEOUT_SECONDS", "90")
            ),
            max_retries=int(
                os.getenv("AGENT_LLM_REQUEST_MAX_RETRIES", "3")
            ),
        )
        self.all_tools, self.tool_health = build_all_tools_with_health(
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            vlm_wire_api=self.vlm_wire_api,
        )
        for tool in self.all_tools.values():
            setter = getattr(tool, "set_source_access_policy", None)
            if callable(setter):
                setter(self.source_access_policy)
        self.tool_health_summary = summarize_health(self.tool_health)
        if validate_startup:
            require_tools(self.tool_health, REQUIRED_TOOLS)
            self._validate_startup_configuration()
        clock = SystemClockClient().now()
        self.date_prefix = (
            f"Current date: {clock['current_date']} ({clock['timezone']}). "
            "Use this runtime date for time-sensitive judgments instead of model memory.\n\n"
        )

    def _validate_startup_configuration(self) -> None:
        tool_thinking_levels = {
            "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL": os.getenv(
                "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL", "minimal"
            ),
            "GEMINI_BROWSE_THINKING_LEVEL": os.getenv(
                "GEMINI_BROWSE_THINKING_LEVEL", "minimal"
            ),
            "GEMINI_VISION_THINKING_LEVEL": os.getenv(
                "GEMINI_VISION_THINKING_LEVEL", "minimal"
            ),
            "GEMINI_REFERENCE_COMPARE_THINKING_LEVEL": os.getenv(
                "GEMINI_REFERENCE_COMPARE_THINKING_LEVEL", "minimal"
            ),
            "GEMINI_VISUAL_ANOMALY_THINKING_LEVEL": os.getenv(
                "GEMINI_VISUAL_ANOMALY_THINKING_LEVEL", "minimal"
            ),
        }
        for env_name, value in tool_thinking_levels.items():
            if value.strip().lower() != "minimal":
                raise ValueError(
                    f"{env_name} must be 'minimal' for the active agent."
                )
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
        runtime_case: ImageOnlyRuntimeCase,
        *,
        decision_policy_version: str = "reinspect-v2",
    ) -> Dict[str, Any]:
        """Run the complete image-only reinspect-v2 investigation path."""

        verify_case_image(runtime_case, image_path)
        state = VerificationState(
            image_path=image_path,
            image_id=runtime_case.case_id,
            runtime_case=runtime_case,
            input_mode="image_only",
            decision_policy_version=decision_policy_version,
            tool_health=self.tool_health_summary,
        )
        self.last_state = state
        started = time.time()
        try:
            if decision_policy_version != "reinspect-v2":
                raise RuntimeError(
                    "image-only runtime requires decision_policy_version=reinspect-v2"
                )
            self._validate_image_only_bootstrap_configuration()
            state.perception = await self._run_perception(state, image_path)
            bootstrap = build_bootstrap_investigation(
                runtime_case,
                state.perception,
            )
            investigation = state_from_bootstrap(bootstrap)
            state.investigation_state = investigation
            self._sync_image_only_state(state, investigation)
            await self._run_image_only_target_planning(
                state,
                investigation,
            )
            await self._run_image_only_investigation(
                state,
                investigation,
                image_path,
                runtime_case,
            )
            self._require_successful_image_only_investigation(state)
            coverage = (
                investigation.coverage_audits[-1]
                if investigation.coverage_audits
                else audit_coverage(investigation)
            )
            compiled_verdict, basis = compile_verdict_basis(investigation)
            judgment = await self._run_image_only_judgment(
                state,
                investigation,
                coverage,
                compiled_verdict,
                basis,
            )
            judgment = self._normalize_incomplete_judgment(
                investigation,
                judgment,
            )
            investigation.judgment = judgment
            state.judgment = judgment  # type: ignore[assignment]
            state.termination = "success"
            self._sync_image_only_state(state, investigation)
        except Exception as exc:
            state.termination = "error"
            state.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            state.stage_timings["total"] = round(time.time() - started, 2)

        return {
            "image_id": state.image_id,
            "image_path": state.image_path,
            "input_mode": state.input_mode,
            "decision_policy_version": state.decision_policy_version,
            "judgment": judgment.model_dump(mode="json"),
            "verdict": judgment.verdict,
            "confidence": judgment.confidence,
            "overall_assessment": judgment.overall_assessment,
            "investigation_status": self._investigation_status(investigation),
            "verification_layers": self._verification_layers(investigation),
            "verdict_basis": basis.model_dump(mode="json"),
            "state": state.to_dict(),
            "termination": state.termination,
            "time_taken": state.stage_timings["total"],
            "token_usage": state.token_usage,
            "total_tool_calls": state.total_tool_calls,
            "total_tool_subcalls": state.total_tool_subcalls,
            "tool_subcalls_by_kind": state.tool_subcalls_by_kind,
            "llm_api_calls": state.llm_api_calls,
            "error": None,
        }

    def _validate_image_only_bootstrap_configuration(self) -> None:
        """Validate only providers exercised by the Phase-B bootstrap."""

        if self.vlm_provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ):
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini perception."
            )
        for tool_name in ("perceive_scene", "ocr_with_position"):
            if tool_name not in self.all_tools:
                raise RuntimeError(
                    f"Required image-only bootstrap tool '{tool_name}' is unavailable: "
                    + str(
                        self.tool_health_summary.get(tool_name, {}).get(
                            "error",
                            "not registered",
                        )
                    )
                )

    async def _run_image_only_target_planning(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
    ) -> None:
        """Let the policy induce bounded factual targets from visible state."""

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_TARGET_PLANNING_PROMPT),
            tools=[],
            output_schema=TargetPlanningOutput,
            max_rounds=2,
            stage_name="image_only_planning",
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_target_planning(
                    investigation,
                    parsed,
                )
            ),
            max_output_tokens=self._stage_output_tokens("PLANNING", 8192),
            generation_config={
                "thinking_level": self._stage_thinking_level("PLANNING")
            },
        )
        parsed, steps = await runner.run(
            render_image_only_target_planning_context(investigation)
        )
        for step in steps:
            if step.action_type != "output_rejected":
                continue
            step.action_type = "planning_revision"
            step.metadata["planning_revision_reason"] = step.metadata.get(
                "rejection_reason",
                "",
            )
        self._record_stage_steps(state, steps)
        if parsed is None:
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "image-only Target Planning did not establish an atomic core fact"
            )
        update = apply_target_planning(investigation, parsed)
        core = next(
            (
                fact
                for fact in investigation.facts
                if fact.fact_id == investigation.core_verdict_fact_id
            ),
            None,
        )
        if (
            not update.get("accepted_fact_ids")
            or core is None
            or core.predicate == "appears_to_depict"
        ):
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "image-only Target Planning produced no externally checkable "
                "atomic core fact"
            )
        self._sync_image_only_state(state, investigation)

    async def _run_image_only_investigation(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
    ) -> None:
        started = time.time()
        prior_evidence_count = len(investigation.evidence)
        prior_finding_count = len(investigation.findings)
        prior_fact_signature = self._image_only_fact_signature(investigation)
        # Establish the no-progress baseline before the first tool call.  Every
        # later accepted action is a decision checkpoint.
        audit_coverage(investigation)
        while not investigation.stop_reason:
            self._check_timeout(started, state)
            if investigation.action_count >= MAX_TOOL_ACTIONS:
                await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger="before_unverifiable",
                    required=True,
                )
                audit_coverage(investigation)
                break
            if not any(
                task.status in {"active", "pending"}
                for task in investigation.tasks
            ):
                reviewed = await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger="before_unverifiable",
                    required=True,
                )
                if reviewed and any(
                    task.status in {"active", "pending"}
                    for task in investigation.tasks
                ):
                    audit_coverage(investigation)
                    self._sync_image_only_state(state, investigation)
                    continue
                audit_coverage(investigation)
                if not investigation.stop_reason:
                    investigation.stop_reason = "information_saturated"
                break

            segment_stop_action = next_action_boundary(
                investigation.action_count
            )
            react_tasks = select_image_only_react_tasks(investigation)
            react_task_ids = {task.task_id for task in react_tasks}
            if not react_task_ids:
                reviewed = await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger="before_unverifiable",
                    required=True,
                )
                if reviewed and select_image_only_react_tasks(investigation):
                    audit_coverage(investigation)
                    self._sync_image_only_state(state, investigation)
                    continue
                audit_coverage(investigation)
                break
            executable_tool_names = self._image_only_executable_tool_names(
                investigation,
                task_ids=react_task_ids,
            )
            if not executable_tool_names:
                reviewed = await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger="before_unverifiable",
                    required=True,
                )
                if reviewed:
                    refreshed_tasks = select_image_only_react_tasks(
                        investigation
                    )
                    refreshed_ids = {
                        item.task_id for item in refreshed_tasks
                    }
                    if self._image_only_executable_tool_names(
                        investigation,
                        task_ids=refreshed_ids,
                    ):
                        audit_coverage(investigation)
                        self._sync_image_only_state(state, investigation)
                        continue
                audit_coverage(investigation)
                break
            task_claims = self._image_only_task_claims(
                investigation,
                task_ids=react_task_ids,
            )
            task_evidence_goals = self._image_only_task_evidence_goals(
                investigation,
                task_ids=react_task_ids,
            )

            observation_update: Dict[str, Any] = {}

            def observation_callback(
                step: StageStep,
                _steps: List[StageStep],
            ) -> Dict[str, Any]:
                nonlocal observation_update
                update = record_tool_observation(
                    investigation,
                    step,
                    image_sha256=runtime_case.image_sha256,
                )
                observation_update = update
                self._sync_image_only_state(state, investigation)
                return update

            runner = StageRunner(
                llm=self.llm,
                system_prompt=self._sp(IMAGE_ONLY_REACT_PROMPT),
                tools=[
                    tool
                    for tool in build_stage_tools(
                        "verification",
                        self.all_tools,
                    )
                    if (
                        tool.name != "current_time"
                        and tool.name in executable_tool_names
                    )
                ],
                output_schema=InvestigationSegmentOutput,
                # Tool availability depends on the reducer state produced by the
                # previous action (for example, a search lead makes page
                # inspection mandatory). End each accepted action at a
                # deterministic boundary so the next request receives a newly
                # compiled tool schema instead of a stale multi-turn schema.
                max_rounds=1,
                image_path=image_path,
                stage_name="verification",
                recent_rounds_to_keep=3,
                tool_cache=self.tool_cache,
                cacheable_tools=list(self.cacheable_tools),
                tool_call_limits=self.verification_tool_limits,
                should_stop=lambda _steps: (
                    bool(investigation.stop_reason)
                    or any(
                        step.action_type == "tool_call"
                        for step in _steps
                    )
                    or investigation.action_count >= segment_stop_action
                    or not any(
                        task.task_id in react_task_ids
                        and task.status in {"active", "pending"}
                        for task in investigation.tasks
                    )
                ),
                min_tool_calls=1,
                attach_image=False,
                prior_steps=[
                    step
                    for step in state.all_steps
                    if getattr(step, "stage_name", "")
                    == "image_only_investigation"
                    or getattr(step, "stage_name", "") == "verification"
                ],
                max_output_tokens=self._stage_output_tokens(
                    "VERIFICATION",
                    16384,
                ),
                generation_config={
                    "thinking_level": self._stage_thinking_level("VERIFICATION")
                },
                observation_callback=observation_callback,
                question_claims=task_claims,
                question_evidence_goals=task_evidence_goals,
                # Dynamic image-only investigation is not a fixed-question
                # scheduler. Task IDs remain schema-validated, while Reflection
                # and decisive-fact Coverage decide which open task must run next.
                priority_question_ids=[],
                supporting_question_ids=[],
                source_access_policy=self.source_access_policy,
                visual_call_validator=lambda tool_name, tool_args: (
                    self._image_only_discovery_route_error(
                        investigation,
                        tool_name,
                        tool_args,
                    )
                ),
                max_protocol_corrections=4,
                max_tool_calls_per_turn=1,
                force_tool_each_round=True,
                question_is_active=lambda task_id: any(
                    task.task_id == task_id
                    and task.status in {"active", "pending"}
                    for task in investigation.tasks
                ),
                stop_output_factory=lambda: InvestigationSegmentOutput(
                    segment_summary=(
                        "The deterministic action or verdict boundary was reached."
                    ),
                    ready_for_reflection=True,
                ),
                request_timeout_seconds=self.stage_request_timeout_seconds,
                tool_timeout_seconds=self.tool_action_timeout_seconds,
                tool_argument_constraints=(
                    self._image_only_tool_argument_constraints(
                        investigation,
                        task_ids=react_task_ids,
                    )
                ),
            )
            try:
                parsed, steps = await runner.run(
                    render_image_only_react_context(investigation)
                )
            except Exception as exc:
                partial_steps = list(
                    getattr(exc, "stage_steps", []) or []
                )
                for step in partial_steps:
                    if step.stage_name == "verification":
                        step.stage_name = "image_only_investigation"
                        step.metadata["stage"] = "image_only_investigation"
                self._record_stage_steps(state, partial_steps)
                raise
            for step in steps:
                if step.stage_name == "verification":
                    step.stage_name = "image_only_investigation"
                    step.metadata["stage"] = "image_only_investigation"
            self._record_stage_steps(state, steps)
            if parsed is None:
                raise RuntimeError(
                    "image-only ReAct segment did not produce valid structured output"
                )
            reflection_boundary = (
                investigation.action_count > 0
                and investigation.action_count % REFLECTION_INTERVAL == 0
            )
            decision_trigger = evidence_decision_checkpoint_reason(
                investigation,
                update=observation_update,
                before_reflection=reflection_boundary,
            )
            if not decision_trigger and not remaining_material_routes(
                investigation,
                fact_id=investigation.core_verdict_fact_id or "",
            ):
                decision_trigger = evidence_decision_checkpoint_reason(
                    investigation,
                    before_unverifiable=True,
                )
            if decision_trigger:
                await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger=decision_trigger,
                    required=decision_trigger == "before_unverifiable",
                )

            audit_coverage(
                investigation,
                decision_checkpoint=True,
            )
            self._sync_image_only_state(state, investigation)

            if (
                reflection_boundary
                and not investigation.stop_reason
            ):
                evidence_gain = len(investigation.evidence) > prior_evidence_count
                decision_gain = (
                    len(investigation.findings) > prior_finding_count
                    or self._image_only_fact_signature(investigation)
                    != prior_fact_signature
                )
                await self._run_image_only_reflection(
                    state,
                    investigation,
                    evidence_gain=evidence_gain,
                    decision_gain=decision_gain,
                )
                audit_coverage(
                    investigation,
                    reflection_checkpoint=True,
                )
                prior_evidence_count = len(investigation.evidence)
                prior_finding_count = len(investigation.findings)
                prior_fact_signature = self._image_only_fact_signature(
                    investigation
                )
                self._sync_image_only_state(state, investigation)

    async def _run_image_only_evidence_decision(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        trigger: str,
        required: bool,
    ) -> bool:
        """Review accumulated Evidence only at a material control boundary."""

        reviewed_evidence_ids = pending_evidence_decision_ids(investigation)
        if not reviewed_evidence_ids:
            return False
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_EVIDENCE_DECISION_PROMPT),
            tools=[],
            output_schema=EvidenceDecisionOutput,
            # One semantic rejection may expose a second, independent schema
            # boundary (for example terminal-vs-refinement, then slot scope).
            # Allow two bounded correction turns without reopening tool use.
            max_rounds=2,
            stage_name="image_only_evidence_decision",
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_evidence_decision(
                    investigation,
                    parsed,
                    reviewed_evidence_ids=reviewed_evidence_ids,
                    trigger=trigger,
                )
            ),
            max_output_tokens=self._stage_output_tokens("VERIFICATION", 8192),
            generation_config={
                "thinking_level": self._stage_thinking_level("VERIFICATION")
            },
        )
        parsed, steps = await runner.run(
            render_image_only_evidence_decision_context(
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
            )
        )
        for step in steps:
            if step.action_type != "output_rejected":
                continue
            step.action_type = "evidence_decision_revision"
            step.metadata["evidence_decision_revision_reason"] = (
                step.metadata.get("rejection_reason", "")
            )
        if parsed is None:
            self._record_stage_steps(state, steps)
            self._sync_image_only_state(state, investigation)
            if required:
                raise RuntimeError(
                    "mandatory semantic Evidence decision did not validate"
                )
            return False
        update = apply_evidence_decision(
            investigation,
            parsed,
            reviewed_evidence_ids=reviewed_evidence_ids,
            trigger=trigger,
        )
        if not update.get("accepted", False):
            self._record_stage_steps(state, steps)
            self._sync_image_only_state(state, investigation)
            if required:
                raise RuntimeError(
                    "mandatory semantic Evidence decision was rejected: "
                    + str(update.get("rejected_reason", "unknown reason"))
                )
            return False
        for step in reversed(steps):
            if step.action_type == "output":
                step.metadata["evidence_decision_state_update"] = update
                break
        self._record_stage_steps(state, steps)
        self._sync_image_only_state(state, investigation)
        return True

    async def _run_image_only_reflection(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        evidence_gain: bool,
        decision_gain: bool,
    ) -> None:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_REFLECTION_PROMPT),
            tools=[],
            output_schema=ReflectionOutput,
            max_rounds=1,
            stage_name="image_only_reflection",
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_reflection(
                    investigation,
                    parsed,
                    evidence_gain=evidence_gain,
                    decision_gain=decision_gain,
                )
            ),
            max_output_tokens=self._stage_output_tokens("REFLECTION", 8192),
            generation_config={
                "thinking_level": self._stage_thinking_level("REFLECTION")
            },
        )
        parsed, steps = await runner.run(
            render_image_only_reflection_context(investigation)
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            investigation.reflection_failure_streak += 1
            if investigation.reflection_failure_streak >= 2:
                raise RuntimeError(
                    "two consecutive image-only Reflection calls failed"
                )
            return
        apply_reflection(
            investigation,
            parsed,
            evidence_gain=evidence_gain,
            decision_gain=decision_gain,
        )

    async def _run_image_only_judgment(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        coverage: Any,
        compiled_verdict: str,
        basis: Any,
    ) -> ImageOnlyJudgment:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_JUDGMENT_PROMPT),
            tools=[],
            output_schema=ImageOnlyJudgment,
            max_rounds=1,
            stage_name="image_only_judgment",
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_judgment(
                    parsed,
                    compiled_verdict=compiled_verdict,
                    basis=basis,
                )
            ),
            max_output_tokens=self._stage_output_tokens("JUDGMENT", 8192),
            generation_config={
                "thinking_level": self._stage_thinking_level("JUDGMENT")
            },
        )
        parsed, steps = await runner.run(
            render_image_only_judgment_context(
                investigation,
                coverage,
                compiled_verdict,
                basis,
            )
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "image-only Judgment did not produce a valid reinspect-v2 output"
            )
        return parsed

    @staticmethod
    def _validate_image_only_reflection(
        investigation: ImageOnlyInvestigationState,
        parsed: ReflectionOutput,
        *,
        evidence_gain: bool,
        decision_gain: bool,
    ) -> tuple[bool, str]:
        candidate = investigation.model_copy(deep=True)
        record = apply_reflection(
            candidate,
            parsed,
            evidence_gain=evidence_gain,
            decision_gain=decision_gain,
        )
        proposed_changes = bool(
            parsed.task_updates
            or parsed.new_tasks
        )
        accepted_changes = bool(
            record.accepted_task_update_ids
            or record.accepted_new_task_ids
        )
        if proposed_changes and not accepted_changes:
            return False, "; ".join(record.rejected_reasons) or (
                "Reflection proposed no valid state transition"
            )
        return True, ""

    @staticmethod
    def _validate_image_only_target_planning(
        investigation: ImageOnlyInvestigationState,
        parsed: TargetPlanningOutput,
    ) -> tuple[bool, str]:
        if len(parsed.proposals) != 1:
            return False, (
                "Target Planning must return exactly one decisive external-world "
                "or source-record proposition"
            )
        proposal = parsed.proposals[0]
        if (
            proposal.decision_relevance != "decisive"
            or proposal.predicate == "visual_integrity"
        ):
            return False, (
                "Target Planning must return one decisive externally checkable "
                "proposition, not a visual-integrity diagnostic"
            )
        candidate = investigation.model_copy(deep=True)
        update = apply_target_planning(candidate, parsed)
        if (
            not update["accepted_fact_ids"]
            or candidate.core_verdict_fact_id not in update["accepted_fact_ids"]
        ):
            return False, "; ".join(update["rejected_reasons"]) or (
                "target planning proposed no externally checkable atomic core fact"
            )
        return True, ""

    @staticmethod
    def _validate_image_only_evidence_decision(
        investigation: ImageOnlyInvestigationState,
        parsed: EvidenceDecisionOutput,
        *,
        reviewed_evidence_ids: Sequence[str],
        trigger: str,
    ) -> tuple[bool, str]:
        candidate = investigation.model_copy(deep=True)
        update = apply_evidence_decision(
            candidate,
            parsed,
            reviewed_evidence_ids=reviewed_evidence_ids,
            trigger=trigger,
        )
        if not update.get("accepted", False):
            return False, str(
                update.get(
                    "rejected_reason",
                    "Evidence decision proposed no valid state transition",
                )
            )
        return True, ""

    @staticmethod
    def _validate_image_only_judgment(
        parsed: ImageOnlyJudgment,
        *,
        compiled_verdict: str,
        basis: Any,
    ) -> tuple[bool, str]:
        if parsed.verdict != compiled_verdict:
            return False, (
                f"verdict must be {compiled_verdict}, received {parsed.verdict}"
            )
        if parsed.policy_rule_id != "reinspect-v2":
            return False, "policy_rule_id must be reinspect-v2"
        checks = (
            (
                set(parsed.selected_fact_ids),
                set(basis.fact_ids),
                "fact",
            ),
            (
                set(parsed.selected_finding_ids),
                set(basis.finding_ids),
                "finding",
            ),
            (
                set(parsed.selected_evidence_ids),
                set(basis.evidence_ids),
                "evidence",
            ),
        )
        for selected, allowed, name in checks:
            if selected != allowed:
                return False, (
                    f"selected {name} ids must exactly match the compiled basis"
                )
        if parsed.unresolved_gaps != basis.unresolved_gaps:
            return False, "unresolved_gaps must match the compiled basis"
        return True, ""

    @staticmethod
    def _image_only_task_claims(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str] | None = None,
    ) -> Dict[str, str]:
        facts = {fact.fact_id: fact for fact in investigation.facts}
        claims: Dict[str, str] = {}
        for task in investigation.tasks:
            if task.status not in {"active", "pending"}:
                continue
            if task_ids is not None and task.task_id not in task_ids:
                continue
            specific_claims = [
                facts[fact_id].statement
                for fact_id in task.fact_ids
                if fact_id in facts
                and (
                    facts[fact_id].origin.type == "web_discovery"
                    or facts[fact_id].predicate
                    not in {
                        "appears_to_depict",
                        "visible_in",
                        "reads",
                        "context_suggested_by_text",
                    }
                )
            ]
            scene_claims = [
                facts[fact_id].statement
                for fact_id in task.fact_ids
                if fact_id in facts
                and facts[fact_id].predicate == "appears_to_depict"
            ]
            claims[task.task_id] = (
                " | ".join(specific_claims or scene_claims)
                if (specific_claims or scene_claims)
                else f"Question to resolve: {task.question}"
            )[:1800]
        return claims

    @staticmethod
    def _image_only_task_evidence_goals(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str] | None = None,
    ) -> Dict[str, str]:
        """Use the source-answerable task question, not the whole image claim."""

        discovery_task_ids = {
            item.task_id
            for item in investigation.discoveries
        }
        goals: Dict[str, str] = {}
        for task in investigation.tasks:
            if (
                task.status not in {"active", "pending"}
                or (task_ids is not None and task.task_id not in task_ids)
            ):
                continue
            if (
                task.task_id in discovery_task_ids
                and Orchestrator._image_only_task_requires_source_goal(
                    investigation,
                    task,
                )
            ):
                goals[task.task_id] = (
                    "Does this candidate public source identify or directly "
                    "describe the same input image or depicted scene? Extract "
                    "only the concrete title, identity, event, place, date, "
                    "creator, or source-record statement that the page itself "
                    "supports."
                )
            else:
                goals[task.task_id] = task.question[:1800]
        return goals

    @staticmethod
    def _image_only_task_requires_source_goal(
        investigation: ImageOnlyInvestigationState,
        task: Any,
    ) -> bool:
        """Keep world-fact extraction tied to its fact rather than its lead."""

        facts = {fact.fact_id: fact for fact in investigation.facts}
        if any(
            facts.get(fact_id) is not None
            and facts[fact_id].predicate
            in {"source_record_matches", "provenance_matches"}
            for fact_id in task.fact_ids
        ):
            return True
        return False

    @staticmethod
    def _image_only_executable_tool_names(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> set[str]:
        """Expose reducer-approved retrieval or lead-inspection tool families."""

        core_id = investigation.core_verdict_fact_id
        if not core_id:
            return set()
        routes = remaining_material_routes(
            investigation,
            fact_id=core_id,
        )
        names: set[str] = set()
        for route in routes:
            parts = route.split(":", 2)
            tool_name = parts[0]
            route_task_id = (
                parts[2]
                if tool_name == "reverse_image_search" and len(parts) >= 3
                else parts[1]
                if len(parts) >= 2
                else ""
            )
            if route_task_id not in task_ids:
                continue
            names.add(tool_name)
        return names

    @staticmethod
    def _image_only_tool_argument_constraints(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> Dict[str, Dict[str, List[Any]]]:
        """Bind schemas to the reducer's actually executable route variants."""

        core_id = investigation.core_verdict_fact_id
        if not core_id:
            return {}
        branches: List[str] = []
        pages: List[str] = []
        references: List[str] = []
        for route in remaining_material_routes(
            investigation,
            fact_id=core_id,
        ):
            parts = route.split(":", 2)
            if len(parts) < 2:
                continue
            if parts[0] == "reverse_image_search" and len(parts) == 3:
                if parts[2] in task_ids:
                    branches.append(parts[1])
            elif parts[0] == "visit" and len(parts) == 3:
                if parts[1] in task_ids:
                    pages.append(parts[2])
            elif parts[0] == "compare_with_reference" and len(parts) == 3:
                if parts[1] in task_ids:
                    references.append(parts[2])
        constraints: Dict[str, Dict[str, List[Any]]] = {}
        if branches:
            constraints["reverse_image_search"] = {
                "branch": list(dict.fromkeys(branches))
            }
        if pages:
            constraints["visit"] = {
                "url": list(dict.fromkeys(pages))
            }
        if references:
            constraints["compare_with_reference"] = {
                "reference_url": list(dict.fromkeys(references))
            }
        return constraints

    @staticmethod
    def _image_only_discovery_route_error(
        investigation: ImageOnlyInvestigationState,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> str:
        """Keep one task from repeatedly searching before inspecting its leads."""

        task_id = str(
            tool_args.get("__question_id")
            or tool_args.get("question_id")
            or tool_args.get("task_id")
            or ""
        ).strip()
        if not task_id:
            return ""
        task = next(
            (
                item
                for item in investigation.tasks
                if item.task_id == task_id
            ),
            None,
        )
        if task is None:
            return f"Unknown image-only task {task_id!r}."
        allowed_tools = runtime_task_tool_names(investigation, task)
        if tool_name not in allowed_tools:
            return (
                f"Tool {tool_name!r} is not enabled for task {task_id!r}. "
                "Use one of that task's planned tools or select another "
                "active task."
            )
        if tool_name in {"check_consistency", "analyze_visual_anomalies"}:
            facts = {fact.fact_id: fact for fact in investigation.facts}
            if not any(
                facts.get(fact_id) is not None
                and facts[fact_id].predicate == "visual_integrity"
                for fact_id in task.fact_ids
            ):
                return (
                    f"Tool {tool_name!r} may inspect pixel integrity only. "
                    "Use external source evidence for identity, location, event, "
                    "date, distribution, habitat, or other depicted-world facts."
                )
        route_constraints = Orchestrator._image_only_tool_argument_constraints(
            investigation,
            task_ids={task_id},
        )
        if tool_name == "visit":
            requested = tool_args.get("url", [])
            requested_urls = (
                [requested]
                if isinstance(requested, str)
                else list(requested)
                if isinstance(requested, list)
                else []
            )
            pending_urls = {
                canonicalize_url(url)
                for url in route_constraints.get("visit", {}).get(
                    "url",
                    [],
                )
            }
            if (
                len(requested_urls) != 1
                or canonicalize_url(str(requested_urls[0])) not in pending_urls
            ):
                return (
                    f"Tool 'visit' must inspect one pending candidate page owned "
                    f"by task {task_id!r}."
                )
            return ""
        if tool_name == "compare_with_reference":
            reference_url = canonicalize_url(
                str(tool_args.get("reference_url", ""))
            )
            pending_urls = {
                canonicalize_url(url)
                for url in route_constraints.get(
                    "compare_with_reference",
                    {},
                ).get("reference_url", [])
            }
            if not reference_url or reference_url not in pending_urls:
                return (
                    "Tool 'compare_with_reference' must inspect a pending "
                    f"reference image owned by task {task_id!r}."
                )
            return ""
        if tool_name not in {"text_search", "reverse_image_search"}:
            return ""
        inspection_tools = {
            route.split(":", 1)[0]
            for route in remaining_material_routes(
                investigation,
                fact_id=investigation.core_verdict_fact_id or "",
            )
            if route.endswith(f":{task_id}")
            or f":{task_id}:" in route
        }
        if not inspection_tools.intersection(
            {"visit", "compare_with_reference"}
        ):
            return ""
        return (
            f"Task {task_id!r} already has uninspected candidate pages or "
            "reference images. Use visit or compare_with_reference for this "
            "task before another retrieval call, or choose a different active "
            "task."
        )

    @staticmethod
    def _image_only_fact_signature(
        investigation: ImageOnlyInvestigationState,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (fact.fact_id, fact.status)
                for fact in investigation.facts
            )
        )

    @staticmethod
    def _attempted_image_only_tool(route: str, tool_name: str) -> bool:
        try:
            parsed = json.loads(route)
        except (TypeError, ValueError):
            return False
        return str(parsed.get("tool", "")).strip() == tool_name

    @staticmethod
    def _image_only_step_succeeded(step: StageStep) -> bool:
        try:
            payload = json.loads(str(step.tool_result or ""))
        except (TypeError, ValueError):
            return False
        return (
            isinstance(payload, dict)
            and payload.get("status") == "success"
        )

    @staticmethod
    def _investigation_status(
        investigation: ImageOnlyInvestigationState,
    ) -> str:
        return {
            "coverage_complete": "complete",
            "verdict_determined": "complete",
            "information_saturated": "bounded_unresolved",
            "hard_budget_exhausted": "incomplete_budget_exhausted",
        }.get(investigation.stop_reason, "incomplete")

    @staticmethod
    def _verification_layers(
        investigation: ImageOnlyInvestigationState,
    ) -> Dict[str, Any]:
        source_facts = [
            fact
            for fact in investigation.facts
            if fact.predicate == "source_record_matches"
        ]
        integrity_facts = [
            fact
            for fact in investigation.facts
            if fact.predicate == "visual_integrity"
        ]
        if not source_facts and not integrity_facts:
            return {}

        def layer(
            facts: Sequence[Any],
            *,
            unresolved_reason: str,
        ) -> Dict[str, Any]:
            if not facts:
                return {
                    "status": "not_assessed",
                    "fact_ids": [],
                    "reason": unresolved_reason,
                }
            statuses = {fact.status for fact in facts}
            if "refuted" in statuses:
                status = "refuted"
            elif statuses == {"supported"}:
                status = "supported"
            elif "conflicted" in statuses:
                status = "conflicted"
            else:
                status = "unresolved"
            return {
                "status": status,
                "fact_ids": [fact.fact_id for fact in facts],
                "reason": (
                    "Resolved from qualified evidence."
                    if status in {"supported", "refuted"}
                    else unresolved_reason
                ),
            }

        return {
            "source_record_match": layer(
                source_facts,
                unresolved_reason=(
                    "The visible account, text, date, and thread relation were "
                    "not fully bound to an original or archived public record."
                ),
            ),
            "visible_integrity": layer(
                integrity_facts,
                unresolved_reason=(
                    "Visible manipulation or layout integrity was not resolved."
                ),
            ),
        }

    @staticmethod
    def _normalize_incomplete_judgment(
        investigation: ImageOnlyInvestigationState,
        judgment: ImageOnlyJudgment,
    ) -> ImageOnlyJudgment:
        if investigation.stop_reason != "hard_budget_exhausted":
            return judgment
        prefix = (
            "Investigation incomplete: the action budget ended before the "
            "remaining evidence routes were resolved. "
        )
        assessment = judgment.overall_assessment
        if not assessment.startswith("Investigation incomplete:"):
            assessment = prefix + assessment
        return judgment.model_copy(
            update={
                "confidence": min(judgment.confidence, 0.65),
                "overall_assessment": assessment[:2000],
            }
        )

    @staticmethod
    def _sync_image_only_state(
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
    ) -> None:
        state.investigation_state = investigation
        state.investigation_brief = investigation.brief
        state.visual_entities = list(investigation.entities)
        state.visual_facts = list(investigation.facts)
        state.research_tasks = list(investigation.tasks)
        state.findings = list(investigation.findings)
        state.retrieval_anchors = list(investigation.retrieval_anchors)

    @staticmethod
    def _require_successful_image_only_investigation(
        state: VerificationState,
    ) -> None:
        investigation_steps = [
            step
            for step in state.all_steps
            if getattr(step, "stage_name", "") == "image_only_investigation"
            and getattr(step, "action_type", "") == "tool_call"
        ]
        if not investigation_steps:
            raise RuntimeError(
                "image-only investigation completed without a real tool attempt"
            )
        if not any(
            Orchestrator._tool_step_succeeded(step)
            for step in investigation_steps
        ):
            raise RuntimeError(
                "every attempted image-only investigation tool call failed"
            )

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
            if (
                not report.scene_description
                and not report.entities
                and not report.text_regions
            ):
                raise RuntimeError(
                    "perception and OCR returned no visible scene, entity, or text"
                )
        finally:
            state.stage_timings["perception"] = round(time.time() - started, 2)
            self._record_stage_steps(state, steps)
        return report

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

    @staticmethod
    def _stage_thinking_level(stage_name: str) -> str:
        value = os.getenv(
            f"GEMINI_{stage_name}_THINKING_LEVEL",
            os.getenv("GEMINI_AGENT_THINKING_LEVEL", "minimal"),
        ).strip().lower()
        if value != "minimal":
            raise ValueError(
                f"GEMINI_{stage_name}_THINKING_LEVEL must be 'minimal' for the active agent."
            )
        return value

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
                result = await asyncio.wait_for(
                    tool.call_async(tool_args),
                    timeout=getattr(
                        self,
                        "tool_action_timeout_seconds",
                        150.0,
                    ),
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, tool.call, tool_args),
                    timeout=getattr(
                        self,
                        "tool_action_timeout_seconds",
                        150.0,
                    ),
                )
        except asyncio.TimeoutError:
            deadline = getattr(
                self,
                "tool_action_timeout_seconds",
                150.0,
            )
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"ToolActionTimeout: {tool_name} exceeded "
                        f"{deadline:.1f}s"
                    ),
                },
                ensure_ascii=False,
            )
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": "ToolActionTimeout",
                "tool_timeout_seconds": deadline,
                "tool_llm_api_calls": 0,
                "tool_tokens": StageRunner._normalize_tool_tokens(None),
            }
        except Exception as exc:
            runtime_metrics = getattr(exc, "_gemini_runtime_metrics", {})
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": type(exc).__name__,
                "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
                "tool_tokens": StageRunner._normalize_tool_tokens(
                    runtime_metrics.get("tokens")
                ),
            }

        runtime_metrics = take_runtime_metrics(result)
        try:
            serialized, succeeded = serialize_tool_result(result)
        except Exception as exc:
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            succeeded = False
            contract_exception = type(exc).__name__
        else:
            contract_exception = ""
        if succeeded and tool_name in self.cacheable_tools:
            self.tool_cache.put(tool_name, cache_args, serialized)
        return serialized, {
            "cache_hit": False,
            "tool_success": succeeded,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "serialized_size": len(serialized),
            "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
            "tool_tokens": StageRunner._normalize_tool_tokens(
                runtime_metrics.get("tokens")
            ),
            **({"tool_exception": contract_exception} if contract_exception else {}),
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
        thought_violation: Optional[StageStep] = None
        for step in steps:
            subcalls = step.metadata.get("tool_subcalls", [])
            if isinstance(subcalls, list):
                for subcall in subcalls:
                    if not isinstance(subcall, dict):
                        continue
                    request_count = max(
                        0,
                        int(subcall.get("request_count", 1) or 0),
                    )
                    state.total_tool_subcalls += request_count
                    kind = str(subcall.get("kind", "unknown")).strip() or "unknown"
                    state.tool_subcalls_by_kind[kind] = (
                        state.tool_subcalls_by_kind.get(kind, 0)
                        + request_count
                    )
            tool_tokens = step.metadata.get("tool_tokens", {})
            if not isinstance(tool_tokens, dict):
                tool_tokens = {}
            for name in ("prompt", "completion", "thought"):
                state.token_usage[name] += step.tokens.get(name, 0)
                state.token_usage[name] += int(tool_tokens.get(name, 0) or 0)
            total_thought = step.tokens.get("thought", 0) + int(
                tool_tokens.get("thought", 0) or 0
            )
            if int(step.metadata.get("tool_llm_api_calls", 0) or 0) > 0:
                step.metadata["total_tokens"] = {
                    name: step.tokens.get(name, 0)
                    + int(tool_tokens.get(name, 0) or 0)
                    for name in ("prompt", "completion", "thought")
                }
            if (
                self.provider == "gemini"
                and str(self.llm.wire_api).lower() == "interactions"
                and total_thought > 0
            ):
                thought_violation = thought_violation or step
        state.llm_api_calls += sum(
            1 for step in steps if step.metadata.get("llm_duration_ms") is not None
        )
        state.llm_api_calls += sum(
            int(step.metadata.get("tool_llm_api_calls", 0) or 0)
            for step in steps
        )
        if thought_violation is not None:
            raise RuntimeError(
                f"Gemini stage '{thought_violation.stage_name}' returned non-zero thought tokens "
                "despite the required minimal thinking policy."
            )

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
    def _safe_json_dict(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _check_timeout(self, started: float, state: VerificationState) -> None:
        if time.time() - started > self.timeout:
            state.termination = "timeout"
            raise TimeoutError(f"Pipeline exceeded timeout of {self.timeout} seconds.")

    @staticmethod
    def _runtime_timeout(env_name: str, default: float) -> float:
        raw = os.getenv(env_name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            value = default
        return max(5.0, value)
