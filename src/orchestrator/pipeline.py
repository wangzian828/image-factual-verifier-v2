# -*- coding: utf-8 -*-
"""VisualFact-driven image-only factual investigation orchestrator."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.integrations.gemini import take_runtime_metrics
from src.integrations.clock.system_clock import SystemClockClient
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.runtime_events import CaseRuntimeStore, current_case_runtime_store
from src.orchestrator.progress_control import (
    record_decision_progress,
)
from src.orchestrator.unified_prompts import (
    UNIFIED_DISCREPANCY_DECISION_PROMPT_VERSION,
    UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
    UNIFIED_JUDGMENT_PROMPT_VERSION,
    UNIFIED_JUDGMENT_SYSTEM_PROMPT,
    UNIFIED_REACT_PROMPT_VERSION,
    UNIFIED_REACT_SYSTEM_PROMPT,
    UNIFIED_REFLECTION_PROMPT_VERSION,
    UNIFIED_REFLECTION_SYSTEM_PROMPT,
)
from src.orchestrator.unified_context import (
    render_unified_discrepancy_decision_context,
    render_unified_judgment_context,
    render_unified_react_context,
    render_unified_reflection_context,
)
from src.orchestrator.investigation_models import (
    ClaimAssessmentProposal,
    DiscrepancyDecisionProposalOutput,
    DiscrepancyDecisionOutput,
    DiscrepancyJudgment,
    DiscrepancyJudgmentOutput,
    ImageOnlyInvestigationState,
    InvestigationSegmentOutput,
    UnifiedReflectionOutput,
    build_discrepancy_decision_output_schema,
)
from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.stage_runner import (
    InteractionSession,
    StageRunner,
    StageStep,
)
from src.orchestrator.tool_execution import run_tool_with_timeout
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
from src.orchestrator.tool_cache import (
    ToolResultCache,
    WEB_EVIDENCE_CONTRACT_VERSION,
)
from src.orchestrator.tool_health import require_tools, summarize_health
from src.orchestrator.tool_registry import (
    REQUIRED_TOOLS,
    build_all_tools_with_health,
)
from src.orchestrator.tool_result import parse_tool_result, serialize_tool_result
from src.orchestrator.unified_react import (
    UNIFIED_REACT_POLICY_VERSION,
    build_unified_react_tools,
    new_unified_react_state,
    reduce_unified_react_action,
    reduce_visual_bootstrap_action,
    unified_react_delta,
    validate_unified_react_action,
)
from src.orchestrator.task_store import (
    MAX_TOOL_ACTIONS,
    REFLECTION_INTERVAL,
    apply_discrepancy_decision,
    bind_discrepancy_decision_runtime_ids,
    claim_owned_visual_evidence_requirements,
    discrepancy_decision_checkpoint_reason,
    discrepancy_decision_evidence_ids,
    remaining_claim_hypothesis_routes,
)
from src.storage import default_tool_cache_dir


MAX_MODEL_TASK_CHOICES_PER_ACTION = 3


class Orchestrator:
    """VisualFact-driven image-only unified-ReAct orchestrator."""

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.7-flash",
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        llm_wire_api: Optional[str] = None,
        vlm_wire_api: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        image_access_mode: str = "direct_multimodal",
        timeout: float = 1800.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        sampling_seed: Optional[int] = None,
        validate_startup: bool = True,
        source_access_policy: Optional[SourceAccessPolicy] = None,
    ):
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.vlm_provider = (vlm_provider or self.provider).lower().strip()
        self.vlm_model = vlm_model or model_name
        self.llm_base_url = llm_base_url
        self.vlm_base_url = vlm_base_url
        self.image_access_mode = str(
            image_access_mode or "direct_multimodal"
        ).strip().lower()
        if self.image_access_mode not in {
            "direct_multimodal",
            "separate_vlm",
        }:
            raise ValueError(
                "image_access_mode must be 'direct_multimodal' or 'separate_vlm'"
            )
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
        self.sampling_seed = sampling_seed
        self._sampling_request_counts: Dict[str, int] = {}
        self.tool_action_timeout_seconds = self._runtime_timeout(
            "AGENT_TOOL_ACTION_TIMEOUT_SECONDS",
            150.0,
        )
        self.stage_request_timeout_seconds = self._runtime_timeout(
            "AGENT_STAGE_REQUEST_TIMEOUT_SECONDS",
            (
                900.0
                if self.provider in {"gemini", "qwen_local", "lmdeploy"}
                else 300.0
            ),
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
                os.getenv("PERCEPTION_CACHE_VERSION", "perception-v1"),
                os.getenv(
                    "OCR_CACHE_VERSION",
                    f"{os.getenv('OCR_BACKEND', 'baidu').strip().lower()}-v1",
                ),
                self.source_access_policy.cache_partition,
            ]
        )
        perception_cache_enabled = os.getenv(
            "PERCEPTION_CACHE_ENABLED",
            "1",
        ).strip().lower() in {"1", "true", "yes"}
        web_cache_enabled = os.getenv(
            "TOOL_CACHE_ENABLED",
            "0",
        ).strip().lower() in {"1", "true", "yes"}
        self.tool_cache = ToolResultCache(
            cache_dir=os.getenv("TOOL_CACHE_DIR", default_tool_cache_dir()),
            enabled=perception_cache_enabled or web_cache_enabled,
            ttl_seconds=float(os.getenv("TOOL_CACHE_TTL_SECONDS", "3600")),
            namespace=cache_namespace,
        )
        self.cacheable_tools = set()
        if perception_cache_enabled:
            self.cacheable_tools.update(
                {
                    "perceive_scene",
                    "ocr_with_position",
                }
            )
        if web_cache_enabled:
            self.cacheable_tools.update(
                {
                    "text_search",
                    "visit",
                    "crop_and_inspect",
                    "focused_visual_inspection",
                    "check_consistency",
                    "analyze_visual_anomalies",
                }
            )
        self.verification_tool_limits = {
            "current_time": 1,
            "ocr_with_position": 3,
            "reverse_image_search": 2,
            "text_search": 16,
            "visit": 16,
            "compare_with_reference": 6,
            "crop_and_inspect": 4,
            "focused_visual_inspection": 2,
            "check_consistency": 3,
            "analyze_visual_anomalies": 3,
        }

        self.llm = APIBackend(
            provider=self.provider,
            model_name=model_name,
            base_url=self.llm_base_url,
            wire_api=self.llm_wire_api,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(
                os.getenv(
                    "AGENT_LLM_REQUEST_TIMEOUT_SECONDS",
                    (
                        "90"
                        if self.provider == "gemini"
                        else "900"
                        if self.provider in {"qwen_local", "lmdeploy"}
                        else "300"
                    ),
                )
            ),
            max_retries=int(
                os.getenv(
                    "AGENT_LLM_REQUEST_MAX_RETRIES",
                    "12" if self.provider == "gemini" else "0",
                )
            ),
        )
        self.all_tools, self.tool_health = build_all_tools_with_health(
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            vlm_wire_api=self.vlm_wire_api,
            vlm_base_url=self.vlm_base_url,
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

    async def aclose(self) -> None:
        """Close per-rollout transports and helper runtimes.

        A batch uses one ``Orchestrator`` per isolated rollout.  Several tool
        wrappers intentionally share a persistent VLM client, Jina reader, and
        API backend within that rollout.  They are referenced from more than
        one wrapper, so collect resources by identity and close each exactly
        once when the rollout completes.
        """

        child_attributes = (
            "client",
            "vlm_client",
            "vlm_backend",
            "browse_client",
            "lens_client",
            "image_search_client",
            "visual_search_client",
            "candidate_reranker",
            "upload_client",
            "serper_lens_client",
            "zhipu_client",
            "baidu_client",
        )
        pending: list[Any] = [getattr(self, "llm", None)]
        pending.extend(getattr(self, "all_tools", {}).values())
        resources: list[Any] = []
        seen: set[int] = set()
        while pending:
            resource = pending.pop()
            if resource is None:
                continue
            resource_id = id(resource)
            if resource_id in seen:
                continue
            seen.add(resource_id)
            resources.append(resource)
            pending.extend(
                getattr(resource, name, None) for name in child_attributes
            )

        for resource in resources:
            async_close = getattr(resource, "aclose", None)
            sync_close = getattr(resource, "close", None)
            try:
                if callable(async_close):
                    result = async_close()
                    if inspect.isawaitable(result):
                        await result
                elif callable(sync_close):
                    await asyncio.to_thread(sync_close)
            except Exception:
                # Cleanup must not replace a completed rollout's actual result.
                # The resource's own atexit guard remains a final best-effort
                # fallback if a provider transport refuses to close here.
                continue

    def _validate_startup_configuration(self) -> None:
        tool_thinking_levels = {
            "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL": os.getenv(
                "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL", "low"
            ),
            "GEMINI_BROWSE_THINKING_LEVEL": os.getenv(
                "GEMINI_BROWSE_THINKING_LEVEL", "low"
            ),
            "GEMINI_VISION_THINKING_LEVEL": os.getenv(
                "GEMINI_VISION_THINKING_LEVEL", "low"
            ),
            "GEMINI_REFERENCE_COMPARE_THINKING_LEVEL": os.getenv(
                "GEMINI_REFERENCE_COMPARE_THINKING_LEVEL", "low"
            ),
            "GEMINI_VISUAL_ANOMALY_THINKING_LEVEL": os.getenv(
                "GEMINI_VISUAL_ANOMALY_THINKING_LEVEL", "low"
            ),
        }
        for env_name, value in tool_thinking_levels.items():
            normalized_value = value.strip().lower()
            if normalized_value == "minimal":
                normalized_value = "low"
            if normalized_value != "low":
                raise ValueError(
                    f"{env_name} must be 'low' for the active agent."
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

    def _uses_separate_vlm(self) -> bool:
        return getattr(self, "image_access_mode", "direct_multimodal") == (
            "separate_vlm"
        )

    def _main_llm_attaches_image(self) -> bool:
        return not self._uses_separate_vlm()

    async def run(
        self,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        *,
        episode_id: Optional[str] = None,
        decision_policy_version: str = UNIFIED_REACT_POLICY_VERSION,
        runtime_store: Optional[CaseRuntimeStore] = None,
    ) -> Dict[str, Any]:
        """Run one explicit image-only decision policy without hidden fallback."""

        verify_case_image(runtime_case, image_path)
        runtime_store = runtime_store or current_case_runtime_store()
        state = VerificationState(
            image_path=image_path,
            image_id=episode_id or runtime_case.case_id,
            runtime_case=runtime_case,
            input_mode="image_only",
            decision_policy_version=decision_policy_version,
            tool_health=self.tool_health_summary,
            runtime_store=runtime_store,
        )
        self.last_state = state
        started = time.time()
        try:
            if decision_policy_version != UNIFIED_REACT_POLICY_VERSION:
                raise RuntimeError(
                    "unsupported decision_policy_version="
                    f"{decision_policy_version!r}; only "
                    f"{UNIFIED_REACT_POLICY_VERSION!r} is supported"
                )
            (
                investigation,
                judgment,
                basis,
                final_visual_audit,
            ) = await self._run_unified_react_policy(
                state,
                image_path=image_path,
                runtime_case=runtime_case,
            )
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
            "case_id": runtime_case.case_id,
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
            "final_visual_audit": final_visual_audit,
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

    async def _run_unified_react_policy(
        self,
        state: VerificationState,
        *,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
    ) -> tuple[
        ImageOnlyInvestigationState,
        DiscrepancyJudgment,
        Any,
        Optional[Dict[str, Any]],
    ]:
        """Run the production unified-react-v1 path from an empty workspace."""

        self._validate_image_only_bootstrap_configuration()
        investigation = new_unified_react_state(runtime_case)
        state.investigation_state = investigation
        state.perception = PerceptionReport(scene_description="")
        self._sync_image_only_state(state, investigation)
        started = time.time()
        interaction_session = InteractionSession()
        last_reflection_action = 0

        try:
            while not investigation.stop_reason:
                self._check_timeout(started, state)
                if investigation.action_count >= MAX_TOOL_ACTIONS and investigation.target_facts:
                    await self._run_unified_discrepancy_decision(
                        state,
                        investigation,
                        trigger="before_unresolved",
                    )
                    audit_discrepancy_coverage(
                        investigation,
                        decision_checkpoint=True,
                    )
                    break

                tools = build_unified_react_tools(investigation, self.all_tools)
                if not tools:
                    if not investigation.target_facts:
                        raise RuntimeError(
                            "unified ReAct has no available tool before the "
                            "initial investigation action"
                        )
                    await self._run_unified_discrepancy_decision(
                        state,
                        investigation,
                        trigger="before_unresolved",
                    )
                    audit_discrepancy_coverage(
                        investigation,
                        decision_checkpoint=True,
                    )
                    if not investigation.stop_reason:
                        investigation.stop_reason = "information_saturated"
                        audit_discrepancy_coverage(
                            investigation,
                            decision_checkpoint=True,
                        )
                    break

                observation_update: Dict[str, Any] = {}

                def observation_callback(
                    step: StageStep,
                    _steps: List[StageStep],
                ) -> Dict[str, Any]:
                    nonlocal observation_update
                    tool_name = str(step.tool_name).strip()
                    if tool_name == "perceive_scene":
                        scene = self._parse_perception_result(step.tool_result)
                        prior = state.perception or PerceptionReport(
                            scene_description=""
                        )
                        state.perception = PerceptionReport(
                            entities=scene.entities,
                            text_regions=list(prior.text_regions),
                            scene_description=scene.scene_description,
                            image_type=scene.image_type,
                        )
                        update = reduce_visual_bootstrap_action(
                            investigation,
                            step=step,
                            runtime_case=runtime_case,
                            perception=state.perception,
                        )
                    elif tool_name == "ocr_with_position" and not (
                        set(investigation.unified_react_bootstrap_tools_completed)
                        == {"perceive_scene", "ocr_with_position"}
                    ):
                        state.perception = self._merge_ocr(
                            state.perception or PerceptionReport(
                                scene_description=""
                            ),
                            step.tool_result,
                        )
                        update = reduce_visual_bootstrap_action(
                            investigation,
                            step=step,
                            runtime_case=runtime_case,
                            perception=state.perception,
                        )
                    else:
                        update = reduce_unified_react_action(
                            investigation,
                            step=step,
                            runtime_case=runtime_case,
                            source_access_policy=self.source_access_policy,
                        )
                    if not update.get("accepted", False):
                        raise RuntimeError(
                            "unified ReAct reducer rejected an executed action: "
                            + str(update.get("rejected_reason", update))
                        )
                    delta = unified_react_delta(step=step, update=update)
                    step.metadata["unified_react_delta"] = delta
                    step.metadata["investigation_state_update"] = delta
                    observation_update = update
                    self._sync_image_only_state(state, investigation)
                    return delta

                runner = StageRunner(
                    llm=self.llm,
                    system_prompt=self._sp(UNIFIED_REACT_SYSTEM_PROMPT),
                    prompt_version=UNIFIED_REACT_PROMPT_VERSION,
                    tools=tools,
                    output_schema=InvestigationSegmentOutput,
                    max_rounds=1,
                    image_path=image_path,
                    stage_name="unified_react",
                    runtime_store=state.runtime_store,
                    handoff_state=investigation,
                    attach_image=False,
                    prior_steps=[
                        step
                        for step in state.all_steps
                        if getattr(step, "stage_name", "") == "unified_react"
                    ],
                    tool_cache=self.tool_cache,
                    cacheable_tools=list(self.cacheable_tools),
                    tool_call_limits=self.verification_tool_limits,
                    min_tool_calls=1,
                    should_stop=lambda steps: any(
                        item.action_type == "tool_call" for item in steps
                    ),
                    max_output_tokens=self._stage_output_tokens(
                        "UNIFIED_REACT",
                        8192,
                    ),
                    generation_config=self._stage_generation_config(
                        "UNIFIED_REACT"
                    ),
                    observation_callback=observation_callback,
                    source_access_policy=self.source_access_policy,
                    visual_call_validator=lambda tool_name, tool_args: (
                        validate_unified_react_action(
                            investigation,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            source_access_policy=self.source_access_policy,
                        )
                    ),
                    max_protocol_corrections=4,
                    max_tool_calls_per_turn=1,
                    force_tool_each_round=True,
                    protocol_exhaustion_boundary=True,
                    interaction_session=interaction_session,
                    stop_output_factory=lambda: InvestigationSegmentOutput(
                        segment_summary=(
                            "The unified-ReAct action selection correction "
                            "budget was exhausted."
                        ),
                        ready_for_reflection=True,
                    ),
                    request_timeout_seconds=self.stage_request_timeout_seconds,
                    tool_timeout_seconds=self.tool_action_timeout_seconds,
                )
                try:
                    parsed, steps = await runner.run(
                        render_unified_react_context(investigation)
                    )
                except Exception as exc:
                    self._record_stage_steps(
                        state,
                        list(getattr(exc, "stage_steps", []) or []),
                    )
                    raise
                self._record_stage_steps(state, steps)
                if parsed is None or not observation_update:
                    raise RuntimeError(
                        "unified ReAct did not complete one accepted action"
                    )
                if investigation.stop_reason == "engineering_error":
                    if observation_update.get("fatal_engineering_error"):
                        raise RuntimeError(
                            "unified ReAct encountered a fatal tool result "
                            "contract error"
                        )
                    raise RuntimeError("a required visual bootstrap tool failed")
                if not investigation.target_facts:
                    continue

                decision_trigger = discrepancy_decision_checkpoint_reason(
                    investigation,
                    update=observation_update,
                )
                if not decision_trigger and not remaining_claim_hypothesis_routes(
                    investigation
                ):
                    decision_trigger = "before_unresolved"
                if decision_trigger:
                    await self._run_unified_discrepancy_decision(
                        state,
                        investigation,
                        trigger=decision_trigger,
                    )
                    audit_discrepancy_coverage(
                        investigation,
                        decision_checkpoint=True,
                    )
                else:
                    audit_discrepancy_coverage(investigation)

                if investigation.stop_reason:
                    break
                if (
                    investigation.action_count
                    and investigation.action_count % REFLECTION_INTERVAL == 0
                    and investigation.action_count != last_reflection_action
                ):
                    await self._run_unified_global_reflection(
                        state,
                        investigation,
                    )
                    last_reflection_action = investigation.action_count
                    self._sync_image_only_state(state, investigation)
        finally:
            state.stage_timings["unified_react"] = round(
                time.time() - started,
                2,
            )

        if not investigation.target_facts:
            raise RuntimeError(
                "unified ReAct completed visual bootstrap without an "
                "investigation target"
            )
        if not investigation.stop_reason:
            await self._run_unified_discrepancy_decision(
                state,
                investigation,
                trigger="before_unresolved",
            )
            audit_discrepancy_coverage(
                investigation,
                decision_checkpoint=True,
            )
        if not investigation.stop_reason:
            investigation.stop_reason = "information_saturated"
            audit_discrepancy_coverage(
                investigation,
                decision_checkpoint=True,
            )
        compiled_verdict, basis = compile_discrepancy_verdict_basis(
            investigation,
            policy_rule_id=UNIFIED_REACT_POLICY_VERSION,
        )
        judgment_started = time.perf_counter()
        try:
            judgment = await self._run_unified_judgment(
                state,
                investigation,
                compiled_verdict,
                basis,
                image_path=image_path,
                final_visual_audit=None,
                interaction_session=None,
                attach_image=False,
                policy_rule_id=UNIFIED_REACT_POLICY_VERSION,
                system_prompt=UNIFIED_JUDGMENT_SYSTEM_PROMPT,
                prompt_version=UNIFIED_JUDGMENT_PROMPT_VERSION,
            )
        finally:
            state.stage_timings["judgment"] = round(
                time.perf_counter() - judgment_started,
                2,
            )
        investigation.discrepancy_judgment = judgment
        state.judgment = judgment
        self._sync_image_only_state(state, investigation)
        return investigation, judgment, basis, None

    async def _run_unified_global_reflection(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
    ) -> UnifiedReflectionOutput:
        """Record one low-frequency global strategy review without route mutation."""

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(UNIFIED_REFLECTION_SYSTEM_PROMPT),
            prompt_version=UNIFIED_REFLECTION_PROMPT_VERSION,
            tools=[],
            output_schema=UnifiedReflectionOutput,
            max_rounds=2,
            stage_name="unified_reflection",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            max_output_tokens=self._stage_output_tokens(
                "UNIFIED_REFLECTION",
                8192,
            ),
            generation_config=self._stage_generation_config(
                "UNIFIED_REFLECTION"
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
        )
        parsed, steps = await runner.run(
            render_unified_reflection_context(investigation)
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "unified global Reflection did not produce valid structured output"
            )
        return parsed

    async def _run_unified_discrepancy_decision(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        trigger: str,
    ) -> Dict[str, Any]:
        """Run the sparse semantic checkpoint retained by unified ReAct."""

        return await self._run_unified_discrepancy_decision_impl(
            state,
            investigation,
            reviewed_evidence_ids=discrepancy_decision_evidence_ids(
                investigation
            ),
            trigger=trigger,
            interaction_session=None,
            stage_name="unified_discrepancy_decision",
            system_prompt=UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
            prompt_version=UNIFIED_DISCREPANCY_DECISION_PROMPT_VERSION,
            allow_new_hypotheses=False,
        )

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

    async def _run_unified_discrepancy_decision_impl(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        reviewed_evidence_ids: Sequence[str],
        trigger: str,
        interaction_session: InteractionSession,
        stage_name: str = "unified_discrepancy_decision",
        system_prompt: str = UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
        prompt_version: str = "",
        allow_new_hypotheses: bool = False,
    ) -> Dict[str, Any]:
        """Run and atomically apply the unified sparse semantic checkpoint."""

        before_signature = self._discrepancy_progress_signature(investigation)
        decision_output_schema = build_discrepancy_decision_output_schema(
            claim_ids=[
                claim.claim_id for claim in investigation.target_facts
            ],
            evidence_ids=list(reviewed_evidence_ids),
            hypothesis_ids=[
                hypothesis.hypothesis_id
                for hypothesis in investigation.search_hypotheses
            ],
            allow_new_hypotheses=allow_new_hypotheses,
        )

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(system_prompt),
            tools=[],
            output_schema=decision_output_schema,
            max_rounds=3,
            stage_name=stage_name,
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_discrepancy_decision(
                    investigation,
                    parsed,
                    reviewed_evidence_ids=reviewed_evidence_ids,
                    trigger=trigger,
                    source_access_policy=self.source_access_policy,
                    allow_new_hypotheses=allow_new_hypotheses,
                )
            ),
            max_output_tokens=self._stage_output_tokens(
                "UNIFIED_DISCREPANCY_DECISION",
                8192,
            ),
            generation_config=self._stage_generation_config(
                "UNIFIED_DISCREPANCY_DECISION"
            ),
            protocol_exhaustion_boundary=True,
            stop_output_factory=lambda: decision_output_schema(
                verdict_proposal="continue",
                rationale=(
                    "No atomic Decision update was accepted; continue with the "
                    "recorded workspace and unresolved gaps."
                ),
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
            prompt_version=prompt_version or f"{stage_name}-v1",
        )
        parsed, steps = await runner.run(
            render_unified_discrepancy_decision_context(
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
                trigger=trigger,
                allow_new_hypotheses=allow_new_hypotheses,
            )
        )
        self._record_stage_steps(state, steps)
        exhaustion_reason = self._protocol_exhaustion_rejection_reason(steps)
        if exhaustion_reason:
            fallback_update = self._apply_discrepancy_decision_fallback(
                state,
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
                rejected_reason=exhaustion_reason,
                before_signature=before_signature,
                trigger=trigger,
                exhaustion_boundary=True,
            )
            if fallback_update is not None:
                return fallback_update
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Discrepancy Decision exhausted protocol correction budget: "
                + exhaustion_reason
            )
        if parsed is None:
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Discrepancy Decision did not produce a valid atomic update"
            )
        bound, binding_error = bind_discrepancy_decision_runtime_ids(
            investigation,
            parsed,
            reviewed_evidence_ids=reviewed_evidence_ids,
        )
        if bound is None:
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Discrepancy Decision lost its runtime ID binding: "
                + binding_error
            )
        update = apply_discrepancy_decision(
            investigation,
            bound,
            reviewed_evidence_ids=reviewed_evidence_ids,
            trigger=trigger,
            source_access_policy=self.source_access_policy,
        )
        if not update.get("accepted", False):
            rejected_reason = str(
                update.get("rejected_reason", "unknown validation error")
            )
            fallback_update = self._apply_discrepancy_decision_fallback(
                state,
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
                rejected_reason=rejected_reason,
                before_signature=before_signature,
                trigger=trigger,
                exhaustion_boundary=False,
            )
            if fallback_update is not None:
                return fallback_update
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Discrepancy Decision failed deterministic application: "
                + rejected_reason
            )
        progress = (
            record_decision_progress(investigation, update)
            if self._discrepancy_progress_signature(investigation)
            != before_signature
            else None
        )
        if progress is not None:
            update["progress"] = progress.model_dump(mode="json")
        self._sync_image_only_state(state, investigation)
        return update

    @staticmethod
    def _protocol_exhaustion_rejection_reason(
        steps: Sequence[StageStep],
    ) -> str:
        if not any(
            bool(step.metadata.get("protocol_correction_exhaustion_boundary"))
            for step in steps
        ):
            return ""
        for step in reversed(steps):
            reason = str(step.metadata.get("rejection_reason", "")).strip()
            if reason:
                return reason
        return "protocol correction budget exhausted before an accepted Decision"

    def _apply_discrepancy_decision_fallback(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        reviewed_evidence_ids: Sequence[str],
        rejected_reason: str,
        before_signature: tuple[Any, ...],
        trigger: str,
        exhaustion_boundary: bool,
    ) -> Optional[Dict[str, Any]]:
        fallback = self._resolved_visual_consumption_fallback(
            investigation,
            reviewed_evidence_ids=reviewed_evidence_ids,
            rejected_reason=rejected_reason,
        )
        fallback_kind = "visual_consumption"
        if fallback is None and exhaustion_boundary:
            fallback = self._decision_correction_exhaustion_fallback(
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
                rejected_reason=rejected_reason,
            )
            fallback_kind = "decision_correction_exhaustion"
        if fallback is None and exhaustion_boundary:
            fallback = DiscrepancyDecisionOutput(
                verdict_proposal="continue",
                rationale=(
                    "Deterministic correction-exhaustion boundary: preserve the "
                    "current workspace and continue without applying an invalid "
                    "Decision update."
                ),
            )
            fallback_kind = "empty_continue"
        if fallback is None:
            return None

        update = apply_discrepancy_decision(
            investigation,
            fallback,
            reviewed_evidence_ids=reviewed_evidence_ids,
            trigger=trigger,
            allow_empty_continue=fallback_kind == "empty_continue",
        )
        if not update.get("accepted", False):
            return None
        if fallback_kind == "visual_consumption":
            update["deterministic_visual_consumption_fallback"] = True
        elif fallback_kind == "decision_correction_exhaustion":
            update["deterministic_decision_exhaustion_fallback"] = True
        else:
            update["deterministic_empty_continue_fallback"] = True
        update["fallback_rejected_reason"] = rejected_reason
        if exhaustion_boundary:
            update["protocol_correction_exhaustion_boundary"] = True
        progress = (
            record_decision_progress(investigation, update)
            if self._discrepancy_progress_signature(investigation)
            != before_signature
            else None
        )
        if progress is not None:
            update["progress"] = progress.model_dump(mode="json")
        self._sync_image_only_state(state, investigation)
        return update

    @staticmethod
    def _resolved_visual_consumption_fallback(
        investigation: ImageOnlyInvestigationState,
        *,
        reviewed_evidence_ids: Sequence[str],
        rejected_reason: str,
    ) -> DiscrepancyDecisionOutput | None:
        """Conservatively consume resolved pixel Evidence after correction exhaustion.

        This fallback is intentionally non-substantive: it never supports, refutes,
        creates a discrepancy, retires routes, or proposes a terminal verdict. It
        only records ``insufficient`` assessments with the exact claim-owned pixel
        Evidence IDs after the model repeatedly failed the same provenance
        requirement. That keeps the visual observations auditable without silently
        downgrading to a source-only Decision.
        """

        if "claim-owned visual Evidence" not in rejected_reason and (
            "resolved focused visual Evidence" not in rejected_reason
        ):
            return None
        requirements = claim_owned_visual_evidence_requirements(
            investigation,
            reviewed_evidence_ids=reviewed_evidence_ids,
        )
        visual_evidence_ids_by_claim: Dict[str, List[str]] = {}
        for requirement in requirements:
            visual_evidence_id = str(requirement["evidence_id"])
            for claim_id in requirement["claim_ids"]:
                selected_ids = visual_evidence_ids_by_claim.setdefault(
                    claim_id,
                    [],
                )
                if visual_evidence_id not in selected_ids:
                    selected_ids.append(visual_evidence_id)
        assessments: List[ClaimAssessmentProposal] = []
        for claim_id, evidence_ids in visual_evidence_ids_by_claim.items():
            # The bounded visual-tool budget keeps this under the model field
            # limit. Fail closed if a future tool change exceeds it, rather
            # than quietly dropping a claim-owned pixel record.
            if len(evidence_ids) > 20:
                return None
            assessments.append(
                ClaimAssessmentProposal(
                    claim_id=claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=evidence_ids,
                    remaining_gap=(
                        "Claim-owned pixel Evidence was recorded, but no "
                        "accepted semantic support/refute/discrepancy update "
                        "survived runtime validation"
                    ),
                    rationale=(
                        "Deterministic fallback after correction exhaustion: "
                        "consume all claim-owned pixel Evidence conservatively "
                        "without changing the Claim to supported or refuted."
                    ),
                )
            )
            if len(assessments) >= 3:
                break
        if not assessments:
            return None
        return DiscrepancyDecisionOutput(
            claim_assessments=assessments,
            verdict_proposal="continue",
            rationale=(
                "Conservative runtime fallback consumed claim-owned visual "
                "Evidence after model correction exhaustion; no terminal verdict "
                "or discrepancy was inferred."
            ),
        )

    @staticmethod
    def _decision_correction_exhaustion_fallback(
        investigation: ImageOnlyInvestigationState,
        *,
        reviewed_evidence_ids: Sequence[str],
        rejected_reason: str,
    ) -> DiscrepancyDecisionOutput | None:
        """Record reviewed Evidence conservatively after Decision retry exhaustion.

        This fallback is intentionally non-substantive. It never supports or
        refutes a Claim, creates a discrepancy, changes routes, or proposes a
        terminal verdict. It only prevents a correction-exhausted Decision
        checkpoint from being recorded as a successful empty no-op.
        """

        if (
            "must request targeted visual_reinspection before semantically "
            "using source Evidence"
            in rejected_reason
        ):
            # A source-to-pixel gate is an executable mechanism requirement,
            # not an optional semantic update. Recording source Evidence as an
            # ordinary insufficient assessment here would silently skip the
            # required tool transition, so fail closed at the caller instead.
            return None
        reviewed = list(dict.fromkeys(str(item) for item in reviewed_evidence_ids))
        if not reviewed:
            return None
        evidence_by_id = {
            item.evidence_id: item for item in investigation.evidence
        }
        claim_ids_by_fact: Dict[str, List[str]] = {}
        for claim in investigation.target_facts:
            claim_ids_by_fact.setdefault(claim.fact_id, []).append(claim.claim_id)
        claim_evidence_ids: Dict[str, List[str]] = {}
        for evidence_id in reviewed:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            for fact_id in evidence.fact_ids:
                for claim_id in claim_ids_by_fact.get(fact_id, []):
                    claim_evidence_ids.setdefault(claim_id, []).append(evidence_id)

        reason = " ".join(str(rejected_reason).split())
        if len(reason) > 420:
            reason = reason[:417].rstrip() + "..."
        assessments: List[ClaimAssessmentProposal] = []
        for claim in investigation.target_facts:
            evidence_ids = list(
                dict.fromkeys(claim_evidence_ids.get(claim.claim_id, []))
            )
            if not evidence_ids:
                continue
            assessments.append(
                ClaimAssessmentProposal(
                    claim_id=claim.claim_id,
                    assessment="insufficient",
                    selected_evidence_ids=evidence_ids[:20],
                    remaining_gap=(
                        "Discrepancy Decision correction budget was exhausted "
                        "before an accepted semantic support/refute/discrepancy "
                        "update; reviewed Evidence is recorded for follow-up."
                    ),
                    rationale=(
                        "Deterministic fallback after Decision correction "
                        "exhaustion: preserve reviewed Evidence without "
                        "inferring support, refutation, discrepancy, or verdict. "
                        f"Last validator feedback: {reason}"
                    ),
                )
            )
            if len(assessments) >= 3:
                break
        if not assessments:
            return None
        return DiscrepancyDecisionOutput(
            claim_assessments=assessments,
            verdict_proposal="continue",
            rationale=(
                "Conservative runtime fallback recorded reviewed Evidence after "
                "Decision correction exhaustion; no terminal verdict or "
                "discrepancy was inferred."
            ),
        )

    @staticmethod
    def _discrepancy_progress_signature(
        investigation: ImageOnlyInvestigationState,
    ) -> tuple[Any, ...]:
        latest = {}
        for item in investigation.claim_assessments:
            latest[item.claim_id] = (
                item.assessment,
                tuple(item.evidence_ids),
                item.remaining_gap,
            )
        return (
            tuple(sorted(latest.items())),
            tuple(
                (
                    item.discrepancy_id,
                    item.status,
                    item.materiality,
                    tuple(item.evidence_ids),
                )
                for item in investigation.material_discrepancies
            ),
            tuple(
                (item.hypothesis_id, item.status)
                for item in investigation.search_hypotheses
            ),
            tuple(
                (item.visual_question_id, item.status)
                for item in investigation.visual_reinspections
            ),
            investigation.proposed_verdict,
        )

    async def _run_unified_judgment(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        compiled_verdict: str,
        basis: Any,
        *,
        image_path: str,
        final_visual_audit: Optional[Dict[str, Any]] = None,
        interaction_session: InteractionSession,
        attach_image: Optional[bool] = None,
        policy_rule_id: str = UNIFIED_REACT_POLICY_VERSION,
        system_prompt: str = UNIFIED_JUDGMENT_SYSTEM_PROMPT,
        prompt_version: str = UNIFIED_JUDGMENT_PROMPT_VERSION,
    ) -> DiscrepancyJudgment:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(system_prompt),
            tools=[],
            output_schema=DiscrepancyJudgmentOutput,
            max_rounds=1,
            image_path=image_path,
            stage_name="unified_judgment",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=(
                bool(image_path) and self._main_llm_attaches_image()
                if attach_image is None
                else bool(attach_image)
            ),
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_discrepancy_judgment(
                    parsed,
                    compiled_verdict=compiled_verdict,
                    basis=basis,
                )
            ),
            max_output_tokens=self._stage_output_tokens(
                "UNIFIED_JUDGMENT",
                8192,
            ),
            generation_config=self._stage_generation_config(
                "UNIFIED_JUDGMENT"
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
            prompt_version=prompt_version or f"{policy_rule_id}-judgment-v1",
        )
        parsed, steps = await runner.run(
            render_unified_judgment_context(
                investigation,
                compiled_verdict,
                basis,
                final_visual_audit=final_visual_audit,
                image_is_attached=(
                    bool(image_path) and self._main_llm_attaches_image()
                    if attach_image is None
                    else bool(attach_image)
                ),
            )
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "unified Judgment did not produce a valid binary judgment"
            )
        # IDs and unresolved gaps are runtime-owned.  Reconstruct the canonical
        # record from the compiled basis instead of asking the model to copy a
        # large, error-prone identifier list.
        return DiscrepancyJudgment(
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            policy_rule_id=policy_rule_id,
            overall_assessment=parsed.overall_assessment,
            selected_claim_ids=list(basis.claim_ids),
            selected_discrepancy_ids=list(basis.discrepancy_ids),
            selected_visual_anchor_fact_ids=list(basis.visual_anchor_fact_ids),
            selected_finding_ids=list(basis.finding_ids),
            selected_evidence_ids=list(basis.evidence_ids),
            unresolved_gaps=list(basis.unresolved_gaps),
            terminal_visual_rationale=parsed.terminal_visual_rationale,
        )

    @staticmethod
    def _validate_discrepancy_decision(
        investigation: ImageOnlyInvestigationState,
        parsed: DiscrepancyDecisionProposalOutput,
        *,
        reviewed_evidence_ids: Sequence[str],
        trigger: str,
        source_access_policy: Optional[SourceAccessPolicy] = None,
        allow_new_hypotheses: bool = True,
    ) -> tuple[bool, str]:
        if not allow_new_hypotheses and parsed.new_hypotheses:
            return False, (
                "unified-react-v1 Discrepancy Decision must not create "
                "new_hypotheses"
            )
        bound, binding_error = bind_discrepancy_decision_runtime_ids(
            investigation,
            parsed,
            reviewed_evidence_ids=reviewed_evidence_ids,
        )
        if bound is None:
            return False, binding_error
        candidate = investigation.model_copy(deep=True)
        update = apply_discrepancy_decision(
            candidate,
            bound,
            reviewed_evidence_ids=reviewed_evidence_ids,
            trigger=trigger,
            source_access_policy=source_access_policy,
        )
        if not update.get("accepted", False):
            return False, str(
                update.get(
                    "rejected_reason",
                    "Discrepancy Decision proposed no valid state update",
                )
            )
        return True, ""

    @staticmethod
    def _validate_discrepancy_judgment(
        parsed: DiscrepancyJudgmentOutput,
        *,
        compiled_verdict: str,
        basis: Any,
    ) -> tuple[bool, str]:
        if compiled_verdict and parsed.verdict != compiled_verdict:
            return False, (
                f"verdict must be {compiled_verdict}, received {parsed.verdict}"
            )
        if not compiled_verdict:
            rationale = parsed.terminal_visual_rationale
            if rationale is None:
                return False, (
                    "bounded binary judgment requires terminal_visual_rationale"
                )
            expected_relation = f"supports_{parsed.verdict}"
            if rationale.relation_to_verdict != expected_relation:
                return False, (
                    "terminal_visual_rationale relation_to_verdict must match "
                    f"the binary verdict ({expected_relation})"
                )
        return True, ""

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
        if state.runtime_store is not None:
            state.runtime_store.write_snapshot(
                "workspace",
                {
                    "action_count": investigation.action_count,
                    "stop_reason": investigation.stop_reason,
                    "investigation_state": investigation.model_dump(mode="json"),
                    "token_usage": state.token_usage,
                    "total_tool_calls": state.total_tool_calls,
                },
            )

    def _stage_output_tokens(self, stage_name: str, default: int) -> int:
        normalized_stage = stage_name.strip().upper()
        if self.provider in {"qwen_local", "lmdeploy"}:
            env_name = f"QWEN_{normalized_stage}_MAX_OUTPUT_TOKENS"
            provider_default = default
        else:
            env_name = f"GEMINI_{normalized_stage}_MAX_OUTPUT_TOKENS"
            provider_default = default
        value = os.getenv(env_name, str(provider_default)).strip()
        try:
            tokens = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{env_name} must be an integer."
            ) from exc
        if tokens < 1:
            raise ValueError(f"{env_name} must be positive.")
        return tokens

    @staticmethod
    def _stage_thinking_level(stage_name: str) -> str:
        normalized_stage = stage_name.strip().upper()
        default = (
            "high"
            if normalized_stage == "UNIFIED_REACT"
            else os.getenv("GEMINI_AGENT_THINKING_LEVEL", "low")
        )
        value = os.getenv(
            f"GEMINI_{normalized_stage}_THINKING_LEVEL",
            default,
        ).strip().lower()
        if value == "minimal":
            value = "low"
        allowed = {"low", "high"}
        if value not in allowed:
            raise ValueError(
                f"GEMINI_{normalized_stage}_THINKING_LEVEL must be one of "
                + ", ".join(sorted(allowed))
                + "."
            )
        return value

    def _stage_generation_config(self, stage_name: str) -> Dict[str, Any]:
        """Map one stage's reasoning policy to the active serving protocol."""

        normalized_stage = stage_name.strip().upper()
        if self.provider in {"qwen_local", "lmdeploy"}:
            # Qwen3.5 uses one checkpoint for both thinking and direct-response
            # modes. Keep deliberation at semantic checkpoints and use direct
            # responses for frequent tool routing. Older Qwen3-VL behavior is
            # retained only for its explicit diagnostic profile.
            qwen35 = "qwen3.5" in str(
                getattr(self, "model_name", "")
            ).lower()
            if qwen35:
                thinking_stages = {
                    "UNIFIED_REACT",
                    "UNIFIED_REFLECTION",
                    "UNIFIED_DISCREPANCY_DECISION",
                    "UNIFIED_JUDGMENT",
                }
                default = (
                    "true" if normalized_stage in thinking_stages else "false"
                )
            else:
                default = "false" if normalized_stage == "PLANNING" else "true"
            raw = os.getenv(
                f"QWEN_{normalized_stage}_ENABLE_THINKING",
                default,
            ).strip().lower()
            if raw not in {"true", "false"}:
                raise ValueError(
                    f"QWEN_{normalized_stage}_ENABLE_THINKING must be true or false."
                )
            enable_thinking = raw == "true"
            config: Dict[str, Any] = {"enable_thinking": enable_thinking}
            if qwen35:
                # Qwen3.5's official recommendations use non-greedy sampling and a
                # presence penalty.  Greedy decoding caused long, repetitive schema
                # deliberation in the real Queen Planning request.  Keep the hard
                # reasoning wall independent of max_tokens so visible JSON always has
                # room to finish.
                config.update(
                    {
                        "temperature": 1.0 if enable_thinking else 0.7,
                        "top_p": 0.95 if enable_thinking else 0.8,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": 1.5,
                        "repetition_penalty": 1.0,
                    }
                )
                if enable_thinking:
                    default_budget = {
                        "UNIFIED_REACT": 2048,
                        "UNIFIED_REFLECTION": 1536,
                        "UNIFIED_DISCREPANCY_DECISION": 2048,
                        "UNIFIED_JUDGMENT": 1024,
                    }.get(normalized_stage, 1024)
                    env_name = f"QWEN_{normalized_stage}_THINKING_TOKEN_BUDGET"
                    raw_budget = os.getenv(env_name, str(default_budget)).strip()
                    try:
                        budget = int(raw_budget)
                    except ValueError as exc:
                        raise ValueError(f"{env_name} must be an integer.") from exc
                    if budget < 1:
                        raise ValueError(f"{env_name} must be positive.")
                    config["thinking_token_budget"] = budget
            sampling_seed = getattr(self, "sampling_seed", None)
            if sampling_seed is not None:
                request_counts = getattr(self, "_sampling_request_counts", {})
                ordinal = request_counts.get(normalized_stage, 0)
                request_counts[normalized_stage] = ordinal + 1
                self._sampling_request_counts = request_counts
                material = (
                    f"{sampling_seed}:{normalized_stage}:{ordinal}"
                ).encode("utf-8")
                config["seed"] = int.from_bytes(
                    hashlib.sha256(material).digest()[:4], "big"
                ) & 0x7FFFFFFF
            return config
        return {
            "thinking_level": self._stage_thinking_level(normalized_stage),
            # Request the provider-generated summary; the hidden chain of
            # thought itself is not available through the API.
            "thinking_summaries": "auto",
        }

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        image_path: str,
        *,
        stage: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        """Execute one tool with cache-miss single-flight protection."""

        if tool_name in self.cacheable_tools and self.tool_cache.enabled:
            cache_tool_args = dict(args)
            properties = self.all_tools[tool_name].parameters.get(
                "properties",
                {},
            )
            if "image_input" in properties and not cache_tool_args.get(
                "image_input"
            ):
                cache_tool_args["image_input"] = image_path
            cache_args = self._build_cache_args(
                tool_name,
                cache_tool_args,
                image_path=image_path,
            )
            async with self.tool_cache.singleflight(tool_name, cache_args):
                return await self._execute_tool_uncached(
                    tool_name,
                    args,
                    image_path,
                    stage=stage,
                )
        return await self._execute_tool_uncached(
            tool_name,
            args,
            image_path,
            stage=stage,
        )

    async def _execute_tool_uncached(
        self,
        tool_name: str,
        args: Dict[str, Any],
        image_path: str,
        *,
        stage: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        tool = self.all_tools[tool_name]
        tool_args = dict(args)
        properties = tool.parameters.get("properties", {})
        if "image_input" in properties and not tool_args.get("image_input"):
            tool_args["image_input"] = image_path
        if hasattr(tool, "image_path") and image_path:
            tool.image_path = image_path

        cache_args = self._build_cache_args(tool_name, tool_args, image_path=image_path)
        started = time.perf_counter()
        runtime_store = current_case_runtime_store()
        if runtime_store is not None:
            recovered = runtime_store.reuse_tool_result(
                stage=stage or "direct_tool",
                tool_name=tool_name,
                tool_args=cache_args,
            )
            if recovered is not None:
                recovered_result = str(recovered["result"])
                _, succeeded = parse_tool_result(recovered_result)
                return recovered_result, {
                    "cache_hit": True,
                    "recovery_cache_hit": True,
                    "recovery_source_attempt_id": recovered.get(
                        "source_attempt_id"
                    ),
                    "recovery_source_memory_id": recovered.get(
                        "source_memory_id"
                    ),
                    "tool_success": succeeded,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    "serialized_size": len(recovered_result),
                }
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
                result = await run_tool_with_timeout(
                    tool.call_async(tool_args),
                    timeout_seconds=getattr(
                        self,
                        "tool_action_timeout_seconds",
                        150.0,
                    ),
                )
            else:
                result = await run_tool_with_timeout(
                    asyncio.to_thread(tool.call, tool_args),
                    timeout_seconds=getattr(
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
        if tool_name in {"visit", "crop_and_search"}:
            cache_args["__web_evidence_contract__"] = WEB_EVIDENCE_CONTRACT_VERSION
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
            if int(step.metadata.get("tool_llm_api_calls", 0) or 0) > 0:
                step.metadata["total_tokens"] = {
                    name: step.tokens.get(name, 0)
                    + int(tool_tokens.get(name, 0) or 0)
                    for name in ("prompt", "completion", "thought")
                }
        state.llm_api_calls += sum(
            1 for step in steps if step.metadata.get("llm_duration_ms") is not None
        )
        state.llm_api_calls += sum(
            int(step.metadata.get("tool_llm_api_calls", 0) or 0)
            for step in steps
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
