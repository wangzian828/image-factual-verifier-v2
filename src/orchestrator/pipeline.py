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
from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import (
    audit_coverage,
    compile_verdict_basis,
)
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.evidence_policy import (
    neutralize_planning_route_text,
    query_policy_violation,
    text_targets_verdict_or_media_origin,
)
from src.orchestrator.evidence_semantics import (
    same_capture_can_support_visual_claim,
)
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.runtime_events import CaseRuntimeStore, current_case_runtime_store
from src.orchestrator.progress_control import (
    record_action_progress,
    record_decision_progress,
)
from src.orchestrator.image_only_prompts import (
    DISCREPANCY_REACT_SYSTEM_PROMPT as IMAGE_ONLY_DISCREPANCY_REACT_PROMPT,
    DISCREPANCY_DECISION_SYSTEM_PROMPT as IMAGE_ONLY_DISCREPANCY_DECISION_PROMPT,
    DISCREPANCY_JUDGMENT_SYSTEM_PROMPT as IMAGE_ONLY_DISCREPANCY_JUDGMENT_PROMPT,
    EVIDENCE_DECISION_SYSTEM_PROMPT as IMAGE_ONLY_EVIDENCE_DECISION_PROMPT,
    IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT as IMAGE_ONLY_IMAGE_ACCOUNT_PLANNING_PROMPT,
    JUDGMENT_SYSTEM_PROMPT as IMAGE_ONLY_JUDGMENT_PROMPT,
    QUERY_CONCEPT_EXTRACTION_SYSTEM_PROMPT as IMAGE_ONLY_QUERY_CONCEPT_EXTRACTION_PROMPT,
    QUERY_REPLAN_SYSTEM_PROMPT as IMAGE_ONLY_QUERY_REPLAN_PROMPT,
    ROUTE_LOCAL_REPLAN_SYSTEM_PROMPT as IMAGE_ONLY_ROUTE_LOCAL_REPLAN_PROMPT,
    REACT_SYSTEM_PROMPT as IMAGE_ONLY_REACT_PROMPT,
    REFLECTION_SYSTEM_PROMPT as IMAGE_ONLY_REFLECTION_PROMPT,
    TARGET_PLANNING_SYSTEM_PROMPT as IMAGE_ONLY_TARGET_PLANNING_PROMPT,
    render_evidence_decision_context as render_image_only_evidence_decision_context,
    render_discrepancy_react_context as render_image_only_discrepancy_react_context,
    render_discrepancy_decision_context as render_image_only_discrepancy_decision_context,
    render_discrepancy_judgment_context as render_image_only_discrepancy_judgment_context,
    render_image_account_planning_context as render_image_only_image_account_planning_context,
    render_judgment_context as render_image_only_judgment_context,
    render_query_concept_extraction_context as render_image_only_query_concept_extraction_context,
    render_query_replan_context as render_image_only_query_replan_context,
    render_route_local_replan_context as render_image_only_route_local_replan_context,
    render_react_context as render_image_only_react_context,
    render_reflection_context as render_image_only_reflection_context,
    render_target_planning_context as render_image_only_target_planning_context,
    select_react_tasks as select_image_only_react_tasks,
    select_discrepancy_react_tasks as select_image_only_discrepancy_react_tasks,
)
from src.orchestrator.investigation_models import (
    ClaimAssessmentProposal,
    EvidenceDecisionOutput,
    DiscrepancyDecisionProposalOutput,
    DiscrepancyDecisionOutput,
    DiscrepancyJudgment,
    DiscrepancyJudgmentOutput,
    build_discrepancy_decision_output_schema,
    build_image_account_planning_output_schema,
    ImageAccountPlanningOutput,
    ImageOnlyInvestigationState,
    ImageOnlyJudgment,
    InvestigationSegmentOutput,
    QueryConceptExtractionOutput,
    QueryReplanOutput,
    ReflectionOutput,
    RouteLocalReplanOutput,
    TargetPlanningOutput,
)
from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.stage_runner import (
    InteractionSession,
    StageRunner,
    StageStep,
)
from src.orchestrator.tool_execution import run_tool_with_timeout
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.source_provenance import canonicalize_url
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
    build_stage_tools,
)
from src.orchestrator.tool_result import parse_tool_result, serialize_tool_result
from src.orchestrator.task_store import (
    MAX_REFLECTIONS,
    MAX_TOOL_ACTIONS,
    REFLECTION_INTERVAL,
    apply_evidence_decision,
    apply_evidence_decision_with_refinement_fallback,
    apply_discrepancy_decision,
    apply_image_account_planning,
    apply_query_replan,
    apply_route_local_replan,
    apply_reflection,
    apply_target_planning,
    archive_recall_available,
    bind_route_local_replan_runtime_ids,
    bind_discrepancy_decision_runtime_ids,
    claim_owned_visual_evidence_requirements,
    discrepancy_decision_checkpoint_reason,
    discrepancy_decision_evidence_ids,
    evidence_decision_checkpoint_reason,
    next_action_boundary,
    pending_evidence_decision_ids,
    pending_query_replan_evidence_ids,
    pending_visual_reinspection,
    query_concept_extraction_error,
    query_replan_candidate_task_ids,
    route_local_replan_candidate,
    route_local_exhausted_candidate,
    remaining_root_image_reverse_branches,
    remaining_material_routes,
    remaining_claim_hypothesis_routes,
    record_route_selection_exhaustion,
    record_tool_observation,
    runtime_task_tool_names,
    state_from_bootstrap,
)
from src.storage import default_tool_cache_dir


MAX_MODEL_TASK_CHOICES_PER_ACTION = 3


class Orchestrator:
    """VisualFact-driven image-only v3 orchestrator."""

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
        decision_policy_version: str = "discrepancy-first-v4",
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
            if decision_policy_version != "discrepancy-first-v4":
                raise RuntimeError(
                    "the public runtime requires decision_policy_version="
                    "discrepancy-first-v4"
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
            planning_started = time.perf_counter()
            try:
                await self._run_image_account_planning(
                    state,
                    investigation,
                    image_path=image_path,
                    interaction_session=None,
                )
            finally:
                state.stage_timings["planning"] = round(
                    time.perf_counter() - planning_started,
                    2,
                )

            investigation_started = time.perf_counter()
            try:
                await self._run_discrepancy_investigation(
                    state,
                    investigation,
                    image_path,
                    runtime_case,
                )
            finally:
                state.stage_timings["investigation"] = round(
                    time.perf_counter() - investigation_started,
                    2,
                )
            self._require_successful_discrepancy_investigation(state)
            compiled_verdict, basis = compile_discrepancy_verdict_basis(
                investigation
            )
            final_visual_audit = None
            if self._uses_separate_vlm():
                final_visual_audit = await self._run_final_visual_audit(
                    state,
                    investigation,
                    compiled_verdict=compiled_verdict,
                    basis=basis,
                    image_path=image_path,
                )
            judgment_started = time.perf_counter()
            try:
                judgment = await self._run_discrepancy_judgment(
                    state,
                    investigation,
                    compiled_verdict,
                    basis,
                    image_path=image_path,
                    final_visual_audit=final_visual_audit,
                    interaction_session=None,
                )
            finally:
                state.stage_timings["judgment"] = round(
                    time.perf_counter() - judgment_started,
                    2,
                )
            investigation.discrepancy_judgment = judgment
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
        *,
        image_path: str = "",
        interaction_session: Optional[InteractionSession] = None,
    ) -> None:
        """Let the policy induce bounded factual targets from visible state."""

        effective_image_path = image_path or state.image_path
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_TARGET_PLANNING_PROMPT),
            tools=[],
            output_schema=TargetPlanningOutput,
            # Initial proposal plus three bounded correction opportunities.
            # Planning remains finite, while one repeated supporting/metadata
            # proposal does not make the whole case an engineering failure.
            max_rounds=3,
            image_path=effective_image_path,
            stage_name="image_only_planning",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=bool(effective_image_path)
            and self._main_llm_attaches_image(),
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_target_planning(
                    investigation,
                    parsed,
                )
            ),
            max_output_tokens=self._stage_output_tokens("PLANNING", 8192),
            generation_config=self._stage_generation_config("PLANNING"),
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

    async def _run_image_account_planning(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        image_path: str = "",
        interaction_session: Optional[InteractionSession] = None,
    ) -> None:
        """Run standalone v4 Planning and atomically install its claim graph."""

        effective_image_path = image_path or state.image_path
        planning_output_schema = build_image_account_planning_output_schema(
            anchor_fact_ids=[
                fact.fact_id
                for fact in investigation.facts
                if fact.origin.type in {"input_image", "ocr"}
            ],
        )
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(
                IMAGE_ONLY_IMAGE_ACCOUNT_PLANNING_PROMPT
            ),
            tools=[],
            output_schema=planning_output_schema,
            max_rounds=self._stage_max_rounds("PLANNING", 2),
            image_path=effective_image_path,
            stage_name="image_account_planning",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=bool(effective_image_path)
            and self._main_llm_attaches_image(),
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_image_account_planning(
                    investigation,
                    parsed,
                    source_access_policy=self.source_access_policy,
                )
            ),
            max_output_tokens=self._stage_output_tokens("PLANNING", 8192),
            generation_config=self._stage_generation_config("PLANNING"),
            request_timeout_seconds=self.stage_request_timeout_seconds,
        )
        parsed, steps = await runner.run(
            render_image_only_image_account_planning_context(
                investigation,
                perception=state.perception,
            )
        )
        for step in steps:
            if step.action_type != "output_rejected":
                continue
            step.action_type = "planning_revision"
            step.metadata["planning_revision_reason"] = step.metadata.get(
                "rejection_reason",
                "",
            )
        planning_request_count = sum(
            1
            for step in steps
            if step.metadata.get("llm_duration_ms") is not None
        )
        planning_revision_count = sum(
            1 for step in steps if step.action_type == "planning_revision"
        )
        for step in steps:
            step.metadata["planning_request_count"] = planning_request_count
            step.metadata["planning_revision_count"] = planning_revision_count
        self._record_stage_steps(state, steps)
        if parsed is None:
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Image Account Planning did not produce a valid claim graph"
            )
        update = apply_image_account_planning(investigation, parsed)
        if not update.get("accepted", False):
            self._sync_image_only_state(state, investigation)
            raise RuntimeError(
                "Image Account Planning failed deterministic application: "
                + str(update.get("rejected_reason", "unknown validation error"))
            )
        self._sync_image_only_state(state, investigation)

    async def _run_image_only_investigation(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        *,
        interaction_session: InteractionSession,
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
            visual_request = pending_visual_reinspection(investigation)
            if visual_request is not None:
                observation_update = (
                    await self._run_image_only_visual_reinspection(
                        state,
                        investigation,
                        image_path=image_path,
                        runtime_case=runtime_case,
                        visual_question_id=visual_request.visual_question_id,
                    )
                )
                decision_trigger = evidence_decision_checkpoint_reason(
                    investigation,
                    update=observation_update,
                )
                if decision_trigger:
                    await self._run_image_only_evidence_decision(
                        state,
                        investigation,
                        trigger=decision_trigger,
                        required=True,
                        interaction_session=interaction_session,
                    )
                audit_coverage(
                    investigation,
                    decision_checkpoint=bool(decision_trigger),
                )
                self._sync_image_only_state(state, investigation)
                continue

            if investigation.action_count >= MAX_TOOL_ACTIONS:
                await self._run_image_only_evidence_decision(
                    state,
                    investigation,
                    trigger="before_unverifiable",
                    required=True,
                    interaction_session=interaction_session,
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
                    interaction_session=interaction_session,
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
                if await self._try_image_only_saturation_reflection(
                    state,
                    investigation,
                    interaction_session=interaction_session,
                ):
                    continue
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
                    interaction_session=interaction_session,
                )
                if reviewed and select_image_only_react_tasks(investigation):
                    audit_coverage(investigation)
                    self._sync_image_only_state(state, investigation)
                    continue
                audit_coverage(investigation)
                if await self._try_image_only_saturation_reflection(
                    state,
                    investigation,
                    interaction_session=interaction_session,
                ):
                    continue
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
                    interaction_session=interaction_session,
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
                if await self._try_image_only_saturation_reflection(
                    state,
                    investigation,
                    interaction_session=interaction_session,
                ):
                    continue
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
                runtime_store=state.runtime_store,
                handoff_state=investigation,
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
                    in {
                        "image_only_investigation",
                        "verification",
                        "image_only_discrepancy_investigation",
                    }
                ],
                max_output_tokens=self._stage_output_tokens(
                    "VERIFICATION",
                    16384,
                ),
                generation_config=self._stage_generation_config("VERIFICATION"),
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
                interaction_session=interaction_session,
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
                    interaction_session=interaction_session,
                )

            query_replan_ran = False
            replan_candidates = query_replan_candidate_task_ids(investigation)
            new_replan_evidence_ids = pending_query_replan_evidence_ids(
                investigation
            )
            if (
                replan_candidates
                and new_replan_evidence_ids
                and (
                    reflection_boundary
                    or not remaining_material_routes(
                        investigation,
                        fact_id=investigation.core_verdict_fact_id or "",
                    )
                )
            ):
                query_replan_ran = await self._run_image_only_query_replan(
                    state,
                    investigation,
                    task_id=replan_candidates[0],
                    new_evidence_ids=new_replan_evidence_ids,
                    trigger="evidence_boundary",
                )

            coverage_audit = audit_coverage(
                investigation,
                decision_checkpoint=True,
            )
            self._sync_image_only_state(state, investigation)
            if (
                coverage_audit.stop_reason == "information_saturated"
                and await self._try_image_only_saturation_reflection(
                    state,
                    investigation,
                    interaction_session=interaction_session,
                )
            ):
                prior_evidence_count = len(investigation.evidence)
                prior_finding_count = len(investigation.findings)
                prior_fact_signature = self._image_only_fact_signature(
                    investigation
                )
                continue

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
                    interaction_session=interaction_session,
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

            if (
                query_replan_ran
                and remaining_material_routes(
                    investigation,
                    fact_id=investigation.core_verdict_fact_id or "",
                )
            ):
                self._sync_image_only_state(state, investigation)
                continue

    async def _run_discrepancy_investigation(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        *,
        interaction_session: Optional[InteractionSession] = None,
    ) -> None:
        """Run the bounded v4 claim/hypothesis loop to a deterministic stop."""

        while not investigation.stop_reason:
            visual_request = pending_visual_reinspection(investigation)
            if visual_request is not None:
                observation_update = await self._run_image_only_visual_reinspection(
                    state,
                    investigation,
                    image_path=image_path,
                    runtime_case=runtime_case,
                    visual_question_id=visual_request.visual_question_id,
                )
                reviewed_ids = discrepancy_decision_evidence_ids(investigation)
                await self._run_discrepancy_decision(
                    state,
                    investigation,
                    reviewed_evidence_ids=reviewed_ids,
                    trigger=(
                        discrepancy_decision_checkpoint_reason(
                            investigation,
                            update=observation_update,
                        )
                        or "scheduled_boundary"
                    ),
                    interaction_session=None,
                )
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
                self._sync_image_only_state(state, investigation)
                continue
            if investigation.proposed_verdict in {"fake", "real"}:
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
                break
            if investigation.action_count >= MAX_TOOL_ACTIONS:
                reviewed_ids = discrepancy_decision_evidence_ids(investigation)
                await self._run_discrepancy_decision(
                    state,
                    investigation,
                    reviewed_evidence_ids=reviewed_ids,
                    trigger="before_unresolved",
                    interaction_session=None,
                )
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
                break
            if investigation.pending_archive_read_ids:
                await self._run_discrepancy_react_action(
                    state,
                    investigation,
                    image_path,
                    runtime_case,
                    interaction_session=InteractionSession(),
                )
                audit_discrepancy_coverage(investigation)
                self._sync_image_only_state(state, investigation)
                continue
            routes = remaining_claim_hypothesis_routes(investigation)
            if not routes:
                exhausted_replan_request = route_local_exhausted_candidate(
                    investigation
                )
                if exhausted_replan_request is not None:
                    route_task_id, route_trigger = exhausted_replan_request
                    if await self._run_image_only_route_local_replan(
                        state,
                        investigation,
                        image_path=image_path,
                        task_id=route_task_id,
                        trigger=route_trigger,
                    ):
                        audit_discrepancy_coverage(
                            investigation,
                            decision_checkpoint=True,
                        )
                        self._sync_image_only_state(state, investigation)
                        continue
                reviewed_ids = discrepancy_decision_evidence_ids(investigation)
                await self._run_discrepancy_decision(
                    state,
                    investigation,
                    reviewed_evidence_ids=reviewed_ids,
                    trigger="before_unresolved",
                    interaction_session=None,
                )
                audit = audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
                self._sync_image_only_state(state, investigation)
                if audit.stop_reason != "continue":
                    break
                continue
            observation_update = await self._run_discrepancy_react_action(
                state,
                investigation,
                image_path,
                runtime_case,
                interaction_session=InteractionSession(),
            )
            if observation_update.get("route_selection_exhausted"):
                route_replan_request = route_local_replan_candidate(
                    investigation,
                    observation_update=observation_update,
                )
                if route_replan_request is not None:
                    route_task_id, route_trigger = route_replan_request
                    if await self._run_image_only_route_local_replan(
                        state,
                        investigation,
                        image_path=image_path,
                        task_id=route_task_id,
                        trigger=route_trigger,
                    ):
                        audit_discrepancy_coverage(
                            investigation,
                            decision_checkpoint=True,
                        )
                        self._sync_image_only_state(state, investigation)
                        continue
                await self._run_discrepancy_decision(
                    state,
                    investigation,
                    reviewed_evidence_ids=(
                        discrepancy_decision_evidence_ids(investigation)
                    ),
                    trigger="scheduled_boundary",
                    interaction_session=None,
                )
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
                self._sync_image_only_state(state, investigation)
                continue
            trigger = discrepancy_decision_checkpoint_reason(
                investigation,
                update=observation_update,
            )
            if trigger:
                await self._run_discrepancy_decision(
                    state,
                    investigation,
                    reviewed_evidence_ids=(
                        discrepancy_decision_evidence_ids(investigation)
                    ),
                    trigger=trigger,
                    interaction_session=None,
                )
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
            else:
                audit_discrepancy_coverage(investigation)

            route_replan_request = route_local_replan_candidate(
                investigation,
                observation_update=observation_update,
            )
            if (
                route_replan_request is not None
                and investigation.proposed_verdict not in {"fake", "real"}
            ):
                route_task_id, route_trigger = route_replan_request
                await self._run_image_only_route_local_replan(
                    state,
                    investigation,
                    image_path=image_path,
                    task_id=route_task_id,
                    trigger=route_trigger,
                )
                audit_discrepancy_coverage(
                    investigation,
                    decision_checkpoint=True,
                )
            self._sync_image_only_state(state, investigation)

            # Recall remains two-step: candidate recall is followed by an exact
            # read on the next action.  A no-gain streak is diagnostic only and
            # must not settle the case while an executable route remains.
            if investigation.pending_archive_read_ids:
                continue

        if investigation.proposed_verdict in {"fake", "real"}:
            audit = (
                investigation.discrepancy_coverage_audits[-1]
                if investigation.discrepancy_coverage_audits
                else audit_discrepancy_coverage(investigation)
            )
            if not audit.complete:
                raise RuntimeError(
                    "v4 model proposed a verdict that failed Coverage preconditions"
                )

    async def _run_discrepancy_react_action(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        *,
        interaction_session: InteractionSession,
    ) -> Dict[str, Any]:
        """Execute one bounded v4 claim/hypothesis action in a short tool chain."""

        if investigation.proposed_verdict in {"fake", "real"}:
            raise RuntimeError("no ReAct action is allowed after a v4 verdict")
        react_tasks = select_image_only_discrepancy_react_tasks(investigation)
        # Give the model a small scheduling window instead of deterministically
        # forcing the first Task. Runtime validation still rejects a tool, URL, or
        # Claim that does not belong to the selected Task.
        react_tasks = react_tasks[:MAX_MODEL_TASK_CHOICES_PER_ACTION]
        task_ids = {task.task_id for task in react_tasks}
        if not task_ids:
            raise RuntimeError("no executable claim/hypothesis task remains")
        executable_tool_names = (
            {"read_evidence"}
            if investigation.pending_archive_read_ids
            else self._discrepancy_executable_tool_names(
                investigation,
                task_ids=task_ids,
            )
        )
        if not executable_tool_names:
            raise RuntimeError("no executable claim/hypothesis route remains")
        recall_available = (
            not investigation.pending_archive_read_ids
            and archive_recall_available(
                investigation,
                task_ids=task_ids,
            )
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
            progress = record_action_progress(investigation, update)
            update["progress"] = progress.model_dump(mode="json")
            observation_update = update
            self._sync_image_only_state(state, investigation)
            return update

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_DISCREPANCY_REACT_PROMPT),
            tools=[
                tool
                for tool in build_stage_tools("verification", self.all_tools)
                if tool.name != "current_time"
                and (
                    tool.name == "read_evidence"
                    if investigation.pending_archive_read_ids
                    else (
                        tool.name in executable_tool_names
                        or (
                            tool.name == "recall_evidence"
                            and recall_available
                        )
                    )
                )
            ],
            output_schema=InvestigationSegmentOutput,
            max_rounds=1,
            image_path=image_path,
            stage_name="verification",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            prior_steps=[
                step
                for step in state.all_steps
                if getattr(step, "stage_name", "")
                in {
                    "image_only_discrepancy_investigation",
                    "image_only_visual_reinspection",
                }
            ],
            tool_cache=self.tool_cache,
            cacheable_tools=list(self.cacheable_tools),
            tool_call_limits=self.verification_tool_limits,
            min_tool_calls=1,
            should_stop=lambda steps: any(
                step.action_type == "tool_call" for step in steps
            ),
            max_output_tokens=self._stage_output_tokens("VERIFICATION", 16384),
            generation_config=self._stage_generation_config("VERIFICATION"),
            observation_callback=observation_callback,
            question_claims=self._discrepancy_task_claims(
                investigation,
                task_ids=task_ids,
            ),
            question_claim_options=self._discrepancy_task_claim_options(
                investigation,
                task_ids=task_ids,
            ),
            question_evidence_goals=self._discrepancy_task_evidence_goals(
                investigation,
                task_ids=task_ids,
            ),
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
            protocol_exhaustion_boundary=True,
            interaction_session=interaction_session,
            question_is_active=lambda task_id: any(
                task.task_id == task_id
                and task.status in {"active", "pending"}
                and task.claim_ids
                and task.hypothesis_id is not None
                for task in investigation.tasks
            ),
            stop_output_factory=lambda: InvestigationSegmentOutput(
                segment_summary=(
                    "The bounded route-selection correction chain was exhausted; "
                    "return to a discrepancy checkpoint."
                ),
                ready_for_reflection=True,
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
            tool_timeout_seconds=self.tool_action_timeout_seconds,
            tool_argument_constraints=(
                {
                    "read_evidence": {
                        "memory_id": list(investigation.pending_archive_read_ids)
                    }
                }
                if investigation.pending_archive_read_ids
                else self._discrepancy_tool_argument_constraints(
                    investigation,
                    task_ids=task_ids,
                )
            ),
        )
        parsed, steps = await runner.run(
            render_image_only_discrepancy_react_context(
                investigation,
                task_ids=task_ids,
            )
        )
        for step in steps:
            if step.stage_name == "verification":
                step.stage_name = "image_only_discrepancy_investigation"
                step.metadata["stage"] = "image_only_discrepancy_investigation"
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "v4 discrepancy ReAct did not reach a valid action boundary"
            )
        if not observation_update:
            boundary_step = next(
                (
                    step
                    for step in reversed(steps)
                    if step.metadata.get(
                        "protocol_correction_exhaustion_boundary"
                    )
                ),
                None,
            )
            if boundary_step is not None:
                request_ids = list(
                    boundary_step.metadata.get(
                        "resolved_rejection_request_ids", []
                    )
                    or []
                )
                observation_update = record_route_selection_exhaustion(
                    investigation,
                    task_id=next(iter(task_ids)),
                    request_id=(
                        str(request_ids[-1])
                        if request_ids
                        else "route-selection-boundary"
                    ),
                )
                boundary_step.metadata["investigation_state_update"] = (
                    observation_update
                )
                self._sync_image_only_state(state, investigation)
                return observation_update
            raise RuntimeError("v4 discrepancy ReAct executed no accepted action")
        return observation_update

    async def _run_discrepancy_decision(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        reviewed_evidence_ids: Sequence[str],
        trigger: str,
        interaction_session: InteractionSession,
    ) -> Dict[str, Any]:
        """Run and atomically apply one sparse v4 multimodal checkpoint."""

        before_signature = self._discrepancy_progress_signature(investigation)
        decision_output_schema = build_discrepancy_decision_output_schema(
            claim_ids=[
                claim.claim_id for claim in investigation.image_claims
            ],
            evidence_ids=list(reviewed_evidence_ids),
            hypothesis_ids=[
                hypothesis.hypothesis_id
                for hypothesis in investigation.search_hypotheses
            ],
        )

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_DISCREPANCY_DECISION_PROMPT),
            tools=[],
            output_schema=decision_output_schema,
            max_rounds=3,
            stage_name="image_only_discrepancy_decision",
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
                )
            ),
            max_output_tokens=self._stage_output_tokens(
                "EVIDENCE_DECISION",
                8192,
            ),
            generation_config=self._stage_generation_config("EVIDENCE_DECISION"),
            protocol_exhaustion_boundary=True,
            stop_output_factory=lambda: DiscrepancyDecisionProposalOutput(
                verdict_proposal="continue",
                rationale=(
                    "No atomic Decision update was accepted; continue with the "
                    "recorded workspace and unresolved gaps."
                ),
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
        )
        parsed, steps = await runner.run(
            render_image_only_discrepancy_decision_context(
                investigation,
                reviewed_evidence_ids=reviewed_evidence_ids,
                trigger=trigger,
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
        for claim in investigation.image_claims:
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
        for claim in investigation.image_claims:
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

    async def _run_image_only_visual_reinspection(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        visual_question_id: str,
    ) -> Dict[str, Any]:
        """Execute one accepted visual question outside the text-search ReAct loop."""

        record = next(
            (
                item
                for item in investigation.visual_reinspections
                if item.visual_question_id == visual_question_id
            ),
            None,
        )
        if record is None or record.status != "pending":
            raise RuntimeError(
                "visual reinspection request is missing or no longer pending"
            )
        task = next(
            (
                item
                for item in investigation.tasks
                if item.task_id == record.task_id
            ),
            None,
        )
        core = next(
            (
                item
                for item in investigation.facts
                if item.fact_id == record.fact_id
            ),
            None,
        )
        if task is None or core is None:
            raise RuntimeError(
                "visual reinspection lost its task or active fact"
            )
        evidence_by_id = {
            item.evidence_id: item
            for item in investigation.evidence
        }
        evidence_context = [
            {
                "evidence_id": evidence_id,
                "source_class": evidence_by_id[evidence_id].source_class,
                "source_url": evidence_by_id[evidence_id].source_url,
                "text": evidence_by_id[evidence_id].exact_text[:900],
            }
            for evidence_id in record.request.grounding_evidence_ids
            if evidence_id in evidence_by_id
        ]
        tool_args: Dict[str, Any] = {
            "visual_question_id": record.visual_question_id,
            "question": record.request.question,
            "expected_property": record.request.expected_property,
            "scope": record.request.scope,
            "anchor_regions": record.anchor_regions,
            "active_fact": core.statement,
            "evidence_context": json.dumps(
                evidence_context,
                ensure_ascii=False,
            )[:4000],
        }
        record.status = "running"
        tool_result, metadata = await self._execute_tool(
            "focused_visual_inspection",
            tool_args,
            image_path,
            stage="image_only_visual_reinspection",
        )
        step = StageStep(
            round=investigation.action_count + 1,
            stage_name="image_only_visual_reinspection",
            action_type="tool_call",
            tool_name="focused_visual_inspection",
            tool_args={
                **tool_args,
                "image_input": image_path,
                "__question_id": task.task_id,
            },
            tool_result=tool_result,
            metadata={
                "stage": "image_only_visual_reinspection",
                "visual_question_id": record.visual_question_id,
                **metadata,
            },
        )
        self._archive_direct_tool_step(
            state,
            step,
            action_index=investigation.action_count + 1,
        )
        update = record_tool_observation(
            investigation,
            step,
            image_sha256=runtime_case.image_sha256,
        )
        visual_failure_guard = not bool(update.get("created_evidence_ids"))
        if visual_failure_guard:
            failure_records = [
                item
                for item in investigation.failures
                if item.failure_id in set(update.get("created_failure_ids", []))
            ]
            guard_message = (
                "Focused visual reinspection produced no pixel Evidence; refusing "
                "to continue into a source-only follow-up Decision."
            )
            if failure_records:
                guard_message += (
                    f" Tool failure [{failure_records[0].code}]: "
                    + failure_records[0].message[:800]
                )
            update["focused_visual_failure_guard"] = guard_message
            update["focused_visual_failure"] = {
                "failure_ids": [
                    item.failure_id for item in failure_records
                ],
                "failure_codes": [
                    item.code for item in failure_records
                ],
                "source_only_follow_up_blocked": True,
            }
            step.metadata["focused_visual_failure_guard"] = guard_message
            step.metadata["focused_visual_failure"] = dict(
                update["focused_visual_failure"]
            )
        progress = record_action_progress(
            investigation,
            update,
            visual_reinspection=True,
        )
        update["progress"] = progress.model_dump(mode="json")
        artifact = step.metadata.get("tool_result_artifact") or {}
        memory_id = str(artifact.get("memory_id", "")).strip()
        if state.runtime_store is not None and memory_id:
            state.runtime_store.bind_archive_lineage(memory_id, update)
        step.metadata["investigation_state_update"] = update
        self._record_stage_steps(state, [step])
        self._sync_image_only_state(state, investigation)
        if visual_failure_guard:
            raise RuntimeError(update["focused_visual_failure_guard"])
        return update

    async def _run_image_only_evidence_decision(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        trigger: str,
        required: bool,
        interaction_session: Optional[InteractionSession] = None,
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
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_evidence_decision(
                    investigation,
                    parsed,
                    reviewed_evidence_ids=reviewed_evidence_ids,
                    trigger=trigger,
                )
            ),
            max_output_tokens=self._stage_output_tokens("VERIFICATION", 8192),
            generation_config=self._stage_generation_config("VERIFICATION"),
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
        update = apply_evidence_decision_with_refinement_fallback(
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
        trigger: str = "interval",
        interaction_session: Optional[InteractionSession] = None,
    ) -> Any:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_REFLECTION_PROMPT),
            tools=[],
            output_schema=ReflectionOutput,
            max_rounds=2,
            stage_name="image_only_reflection",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_reflection(
                    investigation,
                    parsed,
                    evidence_gain=evidence_gain,
                    decision_gain=decision_gain,
                    trigger=trigger,
                )
            ),
            max_output_tokens=self._stage_output_tokens("REFLECTION", 8192),
            generation_config=self._stage_generation_config("REFLECTION"),
        )
        parsed, steps = await runner.run(
            render_image_only_reflection_context(
                investigation,
                trigger=trigger,
            )
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "mandatory image-only Reflection did not produce valid "
                "structured output"
            )
        record = apply_reflection(
            investigation,
            parsed,
            evidence_gain=evidence_gain,
            decision_gain=decision_gain,
            trigger=trigger,
        )
        self._sync_image_only_state(state, investigation)
        return record

    async def _try_image_only_saturation_reflection(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        interaction_session: InteractionSession,
    ) -> bool:
        """Offer one bounded semantic pivot before an unresolved terminal stop."""

        if (
            investigation.action_count >= MAX_TOOL_ACTIONS
            or any(
                item.trigger == "saturation"
                for item in investigation.reflections
            )
            or investigation.stop_reason
            in {
                "verdict_determined",
                "coverage_complete",
                "hard_budget_exhausted",
            }
        ):
            return False
        if not investigation.stop_reason:
            investigation.stop_reason = "information_saturated"
        record = await self._run_image_only_reflection(
            state,
            investigation,
            evidence_gain=False,
            decision_gain=False,
            trigger="saturation",
            interaction_session=interaction_session,
        )
        routes = remaining_material_routes(
            investigation,
            fact_id=investigation.core_verdict_fact_id or "",
        )
        if (
            record.accepted_strategy_decision in {"continue", "replan"}
            and routes
            and not investigation.stop_reason
        ):
            audit_coverage(
                investigation,
                reflection_checkpoint=True,
            )
            self._sync_image_only_state(state, investigation)
            return not investigation.stop_reason
        if not investigation.stop_reason:
            investigation.stop_reason = "information_saturated"
        self._sync_image_only_state(state, investigation)
        return False

    async def _run_image_only_query_replan(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        task_id: str,
        new_evidence_ids: List[str],
        trigger: str,
    ) -> bool:
        """Ask for one bounded evidence-led change in search direction."""

        if not new_evidence_ids:
            raise RuntimeError(
                "image-only Query Replan requires new core Evidence"
            )
        concept_runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(
                IMAGE_ONLY_QUERY_CONCEPT_EXTRACTION_PROMPT
            ),
            tools=[],
            output_schema=QueryConceptExtractionOutput,
            max_rounds=2,
            stage_name="image_only_query_concept_extraction",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_query_concept_extraction(
                    investigation,
                    parsed,
                    task_id=task_id,
                    new_evidence_ids=new_evidence_ids,
                )
            ),
            max_output_tokens=self._stage_output_tokens(
                "QUERY_CONCEPT_EXTRACTION",
                2048,
            ),
            generation_config=self._stage_generation_config(
                "QUERY_CONCEPT_EXTRACTION"
            ),
        )
        concept_extraction, concept_steps = await concept_runner.run(
            render_image_only_query_concept_extraction_context(
                investigation,
                task_id=task_id,
                new_evidence_ids=new_evidence_ids,
            )
        )
        for step in concept_steps:
            if step.action_type == "output_rejected":
                step.action_type = "query_concept_extraction_revision"
                step.metadata["query_concept_extraction_revision_reason"] = (
                    step.metadata.get("rejection_reason", "")
                )
        self._record_stage_steps(state, concept_steps)
        if concept_extraction is None or not concept_extraction.concepts:
            raise RuntimeError(
                "image-only Query Concept Extraction did not produce a usable "
                "Evidence-derived concept"
            )

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_QUERY_REPLAN_PROMPT),
            tools=[],
            output_schema=QueryReplanOutput,
            max_rounds=2,
            stage_name="image_only_query_replan",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_query_replan(
                    investigation,
                    concept_extraction,
                    parsed,
                    task_id=task_id,
                    new_evidence_ids=new_evidence_ids,
                    trigger=trigger,
                    source_access_policy=self.source_access_policy,
                )
            ),
            max_output_tokens=self._stage_output_tokens("QUERY_REPLAN", 2048),
            generation_config=self._stage_generation_config("QUERY_REPLAN"),
        )
        parsed, steps = await runner.run(
            render_image_only_query_replan_context(
                investigation,
                task_id=task_id,
                concept_extraction=concept_extraction,
            )
        )
        for step in steps:
            if step.action_type == "output_rejected":
                step.action_type = "query_replan_revision"
                step.metadata["query_replan_revision_reason"] = (
                    step.metadata.get("rejection_reason", "")
                )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "image-only Query Replan did not produce valid structured output"
            )
        record = apply_query_replan(
            investigation,
            concept_extraction,
            parsed,
            trigger=trigger,
            new_evidence_ids=new_evidence_ids,
            source_access_policy=self.source_access_policy,
        )
        if record.rejected_reason:
            raise RuntimeError(
                "image-only Query Replan was rejected: "
                + record.rejected_reason
            )
        self._sync_image_only_state(state, investigation)
        return bool(record.accepted_queries)

    async def _run_image_only_route_local_replan(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        image_path: str,
        task_id: str,
        trigger: str,
    ) -> bool:
        """Let one stalled route change direction without constraining Planning."""

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_ROUTE_LOCAL_REPLAN_PROMPT),
            tools=[],
            output_schema=RouteLocalReplanOutput,
            max_rounds=2,
            image_path=image_path,
            stage_name="image_only_route_local_replan",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=bool(image_path)
            and self._main_llm_attaches_image(),
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_route_local_replan(
                    investigation,
                    parsed,
                    task_id=task_id,
                    trigger=trigger,
                    source_access_policy=self.source_access_policy,
                )
            ),
            max_output_tokens=self._stage_output_tokens(
                "ROUTE_LOCAL_REPLAN",
                2048,
            ),
            generation_config=self._stage_generation_config(
                "ROUTE_LOCAL_REPLAN"
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
        )
        parsed, steps = await runner.run(
            render_image_only_route_local_replan_context(
                investigation,
                task_id=task_id,
                trigger=trigger,
            )
        )
        for step in steps:
            if step.action_type == "output_rejected":
                step.action_type = "route_local_replan_revision"
                step.metadata["route_local_replan_revision_reason"] = (
                    step.metadata.get("rejection_reason", "")
                )
        self._record_stage_steps(state, steps)
        if parsed is None:
            fallback = RouteLocalReplanOutput(
                task_id=task_id,
                strategy="stop_route",
                rationale=(
                    "Route-local replan did not produce an admissible structured "
                    "output; close only this route and preserve the remaining "
                    "investigation routes."
                ),
            )
            record = apply_route_local_replan(
                investigation,
                fallback,
                trigger=trigger,
                source_access_policy=self.source_access_policy,
            )
            self._sync_image_only_state(state, investigation)
            return record.accepted_strategy != "rejected"
        parsed, binding_error = bind_route_local_replan_runtime_ids(
            investigation,
            parsed,
            task_id=task_id,
        )
        if parsed is None:
            self._sync_image_only_state(state, investigation)
            return False
        record = apply_route_local_replan(
            investigation,
            parsed,
            trigger=trigger,
            source_access_policy=self.source_access_policy,
        )
        self._sync_image_only_state(state, investigation)
        return record.accepted_strategy != "rejected"

    async def _run_final_visual_audit(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        *,
        compiled_verdict: str,
        basis: Any,
        image_path: str,
    ) -> Dict[str, Any]:
        """Ask the VLM for final pixel observations before text judgment."""

        if "focused_visual_inspection" not in self.all_tools:
            raise RuntimeError(
                "separate_vlm mode requires focused_visual_inspection"
            )
        evidence_by_id = {
            item.evidence_id: item for item in investigation.evidence
        }
        evidence_context = [
            {
                "evidence_id": evidence_id,
                "source_class": evidence_by_id[evidence_id].source_class,
                "source_url": evidence_by_id[evidence_id].source_url,
                "exact_text": evidence_by_id[evidence_id].exact_text[:1000],
                "stance": evidence_by_id[evidence_id].stance,
                "relation_scope": evidence_by_id[evidence_id].relation_scope,
                "directness": evidence_by_id[evidence_id].directness,
            }
            for evidence_id in basis.evidence_ids
            if evidence_id in evidence_by_id
        ]
        tool_args: Dict[str, Any] = {
            "image_input": image_path,
            "visual_question_id": "final-visual-audit",
            "question": (
                "Inspect the original pixels as a final visual audit for the "
                "target relation. Report concrete visible observations that "
                "support, contradict, or leave ambiguous the target property. "
                "Do not decide real or fake and do not infer source, creator, "
                "generation method, or any fact that is not visually observable."
            ),
            "expected_property": (
                "The concrete visible subject, relation, value, or scene "
                "condition relevant to the target and its competing alternative."
            ),
            "scope": "scene",
            "anchor_regions": self._final_visual_anchor_regions(state),
            "active_fact": str(
                getattr(basis, "verdict_target", "")
                or compiled_verdict
                or investigation.image_account_summary
            ),
            "evidence_context": json.dumps(
                {
                    "compiled_verdict": compiled_verdict,
                    "verdict_basis": basis.model_dump(mode="json"),
                    "selected_evidence": evidence_context,
                },
                ensure_ascii=False,
            )[:6000],
            "trace_stage": "image_only_final_visual_audit",
            "trace_purpose": "final_judgment_visual_context",
        }
        tool_result, metadata = await self._execute_tool(
            "focused_visual_inspection",
            tool_args,
            image_path,
            stage="image_only_final_visual_audit",
        )
        step = StageStep(
            round=1,
            stage_name="image_only_final_visual_audit",
            action_type="tool_call",
            tool_name="focused_visual_inspection",
            tool_args=dict(tool_args),
            tool_result=tool_result,
            metadata={
                "stage": "image_only_final_visual_audit",
                "non_policy_action": True,
                "excluded_from_investigation_budget": True,
                **metadata,
            },
        )
        self._archive_direct_tool_step(
            state,
            step,
            action_index=state.total_tool_calls + 1,
        )
        self._record_stage_steps(state, [step])
        if not self._tool_step_succeeded(step):
            raise RuntimeError(
                "final VLM visual audit failed: "
                + str(
                    self._safe_json_dict(tool_result).get(
                        "error",
                        tool_result,
                    )
                )[:1200]
            )
        audit = self._safe_json_dict(tool_result)
        audit["stage"] = "image_only_final_visual_audit"
        audit["purpose"] = "final_judgment_visual_context"
        audit["main_llm_received_image"] = False
        state.final_visual_audit = audit
        return audit

    @staticmethod
    def _final_visual_anchor_regions(
        state: VerificationState,
    ) -> List[List[float]]:
        report = state.perception
        if report is None:
            return []
        regions: List[List[float]] = []
        for entity in report.entities:
            if len(entity.bbox) == 4:
                regions.append([round(float(value), 6) for value in entity.bbox])
        for text_region in report.text_regions:
            points = text_region.bbox_quad
            if len(points) < 4:
                continue
            xs = [float(point[0]) for point in points if len(point) >= 2]
            ys = [float(point[1]) for point in points if len(point) >= 2]
            if len(xs) >= 4 and len(ys) >= 4:
                regions.append(
                    [
                        round(max(0.0, min(xs)), 6),
                        round(max(0.0, min(ys)), 6),
                        round(min(1.0, max(xs)), 6),
                        round(min(1.0, max(ys)), 6),
                    ]
                )
        return regions[:4]

    async def _run_image_only_judgment(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        coverage: Any,
        compiled_verdict: str,
        basis: Any,
        *,
        interaction_session: InteractionSession,
    ) -> ImageOnlyJudgment:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_JUDGMENT_PROMPT),
            tools=[],
            output_schema=ImageOnlyJudgment,
            max_rounds=1,
            stage_name="image_only_judgment",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=False,
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_image_only_judgment(
                    parsed,
                    compiled_verdict=compiled_verdict,
                    basis=basis,
                )
            ),
            max_output_tokens=self._stage_output_tokens("JUDGMENT", 8192),
            generation_config=self._stage_generation_config("JUDGMENT"),
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

    async def _run_discrepancy_judgment(
        self,
        state: VerificationState,
        investigation: ImageOnlyInvestigationState,
        compiled_verdict: str,
        basis: Any,
        *,
        image_path: str,
        final_visual_audit: Optional[Dict[str, Any]] = None,
        interaction_session: InteractionSession,
    ) -> DiscrepancyJudgment:
        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(IMAGE_ONLY_DISCREPANCY_JUDGMENT_PROMPT),
            tools=[],
            output_schema=DiscrepancyJudgmentOutput,
            max_rounds=1,
            image_path=image_path,
            stage_name="image_only_discrepancy_judgment",
            runtime_store=state.runtime_store,
            handoff_state=investigation,
            attach_image=bool(image_path) and self._main_llm_attaches_image(),
            interaction_session=interaction_session,
            output_validator=lambda parsed, _steps: (
                self._validate_discrepancy_judgment(
                    parsed,
                    compiled_verdict=compiled_verdict,
                    basis=basis,
                )
            ),
            max_output_tokens=self._stage_output_tokens("JUDGMENT", 8192),
            generation_config=self._stage_generation_config("JUDGMENT"),
            request_timeout_seconds=self.stage_request_timeout_seconds,
        )
        parsed, steps = await runner.run(
            render_image_only_discrepancy_judgment_context(
                investigation,
                compiled_verdict,
                basis,
                final_visual_audit=final_visual_audit,
            )
        )
        self._record_stage_steps(state, steps)
        if parsed is None:
            raise RuntimeError(
                "v4 Judgment did not produce a valid binary judgment"
            )
        # IDs and unresolved gaps are runtime-owned.  Reconstruct the canonical
        # record from the compiled basis instead of asking the model to copy a
        # large, error-prone identifier list.
        return DiscrepancyJudgment(
            verdict=parsed.verdict,
            confidence=parsed.confidence,
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
    def _validate_image_only_reflection(
        investigation: ImageOnlyInvestigationState,
        parsed: ReflectionOutput,
        *,
        evidence_gain: bool,
        decision_gain: bool,
        trigger: str = "interval",
    ) -> tuple[bool, str]:
        candidate = investigation.model_copy(deep=True)
        record = apply_reflection(
            candidate,
            parsed,
            evidence_gain=evidence_gain,
            decision_gain=decision_gain,
            trigger=trigger,
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
        if record.strategy_rejected_reason:
            return False, record.strategy_rejected_reason
        return True, ""

    @staticmethod
    def _validate_image_only_query_concept_extraction(
        investigation: ImageOnlyInvestigationState,
        parsed: QueryConceptExtractionOutput,
        *,
        task_id: str,
        new_evidence_ids: List[str],
    ) -> tuple[bool, str]:
        reason = query_concept_extraction_error(
            investigation,
            parsed,
            task_id=task_id,
            new_evidence_ids=new_evidence_ids,
        )
        return not reason, reason

    @staticmethod
    def _validate_image_only_query_replan(
        investigation: ImageOnlyInvestigationState,
        concept_extraction: QueryConceptExtractionOutput,
        parsed: QueryReplanOutput,
        *,
        task_id: str,
        new_evidence_ids: List[str],
        trigger: str,
        source_access_policy: Optional[SourceAccessPolicy] = None,
    ) -> tuple[bool, str]:
        if parsed.task_id != task_id:
            return False, "Query Replan must update the supplied task_id"
        candidate = investigation.model_copy(deep=True)
        record = apply_query_replan(
            candidate,
            concept_extraction,
            parsed,
            trigger=trigger,
            new_evidence_ids=new_evidence_ids,
            source_access_policy=source_access_policy,
        )
        if record.rejected_reason:
            return False, record.rejected_reason
        return True, ""

    @staticmethod
    def _validate_image_only_route_local_replan(
        investigation: ImageOnlyInvestigationState,
        parsed: RouteLocalReplanOutput,
        *,
        task_id: str,
        trigger: str,
        source_access_policy: Optional[SourceAccessPolicy] = None,
    ) -> tuple[bool, str]:
        parsed, binding_error = bind_route_local_replan_runtime_ids(
            investigation,
            parsed,
            task_id=task_id,
        )
        if parsed is None:
            return False, binding_error
        candidate = investigation.model_copy(deep=True)
        record = apply_route_local_replan(
            candidate,
            parsed,
            trigger=trigger,
            source_access_policy=source_access_policy,
        )
        if record.rejected_reason:
            return False, record.rejected_reason
        return True, ""

    @staticmethod
    def _validate_image_account_planning(
        investigation: ImageOnlyInvestigationState,
        parsed: ImageAccountPlanningOutput,
        *,
        source_access_policy: Optional[SourceAccessPolicy] = None,
    ) -> tuple[bool, str]:
        policy = source_access_policy or SourceAccessPolicy()
        parsed.search_hypotheses = [
            hypothesis.model_copy(
                update={
                    "statement": neutralize_planning_route_text(
                        hypothesis.statement
                    ),
                    "expected_information": neutralize_planning_route_text(
                        hypothesis.expected_information
                    ),
                    "queries": [
                        neutralize_planning_route_text(query)
                        for query in hypothesis.queries
                    ],
                }
            )
            for hypothesis in parsed.search_hypotheses
        ]
        executable_hypotheses = []
        for hypothesis in parsed.search_hypotheses:
            route_values = (
                hypothesis.statement,
                hypothesis.expected_information,
                *hypothesis.queries,
            )
            route_is_blocked = (
                hypothesis.route_focus == "media_origin"
                or any(
                    query_policy_violation(
                        query,
                        source_access_policy=policy,
                    )
                    for query in hypothesis.queries
                )
                or any(
                    text_targets_verdict_or_media_origin(value)
                    for value in route_values
                )
            )
            if not route_is_blocked:
                executable_hypotheses.append(hypothesis)
        if not executable_hypotheses:
            return False, (
                "Image Account Planning produced no executable neutral "
                "investigation route for its target fact."
            )
        # A bad optional route is not a reason to discard an otherwise valid
        # image account.  Remove it before state reduction so it cannot become
        # a later search action.  The original model output remains recorded in
        # the rejected-policy trace snapshot; only executable routes enter state.
        if len(executable_hypotheses) != len(parsed.search_hypotheses):
            parsed.search_hypotheses = executable_hypotheses
        candidate = investigation.model_copy(deep=True)
        update = apply_image_account_planning(candidate, parsed)
        if not update.get("accepted", False):
            return False, str(
                update.get(
                    "rejected_reason",
                    "Image Account Planning proposed no valid claim graph",
                )
            )
        if not candidate.image_claims or not any(
            claim.salience == "high" for claim in candidate.image_claims
        ):
            return False, "Image Account Planning requires a high-salience target fact"
        if candidate.core_verdict_fact_id is not None:
            return False, "Image Account Planning must not select a core verdict fact"
        return True, ""

    @staticmethod
    def _validate_discrepancy_decision(
        investigation: ImageOnlyInvestigationState,
        parsed: DiscrepancyDecisionProposalOutput,
        *,
        reviewed_evidence_ids: Sequence[str],
        trigger: str,
        source_access_policy: Optional[SourceAccessPolicy] = None,
    ) -> tuple[bool, str]:
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
        if proposal.decision_relevance != "decisive":
            return False, (
                "Target Planning must choose a decisive central image relation; "
                "a supporting attribute, serial number, label, or incidental OCR "
                "detail may guide retrieval but cannot own the verdict"
            )
        if proposal.predicate == "visual_integrity":
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
        update = apply_evidence_decision_with_refinement_fallback(
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
                    or (
                        facts[fact_id].predicate != "context_suggested_by_text"
                        and not same_capture_can_support_visual_claim(
                            facts[fact_id]
                        )
                    )
                )
            ]
            scene_claims = [
                facts[fact_id].statement
                for fact_id in task.fact_ids
                if fact_id in facts
                and same_capture_can_support_visual_claim(facts[fact_id])
            ]
            claims[task.task_id] = (
                " | ".join(specific_claims or scene_claims)
                if (specific_claims or scene_claims)
                else f"Question to resolve: {task.question}"
            )[:1800]
        return claims

    @staticmethod
    def _discrepancy_task_claims(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> Dict[str, str]:
        claims = {item.claim_id: item for item in investigation.image_claims}
        result: Dict[str, str] = {}
        for task in investigation.tasks:
            if task.task_id not in task_ids:
                continue
            owned_claims = [
                claims[claim_id].statement
                for claim_id in task.claim_ids
                if claim_id in claims
            ]
            result[task.task_id] = (
                "ImageClaims: "
                + " | ".join(owned_claims)
            )[:1800]
        return result

    @staticmethod
    def _discrepancy_task_claim_options(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> Dict[str, Dict[str, str]]:
        claims = {item.claim_id: item for item in investigation.image_claims}
        return {
            task.task_id: {
                claim_id: claims[claim_id].statement
                for claim_id in task.claim_ids
                if claim_id in claims
            }
            for task in investigation.tasks
            if task.task_id in task_ids
        }

    @staticmethod
    def _discrepancy_task_evidence_goals(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> Dict[str, str]:
        hypotheses = {
            item.hypothesis_id: item
            for item in investigation.search_hypotheses
        }
        return {
            task.task_id: (
                hypotheses[task.hypothesis_id].expected_information
                if task.hypothesis_id in hypotheses
                else task.question
            )[:1800]
            for task in investigation.tasks
            if task.task_id in task_ids
        }

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
    def _discrepancy_executable_tool_names(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> set[str]:
        names: set[str] = set()
        for route in remaining_claim_hypothesis_routes(
            investigation,
            task_ids=task_ids,
        ):
            parts = route.split(":", 2)
            tool_name = parts[0]
            route_task_id = (
                parts[2]
                if tool_name == "reverse_image_search" and len(parts) >= 3
                else parts[1]
                if len(parts) >= 2
                else ""
            )
            if route_task_id in task_ids:
                names.add(tool_name)
        return names

    @staticmethod
    def _discrepancy_tool_argument_constraints(
        investigation: ImageOnlyInvestigationState,
        *,
        task_ids: set[str],
    ) -> Dict[str, Dict[str, List[Any]]]:
        branches: List[str] = []
        pages: List[str] = []
        references: List[str] = []
        for route in remaining_claim_hypothesis_routes(
            investigation,
            task_ids=task_ids,
        ):
            parts = route.split(":", 2)
            if len(parts) < 2:
                continue
            if parts[0] == "reverse_image_search" and len(parts) == 3:
                if parts[2] in task_ids:
                    branches.extend(
                        remaining_root_image_reverse_branches(investigation)
                    )
            elif parts[0] == "visit" and len(parts) == 3:
                if parts[1] in task_ids:
                    pages.append(parts[2])
            elif parts[0] == "compare_with_reference" and len(parts) == 3:
                if parts[1] in task_ids:
                    references.append(parts[2])
        route_task_ids = {
            str(route.split(":", 2)[2])
            if route.startswith("reverse_image_search:")
            and len(route.split(":", 2)) == 3
            else str(route.split(":", 2)[1])
            for route in remaining_claim_hypothesis_routes(
                investigation,
                task_ids=task_ids,
            )
            if len(route.split(":", 2)) >= 2
        }
        constraints: Dict[str, Dict[str, List[Any]]] = {}
        if branches:
            constraints["reverse_image_search"] = {
                "branch": list(dict.fromkeys(branches)),
                "question_id": list(dict.fromkeys(route_task_ids)),
            }
        if pages:
            page_task_ids = list(
                dict.fromkeys(
                    route.split(":", 2)[1]
                    for route in remaining_claim_hypothesis_routes(
                        investigation,
                        task_ids=task_ids,
                    )
                    if route.startswith("visit:")
                    and len(route.split(":", 2)) == 3
                )
            )
            constraints["visit"] = {
                "url": list(dict.fromkeys(pages)),
                "question_id": page_task_ids,
            }
        if references:
            reference_task_ids = list(
                dict.fromkeys(
                    route.split(":", 2)[1]
                    for route in remaining_claim_hypothesis_routes(
                        investigation,
                        task_ids=task_ids,
                    )
                    if route.startswith("compare_with_reference:")
                    and len(route.split(":", 2)) == 3
                )
            )
            constraints["compare_with_reference"] = {
                "reference_url": list(dict.fromkeys(references)),
                "question_id": reference_task_ids,
            }
        for tool_name in Orchestrator._discrepancy_executable_tool_names(
            investigation,
            task_ids=task_ids,
        ):
            tool_task_ids = list(
                dict.fromkeys(
                    (
                        parts[2]
                        if tool_name == "reverse_image_search" and len(parts) == 3
                        else parts[1]
                    )
                    for route in remaining_claim_hypothesis_routes(
                        investigation,
                        task_ids=task_ids,
                    )
                    if (parts := route.split(":", 2))[0] == tool_name
                    and len(parts) >= 2
                )
            )
            if tool_task_ids:
                constraints.setdefault(tool_name, {})["question_id"] = tool_task_ids
        return constraints

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
        replanned_queries: List[str] = []
        task_by_id = {
            task.task_id: task
            for task in investigation.tasks
            if task.task_id in task_ids
        }
        for route in remaining_material_routes(
            investigation,
            fact_id=core_id,
        ):
            parts = route.split(":", 2)
            if len(parts) < 2:
                continue
            if parts[0] == "reverse_image_search" and len(parts) == 3:
                if parts[2] in task_ids:
                    branches.extend(
                        remaining_root_image_reverse_branches(investigation)
                    )
            elif parts[0] == "visit" and len(parts) == 3:
                if parts[1] in task_ids:
                    pages.append(parts[2])
            elif parts[0] == "compare_with_reference" and len(parts) == 3:
                if parts[1] in task_ids:
                    references.append(parts[2])
            elif parts[0] == "text_search" and len(parts) == 2:
                task = task_by_id.get(parts[1])
                if task is not None and task.query_replan_count:
                    replanned_queries.extend(task.suggested_queries)
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
        if replanned_queries:
            constraints["text_search"] = {
                "queries": list(dict.fromkeys(replanned_queries))
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
        if tool_name in {"recall_evidence", "read_evidence"}:
            return ""
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
        route_constraints = (
            Orchestrator._discrepancy_tool_argument_constraints(
                investigation,
                task_ids={task_id},
            )
            if investigation.image_claims
            else Orchestrator._image_only_tool_argument_constraints(
                investigation,
                task_ids={task_id},
            )
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
            normalized_requested_urls = list(
                dict.fromkeys(
                    canonicalize_url(str(url))
                    for url in requested_urls
                    if canonicalize_url(str(url))
                )
            )
            if (
                not normalized_requested_urls
                or len(normalized_requested_urls) > 3
                or not set(normalized_requested_urls) <= pending_urls
            ):
                return (
                    f"Tool 'visit' must inspect one to three pending candidate "
                    f"pages owned by task {task_id!r}."
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
        if tool_name == "reverse_image_search":
            requested_branch = (
                str(tool_args.get("branch", "lens")).strip().lower()
            )
            pending_branches = set(
                route_constraints.get("reverse_image_search", {}).get(
                    "branch",
                    [],
                )
            )
            if requested_branch not in pending_branches:
                return (
                    "Tool 'reverse_image_search' must use an untried branch "
                    f"for task {task_id!r}; pending branches are "
                    f"{sorted(pending_branches) or ['none']}."
                )
        if tool_name not in {"text_search", "reverse_image_search"}:
            return ""
        route_inventory = (
            remaining_claim_hypothesis_routes(
                investigation,
                task_ids={task_id},
            )
            if investigation.image_claims
            else remaining_material_routes(
                investigation,
                fact_id=investigation.core_verdict_fact_id or "",
            )
        )
        inspection_tools = {
            route.split(":", 1)[0]
            for route in route_inventory
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

    @staticmethod
    def _require_successful_discrepancy_investigation(
        state: VerificationState,
    ) -> None:
        investigation_steps = [
            step
            for step in state.all_steps
            if getattr(step, "stage_name", "")
            in {
                "image_only_discrepancy_investigation",
                "image_only_visual_reinspection",
            }
            and getattr(step, "action_type", "") == "tool_call"
        ]
        if not investigation_steps:
            raise RuntimeError(
                "v4 discrepancy investigation completed without a real tool attempt"
            )
        if not any(
            Orchestrator._tool_step_succeeded(step)
            for step in investigation_steps
        ):
            raise RuntimeError(
                "every attempted v4 discrepancy investigation tool call failed"
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
            if "ocr_with_position" not in self.all_tools:
                # Preserve the historical failure ordering for incomplete
                # test/startup registries: scene perception is still attempted
                # so its provider failure remains the primary diagnostic.
                tool_result, metadata = await self._execute_tool(
                    "perceive_scene",
                    {"image_input": image_path},
                    image_path,
                    stage="perception",
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
                self._archive_direct_tool_step(state, step, action_index=1)
                steps.append(step)
                if not self._tool_step_succeeded(step):
                    raise RuntimeError(
                        f"perceive_scene failed: {tool_result[:1000]}"
                    )
                raise RuntimeError(
                    "Required perception tool 'ocr_with_position' is unavailable: "
                    + str(
                        self.tool_health_summary.get("ocr_with_position", {}).get(
                            "error", "not registered"
                        )
                    )
                )

            # Scene perception and positioned OCR are independent observations
            # of the same immutable input image. Run them concurrently, then
            # archive and merge them in a fixed order for deterministic traces.
            (scene_result, scene_metadata), (ocr_result, ocr_metadata) = (
                await asyncio.gather(
                    self._execute_tool(
                        "perceive_scene",
                        {"image_input": image_path},
                        image_path,
                        stage="perception",
                    ),
                    self._execute_tool(
                        "ocr_with_position",
                        {"image_input": image_path},
                        image_path,
                        stage="perception",
                    ),
                )
            )

            scene_step = StageStep(
                round=1,
                stage_name="perception",
                action_type="tool_call",
                tool_name="perceive_scene",
                tool_args={"image_input": image_path},
                tool_result=scene_result,
                metadata={"stage": "perception", **scene_metadata},
            )
            ocr_step = StageStep(
                round=2,
                stage_name="perception",
                action_type="tool_call",
                tool_name="ocr_with_position",
                tool_args={"image_input": image_path},
                tool_result=ocr_result,
                metadata={"stage": "perception", **ocr_metadata},
            )

            for action_index, step in enumerate((scene_step, ocr_step), start=1):
                self._archive_direct_tool_step(
                    state,
                    step,
                    action_index=action_index,
                )
                steps.append(step)

            if not self._tool_step_succeeded(scene_step):
                raise RuntimeError(
                    f"perceive_scene failed: {scene_result[:1000]}"
                )
            if not self._tool_step_succeeded(ocr_step):
                raise RuntimeError(
                    f"ocr_with_position failed: {ocr_result[:1000]}"
                )

            report = self._parse_perception_result(scene_result)
            report = self._merge_ocr(report, ocr_result)
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

    def _stage_output_tokens(self, stage_name: str, default: int) -> int:
        normalized_stage = stage_name.strip().upper()
        if self.provider in {"qwen_local", "lmdeploy"}:
            env_name = f"QWEN_{normalized_stage}_MAX_OUTPUT_TOKENS"
            qwen35 = "qwen3.5" in str(
                getattr(self, "model_name", "")
            ).lower()
            provider_default = (
                8192
                if qwen35 and normalized_stage == "PLANNING"
                else 32768
                if normalized_stage == "PLANNING"
                else 8192
                if normalized_stage == "VERIFICATION"
                else default
            )
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

    def _stage_max_rounds(self, stage_name: str, default: int) -> int:
        """Read a bounded retry budget without changing stage semantics."""

        normalized_stage = stage_name.strip().upper()
        env_name = (
            f"QWEN_{normalized_stage}_MAX_ROUNDS"
            if self.provider in {"qwen_local", "lmdeploy"}
            else f"GEMINI_{normalized_stage}_MAX_ROUNDS"
        )
        raw = os.getenv(env_name, str(default)).strip()
        try:
            rounds = int(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer.") from exc
        if rounds < 1:
            raise ValueError(f"{env_name} must be positive.")
        return rounds

    @staticmethod
    def _archive_direct_tool_step(
        state: VerificationState,
        step: StageStep,
        *,
        action_index: int,
    ) -> None:
        if state.runtime_store is None or step.action_type != "tool_call":
            return
        descriptor = state.runtime_store.archive_tool_result(
            stage=step.stage_name,
            action_index=action_index,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            tool_result=step.tool_result,
            metadata={
                "tool_success": bool(step.metadata.get("tool_success", False)),
                "cache_hit": bool(step.metadata.get("cache_hit", False)),
                "function_call_id": step.metadata.get("function_call_id"),
            },
        )
        step.metadata["tool_result_artifact"] = descriptor

    @staticmethod
    def _stage_thinking_level(stage_name: str) -> str:
        normalized_stage = stage_name.strip().upper()
        fallback = (
            "high"
            if normalized_stage == "PLANNING"
            else os.getenv("GEMINI_AGENT_THINKING_LEVEL", "low")
        )
        value = os.getenv(
            f"GEMINI_{normalized_stage}_THINKING_LEVEL",
            fallback,
        ).strip().lower()
        if value == "minimal":
            value = "low"
        allowed = (
            {"low", "medium", "high"}
            if normalized_stage == "PLANNING"
            else {"low"}
        )
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
                    "PLANNING",
                    "EVIDENCE_DECISION",
                    "REFLECTION",
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
                        "PLANNING": 1024,
                        "EVIDENCE_DECISION": 2048,
                        "REFLECTION": 1536,
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
