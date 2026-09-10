"""Qwen Chat Completions continuation adapter for IFV OPSD.

The adapter intentionally delegates model calls and tool execution to the
existing runtime StageRunner.  It is not a second tool protocol and it does
not support Gemini interaction IDs as an arbitrary history-replay mechanism.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Sequence

from . import _repo_import  # noqa: F401
from src.orchestrator.source_access import SourceAccessPolicy
from src.orchestrator.investigation_models import InvestigationSegmentOutput
from src.orchestrator.stage_runner import StageRunner, StageStep
from src.orchestrator.tool_cache import ToolResultCache
from src.tools.base import BaseTool

from .opsd import (
    FailureSite,
    HintProposal,
    build_proposer_prompt,
    build_student_messages,
    build_teacher_messages,
    build_hint_proposal,
    parse_proposer_response,
    restore_failure_site_archive,
)


@dataclass(frozen=True)
class ContinuationResult:
    teacher_steps: List[StageStep]
    student_steps: List[StageStep]
    teacher_history: List[Dict[str, Any]]
    student_history: List[Dict[str, Any]]
    hint: str
    teacher_complete: bool = False
    student_complete: bool = False
    stop_reason: str = ""


class QwenContinuationAdapter:
    """Run a hinted suffix and a no-hint continuation from the same raw state."""

    def __init__(
        self,
        *,
        llm: Any,
        tools: Sequence[BaseTool],
        image_path: str,
        runtime_store: Any = None,
        tool_cache: ToolResultCache | None = None,
        cacheable_tools: Sequence[str] = (),
        source_access_policy: SourceAccessPolicy | None = None,
        tool_call_limits: Mapping[str, int] | None = None,
        max_output_tokens: int = 8192,
        generation_config: Mapping[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
        tool_timeout_seconds: float | None = None,
        require_runtime_archive: bool = True,
        source_runtime_store_path: str = "",
        policy_topk: int = 20,
    ) -> None:
        provider = str(getattr(llm, "provider", "")).strip().lower()
        wire_api = str(getattr(llm, "wire_api", "")).strip().lower()
        if provider not in {"qwen_local", "lmdeploy"} or wire_api != "chat_completions":
            raise ValueError(
                "OPSD continuation requires a Qwen-compatible Chat Completions backend"
            )
        self.llm = llm
        self.tools = list(tools)
        self.image_path = image_path
        self.runtime_store = runtime_store
        self.tool_cache = tool_cache
        self.cacheable_tools = list(cacheable_tools)
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.tool_call_limits = dict(tool_call_limits or {})
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.generation_config = dict(generation_config or {})
        self.request_timeout_seconds = request_timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.require_runtime_archive = bool(require_runtime_archive)
        self.source_runtime_store_path = str(source_runtime_store_path or "").strip()
        self.policy_topk = max(1, min(100, int(policy_topk)))

    def _site(self, site: FailureSite) -> FailureSite:
        if (
            self.source_runtime_store_path
            and site.runtime_store_path != self.source_runtime_store_path
        ):
            site = replace(
                site,
                runtime_store_path=self.source_runtime_store_path,
            )
        return restore_failure_site_archive(
            site,
            require=self.require_runtime_archive,
        )

    def _runner(
        self,
        *,
        site: FailureSite,
        history: Sequence[Mapping[str, Any]],
        include_hint_as_pending_user: bool,
        stage_name: str,
        prior_steps: Sequence[StageStep] = (),
    ) -> StageRunner:
        return StageRunner(
            llm=self.llm,
            system_prompt=_text(site.policy_input.get("system_instruction")),
            tools=list(self.tools),
            output_schema=InvestigationSegmentOutput,
            max_rounds=1,
            image_path=self.image_path,
            stage_name=stage_name,
            runtime_store=self.runtime_store,
            attach_image=False,
            prior_steps=list(prior_steps),
            tool_cache=self.tool_cache,
            cacheable_tools=self.cacheable_tools,
            tool_call_limits=self.tool_call_limits,
            min_tool_calls=1,
            should_stop=lambda steps: any(
                step.action_type == "tool_call" for step in steps
            ),
            force_tool_each_round=True,
            stop_output_factory=lambda: InvestigationSegmentOutput(
                segment_summary="OPSD continuation action completed.",
                action_completed=True,
            ),
            protocol_exhaustion_boundary=True,
            max_tool_calls_per_turn=1,
            max_output_tokens=self.max_output_tokens,
            generation_config=self.generation_config,
            source_access_policy=self.source_access_policy,
            request_timeout_seconds=self.request_timeout_seconds,
            tool_timeout_seconds=self.tool_timeout_seconds,
            capture_policy_tokens=True,
            policy_topk=self.policy_topk,
            native_history=[dict(item) for item in history],
            native_system_instruction=_text(site.policy_input.get("system_instruction")),
            native_history_includes_pending_user=include_hint_as_pending_user,
        )

    async def propose_hints(
        self,
        *,
        failure_site: FailureSite,
        public_trace_context: Mapping[str, Any],
        private_context: Mapping[str, Any] | None = None,
        hint_count: int = 4,
        hint_level: int = 1,
    ) -> list[HintProposal]:
        """Ask the privileged Qwen endpoint for audited procedural hints."""

        failure_site = self._site(failure_site)
        prompt = build_proposer_prompt(
            failure_site=failure_site,
            public_trace_context=public_trace_context,
            private_context=private_context,
            hint_count=hint_count,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Return JSON only. You are the privileged repair proposer; "
                    "the generated hints must not reveal the answer or exact action."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = await self._call_proposer(messages)
        try:
            payload = json.loads(response.text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("privileged proposer returned non-JSON output") from exc
        proposals: list[HintProposal] = []
        candidate_id = failure_site.step_id
        for text in parse_proposer_response(payload, limit=hint_count):
            try:
                proposals.append(
                    build_hint_proposal(
                        text=text,
                        level=hint_level,
                        provider=str(getattr(self.llm, "provider", "qwen_local")),
                        model=str(getattr(self.llm, "model_name", "")),
                        candidate_id=candidate_id,
                        public_failure_context={
                            "failure_site": failure_site.public_record(),
                            "trace_context": dict(public_trace_context),
                        },
                        private_context=private_context,
                    )
                )
            except ValueError:
                continue
        return proposals

    async def _call_proposer(
        self,
        messages: list[dict[str, Any]],
    ) -> Any:
        """Call the privileged proposer while preserving an auditable request."""

        request_id = ""
        response_format = {"type": "json_object"}
        generation_config = {"enable_thinking": False}
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage="opsd_proposer",
                lifecycle_kind="privileged_proposer",
                system_instruction=messages[0].get("content", "") if messages else "",
                input_payload=messages[1:] if len(messages) > 1 else messages,
                tools=[],
                response_format=response_format,
                generation_config=generation_config,
                max_output_tokens=min(self.max_output_tokens, 2048),
                model=str(getattr(self.llm, "model_name", "")),
                prompt_version="ifv-opsd-proposer-v1",
            )
        started = time.perf_counter()
        try:
            response = await self.llm.get_response(
                messages,
                max_tokens=min(self.max_output_tokens, 2048),
                response_format=response_format,
                generation_config=generation_config,
            )
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    response_metadata={
                        "opsd_proposer": True,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
            raise
        if request_id:
            raw = response.raw if isinstance(getattr(response, "raw", None), dict) else {}
            usage = raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {
                "input_tokens": getattr(response, "prompt_tokens", 0),
                "output_tokens": getattr(response, "completion_tokens", 0),
            }
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=usage,
                status="completed",
                response_metadata={
                    "opsd_proposer": True,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "response_content_chars": len(str(getattr(response, "text", "") or "")),
                },
            )
        return response

    async def run_hinted_suffix(
        self,
        *,
        failure_site: FailureSite,
        hint: HintProposal,
        prior_steps: Sequence[StageStep] = (),
    ) -> ContinuationResult:
        failure_site = self._site(failure_site)
        teacher_history = build_teacher_messages(failure_site, hint)
        teacher_runner = self._runner(
            site=failure_site,
            history=teacher_history,
            include_hint_as_pending_user=True,
            stage_name="opsd_teacher_repair",
            prior_steps=prior_steps,
        )
        _, teacher_steps = await teacher_runner.run("")
        if not teacher_runner.last_native_history:
            raise RuntimeError("teacher continuation did not retain native history")
        # Student and teacher must be sampled at the identical failed state.
        # Removing the hint from the teacher's post-action history would instead
        # expose the teacher's repaired tool call to the student.
        student_history = build_student_messages(failure_site)
        student_runner = self._runner(
            site=failure_site,
            history=student_history,
            include_hint_as_pending_user=True,
            stage_name="opsd_student_continuation",
            prior_steps=list(prior_steps),
        )
        _, student_steps = await student_runner.run("")
        return ContinuationResult(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_history=teacher_runner.last_native_history,
            student_history=student_runner.last_native_history,
            hint=hint.text,
        )

    async def run_hinted_episode(
        self,
        *,
        failure_site: FailureSite,
        hint: HintProposal,
        prior_steps: Sequence[StageStep] = (),
        max_suffix_actions: int = 8,
    ) -> ContinuationResult:
        """Continue teacher and student independently from the same failure state.

        Each call executes at most one native tool action. Re-entering
        ``StageRunner`` preserves the provider message history while keeping
        the existing runtime validators, tool budgets, archive and media
        handling in the loop. The result is explicitly incomplete unless a
        caller's episode verifier proves termination.
        """

        if max_suffix_actions < 1:
            raise ValueError("max_suffix_actions must be positive")
        site = self._site(failure_site)
        teacher_history = build_teacher_messages(site, hint)
        student_history = build_student_messages(site)
        teacher_steps: list[StageStep] = []
        student_steps: list[StageStep] = []
        teacher_complete = False
        student_complete = False

        for _ in range(max_suffix_actions):
            runner = self._runner(
                site=site,
                history=teacher_history,
                include_hint_as_pending_user=True,
                stage_name="opsd_teacher_repair",
                prior_steps=[*prior_steps, *teacher_steps],
            )
            _, steps = await runner.run("")
            teacher_steps.extend(steps)
            if not runner.last_native_history:
                raise RuntimeError("teacher continuation did not retain native history")
            teacher_history = runner.last_native_history
            if not any(step.action_type == "tool_call" for step in steps):
                teacher_complete = True
                break

        for _ in range(max_suffix_actions):
            runner = self._runner(
                site=site,
                history=student_history,
                include_hint_as_pending_user=True,
                stage_name="opsd_student_continuation",
                prior_steps=[*prior_steps, *student_steps],
            )
            _, steps = await runner.run("")
            student_steps.extend(steps)
            if not runner.last_native_history:
                raise RuntimeError("student continuation did not retain native history")
            student_history = runner.last_native_history
            if not any(step.action_type == "tool_call" for step in steps):
                student_complete = True
                break

        return ContinuationResult(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_history=teacher_history,
            student_history=student_history,
            hint=hint.text,
            teacher_complete=teacher_complete,
            student_complete=student_complete,
            stop_reason=(
                "both_suffixes_reached_non_tool_boundary"
                if teacher_complete and student_complete
                else "bounded_suffix_exhausted"
            ),
        )


def _text(value: Any) -> str:
    return str(value or "").strip()
