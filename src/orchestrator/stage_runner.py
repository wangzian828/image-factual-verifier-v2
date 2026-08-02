# -*- coding: utf-8 -*-
"""Single-path stage runner for multi-round ReAct execution."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from src.integrations.gemini import (
    extract_function_calls,
    extract_text,
    missing_required_paths,
    normalize_json_schema,
    validate_interaction_response,
    exception_runtime_metrics,
    take_runtime_metrics,
)
from src.orchestrator.evidence_policy import query_targets_fact_check_answer
from src.orchestrator.llm_backend import LLMBackend, LLMResponse
from src.orchestrator.route_policy import routes_semantically_equivalent
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.orchestrator.context_workspace import (
    StageHandoffPacket,
    build_stage_handoff,
    fit_stage_handoff_to_budget,
    render_stage_handoff,
    render_stage_request,
)
from src.orchestrator.tool_cache import (
    ToolResultCache,
    WEB_EVIDENCE_CONTRACT_VERSION,
)
from src.orchestrator.tool_result import ToolResultContractError, parse_tool_result, serialize_tool_result
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool
from src.tools.vision_utils import controlled_image_to_data_url


def _bounded_timeout(
    value: Optional[float],
    *,
    env_name: str,
    default: float,
) -> float:
    """Resolve an action deadline without allowing a zero/negative timeout."""

    if value is None:
        raw = os.getenv(env_name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            value = default
    return max(5.0, float(value))


@dataclass
class StageStep:
    """A single stage step."""

    round: int = 0
    stage_name: str = ""
    thought: str = ""
    action_type: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    output: Optional[Dict[str, Any]] = None
    tokens: Dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "thought": 0}
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionSession:
    """State shared by policy stages in one Gemini Interactions investigation."""

    previous_interaction_id: Optional[str] = None
    pending_input: List[Dict[str, Any]] = field(default_factory=list)


class StageRunner:
    """A compact multi-round agent loop for one stage."""

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str,
        tools: List[BaseTool],
        output_schema: Optional[Type[BaseModel]] = None,
        max_rounds: int = 5,
        image_path: str = "",
        stage_name: str = "",
        recent_rounds_to_keep: int = 3,
        tool_cache: Optional[ToolResultCache] = None,
        cacheable_tools: Optional[List[str]] = None,
        tool_call_limits: Optional[Dict[str, int]] = None,
        should_stop: Optional[Callable[[List[StageStep]], bool]] = None,
        output_validator: Optional[Callable[[BaseModel, List[StageStep]], Tuple[bool, str]]] = None,
        min_tool_calls: int = 0,
        attach_image: bool = True,
        tool_response_max_chars: int = 6000,
        prior_steps: Optional[List[StageStep]] = None,
        max_output_tokens: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        final_output_max_tokens: Optional[int] = None,
        final_output_generation_config: Optional[Dict[str, Any]] = None,
        observation_callback: Optional[Callable[[StageStep, List[StageStep]], Optional[Dict[str, Any]]]] = None,
        visual_call_validator: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        question_claims: Optional[Dict[str, str]] = None,
        question_claim_options: Optional[Dict[str, Dict[str, str]]] = None,
        priority_question_ids: Optional[List[str]] = None,
        resolved_priority_question_ids: Optional[List[str]] = None,
        supporting_question_ids: Optional[List[str]] = None,
        resolved_supporting_question_ids: Optional[List[str]] = None,
        source_access_policy: Optional[SourceAccessPolicy] = None,
        question_evidence_goals: Optional[Dict[str, str]] = None,
        max_protocol_corrections: int = 4,
        max_tool_calls_per_turn: Optional[int] = None,
        force_tool_each_round: bool = False,
        question_is_active: Optional[Callable[[str], bool]] = None,
        stop_output_factory: Optional[Callable[[], BaseModel]] = None,
        protocol_exhaustion_boundary: bool = False,
        request_timeout_seconds: Optional[float] = None,
        tool_timeout_seconds: Optional[float] = None,
        tool_argument_constraints: Optional[
            Dict[str, Dict[str, List[Any]]]
        ] = None,
        interaction_session: Optional[InteractionSession] = None,
        runtime_store: Optional[CaseRuntimeStore] = None,
        prompt_version: str = "",
        handoff_packet: Optional[StageHandoffPacket] = None,
        handoff_state: Optional[Any] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools
        self.output_schema = output_schema
        self.max_rounds = max_rounds
        self.image_path = image_path
        self.stage_name = stage_name
        self.recent_rounds_to_keep = max(1, recent_rounds_to_keep)
        self.tool_cache = tool_cache
        self.cacheable_tools = set(cacheable_tools or [])
        self.tool_call_limits = dict(tool_call_limits or {})
        self.should_stop = should_stop
        self.output_validator = output_validator
        self.min_tool_calls = max(0, int(min_tool_calls))
        self.attach_image = attach_image
        self.tool_response_max_chars = max(1200, int(tool_response_max_chars))
        self.prior_steps = list(prior_steps or [])
        self.max_output_tokens = (
            max(1, int(max_output_tokens)) if max_output_tokens is not None else None
        )
        self.generation_config = dict(generation_config or {})
        self.final_output_max_tokens = (
            max(1, int(final_output_max_tokens))
            if final_output_max_tokens is not None
            else self.max_output_tokens
        )
        self.final_output_generation_config = {
            **self.generation_config,
            **dict(final_output_generation_config or {}),
        }
        self.active_question_ids: List[str] = []
        self.llm_api_calls = 0
        self.observation_callback = observation_callback
        self.visual_call_validator = visual_call_validator
        self.question_claims = dict(question_claims or {})
        self.question_claim_options = deepcopy(question_claim_options or {})
        self.priority_question_ids = list(dict.fromkeys(priority_question_ids or []))
        self.resolved_priority_question_ids = set(resolved_priority_question_ids or [])
        self.supporting_question_ids = list(dict.fromkeys(supporting_question_ids or []))
        self.resolved_supporting_question_ids = set(resolved_supporting_question_ids or [])
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.question_evidence_goals = dict(question_evidence_goals or {})
        self.max_protocol_corrections = max(0, int(max_protocol_corrections))
        self.max_tool_calls_per_turn = (
            max(1, int(max_tool_calls_per_turn))
            if max_tool_calls_per_turn is not None
            else None
        )
        self.force_tool_each_round = bool(force_tool_each_round)
        self.question_is_active = question_is_active
        self.stop_output_factory = stop_output_factory
        self.protocol_exhaustion_boundary = bool(protocol_exhaustion_boundary)
        self.tool_argument_constraints = deepcopy(
            tool_argument_constraints or {}
        )
        self.interaction_session = interaction_session
        self.runtime_store = runtime_store
        self.prompt_version = prompt_version or f"{stage_name or 'stage'}-v1"
        self._last_context_request_id = ""
        self._last_interaction_lifecycle_kind = ""
        self._last_image_view: Dict[str, Any] = {}
        self.handoff_packet = handoff_packet
        self.handoff_state = handoff_state
        self.request_timeout_seconds = _bounded_timeout(
            request_timeout_seconds,
            env_name="AGENT_STAGE_REQUEST_TIMEOUT_SECONDS",
            default=120.0,
        )
        self.tool_timeout_seconds = _bounded_timeout(
            tool_timeout_seconds,
            env_name="AGENT_TOOL_ACTION_TIMEOUT_SECONDS",
            default=150.0,
        )

    async def run(self, input_context: str) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run the ReAct loop."""
        recalled_materials = self._recent_recalled_materials()
        if self.handoff_packet is None and self.handoff_state is not None:
            self.handoff_packet = build_stage_handoff(
                self.handoff_state,
                target_stage=self.stage_name,
                stage_input=input_context,
                available_tools=self.tools,
                output_contract=(
                    self.output_schema.__name__ if self.output_schema is not None else ""
                ),
                recent_steps=self.prior_steps,
                recalled_materials=recalled_materials,
            )
        if self.handoff_packet is not None:
            budget_result = fit_stage_handoff_to_budget(self.handoff_packet)
            self.handoff_packet = budget_result.packet
            if not budget_result.all_protected_items_reachable:
                raise RuntimeError(
                    "stage handoff lost protected context: "
                    + ", ".join(self.handoff_packet.missing_protected_ids)
                )
            if self.runtime_store is not None and (
                budget_result.removed_item_ids
                or budget_result.protected_context_overflow
            ):
                self.runtime_store.append_event(
                    "context_compaction",
                    {
                        "handoff_id": self.handoff_packet.handoff_id,
                        **self.handoff_packet.compaction,
                    },
                )
        if self.runtime_store is not None and self.handoff_packet is not None:
            handoff_artifact = self.runtime_store.artifacts.put_text(
                render_stage_handoff(self.handoff_packet),
                media_type="application/json; charset=utf-8",
                suffix=".json",
                metadata={
                    "kind": "stage_handoff_shadow",
                    "handoff_id": self.handoff_packet.handoff_id,
                    "target_stage": self.handoff_packet.target_stage,
                },
            )
            self.runtime_store.append_event(
                "stage_handoff_shadow",
                {
                    "handoff_id": self.handoff_packet.handoff_id,
                    "target_stage": self.handoff_packet.target_stage,
                    "workspace_version": self.handoff_packet.workspace.workspace_version,
                    "protected_coverage": self.handoff_packet.protected_coverage,
                    "protected_ids": self.handoff_packet.protected_ids,
                    "included_protected_ids": self.handoff_packet.included_protected_ids,
                    "missing_protected_ids": self.handoff_packet.missing_protected_ids,
                    "estimated_tokens": self.handoff_packet.estimated_tokens,
                    "compaction": self.handoff_packet.compaction,
                    "legacy_input_chars": len(str(input_context)),
                    "legacy_input_sha256": hashlib.sha256(
                        str(input_context).encode("utf-8")
                    ).hexdigest(),
                    "handoff_artifact": handoff_artifact,
                },
            )
        if str(getattr(self.llm, "provider", "")).lower() == "gemini" and str(
            getattr(self.llm, "wire_api", "")
        ).lower() != "interactions":
            raise RuntimeError("Gemini stages require wire_api='interactions'.")
        if self.handoff_packet is not None:
            input_context = render_stage_request(self.handoff_packet)
        configured_question_ids = [
            question_id
            for question_id in self.question_claims
            if not self.question_is_active
            or self.question_is_active(question_id)
        ]
        context_question_ids = re.findall(
            r"\[((?:q[^\]]*)|(?:task-[^\]]+))\]",
            input_context,
        )
        self.active_question_ids = list(
            dict.fromkeys(configured_question_ids or context_question_ids)
        )
        if self._uses_native_interactions():
            try:
                return await self._run_native_interactions(input_context)
            except RuntimeError as exc:
                partial_steps = list(getattr(exc, "stage_steps", []) or [])
                boundary = self._protocol_exhaustion_stage_boundary(
                    partial_steps
                )
                if boundary is not None:
                    return boundary
                raise
        if self._uses_native_structured_output():
            return await self._run_native_structured_output(input_context)

        steps: List[StageStep] = []
        native_chat = self._uses_native_chat_completions()
        system_msg = {
            "role": "system",
            "content": (
                self._build_native_chat_system_content()
                if native_chat
                else self._build_system_content()
            ),
        }
        user_msg = self._build_user_message(input_context)
        history: List[Dict[str, Any]] = [system_msg, user_msg]
        evidence_so_far: List[str] = []

        # Chat Completions has no provider-side Interaction lifecycle to
        # distinguish a rejected protocol turn from an accepted ReAct action.
        # Keep the same bounded correction semantics as native Interactions:
        # correction-only requests do not consume the action budget.
        action_turns = 0
        correction_turns = 0
        request_index = 0
        correction_only_turns = native_chat and bool(self.tools_list)
        next_lifecycle_kind = (
            "tool_roundtrip"
            if native_chat and bool(self.tools_list)
            else "standalone_request"
        )
        next_parent_context_request_id = ""
        next_generation_config: Optional[Dict[str, Any]] = None

        def direct_schema_generation_config() -> Optional[Dict[str, Any]]:
            """Disable hidden reasoning only for a rejected schema retry.

            A Qwen Chat Completions response may contain a candidate solely in
            its reasoning field.  Valid candidates still pass the normal schema
            validator.  When that candidate is invalid, repeating the original
            thinking request can replay the same stale schema indefinitely.
            Keep the first request unchanged, but make its correction a direct
            structured-output request.  This changes decoding only; it neither
            accepts nor translates an invalid response.
            """

            if not (
                native_chat
                and not self.tools_list
                and self.output_schema is not None
                and bool(self.generation_config.get("enable_thinking"))
            ):
                return None
            config = dict(self.generation_config)
            config["enable_thinking"] = False
            config.pop("thinking_token_budget", None)
            return config

        def needs_direct_schema_correction(affected: StageStep) -> bool:
            return bool(
                affected.metadata.get("response_content_chars", 0) == 0
                and affected.metadata.get("response_reasoning_chars", 0) > 0
                and direct_schema_generation_config() is not None
            )

        def request_chat_protocol_correction(
            affected: StageStep,
            reason: str,
            *,
            direct_schema_output: bool = False,
        ) -> bool:
            nonlocal correction_turns
            nonlocal next_lifecycle_kind
            nonlocal next_parent_context_request_id
            nonlocal next_generation_config
            if direct_schema_output:
                next_generation_config = direct_schema_generation_config()
            if not correction_only_turns:
                next_lifecycle_kind = "protocol_correction"
                next_parent_context_request_id = str(
                    affected.metadata.get("context_request_id", "")
                ).strip()
                return True
            affected.metadata["react_action_turn"] = action_turns
            affected.metadata["chat_request_index"] = request_index
            if correction_turns >= self.max_protocol_corrections:
                affected.metadata["protocol_corrections_used"] = correction_turns
                affected.metadata["correction_budget_exhausted"] = True
                affected.metadata["termination_reason"] = (
                    "protocol_correction_budget_exhausted"
                )
                affected.metadata.setdefault("rejection_reason", reason)
                return False
            correction_turns += 1
            affected.metadata["protocol_corrections_used"] = correction_turns
            next_lifecycle_kind = "protocol_correction"
            next_parent_context_request_id = str(
                affected.metadata.get("context_request_id", "")
            ).strip()
            return True

        while (
            action_turns if correction_only_turns else request_index
        ) < self.max_rounds:
            request_index += 1
            round_num = request_index
            messages = self._build_round_messages(system_msg, user_msg, history, evidence_so_far)
            completed_tool_calls = sum(
                1 for item in steps if item.action_type == "tool_call"
            )
            response, llm_metadata = await self._call_llm(
                messages,
                require_tool=(
                    native_chat
                    and bool(self.tools_list)
                    and (
                        self.force_tool_each_round
                        or completed_tool_calls < self.min_tool_calls
                    )
                ),
                lifecycle_kind=next_lifecycle_kind,
                parent_context_request_id=next_parent_context_request_id,
                generation_config=next_generation_config,
            )
            next_lifecycle_kind = (
                "tool_roundtrip"
                if native_chat and bool(self.tools_list)
                else "standalone_request"
            )
            next_parent_context_request_id = ""
            next_generation_config = None
            step = StageStep(
                round=round_num,
                stage_name=self.stage_name,
                tokens={"prompt": response.prompt_tokens, "completion": response.completion_tokens},
                metadata={
                    "stage": self.stage_name,
                    **llm_metadata,
                    "policy_input": self._policy_input_snapshot(
                        system_instruction=system_msg["content"],
                        input_payload=messages[1:],
                        tools=(
                            self._build_native_tool_schemas()
                            if native_chat and self.tools_list
                            else []
                        ),
                        response_format=(
                            self._openai_response_format()
                            if native_chat and not self.tools_list
                            else None
                        ),
                    ),
                },
            )

            content = (response.text or "").strip()
            if not content:
                step.action_type = "format_error"
                step.thought = "(empty response)"
                steps.append(step)
                history.append({"role": "assistant", "content": ""})
                history.append({"role": "user", "content": "Response was empty. Produce one valid tool call or final output."})
                if request_chat_protocol_correction(
                    step,
                    "the model returned an empty response",
                ):
                    continue
                break

            step.thought = self._extract_think(content)
            native_assistant = (
                self._openai_assistant_message(response.raw)
                if native_chat
                else None
            )
            native_call = (
                self._openai_single_tool_call(response.raw)
                if native_chat
                else None
            )

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_name, tool_args = self._parse_tool_call(content)
                if tool_name not in self.tools:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["invalid_tool_name"] = tool_name
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": self._unknown_tool_message(tool_name)})
                    if request_chat_protocol_correction(
                        step,
                        f"unknown tool {tool_name!r}",
                    ):
                        continue
                    break

                step.action_type = "tool_call"
                step.tool_name = tool_name
                step.tool_args = self._prepare_tool_args(tool_name, dict(tool_args), input_context)
                step.metadata["policy_action"] = {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": deepcopy(step.tool_args),
                }
                if native_call is not None:
                    step.metadata["native_chat_completions"] = True
                    step.metadata["function_call_id"] = str(
                        native_call.get("id", "")
                    ).strip()
                step.tool_args = self._bind_pending_visual_args(tool_name, step.tool_args)
                question_error = self._question_id_error(step.tool_args)
                if question_error:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["invalid_question_id"] = True
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                question_error
                            ),
                        }
                    )
                    if request_chat_protocol_correction(step, question_error):
                        continue
                    break
                if self._has_duplicate_tool_call(steps, tool_name, step.tool_args):
                    duplicate_message = self._duplicate_tool_message(
                        tool_name,
                        step.tool_args,
                    )
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["duplicate_tool_call"] = True
                    step.metadata["rejection_reason"] = duplicate_message
                    step.tool_result = json.dumps(
                        {"status": "error", "error": duplicate_message},
                        ensure_ascii=False,
                    )
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {"role": "user", "content": duplicate_message}
                    )
                    if request_chat_protocol_correction(
                        step,
                        duplicate_message,
                    ):
                        continue
                    break
                if self._tool_budget_reached(steps, tool_name):
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["tool_budget_reached"] = True
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": self._tool_budget_message(tool_name)})
                    if request_chat_protocol_correction(
                        step,
                        f"{tool_name} budget reached",
                    ):
                        continue
                    break
                filtered_query_count = self._sanitize_search_queries(
                    tool_name,
                    step.tool_args,
                )
                if filtered_query_count:
                    step.metadata["policy_filtered_query_count"] = filtered_query_count
                search_policy_error = self._search_policy_error(tool_name, step.tool_args)
                if search_policy_error:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["search_policy_rejection"] = True
                    step.tool_result = json.dumps(
                        {"status": "error", "error": search_policy_error},
                        ensure_ascii=False,
                    )
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": search_policy_error})
                    if request_chat_protocol_correction(step, search_policy_error):
                        continue
                    break
                if self.visual_call_validator is not None:
                    visual_error = self.visual_call_validator(tool_name, step.tool_args)
                    if visual_error:
                        step.action_type = "format_error"
                        step.metadata["error_class"] = "protocol_error"
                        step.metadata["invalid_visual_question"] = True
                        step.tool_result = json.dumps(
                            {"status": "error", "error": visual_error},
                            ensure_ascii=False,
                        )
                        steps.append(step)
                        history.append({"role": "assistant", "content": content})
                        history.append({"role": "user", "content": visual_error})
                        if request_chat_protocol_correction(step, visual_error):
                            continue
                        break
                coverage_error = self._priority_coverage_error(step.tool_args, steps)
                if coverage_error:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["unbalanced_priority_coverage"] = True
                    step.tool_result = json.dumps(
                        {"status": "error", "error": coverage_error},
                        ensure_ascii=False,
                    )
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": coverage_error})
                    if request_chat_protocol_correction(step, coverage_error):
                        continue
                    break
                serialized, tool_metadata = await self._execute_tool(tool_name, dict(step.tool_args))
                step.tool_result = serialized
                step.metadata.update(tool_metadata)
                self._archive_tool_step(
                    step,
                    action_index=len(self.prior_steps) + len(steps) + 1,
                )
                step.metadata.setdefault(
                    "function_call_id",
                    (
                        f"legacy-{self.stage_name or 'stage'}-{round_num}-"
                        f"{len(self.prior_steps) + len(steps) + 1}"
                    ),
                )
                steps.append(step)
                state_update = self._record_observation_update(step, steps)
                evidence_so_far.append(self._summarize_tool_result(tool_name, step.tool_args, serialized))
                tool_response = self._build_tool_response_message(
                    tool_name,
                    step.tool_args,
                    serialized,
                    function_call_id=str(step.metadata["function_call_id"]),
                    state_update=state_update,
                    native_chat=(
                        native_assistant is not None and native_call is not None
                    ),
                )
                if native_assistant is not None and native_call is not None:
                    history.append(native_assistant)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(step.metadata["function_call_id"]),
                            "name": tool_name,
                            "content": tool_response,
                        }
                    )
                else:
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": tool_response})

                action_turns += 1
                step.metadata["react_action_turn"] = action_turns
                step.metadata["protocol_corrections_used"] = correction_turns
                step.metadata["chat_request_index"] = request_index
                if self.should_stop and self.should_stop(steps):
                    if self.stop_output_factory is not None:
                        parsed = self.stop_output_factory()
                        steps.append(
                            StageStep(
                                round=round_num + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                metadata={
                                    "stage": self.stage_name,
                                    "deterministic_segment_boundary": True,
                                },
                            )
                        )
                        return parsed, steps
                    break
                next_lifecycle_kind = "tool_roundtrip"
                next_parent_context_request_id = str(
                    step.metadata.get("context_request_id", "")
                ).strip()
                continue

            output_json = self._extract_output(content)
            format_repair = ""
            if output_json is None:
                output_json, format_repair = (
                    self._try_parse_bare_json_with_repair(content)
                )
            if output_json is not None:
                step.action_type = "output"
                step.output = output_json
                step.metadata["policy_action"] = deepcopy(output_json)
                if format_repair:
                    step.metadata["format_repair"] = format_repair
                steps.append(step)
                parsed, schema_error = self._validate_output_with_error(
                    output_json
                )
                if parsed is not None:
                    accepted, reason = self._accept_output(parsed, steps)
                    if accepted:
                        return parsed, steps
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = reason
                    history.append({"role": "assistant", "content": content})
                    history.append(
                        {
                            "role": "user",
                            "content": self._structured_output_correction_prompt(
                                reason
                            ),
                        }
                    )
                    if request_chat_protocol_correction(step, reason):
                        continue
                    break
                step.action_type = "output_rejected"
                rejection_reason = (
                    schema_error or "output schema was invalid or incomplete"
                )
                step.metadata["rejection_reason"] = rejection_reason
                self._attach_invalid_response_preview(step, content)
                history.append({"role": "assistant", "content": content})
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Output schema was invalid: "
                            + rejection_reason
                            + " Return one corrected complete JSON object."
                        ),
                    }
                )
                if request_chat_protocol_correction(
                    step,
                    rejection_reason,
                    direct_schema_output=needs_direct_schema_correction(step),
                ):
                    continue
                break

            step.action_type = "format_error"
            self._attach_invalid_response_preview(step, content)
            steps.append(step)
            history.append({"role": "assistant", "content": content})
            history.append(
                {
                    "role": "user",
                    "content": (
                        "Use exactly one native function call or one JSON object."
                        if native_chat
                        else "Use exactly one <tool_call> or one <output> block."
                    ),
                }
            )
            if request_chat_protocol_correction(
                step,
                "the response was neither a function call nor valid JSON",
                direct_schema_output=needs_direct_schema_correction(step),
            ):
                continue
            break

        boundary = self._protocol_exhaustion_stage_boundary(steps)
        if boundary is not None:
            return boundary

        last_rejection_reason = ""
        for prior_step in reversed(steps):
            candidate_reason = str(
                (prior_step.metadata or {}).get("rejection_reason", "")
            ).strip()
            if candidate_reason:
                last_rejection_reason = candidate_reason
                break
        last_request_step = next(
            (
                item
                for item in reversed(steps)
                if str(item.metadata.get("context_request_id", "")).strip()
            ),
            None,
        )
        forced_is_correction = bool(
            last_request_step is not None
            and (
                last_request_step.action_type
                in {"format_error", "output_rejected"}
                or last_request_step.metadata.get("rejection_reason")
            )
        )
        forced_lifecycle_kind = (
            "protocol_correction"
            if forced_is_correction
            else "tool_roundtrip"
            if native_chat and bool(self.tools_list)
            else "standalone_request"
        )
        forced_parent_context_request_id = (
            str(
                last_request_step.metadata.get("context_request_id", "")
            ).strip()
            if last_request_step is not None
            and forced_lifecycle_kind != "standalone_request"
            else ""
        )
        forced, forced_meta = await self._force_output(
            system_msg,
            user_msg,
            history,
            evidence_so_far,
            last_rejection_reason=last_rejection_reason,
            lifecycle_kind=forced_lifecycle_kind,
            parent_context_request_id=forced_parent_context_request_id,
        )
        if forced is not None:
            accepted, reason = self._accept_output(forced, steps, final_attempt=True)
            if not accepted:
                forced_meta["rejection_reason"] = reason
                forced_meta["policy_action"] = forced.model_dump()
                forced = None
        rejected_output = forced_meta.get("policy_action")
        final_step = StageStep(
            round=len(steps) + 1,
            stage_name=self.stage_name,
            action_type=(
                "output"
                if forced is not None
                else "output_rejected"
                if isinstance(rejected_output, dict)
                else "format_error"
            ),
            output=(
                forced.model_dump()
                if forced is not None
                else rejected_output
                if isinstance(rejected_output, dict)
                else None
            ),
            metadata={"stage": self.stage_name, **forced_meta},
        )
        steps.append(final_step)
        if forced is None and self.protocol_exhaustion_boundary:
            # A semantic checkpoint may opt into a deterministic no-op boundary
            # after its last schema/validator retry. Do not issue a third provider
            # request or turn an invalid checkpoint into a factual decision.
            final_step.metadata["correction_budget_exhausted"] = True
            final_step.metadata["termination_reason"] = (
                "protocol_correction_budget_exhausted"
            )
            boundary = self._protocol_exhaustion_stage_boundary(steps)
            if boundary is not None:
                return boundary
        return forced, steps

    def _protocol_exhaustion_stage_boundary(
        self,
        steps: List[StageStep],
    ) -> Optional[Tuple[BaseModel, List[StageStep]]]:
        """Close one failed correction chain without inventing a tool action.

        This is opt-in because a generic structured-output stage should still fail
        closed.  A bounded ReAct orchestrator may instead return to its semantic
        checkpoint after the model repeatedly fails to select a new executable
        route.  The boundary records every rejected provider request it resolves;
        it does not count as an action or claim that a retrieval route succeeded.
        """

        if not self.protocol_exhaustion_boundary or self.stop_output_factory is None:
            return None
        exhausted = [
            step
            for step in steps
            if bool(step.metadata.get("correction_budget_exhausted"))
        ]
        if not exhausted:
            return None
        resolved_request_ids = list(
            dict.fromkeys(
                str(step.metadata.get("context_request_id", "")).strip()
                for step in steps
                if step.action_type in {"format_error", "output_rejected"}
                and str(step.metadata.get("context_request_id", "")).strip()
            )
        )
        parsed = self.stop_output_factory()
        boundary_step = StageStep(
            round=len(steps) + 1,
            stage_name=self.stage_name,
            action_type="output",
            output=parsed.model_dump(),
            metadata={
                "stage": self.stage_name,
                "deterministic_segment_boundary": True,
                "protocol_correction_exhaustion_boundary": True,
                "resolved_rejection_request_ids": resolved_request_ids,
                "protocol_corrections_used": max(
                    int(step.metadata.get("protocol_corrections_used", 0) or 0)
                    for step in exhausted
                ),
                "termination_reason": "protocol_correction_budget_exhausted",
            },
        )
        steps.append(boundary_step)
        return parsed, steps

    def _uses_native_interactions(self) -> bool:
        """Use Gemini's native function protocol when this stage has tools."""
        return bool(
            self.tools_list
            and str(getattr(self.llm, "provider", "")).lower() == "gemini"
            and str(getattr(self.llm, "wire_api", "")).lower() == "interactions"
            and callable(getattr(self.llm, "create_interaction", None))
        )

    def _uses_native_structured_output(self) -> bool:
        return bool(
            not self.tools_list
            and self.output_schema is not None
            and str(getattr(self.llm, "provider", "")).lower() == "gemini"
            and str(getattr(self.llm, "wire_api", "")).lower() == "interactions"
            and callable(getattr(self.llm, "create_interaction", None))
        )

    async def _run_native_structured_output(
        self,
        input_context: str,
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run a no-tool stage with Interactions JSON schema output."""
        steps: List[StageStep] = []
        previous_interaction_id = self._session_previous_interaction_id()
        next_input: Any = self._build_session_input(input_context)

        for round_num in range(1, self.max_rounds + 2):
            request_previous_interaction_id = previous_interaction_id
            started = time.perf_counter()
            self.llm_api_calls += 1
            system_instruction = self.system_prompt
            response_format = self._native_response_format()
            payload = await self._create_interaction(
                input_payload=next_input,
                system_instruction=system_instruction,
                previous_interaction_id=request_previous_interaction_id,
                response_format=response_format,
                store=True,
                max_tokens=self.max_output_tokens,
                generation_config=self.generation_config,
            )
            interaction_id, status = validate_interaction_response(payload)
            self._advance_interaction_session(interaction_id)
            if self._extract_native_function_calls(payload):
                raise RuntimeError("No-tool stage received an unexpected function_call response.")
            if status != "completed":
                raise RuntimeError(
                    f"No-tool stage requires status=completed, received status={status}."
                )

            usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
            content = self._extract_native_text(payload).strip()
            output_json = self._extract_output(content) or self._try_parse_bare_json(content)
            step = StageStep(
                round=round_num,
                stage_name=self.stage_name,
                thought=self._extract_native_thought(payload),
                action_type="output" if output_json is not None else "format_error",
                output=output_json,
                tokens=self._usage_tokens(usage),
                metadata={
                    "stage": self.stage_name,
                    "native_interactions": True,
                    "structured_output": True,
                    "previous_interaction_id": request_previous_interaction_id,
                    "interaction_id": interaction_id,
                    "interaction_status": status,
                    "llm_duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "policy_input": self._policy_input_snapshot(
                        system_instruction=system_instruction,
                        input_payload=next_input,
                        tools=[],
                        response_format=response_format,
                    ),
                    "policy_action": deepcopy(output_json),
                    "context_request_id": self._last_context_request_id,
                    "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
                },
            )
            steps.append(step)
            parsed = self._validate_output(output_json) if output_json is not None else None
            if parsed is not None:
                accepted, reason = self._accept_output(parsed, steps)
                if accepted:
                    return parsed, steps
                step.action_type = "output_rejected"
                step.metadata["rejection_reason"] = reason
                next_input = self._structured_output_correction_prompt(reason)
            else:
                if output_json is not None:
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = (
                        "output schema was invalid or incomplete"
                    )
                next_input = "Return one JSON object that exactly matches the required schema."
            previous_interaction_id = interaction_id
        return None, steps

    async def _run_native_interactions(
        self,
        input_context: str,
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run a ReAct loop using Gemini Interactions native function calls."""
        steps: List[StageStep] = []
        evidence_so_far: List[str] = []
        previous_interaction_id = self._session_previous_interaction_id()
        next_input: Any = self._build_session_input(input_context)
        native_tools = self._build_native_tool_schemas()
        system_suffix = ""

        action_turns = 0
        correction_turns = 0
        request_index = 0

        def request_protocol_correction(
            affected_steps: List[StageStep],
            reason: str,
        ) -> None:
            """Reserve one correction-only request or fail without forcing output."""

            nonlocal correction_turns
            if correction_turns >= self.max_protocol_corrections:
                for affected in affected_steps:
                    affected.metadata["react_action_turn"] = action_turns
                    affected.metadata["protocol_corrections_used"] = correction_turns
                    affected.metadata["interaction_request_index"] = request_index
                    affected.metadata["correction_budget_exhausted"] = True
                    affected.metadata["termination_reason"] = (
                        "protocol_correction_budget_exhausted"
                    )
                exc = RuntimeError(
                    f"{self.stage_name or 'stage'} exhausted its protocol correction "
                    f"budget ({self.max_protocol_corrections}): {reason}"
                )
                self._attach_partial_steps(exc, steps)
                raise exc
            correction_turns += 1
            for affected in affected_steps:
                affected.metadata["react_action_turn"] = action_turns
                affected.metadata["protocol_corrections_used"] = correction_turns
                affected.metadata["interaction_request_index"] = request_index

        while action_turns < self.max_rounds:
            request_index += 1
            request_previous_interaction_id = previous_interaction_id
            started = time.perf_counter()
            self.llm_api_calls += 1
            system_instruction = (
                self._build_native_system_content() + system_suffix
            )
            # Gemini Interactions rejects requests that combine native function
            # tools with a structured ``response_format``. Keep tool-bearing
            # ReAct turns unconstrained at the transport layer and validate any
            # completed JSON locally. The forced no-tool output turn below still
            # uses the exact structured response schema.
            response_format = None
            request_generation_config = dict(self.generation_config)
            if self.force_tool_each_round or action_turns < self.min_tool_calls:
                request_generation_config["tool_choice"] = "any"
            try:
                payload = await self._create_interaction(
                    input_payload=next_input,
                    system_instruction=system_instruction,
                    tools=native_tools,
                    previous_interaction_id=request_previous_interaction_id,
                    response_format=response_format,
                    store=True,
                    max_tokens=self.max_output_tokens,
                    generation_config=request_generation_config,
                )
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            try:
                interaction_id, interaction_status = validate_interaction_response(payload)
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            self._advance_interaction_session(interaction_id)

            usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
            tokens = self._usage_tokens(usage)
            common_metadata = {
                "stage": self.stage_name,
                "native_interactions": True,
                "previous_interaction_id": request_previous_interaction_id,
                "interaction_id": interaction_id,
                "interaction_status": interaction_status,
                "llm_duration_ms": duration_ms,
                "policy_input": self._policy_input_snapshot(
                    system_instruction=system_instruction,
                    input_payload=next_input,
                    tools=native_tools,
                    response_format=response_format,
                ),
                "context_request_id": self._last_context_request_id,
                "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
            }
            thought = self._extract_native_thought(payload)
            function_calls = self._extract_native_function_calls(payload)

            if function_calls:
                if (
                    self.max_tool_calls_per_turn is not None
                    and len(function_calls) > self.max_tool_calls_per_turn
                ):
                    step = StageStep(
                        round=request_index,
                        stage_name=self.stage_name,
                        thought=thought,
                        action_type="format_error",
                        tokens=tokens,
                        metadata={
                            **common_metadata,
                            "error_class": "protocol_error",
                            "parallel_tool_calls_rejected": len(function_calls),
                            "policy_action": {
                                "type": "parallel_tool_calls",
                                "calls": deepcopy(function_calls),
                            },
                        },
                    )
                    steps.append(step)
                    previous_interaction_id = interaction_id
                    request_protocol_correction(
                        [step],
                        "only one tool call is allowed per image-only action turn",
                    )
                    next_input = (
                        "Choose exactly one current task and invoke exactly one "
                        "function for the next action."
                    )
                    system_suffix = ""
                    continue
                missing_call_ids = [
                    call for call in function_calls if not str(call.get("id", "")).strip()
                ]
                if missing_call_ids:
                    raise RuntimeError(
                        "Gemini Interactions returned a function_call without a call id."
                    )

                function_results: List[Dict[str, Any]] = []
                response_steps: List[StageStep] = []
                for call_index, call in enumerate(function_calls):
                    call_id = str(call.get("id", "")).strip()
                    tool_name = str(call.get("name", "")).strip()
                    tool_args = self._coerce_native_arguments(call.get("arguments", {}))
                    native_args = self._bind_pending_visual_args(
                        tool_name,
                        dict(tool_args),
                    )
                    prepared_args = self._prepare_tool_args(
                        tool_name,
                        dict(native_args),
                        input_context,
                    )
                    step = StageStep(
                        round=request_index,
                        stage_name=self.stage_name,
                        thought=thought if call_index == 0 else "",
                        tool_name=tool_name,
                        tool_args=dict(prepared_args),
                        tokens=(
                            tokens
                            if call_index == 0
                            else {"prompt": 0, "completion": 0, "thought": 0}
                        ),
                        metadata={
                            **common_metadata,
                            "function_call_id": call_id,
                            "function_call_index": call_index,
                            "function_call_count": len(function_calls),
                            "policy_action": {
                                "type": "tool_call",
                                "name": tool_name,
                                "arguments": deepcopy(tool_args),
                            },
                        },
                    )
                    if call_index > 0:
                        step.metadata.pop("llm_duration_ms", None)

                    error_message = ""
                    if tool_name not in self.tools:
                        error_message = self._unknown_tool_message(tool_name)
                        step.metadata["invalid_tool_name"] = tool_name
                        step.metadata["error_class"] = "protocol_error"
                    else:
                        schema_error = self._validate_native_tool_args(tool_name, native_args)
                        if schema_error:
                            error_message = schema_error
                            step.metadata["invalid_tool_arguments"] = True
                            step.metadata["error_class"] = "protocol_error"
                    if not error_message and self._has_duplicate_tool_call(steps, tool_name, prepared_args):
                        error_message = self._duplicate_tool_message(
                            tool_name,
                            prepared_args,
                        )
                        step.metadata["duplicate_tool_call"] = True
                        step.metadata["error_class"] = "protocol_error"
                    elif not error_message and self._tool_budget_reached(steps, tool_name):
                        error_message = self._tool_budget_message(tool_name)
                        step.metadata["tool_budget_reached"] = True
                        step.metadata["error_class"] = "protocol_error"
                    if not error_message and self.visual_call_validator is not None:
                        visual_error = self.visual_call_validator(tool_name, prepared_args)
                        if visual_error:
                            error_message = visual_error
                            step.metadata["invalid_visual_question"] = True
                            step.metadata["error_class"] = "protocol_error"

                    if error_message:
                        step.action_type = "format_error"
                        step.tool_result = json.dumps(
                            {"status": "error", "error": error_message},
                            ensure_ascii=False,
                        )
                    else:
                        step.tool_args = prepared_args
                        question_error = self._question_id_error(prepared_args)
                        if question_error:
                            step.action_type = "format_error"
                            step.metadata["error_class"] = "protocol_error"
                            step.metadata["invalid_question_id"] = True
                            step.tool_result = json.dumps(
                                {
                                    "status": "error",
                                    "error": question_error,
                                },
                                ensure_ascii=False,
                            )
                        else:
                            filtered_query_count = self._sanitize_search_queries(
                                tool_name,
                                prepared_args,
                            )
                            if filtered_query_count:
                                step.metadata["policy_filtered_query_count"] = (
                                    filtered_query_count
                                )
                            search_policy_error = self._search_policy_error(
                                tool_name,
                                prepared_args,
                            )
                            if search_policy_error:
                                step.action_type = "format_error"
                                step.metadata["error_class"] = "protocol_error"
                                step.metadata["search_policy_rejection"] = True
                                step.tool_result = json.dumps(
                                    {
                                        "status": "error",
                                        "error": search_policy_error,
                                    },
                                    ensure_ascii=False,
                                )
                            else:
                                step.action_type = "tool_call"
                                serialized, tool_metadata = await self._execute_tool(
                                    tool_name,
                                    dict(prepared_args),
                                )
                                step.tool_result = serialized
                                step.metadata.update(tool_metadata)
                                self._archive_tool_step(
                                    step,
                                    action_index=(
                                        len(self.prior_steps) + len(steps) + 1
                                    ),
                                )
                                evidence_so_far.append(
                                    self._summarize_tool_result(
                                        tool_name,
                                        prepared_args,
                                        serialized,
                                    )
                                )

                    steps.append(step)
                    response_steps.append(step)
                    state_update = self._record_observation_update(step, steps)
                    function_results.append(
                        self._build_native_function_result(
                            call_id=call_id,
                            tool_name=tool_name,
                            tool_args=step.tool_args,
                            result=step.tool_result,
                            state_update=state_update,
                            control_step=step,
                        )
                    )

                if any(item.action_type == "tool_call" for item in response_steps):
                    action_turns += 1
                else:
                    request_protocol_correction(
                        response_steps,
                        "the model returned no executable function call",
                    )
                for item in response_steps:
                    item.metadata["react_action_turn"] = action_turns
                    item.metadata["protocol_corrections_used"] = correction_turns
                    item.metadata["interaction_request_index"] = request_index
                previous_interaction_id = interaction_id
                next_input = function_results
                system_suffix = ""
                if self.should_stop and self.should_stop(steps):
                    self._set_session_pending_input(function_results)
                    if self.stop_output_factory is not None:
                        parsed = self.stop_output_factory()
                        steps.append(
                            StageStep(
                                round=request_index + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                metadata={
                                    "stage": self.stage_name,
                                    "deterministic_segment_boundary": True,
                                    "previous_interaction_id": (
                                        previous_interaction_id
                                    ),
                                },
                            )
                        )
                        return parsed, steps
                    return await self._force_native_output(
                        previous_interaction_id=previous_interaction_id,
                        pending_input=next_input,
                        steps=steps,
                        evidence_so_far=evidence_so_far,
                    )
                continue

            content = self._extract_native_text(payload).strip()
            step = StageStep(
                round=request_index,
                stage_name=self.stage_name,
                thought=thought,
                tokens=tokens,
                metadata=common_metadata,
            )
            if not content:
                step.action_type = "format_error"
                step.thought = step.thought or "(empty native response)"
                steps.append(step)
                previous_interaction_id = interaction_id
                request_protocol_correction([step], "the model returned an empty response")
                next_input = (
                    "Do not return final output yet. Invoke exactly one available "
                    "function for an active task."
                    if (
                        self.force_tool_each_round
                        or action_turns < self.min_tool_calls
                    )
                    else (
                        "Return one valid final JSON object, or call one available "
                        "function."
                    )
                )
                system_suffix = ""
                continue

            output_json = self._extract_output(content) or self._try_parse_bare_json(content)
            if output_json is not None:
                step.action_type = "output"
                step.output = output_json
                step.metadata["policy_action"] = deepcopy(output_json)
                steps.append(step)
                parsed = self._validate_output(output_json)
                if parsed is not None:
                    accepted, reason = self._accept_output(parsed, steps)
                    if accepted:
                        return parsed, steps
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = reason
                    previous_interaction_id = interaction_id
                    request_protocol_correction([step], reason)
                    next_input = f"Output rejected: {reason} Continue investigating with one function call."
                    system_suffix = ""
                    continue
                step.action_type = "output_rejected"
                step.metadata["rejection_reason"] = (
                    "output schema was invalid or incomplete"
                )
                next_input = (
                    "The output schema was invalid, and this segment has not yet "
                    "completed its required tool action. Do not return output. "
                    "Invoke exactly one available function for an active task."
                    if (
                        self.force_tool_each_round
                        or action_turns < self.min_tool_calls
                    )
                    else (
                        "The output schema was invalid. Return one valid JSON "
                        "object."
                    )
                )
                previous_interaction_id = interaction_id
                request_protocol_correction(
                    [step],
                    "the output schema was invalid or incomplete",
                )
                system_suffix = ""
                continue

            step.action_type = "format_error"
            step.metadata["native_text_preview"] = content[:500]
            steps.append(step)
            previous_interaction_id = interaction_id
            request_protocol_correction(
                [step],
                "the model returned neither a function call nor valid JSON",
            )
            next_input = (
                "Do not return final output yet. Invoke exactly one available "
                "function for an active task."
                if (
                    self.force_tool_each_round
                    or action_turns < self.min_tool_calls
                )
                else (
                    "Use a native function call, or return exactly one valid final "
                    "JSON object."
                )
            )
            system_suffix = ""

        return await self._force_native_output(
            previous_interaction_id=previous_interaction_id,
            pending_input=next_input,
            steps=steps,
            evidence_so_far=evidence_so_far,
        )

    @staticmethod
    def _attach_partial_steps(exc: Exception, steps: List[StageStep]) -> None:
        if steps and not hasattr(exc, "stage_steps"):
            setattr(exc, "stage_steps", list(steps))

    def _build_native_system_content(self) -> str:
        prompt = re.sub(
            r"Return exactly one JSON object inside <output>\.\.\.</output>",
            "Return exactly one JSON object",
            self.system_prompt,
        )
        return (
            prompt
            + "\n\nNative Gemini Interactions protocol:\n"
            + "- Invoke tools through native function calls. Never write <tool_call> markup.\n"
            + "- You may invoke multiple independent functions in one turn; every call will be executed and returned.\n"
            + (
                f"- Before final output, this segment requires at least "
                f"{self.min_tool_calls} executable function call(s).\n"
                if self.min_tool_calls
                else ""
            )
            + (
                "- Every action turn in this segment must invoke exactly one "
                "function. Final JSON is requested separately after the action "
                "budget.\n"
                if self.force_tool_each_round
                else ""
            )
            + "- When the investigation is complete, return exactly one JSON object. "
            + "Do not wrap it in markdown.\n"
            + "- Tool failures are observations to react to, not successful evidence."
        )

    def _build_native_chat_system_content(self) -> str:
        prompt = re.sub(
            r"Return exactly one JSON object inside <output>\.\.\.</output>",
            "Return exactly one JSON object",
            self.system_prompt,
        )
        if not self.tools_list:
            return prompt + "\n\nReturn one JSON object matching the response schema."
        return (
            prompt
            + "\n\nUse native function calls for tools. Return one JSON object "
            + "when finished; tool failures are not evidence."
        )

    def _openai_response_format(self) -> Optional[Dict[str, Any]]:
        if self.output_schema is None:
            return None
        schema = self._normalized_output_schema()
        if str(getattr(self.llm, "provider", "")).lower() in {
            "qwen_local",
            "lmdeploy",
        }:
            schema = self._lmdeploy_response_schema(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_schema.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    @classmethod
    def _lmdeploy_response_schema(cls, schema: Any) -> Any:
        """Project strict application schemas onto LMDeploy's grammar subset.

        LMDeploy 0.13 warns that string length, pattern and format keywords are
        unsupported. Passing them in nested schemas can make guided decoding
        emit whitespace until the output budget is exhausted. Structural,
        enum, array and numeric constraints remain on the wire; Pydantic still
        enforces the complete original model after decoding.
        """

        if isinstance(schema, list):
            return [cls._lmdeploy_response_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        unsupported = {"minLength", "maxLength", "pattern", "format"}
        return {
            key: cls._lmdeploy_response_schema(value)
            for key, value in schema.items()
            if key not in unsupported
        }

    @staticmethod
    def _openai_assistant_message(
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        result = deepcopy(message)
        result["role"] = "assistant"
        return result

    @classmethod
    def _openai_single_tool_call(
        cls,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        message = cls._openai_assistant_message(payload)
        if message is None:
            return None
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        return calls[0] if isinstance(calls[0], dict) else None

    def _build_native_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in self.tools_list:
            parameters = deepcopy(tool.parameters or {})
            parameters.setdefault("type", "object")
            properties = parameters.setdefault("properties", {})
            required = list(parameters.get("required", []) or [])
            for property_name, allowed_values in (
                self.tool_argument_constraints.get(tool.name, {}).items()
            ):
                if property_name not in properties or not allowed_values:
                    continue
                property_schema = properties[property_name]
                property_type = property_schema.get("type")
                if (
                    property_type == "array"
                    or isinstance(property_type, list)
                    and "array" in property_type
                ):
                    property_schema.setdefault("items", {"type": "string"})
                    property_schema["items"]["enum"] = list(allowed_values)
                else:
                    property_schema["enum"] = list(allowed_values)
                # A runtime constraint represents a concrete executable route,
                # not a hint.  If the base tool marks that selector optional,
                # guided decoding may legally omit it and produce a call the
                # route validator must reject (for example a reference compare
                # without its pending reference URL).
                if property_name not in required:
                    required.append(property_name)
            properties.pop("image_input", None)
            required = [name for name in required if name != "image_input"]
            parameters = self._normalize_native_schema(parameters)
            properties = parameters.setdefault("properties", {})
            if self.stage_name == "verification":
                constrained_question_ids = list(
                    self.tool_argument_constraints.get(tool.name, {}).get(
                        "question_id",
                        [],
                    )
                )
                question_schema: Dict[str, Any] = {
                    "type": "string",
                    "description": (
                        "Active runtime task/question id advanced by this call. "
                        "When enum values are supplied, use one exactly."
                    ),
                }
                if constrained_question_ids:
                    question_schema["enum"] = constrained_question_ids
                elif self.active_question_ids:
                    question_schema["enum"] = self.active_question_ids
                properties["question_id"] = question_schema
                if "question_id" not in required:
                    required.append("question_id")
                claim_ids = list(
                    dict.fromkeys(
                        claim_id
                        for question_id in (
                            constrained_question_ids or self.active_question_ids
                        )
                        for claim_id in self.question_claim_options.get(
                            question_id,
                            {},
                        )
                    )
                )
                if tool.name in {"visit", "crop_and_search"} and claim_ids:
                    properties["claim_id"] = {
                        "type": "string",
                        "enum": claim_ids,
                        "description": (
                            "One runtime-owned ImageClaim whose stance this "
                            "inspection should evaluate."
                        ),
                    }
                    if "claim_id" not in required:
                        required.append("claim_id")
                constrained_fields = self.tool_argument_constraints.get(
                    tool.name,
                    {},
                )
                runtime_bound = {
                    name
                    for name in self._runtime_bound_visual_fields(tool.name)
                    if not constrained_fields.get(name)
                }
                if runtime_bound:
                    required = [name for name in required if name not in runtime_bound]
                    for name in runtime_bound:
                        if name in {"image_claim", "retrieval_goal"}:
                            properties.pop(name, None)
            parameters["required"] = required
            parameters["additionalProperties"] = False
            schemas.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                }
            )
        return schemas

    @classmethod
    def _normalize_native_schema(cls, schema: Any) -> Any:
        if isinstance(schema, list):
            return [cls._normalize_native_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        normalized = {key: cls._normalize_native_schema(value) for key, value in schema.items()}
        schema_type = normalized.get("type")
        if isinstance(schema_type, list):
            if "array" in schema_type:
                normalized["type"] = "array"
                normalized.setdefault("items", {"type": "string"})
            elif "string" in schema_type:
                normalized["type"] = "string"
            elif schema_type:
                normalized["type"] = schema_type[0]
        if normalized.get("type") == "array":
            normalized.setdefault("items", {"type": "string"})
        return normalized

    def _validate_native_tool_args(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return self._unknown_tool_message(tool_name)
        schema = next(
            item["parameters"]
            for item in self._build_native_tool_schemas()
            if item["name"] == tool_name
        )
        properties = schema.get("properties", {}) or {}
        unknown = sorted(set(tool_args) - set(properties))
        if unknown:
            return f"Unknown argument(s) for {tool_name}: {', '.join(unknown)}"
        missing = [name for name in schema.get("required", []) if name not in tool_args]
        if missing:
            return f"Missing required argument(s) for {tool_name}: {', '.join(missing)}"
        for name, value in tool_args.items():
            error = self._validate_schema_value(
                value,
                properties.get(name, {}),
                path=f"Argument '{name}' for {tool_name}",
            )
            if error:
                return error
        return ""

    @classmethod
    def _validate_schema_value(
        cls,
        value: Any,
        spec: Dict[str, Any],
        *,
        path: str,
    ) -> str:
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{path} must be a string."
        if expected == "array":
            if not isinstance(value, list):
                return f"{path} must be an array."
            minimum = spec.get("minItems")
            maximum = spec.get("maxItems")
            if isinstance(minimum, int) and len(value) < minimum:
                return f"{path} must contain at least {minimum} items."
            if isinstance(maximum, int) and len(value) > maximum:
                return f"{path} must contain at most {maximum} items."
            item_spec = spec.get("items")
            if isinstance(item_spec, dict):
                for index, item in enumerate(value):
                    error = cls._validate_schema_value(
                        item,
                        item_spec,
                        path=f"{path}[{index}]",
                    )
                    if error:
                        return error
        if expected == "object":
            if not isinstance(value, dict):
                return f"{path} must be an object."
            properties = spec.get("properties", {}) or {}
            unknown = (
                sorted(set(value) - set(properties))
                if spec.get("additionalProperties") is False
                else []
            )
            if unknown:
                return f"{path} has unknown field(s): {', '.join(unknown)}."
            for name in spec.get("required", []) or []:
                if name not in value:
                    return f"{path} is missing required field '{name}'."
            for name, child in value.items():
                child_spec = properties.get(name)
                if isinstance(child_spec, dict):
                    error = cls._validate_schema_value(
                        child,
                        child_spec,
                        path=f"{path}.{name}",
                    )
                    if error:
                        return error
        if expected == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return f"{path} must be a number."
        if expected == "integer" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return f"{path} must be an integer."
        if expected == "boolean" and not isinstance(value, bool):
            return f"{path} must be a boolean."
        allowed = spec.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            return f"{path} must be one of: " + ", ".join(str(item) for item in allowed)
        return ""

    @staticmethod
    def _usage_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
        return {
            "prompt": int(
                usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0
            ),
            "completion": int(
                usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0
            ),
            "thought": int(
                usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0
            ),
        }

    def _build_native_input(self, input_context: str) -> Any:
        if not (self.attach_image and self.image_path):
            return input_context
        image_url, view = controlled_image_to_data_url(self.image_path)
        self._record_image_view(view, purpose=self.stage_name or "stage")
        if not (image_url.startswith("data:") and ";base64," in image_url):
            return [
                {"type": "text", "text": input_context},
                {"type": "image", "uri": image_url},
            ]
        header, data = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return [
            {"type": "text", "text": input_context},
            {"type": "image", "mime_type": mime_type, "data": data},
        ]

    def _session_previous_interaction_id(self) -> Optional[str]:
        if self.interaction_session is None:
            return None
        return self.interaction_session.previous_interaction_id

    def _build_session_input(self, input_context: str) -> Any:
        current_input = self._build_native_input(input_context)
        if (
            self.interaction_session is None
            or not self.interaction_session.pending_input
        ):
            return current_input
        pending = deepcopy(self.interaction_session.pending_input)
        if isinstance(current_input, list):
            user_content = current_input
        else:
            user_content = [{"type": "text", "text": str(current_input)}]
        return [
            *pending,
            {"type": "user_input", "content": user_content},
        ]

    def _advance_interaction_session(self, interaction_id: str) -> None:
        if self.interaction_session is None:
            return
        self.interaction_session.previous_interaction_id = interaction_id
        self.interaction_session.pending_input = []

    def _set_session_pending_input(
        self,
        pending_input: List[Dict[str, Any]],
    ) -> None:
        if self.interaction_session is None:
            return
        self.interaction_session.pending_input = deepcopy(pending_input)

    @staticmethod
    def _coerce_native_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _extract_native_function_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return extract_function_calls(payload)

    @staticmethod
    def _extract_native_text(payload: Dict[str, Any]) -> str:
        return extract_text(payload)

    @staticmethod
    def _extract_native_thought(payload: Dict[str, Any]) -> str:
        chunks: List[str] = []
        for step in payload.get("steps", []) or []:
            if not isinstance(step, dict) or step.get("type") != "thought":
                continue
            direct_text = step.get("text")
            if isinstance(direct_text, str) and direct_text.strip():
                chunks.append(direct_text)
            for item in step.get("content", []) or []:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    if item["text"].strip():
                        chunks.append(item["text"])
        return "\n".join(chunks)

    def _build_native_function_result(
        self,
        *,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        state_update: Optional[Dict[str, Any]] = None,
        control_step: Optional[StageStep] = None,
    ) -> Dict[str, Any]:
        recorded_step = control_step or StageStep(
            action_type="tool_call" if not self._tool_result_is_error(result) else "format_error",
            tool_name=tool_name,
            tool_args=dict(tool_args),
            metadata=(
                {"investigation_state_update": state_update}
                if state_update
                else {}
            ),
        )
        self._control_steps = list(getattr(self, "_control_steps", [])) + [recorded_step]
        compact = self._compact_tool_result_for_context(tool_name, result)
        question_id = str(tool_args.get("__question_id", "")).strip()
        content: Dict[str, Any] = {
            "function_call_id": call_id,
            "result": compact,
        }
        if question_id:
            content["question_id"] = question_id
        if state_update:
            content["investigation_state_update"] = state_update
        content["agent_control_state"] = self._agent_control_state()
        text = json.dumps(content, ensure_ascii=False, default=str)
        if len(text) > self.tool_response_max_chars:
            content = {
                "function_call_id": call_id,
                "question_id": question_id,
                "agent_control_state": self._agent_control_state(),
                "result": {
                    "truncated": True,
                    "preview": text[: self.tool_response_max_chars - 160],
                },
            }
            text = json.dumps(content, ensure_ascii=False, default=str)
        return {
            "type": "function_result",
            "name": tool_name,
            "call_id": call_id,
            "result": [{"type": "text", "text": text}],
            **({"is_error": True} if self._tool_result_is_error(result) else {}),
        }

    @staticmethod
    def _tool_result_is_error(result: str) -> bool:
        try:
            _, succeeded = parse_tool_result(result)
        except Exception:
            return True
        return not succeeded

    async def _force_native_output(
        self,
        *,
        previous_interaction_id: Optional[str],
        pending_input: Any,
        steps: List[StageStep],
        evidence_so_far: List[str],
    ) -> Tuple[Optional[BaseModel], List[StageStep]]:
        if not previous_interaction_id:
            return None, steps
        directive = (
            "No more tool turns remain. Use the collected evidence and return exactly one "
            "final JSON object now. Be explicit about uncertainty."
        )
        if evidence_so_far:
            directive += "\nCollected evidence:\n" + "\n".join(evidence_so_far[-12:])
        if isinstance(pending_input, list):
            forced_input = list(pending_input)
        else:
            forced_input = f"{pending_input}\n\n{directive}" if pending_input else directive

        system_instruction = (
            self._build_native_system_content() + "\n\n" + directive
        )
        response_format = self._native_response_format()
        request_input: Any = forced_input
        request_parent = previous_interaction_id

        for correction_index in range(2):
            started = time.perf_counter()
            self.llm_api_calls += 1
            try:
                payload = await self._create_interaction(
                    input_payload=request_input,
                    system_instruction=system_instruction,
                    tools=[],
                    previous_interaction_id=request_parent,
                    response_format=response_format,
                    store=True,
                    max_tokens=self.final_output_max_tokens,
                    generation_config=self.final_output_generation_config,
                )
                interaction_id, interaction_status = (
                    validate_interaction_response(payload)
                )
            except Exception as exc:
                self._attach_partial_steps(exc, steps)
                raise
            usage = (
                payload.get("usage", {})
                if isinstance(payload.get("usage"), dict)
                else {}
            )
            metadata = {
                "stage": self.stage_name,
                "native_interactions": True,
                "forced_output": True,
                "forced_output_correction": correction_index > 0,
                "previous_interaction_id": request_parent,
                "interaction_id": interaction_id,
                "interaction_status": interaction_status,
                "llm_duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "max_output_tokens": self.final_output_max_tokens,
                "thinking_level": self.final_output_generation_config.get(
                    "thinking_level"
                ),
                "policy_input": self._policy_input_snapshot(
                    system_instruction=system_instruction,
                    input_payload=request_input,
                    tools=[],
                    response_format=response_format,
                ),
                "context_request_id": self._last_context_request_id,
                "interaction_lifecycle_kind": self._last_interaction_lifecycle_kind,
            }
            tokens = self._usage_tokens(usage)
            if self._extract_native_function_calls(payload):
                function_calls = self._extract_native_function_calls(payload)
                reason = (
                    "model requested another function after the tool budget ended"
                )
                metadata["rejection_reason"] = reason
                metadata["policy_action"] = {
                    "type": (
                        "parallel_tool_calls"
                        if len(function_calls) > 1
                        else "tool_call"
                    ),
                    "calls": deepcopy(function_calls),
                }
                output_json = None
                parsed = None
            else:
                content = self._extract_native_text(payload)
                output_json = (
                    self._extract_output(content)
                    or self._try_parse_bare_json(content)
                )
                if output_json is not None:
                    metadata["policy_action"] = deepcopy(output_json)
                parsed = (
                    self._validate_output(output_json)
                    if output_json is not None
                    else None
                )
                if parsed is not None:
                    accepted, reason = self._accept_output(
                        parsed,
                        steps,
                        final_attempt=True,
                    )
                    if accepted:
                        steps.append(
                            StageStep(
                                round=len(steps) + 1,
                                stage_name=self.stage_name,
                                action_type="output",
                                output=parsed.model_dump(),
                                tokens=tokens,
                                metadata=metadata,
                            )
                        )
                        return parsed, steps
                    metadata["rejection_reason"] = reason
                else:
                    reason = (
                        "output schema was invalid or incomplete"
                        if output_json is not None
                        else "model returned no structured output"
                    )
                    metadata["rejection_reason"] = reason

            steps.append(
                StageStep(
                    round=len(steps) + 1,
                    stage_name=self.stage_name,
                    action_type=(
                        "output_rejected"
                        if output_json is not None
                        else "format_error"
                    ),
                    output=output_json,
                    tokens=tokens,
                    metadata=metadata,
                )
            )
            if correction_index == 0:
                request_parent = interaction_id
                request_input = (
                    f"Output rejected: {reason}. Return a corrected JSON object. "
                    "Use only fields allowed by the response schema and do not "
                    "invent runtime ids or state transitions."
                )
        return None, steps

    def _native_response_format(self) -> Optional[Dict[str, Any]]:
        if self.output_schema is None:
            return None
        return {
            "type": "text",
            "mime_type": "application/json",
            "schema": self._gemini_response_schema(
                self._normalized_output_schema()
            ),
        }

    @classmethod
    def _gemini_response_schema(cls, schema: Any) -> Any:
        """Remove response-schema keywords rejected by Gemini Interactions.

        ``maxItems`` is accepted in tool schemas and some direct media requests,
        but Gemini rejects it in a structured response schema that continues a
        native function-call interaction. Pydantic still enforces the original
        list bounds after the response is returned.
        """

        if isinstance(schema, list):
            return [cls._gemini_response_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        return {
            key: cls._gemini_response_schema(value)
            for key, value in schema.items()
            if key != "maxItems"
        }

    @staticmethod
    def _policy_input_snapshot(
        *,
        system_instruction: str,
        input_payload: Any,
        tools: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist model-visible structure without embedding binary media payloads."""

        return {
            "system_instruction": system_instruction,
            "input_payload": StageRunner._snapshot_input_payload(input_payload),
            "tools": deepcopy(tools),
            "response_format": deepcopy(response_format),
        }

    @staticmethod
    def _snapshot_input_payload(value: Any) -> Any:
        """Replace image bytes/URIs with a stable runtime-image reference."""

        if isinstance(value, list):
            return [
                StageRunner._snapshot_input_payload(item)
                for item in value
            ]
        if not isinstance(value, dict):
            return deepcopy(value)
        media_type = str(value.get("type", "")).strip().lower()
        if media_type in {"image", "image_url"}:
            snapshot: Dict[str, Any] = {
                "type": media_type,
                "runtime_image": True,
            }
            mime_type = str(value.get("mime_type", "")).strip()
            if mime_type:
                snapshot["mime_type"] = mime_type
            return snapshot
        return {
            key: StageRunner._snapshot_input_payload(item)
            for key, item in value.items()
        }

    def _normalized_output_schema(self) -> Dict[str, Any]:
        if self.output_schema is None:
            return {}
        return normalize_json_schema(
            self.output_schema.model_json_schema(),
            require_all_properties=True,
        )

    def _build_system_content(self) -> str:
        if not self.tools_list:
            return self.system_prompt + self._output_format_instructions()

        lines = [self.system_prompt, "", "Available tools:"]
        for tool in self.tools_list:
            required = tool.parameters.get("required", [])
            props = list((tool.parameters.get("properties", {}) or {}).keys())
            line = f"- {tool.name}"
            if required:
                line += f" | required: {', '.join(required)}"
            if props:
                line += f" | args: {', '.join(props[:5])}"
            lines.append(line)
        lines.append("")
        lines.append(self._output_format_instructions())
        return "\n".join(lines)

    def _accept_output(
        self,
        parsed: BaseModel,
        steps: List[StageStep],
        *,
        final_attempt: bool = False,
    ) -> Tuple[bool, str]:
        tool_calls = sum(1 for step in steps if step.action_type == "tool_call")
        if tool_calls < self.min_tool_calls:
            return False, f"at least {self.min_tool_calls} tool calls are required; only {tool_calls} completed"
        required_question_error = self._required_question_output_error(steps)
        if required_question_error:
            suffix = (
                " No more tool turns remain."
                if final_attempt and self.tools_list
                else " This is the final correction attempt for this stage; a "
                "non-terminal output remains valid."
                if final_attempt
                else ""
            )
            return False, required_question_error + suffix
        if self.output_validator:
            accepted, reason = self.output_validator(parsed, steps)
            if not accepted:
                suffix = (
                    " No more tool turns remain."
                    if final_attempt and self.tools_list
                    else " This is the final correction attempt for this stage; a "
                    "non-terminal output remains valid."
                    if final_attempt
                    else ""
                )
                return False, reason + suffix
        return True, ""

    def _prepare_tool_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        input_context: str,
    ) -> Dict[str, Any]:
        if self.stage_name != "verification":
            return tool_args
        question_id = str(tool_args.pop("question_id", "")).strip()
        if question_id:
            tool_args["__question_id"] = question_id
            selected_claim_id = str(tool_args.pop("claim_id", "")).strip()
            claim_options = self.question_claim_options.get(question_id, {})
            claim_text = str(
                claim_options.get(selected_claim_id)
                or self.question_claims.get(question_id, "")
            ).strip()
            if selected_claim_id:
                tool_args["__claim_id"] = selected_claim_id
            if claim_text:
                tool_args["__claim_text"] = claim_text
            evidence_goal = self.question_evidence_goals.get(question_id, "").strip()
            if evidence_goal:
                tool_args["__evidence_goal"] = evidence_goal
            tool = self.tools.get(tool_name)
            runtime_properties = (
                tool.parameters.get("properties", {})
                if tool is not None
                else {}
            )
            if (
                tool_name in {"visit", "crop_and_search"}
                and "image_claim" in runtime_properties
                and "retrieval_goal" in runtime_properties
            ):
                tool_args.pop("goal", None)
                if claim_text:
                    tool_args["image_claim"] = claim_text
                requested_goal = str(tool_args.get("retrieval_goal", "")).strip()
                if requested_goal or evidence_goal or claim_text:
                    tool_args["retrieval_goal"] = (
                        requested_goal or evidence_goal or claim_text
                    )
            elif tool_name == "text_search":
                immutable_goal = evidence_goal or claim_text
                if immutable_goal:
                    tool_args["goal"] = immutable_goal
        return tool_args

    def _bind_pending_visual_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        bound = dict(tool_args)
        visual_question_id = str(bound.get("visual_question_id", "")).strip()
        if self.stage_name != "verification" or not visual_question_id:
            return bound
        spec = next(
            (
                item
                for item in self._pending_visual_questions()
                if str(item.get("visual_question_id", "")).strip() == visual_question_id
            ),
            None,
        )
        if not spec:
            return bound
        question_id = str(bound.get("__question_id", "")).strip()
        claim_id = str(spec.get("claim_id", "")).strip()
        if question_id and claim_id and claim_id != f"claim-{question_id}":
            return bound

        source_evidence_id = str(spec.get("source_evidence_id", "")).strip()
        source_discovery_id = str(spec.get("source_discovery_id", "")).strip()
        expected_property = str(spec.get("expected_property", "")).strip()
        target_bbox = list(spec.get("target_bbox") or [])
        reference_url = str(spec.get("reference_image_url", "")).strip()

        if source_evidence_id:
            bound["source_evidence_id"] = source_evidence_id
        if source_discovery_id:
            bound["source_discovery_id"] = source_discovery_id
        if expected_property:
            bound["expected_property"] = expected_property
        if target_bbox and tool_name in {"crop_and_inspect", "crop_and_search", "count_objects", "ocr_with_position"}:
            bound["bbox"] = target_bbox
        if reference_url and tool_name == "compare_with_reference":
            bound["reference_url"] = reference_url
        if expected_property:
            if tool_name == "crop_and_inspect":
                bound.setdefault("focus_question", expected_property)
            elif tool_name == "ocr_with_position":
                bound.setdefault("goal", expected_property)
            elif tool_name == "count_objects":
                bound.setdefault("target_object", expected_property[:200])
            elif tool_name == "compare_with_reference":
                bound.setdefault("focus", expected_property[:400])
        return bound

    def _runtime_bound_visual_fields(self, tool_name: str) -> set[str]:
        if tool_name == "visit":
            return (
                {"image_claim"}
                if self.question_claim_options
                else {"image_claim", "retrieval_goal"}
            )
        if tool_name == "compare_with_reference":
            return {"reference_url"}
        if tool_name == "crop_and_inspect":
            return {"bbox", "focus_question"}
        if tool_name == "crop_and_search":
            return (
                {"bbox", "image_claim"}
                if self.question_claim_options
                else {"bbox", "image_claim", "retrieval_goal"}
            )
        if tool_name == "ocr_with_position":
            return {"bbox", "goal"}
        if tool_name == "count_objects":
            return {"bbox", "target_object"}
        return set()

    def _question_id_error(self, tool_args: Dict[str, Any]) -> str:
        if self.stage_name != "verification":
            return ""
        question_id = str(tool_args.get("__question_id", "")).strip()
        if not question_id:
            return "question_id is required for every verification tool call."
        if self.active_question_ids and question_id not in self.active_question_ids:
            return (
                f"Unknown question_id '{question_id}'. Valid ids: "
                + ", ".join(self.active_question_ids)
            )
        if (
            self.question_is_active is not None
            and not self.question_is_active(question_id)
        ):
            return (
                f"Task '{question_id}' is already resolved, blocked, or exhausted. "
                "Choose a currently active task."
            )
        return ""

    def _priority_coverage_error(
        self,
        tool_args: Dict[str, Any],
        current_steps: List[StageStep],
    ) -> str:
        """Keep ReAct tool selection free; coverage is enforced before output."""

        # The outer Coverage Audit validates service and evidence after each iteration.
        # Enforcing a P1/P2 call order here turns the agent into a fixed scheduler and
        # creates avoidable rejected turns.
        return ""

    def _search_policy_error(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if not self.source_access_policy.active or tool_name != "text_search":
            return ""
        queries = tool_args.get("queries", [])
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            return ""
        usable = [str(query).strip() for query in queries if str(query).strip()]
        if usable and not any(
            query_targets_fact_check_answer(query)
            or self.source_access_policy.blocked_query_reference(query)
            for query in usable
        ):
            return ""
        return (
            "Search policy removed every query because it targeted a ready-made "
            "fact-check verdict or an excluded source. Reformulate with claim terms, "
            "an original statement, an official record, a primary source, or "
            "independent reporting."
        )

    def _sanitize_search_queries(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> int:
        """Drop prohibited retrieval strings without discarding usable sibling queries."""

        if not self.source_access_policy.active or tool_name != "text_search":
            return 0
        raw_queries = tool_args.get("queries", [])
        queries = [raw_queries] if isinstance(raw_queries, str) else raw_queries
        if not isinstance(queries, list):
            return 0
        allowed: List[str] = []
        rejected: List[str] = []
        for raw_query in queries:
            query = str(raw_query).strip()
            if not query:
                continue
            if (
                query_targets_fact_check_answer(query)
                or self.source_access_policy.blocked_query_reference(query)
            ):
                rejected.append(query)
            else:
                allowed.append(query)
        if rejected:
            tool_args["queries"] = allowed
        return len(rejected)

    def _resolved_question_ids(self, current_steps: List[StageStep]) -> set[str]:
        resolved = set(self.resolved_priority_question_ids)
        resolved.update(self.resolved_supporting_question_ids)
        for step in [*self.prior_steps, *current_steps, *list(getattr(self, "_control_steps", []))]:
            update = (step.metadata or {}).get("investigation_state_update", {})
            if not isinstance(update, dict):
                continue
            delta = update.get("belief_delta", {})
            if not isinstance(delta, dict):
                continue
            claim_id = str(delta.get("claim_id", ""))
            if (
                claim_id.startswith("claim-")
                and str(delta.get("new_status", "")) in {"supported", "refuted"}
            ):
                resolved.add(claim_id.removeprefix("claim-"))
        return resolved

    def _required_question_output_error(self, current_steps: List[StageStep]) -> str:
        if self.stage_name != "verification":
            return ""
        resolved = self._resolved_question_ids(current_steps)
        attempted = {
            str(step.tool_args.get("__question_id", "")).strip()
            for step in [*self.prior_steps, *current_steps]
            if step.action_type == "tool_call"
            and str(step.tool_args.get("__question_id", "")).strip()
        }
        missing_p1 = [
            item
            for item in self.priority_question_ids
            if item not in resolved and item not in attempted
        ]
        missing_p2 = [
            item
            for item in self.supporting_question_ids
            if item not in resolved and item not in attempted
        ]
        if not missing_p1 and not missing_p2:
            return ""
        parts = []
        if missing_p1:
            parts.append("untouched P1: " + ", ".join(missing_p1))
        if missing_p2:
            parts.append("untouched P2: " + ", ".join(missing_p2))
        return "required investigation questions still need a tool attempt (" + "; ".join(parts) + ")"

    def _pending_visual_call_is_valid(self, tool_args: Dict[str, Any]) -> bool:
        visual_question_id = str(tool_args.get("visual_question_id", "")).strip()
        question_id = str(tool_args.get("__question_id", "")).strip()
        for step in [*self.prior_steps, *list(getattr(self, "_control_steps", []))]:
            update = (step.metadata or {}).get("investigation_state_update", {})
            if not isinstance(update, dict):
                continue
            created = update.get("created_visual_questions", []) or []
            for item in created:
                if not isinstance(item, dict):
                    continue
                if (
                    str(item.get("visual_question_id", "")) == visual_question_id
                    and str(item.get("claim_id", "")) == f"claim-{question_id}"
                    and str(item.get("status", "pending")) == "pending"
                ):
                    return visual_question_id in self._pending_visual_question_ids()
        return False

    def _agent_control_state(self) -> Dict[str, Any]:
        steps = list(self.prior_steps) + list(getattr(self, "_control_steps", []))
        attempts = {question_id: 0 for question_id in self.active_question_ids}
        for step in steps:
            if step.action_type != "tool_call":
                continue
            question_id = str(step.tool_args.get("__question_id", "")).strip()
            if question_id:
                attempts[question_id] = attempts.get(question_id, 0) + 1
        untouched_priority = [
            question_id
            for question_id in self.priority_question_ids
            if attempts.get(question_id, 0) == 0
        ]
        untouched_supporting = [
            question_id
            for question_id in self.supporting_question_ids
            if attempts.get(question_id, 0) == 0
        ]
        remaining = {}
        for tool_name, limit in self.tool_call_limits.items():
            used = sum(
                1
                for step in steps
                if step.action_type == "tool_call" and step.tool_name == tool_name
            )
            remaining[tool_name] = max(0, int(limit) - used)
        claim_states = self._claim_control_states()
        return {
            "question_attempts": attempts,
            "untouched_priority_question_ids": untouched_priority,
            "untouched_supporting_question_ids": untouched_supporting,
            "remaining_tool_budgets": remaining,
            "pending_visual_question_ids": self._pending_visual_question_ids(),
            "pending_visual_questions": self._pending_visual_questions(),
            "claim_states": claim_states,
            "next_action_guidance": self._next_action_guidance(
                untouched_priority,
                untouched_supporting,
                claim_states,
            ),
        }

    def _next_action_guidance(
        self,
        untouched_priority: List[str],
        untouched_supporting: List[str],
        claim_states: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if untouched_priority:
            return "Attempt one untouched P1 question before resampling."
        if untouched_supporting:
            return "Attempt one untouched P2 question before further P1 resampling."
        pending = self._pending_visual_questions()
        if pending:
            return "Resolve a pending ReInspect specification with its recommended tool."
        unresolved_priority = [
            item
            for item in self.priority_question_ids
            if item not in self._resolved_question_ids([])
        ]
        if unresolved_priority:
            unresolved_rows = [
                item for item in (claim_states or [])
                if item.get("question_id") in unresolved_priority
            ]
            if unresolved_rows:
                details = "; ".join(
                    f"{item['question_id']}: {item.get('remaining_gap', 'needs decisive evidence')}"
                    for item in unresolved_rows
                )
                return (
                    "Collect an independent direct official/news source for unresolved P1; "
                    "do not resubmit rejected or UGC-only evidence. " + details
                )
            return (
                "Choose the highest-value direct-source follow-up for unresolved P1: "
                + ", ".join(unresolved_priority)
            )
        return "Finalize when the evidence ledger can support the required output."

    def _claim_control_states(self) -> List[Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for step in [*self.prior_steps, *list(getattr(self, "_control_steps", []))]:
            update = (step.metadata or {}).get("investigation_state_update", {})
            if not isinstance(update, dict):
                continue
            delta = update.get("belief_delta", {})
            if not isinstance(delta, dict):
                continue
            claim_id = str(delta.get("claim_id", "")).strip()
            if not claim_id.startswith("claim-"):
                continue
            question_id = claim_id.removeprefix("claim-")
            latest[question_id] = {
                "question_id": question_id,
                "status": str(delta.get("new_status", "open")),
                "operation": str(delta.get("operation", "zero")),
                "remaining_gap": str(delta.get("explanation", ""))[:500],
            }
        return [latest[key] for key in sorted(latest)]

    def _pending_visual_question_ids(self) -> List[str]:
        return [
            str(item["visual_question_id"])
            for item in self._pending_visual_questions()
        ]

    def _pending_visual_questions(self) -> List[Dict[str, Any]]:
        pending: List[str] = []
        specs: Dict[str, Dict[str, Any]] = {}
        for step in [*self.prior_steps, *list(getattr(self, "_control_steps", []))]:
            update = (step.metadata or {}).get("investigation_state_update", {})
            if not isinstance(update, dict):
                continue
            for item in update.get("created_visual_questions", []) or []:
                if isinstance(item, dict) and item.get("visual_question_id"):
                    visual_id = str(item["visual_question_id"])
                    pending.append(visual_id)
                    specs[visual_id] = dict(item)
            for item in update.get("resolved_visual_questions", []) or []:
                if isinstance(item, dict) and item.get("visual_question_id"):
                    resolved_id = str(item["visual_question_id"])
                    pending = [value for value in pending if value != resolved_id]
                    specs.pop(resolved_id, None)
        return [specs[value] for value in dict.fromkeys(pending) if value in specs]

    @staticmethod
    def _output_format_instructions() -> str:
        return (
            "Each round must produce exactly one action:\n"
            "1. <think>reasoning</think><tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n"
            "2. <think>reasoning</think><output>{...}</output>"
        )

    def _build_user_message(self, input_context: str) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if self.attach_image and self.image_path:
            try:
                image_url, view = controlled_image_to_data_url(self.image_path)
                self._record_image_view(view, purpose=self.stage_name or "stage")
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
            except Exception:
                pass
        parts.append({"type": "text", "text": input_context})
        if len(parts) == 1 and parts[0]["type"] == "text":
            return {"role": "user", "content": parts[0]["text"]}
        return {"role": "user", "content": parts}

    def _record_image_view(self, view: Dict[str, Any], *, purpose: str) -> None:
        fingerprint = json.dumps(view, sort_keys=True, default=str)
        if fingerprint == self._last_image_view.get("fingerprint"):
            return
        self._last_image_view = {"fingerprint": fingerprint, **dict(view)}
        if self.runtime_store is None:
            return
        self.runtime_store.append_event(
            "image_view",
            {
                "stage": self.stage_name,
                "purpose": purpose,
                "image_id": "input-image",
                "crop": None,
                "before_understanding_version": None,
                "after_understanding_version": None,
                "decision_impact": "pending",
                **dict(view),
            },
        )

    def _uses_native_chat_completions(self) -> bool:
        return bool(
            str(getattr(self.llm, "provider", "")).lower() == "qwen_local"
            and str(getattr(self.llm, "wire_api", "")).lower()
            == "chat_completions"
        )

    def _recent_recalled_materials(self) -> List[Dict[str, Any]]:
        materials: List[Dict[str, Any]] = []
        for step in self.prior_steps:
            if getattr(step, "action_type", "") != "tool_call":
                continue
            if getattr(step, "tool_name", "") != "read_evidence":
                continue
            try:
                payload = json.loads(str(getattr(step, "tool_result", "")))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "success":
                materials.append(
                    {
                        "memory_id": payload.get("memory_id"),
                        "artifact": payload.get("artifact", {}),
                        "offset": payload.get("offset", 0),
                        "end": payload.get("end", 0),
                        "content": str(payload.get("content", ""))[:24000],
                    }
                )
        return materials[-4:]

    def _build_round_messages(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        history: List[Dict[str, Any]],
        evidence_so_far: List[str],
    ) -> List[Dict[str, Any]]:
        messages = [system_msg, user_msg]
        if evidence_so_far:
            summary = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(evidence_so_far[-8:]))
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>Reviewing collected evidence.</think>\n## Evidence so far\n{summary}",
                }
            )
            messages.append({"role": "user", "content": "Continue from the current strongest next step."})

        recent = history[2:]
        if len(recent) > self.recent_rounds_to_keep * 2:
            recent = recent[-(self.recent_rounds_to_keep * 2):]
        messages.extend(recent)
        return messages

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        *,
        require_tool: bool = False,
        max_tokens: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        lifecycle_kind: str = "",
        parent_context_request_id: str = "",
        suppress_tools: bool = False,
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        started = time.perf_counter()
        self.llm_api_calls += 1
        request_kwargs: Dict[str, Any] = {}
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.max_output_tokens
        )
        effective_generation_config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        if effective_generation_config:
            request_kwargs["generation_config"] = dict(
                effective_generation_config
            )
        response_format: Optional[Dict[str, Any]] = None
        if self._uses_native_chat_completions():
            if self.tools_list and not suppress_tools:
                request_kwargs["tools"] = self._build_native_tool_schemas()
                request_kwargs["tool_choice"] = (
                    "required" if require_tool else "auto"
                )
            elif self.output_schema is not None:
                response_format = self._openai_response_format()
                request_kwargs["response_format"] = response_format
        request_id = ""
        effective_lifecycle_kind = lifecycle_kind.strip() or (
            "tool_roundtrip"
            if self._uses_native_chat_completions() and bool(self.tools_list)
            else "standalone_request"
        )
        effective_parent_request_id = parent_context_request_id.strip() or None
        if (
            effective_lifecycle_kind == "protocol_correction"
            and effective_parent_request_id is None
            and self.runtime_store is not None
        ):
            raise RuntimeError(
                "protocol_correction requires a parent context request"
            )
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage=self.stage_name,
                lifecycle_kind=effective_lifecycle_kind,
                system_instruction="",
                input_payload=messages,
                tools=request_kwargs.get("tools"),
                response_format=response_format,
                generation_config=request_kwargs.get("generation_config"),
                max_output_tokens=effective_max_tokens,
                parent_request_id=effective_parent_request_id,
                model=str(getattr(self.llm, "model_name", "")),
                prompt_version=self.prompt_version,
            )
        try:
            response = await asyncio.wait_for(
                self.llm.get_response(messages, **request_kwargs),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error="request timeout",
                )
            raise TimeoutError(
                f"{self.stage_name or 'stage'} model request exceeded "
                f"{self.request_timeout_seconds:.1f}s"
            ) from exc
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        raw = response.raw if isinstance(response.raw, dict) else {}
        choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        reasoning_field = ""
        reasoning_text = ""
        for candidate_field in ("reasoning", "reasoning_content"):
            candidate = message.get(candidate_field)
            if isinstance(candidate, str) and candidate.strip():
                reasoning_field = candidate_field
                reasoning_text = candidate
                break
        response_content_chars = len(str(message.get("content") or ""))
        response_reasoning_chars = len(reasoning_text)
        reasoning_artifact: Optional[Dict[str, Any]] = None
        if self.runtime_store is not None and reasoning_text:
            reasoning_artifact = self.runtime_store.artifacts.put_text(
                reasoning_text,
                media_type="text/plain; charset=utf-8",
                suffix=".txt",
                metadata={
                    "kind": "model_reasoning",
                    "stage": self.stage_name,
                    "request_id": request_id,
                    "provider_field": reasoning_field,
                    "excluded_from_model_context": True,
                },
            )
            self.runtime_store.append_event(
                "model_reasoning_archived",
                {
                    "stage": self.stage_name,
                    "request_id": request_id,
                    "provider_field": reasoning_field,
                    "char_count": response_reasoning_chars,
                    "artifact": reasoning_artifact,
                    "excluded_from_model_context": True,
                },
            )
        if request_id:
            usage = (
                raw.get("usage", {})
                if isinstance(raw.get("usage"), dict)
                else {
                    "input_tokens": response.prompt_tokens,
                    "output_tokens": response.completion_tokens,
                }
            )
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=usage,
                status="completed",
                response_metadata={
                    "response_content_chars": response_content_chars,
                    "response_reasoning_chars": response_reasoning_chars,
                    "reasoning_artifact": reasoning_artifact,
                },
            )
        self._last_context_request_id = request_id
        return response, {
            "llm_duration_ms": duration_ms,
            "context_request_id": request_id,
            "parent_context_request_id": effective_parent_request_id,
            "interaction_lifecycle_kind": effective_lifecycle_kind,
            "native_chat_completions": self._uses_native_chat_completions(),
            "finish_reason": str(choice.get("finish_reason", "")),
            "response_content_chars": response_content_chars,
            "response_reasoning_chars": response_reasoning_chars,
            "reasoning_artifact": reasoning_artifact,
        }

    async def _create_interaction(self, **kwargs: Any) -> Dict[str, Any]:
        """Apply one wall-clock deadline to every native Gemini request."""

        explicit_lifecycle = str(kwargs.pop("lifecycle_kind", "")).strip()
        request_id = ""
        tools = kwargs.get("tools") or []
        previous_interaction_id = kwargs.get("previous_interaction_id")
        lifecycle_kind = explicit_lifecycle or (
            "tool_roundtrip"
            if tools
            else "protocol_correction"
            if previous_interaction_id
            else "standalone_request"
        )
        if (
            previous_interaction_id
            and lifecycle_kind == "standalone_request"
        ):
            raise RuntimeError(
                "standalone_request cannot inherit a previous interaction"
            )
        self._last_interaction_lifecycle_kind = lifecycle_kind
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage=self.stage_name,
                lifecycle_kind=lifecycle_kind,
                system_instruction=kwargs.get("system_instruction"),
                input_payload=kwargs.get("input_payload"),
                tools=tools,
                response_format=kwargs.get("response_format"),
                generation_config=kwargs.get("generation_config"),
                max_output_tokens=kwargs.get("max_tokens"),
                previous_interaction_id=previous_interaction_id,
                model=str(getattr(self.llm, "model_name", "")),
                prompt_version=self.prompt_version,
            )
        try:
            payload = await asyncio.wait_for(
                self.llm.create_interaction(**kwargs),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error="request timeout",
                )
            raise TimeoutError(
                f"{self.stage_name or 'stage'} Gemini request exceeded "
                f"{self.request_timeout_seconds:.1f}s"
            ) from exc
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        if request_id:
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=(
                    payload.get("usage", {})
                    if isinstance(payload.get("usage"), dict)
                    else {}
                ),
                interaction_id=str(payload.get("id", "")).strip() or None,
                status=str(payload.get("status", "completed")),
            )
        self._last_context_request_id = request_id
        return payload

    @staticmethod
    def _extract_think(content: str) -> str:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_output(content: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        if not match:
            return None
        block = StageRunner._strip_markdown_fence(match.group(1).strip())
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_tool_call(content: str) -> Tuple[str, Dict[str, Any]]:
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not match:
            return "", {}
        block = StageRunner._strip_markdown_fence(match.group(1).strip())
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return "", {}
        return str(parsed.get("name", "")).strip(), parsed.get("arguments", {}) or {}

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        tool = self.tools[tool_name]
        tool_args.pop("__question_id", None)
        tool_args.pop("__claim_id", None)
        tool_args.pop("__claim_text", None)
        tool_args.pop("__evidence_goal", None)
        properties = tool.parameters.get("properties", {})
        if "image_input" in properties:
            tool_args["image_input"] = self.image_path
        if hasattr(tool, "image_path") and self.image_path:
            tool.image_path = self.image_path

        cache_args = self._build_cache_args(tool_name, tool_args)
        started = time.perf_counter()
        if self.tool_cache and tool_name in self.cacheable_tools:
            cached = self.tool_cache.get(tool_name, cache_args)
            if cached is not None:
                cached_result, succeeded = parse_tool_result(cached)
                if succeeded and self.source_access_policy.active:
                    sanitized, filtered_count = self.source_access_policy.sanitize_payload(cached_result)
                    if sanitized is None:
                        succeeded = False
                        cached = json.dumps(
                            {
                                "status": "error",
                                "error": "Cached tool result blocked by the active source access policy.",
                            },
                            ensure_ascii=False,
                        )
                    else:
                        if filtered_count:
                            sanitized["policy_filtered_count"] = int(
                                sanitized.get("policy_filtered_count", 0) or 0
                            ) + filtered_count
                        cached, succeeded = serialize_tool_result(sanitized)
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
                    timeout=self.tool_timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.call, tool_args),
                    timeout=self.tool_timeout_seconds,
                )
        except asyncio.TimeoutError:
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"ToolActionTimeout: {tool_name} exceeded "
                        f"{self.tool_timeout_seconds:.1f}s"
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
                "tool_timeout_seconds": self.tool_timeout_seconds,
                "tool_llm_api_calls": 0,
                "tool_tokens": self._normalize_tool_tokens(None),
            }
        except Exception as exc:
            serialized = json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
            runtime_metrics = exception_runtime_metrics(exc)
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": type(exc).__name__,
                "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
                "tool_tokens": self._normalize_tool_tokens(runtime_metrics.get("tokens")),
            }

        if not isinstance(result, dict):
            runtime_metrics: Dict[str, Any] = {}
            tool_subcalls: List[Dict[str, Any]] = []
        else:
            raw_subcalls = result.get("subcalls", [])
            tool_subcalls = [
                dict(item)
                for item in raw_subcalls
                if isinstance(item, dict)
            ] if isinstance(raw_subcalls, list) else []
            runtime_metrics = take_runtime_metrics(result)
        try:
            serialized, succeeded = serialize_tool_result(result)
        except ToolResultContractError as exc:
            serialized = json.dumps(
                {"status": "error", "error": f"ToolResultContractError: {exc}"},
                ensure_ascii=False,
            )
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": "ToolResultContractError",
                "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
                "tool_tokens": self._normalize_tool_tokens(runtime_metrics.get("tokens")),
                "tool_subcalls": tool_subcalls,
            }
        if succeeded:
            parsed_result, _ = parse_tool_result(serialized)
            canonical = self._canonical_tool_result(
                tool_name,
                parsed_result,
            )
            serialized, succeeded = serialize_tool_result(canonical)
        if succeeded and self.source_access_policy.active:
            parsed_result, _ = parse_tool_result(serialized)
            sanitized, filtered_count = self.source_access_policy.sanitize_payload(parsed_result)
            if sanitized is None:
                serialized = json.dumps(
                    {
                        "status": "error",
                        "error": "Tool result blocked by the active source access policy.",
                    },
                    ensure_ascii=False,
                )
                succeeded = False
            else:
                if filtered_count:
                    sanitized["policy_filtered_count"] = int(
                        sanitized.get("policy_filtered_count", 0) or 0
                    ) + filtered_count
                serialized, succeeded = serialize_tool_result(sanitized)
        if succeeded and self.tool_cache and tool_name in self.cacheable_tools:
            self.tool_cache.put(tool_name, cache_args, serialized)
        return serialized, {
            "cache_hit": False,
            "tool_success": succeeded,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "serialized_size": len(serialized),
            "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
            "tool_tokens": self._normalize_tool_tokens(runtime_metrics.get("tokens")),
            "tool_subcalls": tool_subcalls,
        }

    @staticmethod
    def _normalize_tool_tokens(value: Any) -> Dict[str, int]:
        value = value if isinstance(value, dict) else {}
        return {
            name: int(value.get(name, 0) or 0)
            for name in ("prompt", "completion", "thought")
        }

    def _build_cache_args(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(tool_args)
        args.pop("__question_id", None)
        args.pop("__claim_id", None)
        args.pop("__claim_text", None)
        args.pop("__evidence_goal", None)
        if tool_name in {"compare_with_reference", "analyze_visual_anomalies"} and self.image_path:
            args["__image_input__"] = self.image_path
        if tool_name in {"visit", "crop_and_search"}:
            args["__web_evidence_contract__"] = WEB_EVIDENCE_CONTRACT_VERSION
        return args

    def _build_tool_response_message(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        *,
        function_call_id: str,
        state_update: Optional[Dict[str, Any]] = None,
        native_chat: bool = False,
    ) -> str:
        compact = self._compact_tool_result_for_context(tool_name, result)
        payload = {
            "function_call_id": function_call_id,
            "tool": tool_name,
            "arguments": tool_args,
            "result": compact,
        }
        if state_update:
            payload["investigation_state_update"] = state_update
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if native_chat else 2,
            separators=(",", ":") if native_chat else None,
        )
        if not native_chat and len(text) > self.tool_response_max_chars:
            text = text[: self.tool_response_max_chars] + "\n...<truncated>"
        if native_chat:
            return text
        return f"<tool_response>\n{text}\n</tool_response>"

    def _record_observation_update(
        self,
        step: StageStep,
        steps: List[StageStep],
    ) -> Optional[Dict[str, Any]]:
        if self.observation_callback is None or step.action_type != "tool_call":
            return None
        update = self.observation_callback(step, list(self.prior_steps) + list(steps))
        if update:
            step.metadata["investigation_state_update"] = update
            memory_id = str(
                (step.metadata.get("tool_result_artifact") or {}).get(
                    "memory_id", ""
                )
            ).strip()
            if self.runtime_store is not None and memory_id:
                self.runtime_store.bind_archive_lineage(memory_id, update)
        return update

    def _archive_tool_step(self, step: StageStep, *, action_index: int) -> None:
        if self.runtime_store is None or step.action_type != "tool_call":
            return
        descriptor = self.runtime_store.archive_tool_result(
            stage=self.stage_name,
            action_index=action_index,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            tool_result=step.tool_result,
            metadata={
                "context_request_id": self._last_context_request_id,
                "cache_hit": bool(step.metadata.get("cache_hit", False)),
                "tool_success": bool(step.metadata.get("tool_success", False)),
                "function_call_id": step.metadata.get("function_call_id"),
            },
        )
        step.metadata["tool_result_artifact"] = descriptor

    def _compact_tool_result_for_context(self, tool_name: str, result: str) -> Any:
        try:
            data = json.loads(result)
        except Exception:
            data = None

        if isinstance(data, dict) and str(data.get("status", "")).lower() in {
            "error",
            "failed",
            "failure",
        }:
            return data
        if data is None and str(result).strip().lower().startswith("error"):
            return {"status": "error", "error": str(result).strip()}

        if isinstance(data, (dict, list)):
            raw = json.dumps(data, ensure_ascii=False)
            if len(raw) <= self.tool_response_max_chars:
                return data
            return {"preview": raw[: self.tool_response_max_chars - 32] + "...<truncated>"}
        return str(result)[: self.tool_response_max_chars]

    def _canonical_tool_result(
        self,
        tool_name: str,
        data: Any,
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ToolResultContractError(
                f"{tool_name} result must be an object"
            )
        status = str(data.get("status", "")).strip().lower()
        if status != "success":
            return dict(data)
        if tool_name == "text_search":
            return self._canonical_search_result(data)
        if tool_name == "visit":
            return self._canonical_visit_result(data)
        if tool_name == "reverse_image_search":
            return self._canonical_reverse_image_result(data)
        if tool_name == "crop_and_search":
            return self._canonical_crop_and_search_result(data)
        return dict(data)

    @staticmethod
    def _canonical_search_result(data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data.get("queries"), list):
            return dict(data)
        queries: List[Dict[str, Any]] = []
        for item in (data.get("queries", []) or [])[:1]:
            if not isinstance(item, dict):
                continue
            rows = []
            for row in (item.get("results", []) or [])[:5]:
                if isinstance(row, dict):
                    rows.append(
                        {
                            "title": str(row.get("title", "")),
                            "url": str(row.get("url", "")),
                            "snippet": str(row.get("snippet", ""))[:220],
                        }
                    )
            queries.append(
                {
                    "query": str(item.get("query", "")),
                    "provider": str(item.get("provider", "")),
                    "results": rows,
                    "search_error": str(item.get("search_error", "")),
                    "timings": dict(item.get("timings", {}) or {}),
                }
            )
        return {
            "status": "success",
            "queries": queries,
            "subcalls": [
                dict(item)
                for item in (data.get("subcalls", []) or [])
                if isinstance(item, dict)
            ],
        }

    def _canonical_visit_result(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        canonical = self._compact_visit_result(data)
        canonical.update(
            {
                "status": "success",
                "url": str(data.get("url", "")),
                "selected_url": str(
                    data.get("selected_url", "")
                    or data.get("url", "")
                ),
                "provider": str(data.get("provider", "")),
            }
        )
        return canonical

    @staticmethod
    def _canonical_reverse_image_result(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "branch": str(data.get("branch", "lens")),
            "subcalls": list(data.get("subcalls", []) or []),
            **StageRunner._compact_reverse_image_result(data),
        }

    @staticmethod
    def _canonical_crop_and_search_result(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            **StageRunner._compact_crop_and_search_result(data),
        }

    def _compact_search_result(self, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            responses = data["queries"]
        else:
            responses = data if isinstance(data, list) else [data]
        compacted = []
        for item in responses[:2]:
            if not isinstance(item, dict):
                continue
            results = item.get("results", [])
            top_results = []
            if isinstance(results, list):
                for row in results[:3]:
                    if isinstance(row, dict):
                        top_results.append(
                            {
                                "title": row.get("title", ""),
                                "url": row.get("url", ""),
                                "snippet": str(row.get("snippet", ""))[:220],
                            }
                        )
            compacted.append(
                {
                    "query": item.get("query", ""),
                    "top_results": top_results,
                    "summary": (
                        ""
                        if item.get("injection_flags")
                        else str(item.get("summary", ""))[:320]
                    ),
                    "evidence": (
                        ""
                        if item.get("injection_flags")
                        else str(item.get("evidence", ""))[:320]
                    ),
                    "selected_url": item.get("selected_url", ""),
                    "stance": item.get("stance", "unclear"),
                    "directness": item.get("directness", "none"),
                    "temporal_alignment": item.get(
                        "temporal_alignment",
                        "not_applicable",
                    ),
                    "relevance": item.get("relevance", "low"),
                    "artifact_sha256": item.get("artifact_sha256", ""),
                    "evidence_span": item.get("evidence_span", {}),
                    "retrieved_at": item.get("retrieved_at", ""),
                    "injection_flags": item.get("injection_flags", []),
                    "evidence_eligible": bool(item.get("evidence_eligible", False)),
                }
            )
        return {
            "queries": compacted,
            "validated_claim_state": self._claim_control_states(),
        }

    def _compact_visit_result(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        evidence_records = [
            {
                "url": item.get("url", data.get("url", "")),
                "selected_url": item.get(
                    "selected_url",
                    data.get("selected_url", "") or data.get("url", ""),
                ),
                "evidence": str(item.get("evidence", "")),
                "image_claim": str(item.get("image_claim", "")),
                "retrieval_goal": str(item.get("retrieval_goal", "")),
                "relevance": item.get("relevance", "low"),
                "stance": item.get("stance", "unclear"),
                "relation_scope": item.get("relation_scope", "unclear"),
                "relation_stance": item.get("relation_stance", "unclear"),
                "directness": item.get("directness", "none"),
                "context_only": bool(item.get("context_only", False)),
                "temporal_alignment": item.get(
                    "temporal_alignment",
                    "not_applicable",
                ),
                "artifact_sha256": item.get("artifact_sha256", ""),
                "evidence_span": item.get("evidence_span", {}),
                "retrieved_at": item.get("retrieved_at", ""),
                "injection_flags": item.get("injection_flags", []),
                "evidence_eligible": bool(
                    item.get("evidence_eligible", False)
                ),
            }
            for item in (data.get("evidence_records", []) or [])[:3]
            if isinstance(item, dict)
        ]
        visits = []
        for item in (data.get("visits", []) or [])[:3]:
            if isinstance(item, dict):
                unsafe = bool(item.get("injection_flags"))
                visits.append(
                    {
                        "url": item.get("url", ""),
                        "summary": "" if unsafe else str(item.get("summary", ""))[:180],
                        "evidence": "" if unsafe else str(item.get("evidence", ""))[:180],
                        "image_claim": str(item.get("image_claim", "")),
                        "retrieval_goal": str(item.get("retrieval_goal", "")),
                        "relevance": item.get("relevance", "low"),
                        "stance": item.get("stance", "unclear"),
                        "relation_scope": item.get(
                            "relation_scope",
                            "unclear",
                        ),
                        "relation_stance": item.get(
                            "relation_stance",
                            "unclear",
                        ),
                        "directness": item.get("directness", "none"),
                        "temporal_alignment": item.get(
                            "temporal_alignment",
                            "not_applicable",
                        ),
                        "artifact_sha256": item.get("artifact_sha256", ""),
                        "evidence_span": item.get("evidence_span", {}),
                        "retrieved_at": item.get("retrieved_at", ""),
                        "injection_flags": item.get("injection_flags", []),
                        "evidence_eligible": bool(item.get("evidence_eligible", False)),
                    }
                )
        unsafe = bool(data.get("injection_flags"))
        return {
            "selected_url": data.get("selected_url", ""),
            "summary": "" if unsafe else str(data.get("summary", ""))[:320],
            "evidence": "" if unsafe else str(data.get("evidence", "")),
            "image_claim": str(data.get("image_claim", "")),
            "retrieval_goal": str(data.get("retrieval_goal", "")),
            "stance": data.get("stance", "unclear"),
            "relation_scope": data.get("relation_scope", "unclear"),
            "relation_stance": data.get("relation_stance", "unclear"),
            "directness": data.get("directness", "none"),
            "temporal_alignment": data.get(
                "temporal_alignment",
                "not_applicable",
            ),
            "relevance": data.get("relevance", "low"),
            "artifact_sha256": data.get("artifact_sha256", ""),
            "evidence_span": data.get("evidence_span", {}),
            "retrieved_at": data.get("retrieved_at", ""),
            "injection_flags": data.get("injection_flags", []),
            "evidence_eligible": bool(data.get("evidence_eligible", False)),
            "evidence_records": evidence_records,
            "visits": visits,
            "validated_claim_state": self._claim_control_states(),
        }

    @staticmethod
    def _compact_reverse_image_result(data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def _rows(items: Any) -> List[Dict[str, Any]]:
            compacted: List[Dict[str, Any]] = []
            if not isinstance(items, list):
                return compacted
            for item in items[:3]:
                if isinstance(item, dict):
                    compacted.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "image_url": item.get("image_url", ""),
                        }
                    )
            return compacted

        return {
            "vlm_query": data.get("vlm_query", ""),
            "vlm_error": data.get("vlm_error", ""),
            "lens_error": data.get("lens_error", ""),
            "candidate_page_urls": (data.get("candidate_page_urls", []) or [])[:5],
            "reference_image_candidates": (
                data.get("reference_image_candidates", []) or []
            )[:5],
            "reference_image_url": data.get("reference_image_url", ""),
            "lens_results": _rows(data.get("lens_results", [])),
            "semantic_results": _rows(data.get("semantic_results", [])),
        }

    @staticmethod
    def _compact_crop_and_search_result(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        regions = []
        for item in (data.get("regions", []) or [])[:2]:
            if isinstance(item, dict):
                regions.append(
                    {
                        "bbox": item.get("bbox", []),
                        "crop_query": item.get("crop_query", ""),
                        "summary": str(item.get("summary", ""))[:180],
                        "selected_url": item.get("selected_url", ""),
                        "vlm_query_error": item.get("vlm_query_error", ""),
                    }
                )
        return {
            "regions": regions,
            "summary": str(data.get("summary", ""))[:320],
            "evidence": str(data.get("evidence", "")),
            "image_claim": str(data.get("image_claim", "")),
            "retrieval_goal": str(data.get("retrieval_goal", "")),
            "selected_url": str(data.get("selected_url", "")),
            "relevance": data.get("relevance", "low"),
            "stance": data.get("stance", "unclear"),
            "relation_scope": data.get("relation_scope", "unclear"),
            "relation_stance": data.get("relation_stance", "unclear"),
            "directness": data.get("directness", "none"),
            "context_only": bool(data.get("context_only", False)),
            "temporal_alignment": data.get(
                "temporal_alignment",
                "not_applicable",
            ),
            "artifact_sha256": data.get("artifact_sha256", ""),
            "evidence_span": data.get("evidence_span", {}),
            "retrieved_at": data.get("retrieved_at", ""),
            "injection_flags": data.get("injection_flags", []),
            "evidence_eligible": bool(data.get("evidence_eligible", False)),
        }

    def _validate_output_with_error(
        self,
        output_json: Dict[str, Any],
    ) -> Tuple[Optional[BaseModel], str]:
        if self.output_schema is None:
            return None, "no output schema is configured"
        if not isinstance(output_json, dict):
            return None, "output must be a JSON object"
        missing = missing_required_paths(
            output_json,
            self._normalized_output_schema(),
        )
        if missing:
            return None, "missing required fields: " + ", ".join(missing[:12])
        try:
            return self.output_schema.model_validate(output_json), ""
        except Exception as exc:
            compact = " ".join(str(exc).split())
            return None, compact[:1200]

    def _validate_output(self, output_json: Dict[str, Any]) -> Optional[BaseModel]:
        parsed, _reason = self._validate_output_with_error(output_json)
        return parsed

    @staticmethod
    def _try_parse_bare_json(content: str) -> Optional[Dict[str, Any]]:
        parsed, _repair = StageRunner._try_parse_bare_json_with_repair(content)
        return parsed

    @staticmethod
    def _try_parse_bare_json_with_repair(
        content: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        text = StageRunner._strip_markdown_fence(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return (parsed, "") if isinstance(parsed, dict) else (None, "")
            except json.JSONDecodeError:
                pass

        # LMDeploy guided decoding can omit the fixed opening token while
        # returning the rest of a complete top-level object. Repair only that
        # syntax defect; never infer fields, values or semantic content.
        if start == -1 and text.startswith('"') and text.endswith("}"):
            try:
                parsed = json.loads("{" + text)
                if isinstance(parsed, dict):
                    return parsed, "prepended_missing_top_level_open_brace"
            except json.JSONDecodeError:
                pass
        return None, ""

    @staticmethod
    def _attach_invalid_response_preview(
        step: StageStep,
        content: str,
    ) -> None:
        """Persist a bounded model-response preview for protocol diagnosis."""

        text = str(content or "")
        step.metadata["invalid_response"] = {
            "content_chars": len(text),
            "head": text[:600],
            "tail": text[-600:] if len(text) > 600 else "",
        }

    def _summarize_tool_result(self, tool_name: str, tool_args: Dict[str, Any], result: str) -> str:
        try:
            data = json.loads(result)
        except Exception:
            data = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]

        if tool_name == "text_search":
            query = tool_args.get("queries", tool_args.get("query", ""))
            if isinstance(query, list):
                query = query[0] if query else ""
            if isinstance(data, dict):
                responses = data.get("queries", [])
                if isinstance(responses, list) and responses and isinstance(responses[0], dict):
                    data = responses[0]
                summary = str(data.get("summary", "") or data.get("evidence", "")).strip()
                if summary:
                    return f"[text_search] {query}: {summary[:120]}"
                results = data.get("results", [])
                if isinstance(results, list) and results:
                    first = results[0]
                    return f"[text_search] {query}: {str(first.get('title', '') or first.get('snippet', ''))[:120]}"
            return f"[text_search] {query}: no useful result"

        if tool_name == "reverse_image_search" and isinstance(data, dict):
            urls = data.get("candidate_page_urls", []) or []
            return f"[reverse_image_search] candidates={len(urls)} ref={str(data.get('reference_image_url', ''))[:80]}"

        if tool_name == "visit" and isinstance(data, dict):
            return f"[visit] {str(data.get('summary', '') or data.get('evidence', ''))[:120]}"

        if tool_name == "compare_with_reference":
            return f"[compare_with_reference] {str(result)[:120]}"

        if tool_name == "check_consistency" and isinstance(data, dict):
            return f"[check_consistency] consistent={data.get('consistent', True)} {str(data.get('details', ''))[:100]}"

        if tool_name == "analyze_visual_anomalies":
            return f"[analyze_visual_anomalies] {str(result)[:120]}"

        if tool_name == "ocr_with_position" and isinstance(data, dict):
            return f"[ocr_with_position] regions={data.get('total_regions', 0)} text={str(data.get('full_text', ''))[:100]}"

        if tool_name == "perceive_scene" and isinstance(data, dict):
            return f"[perceive_scene] scene={str(data.get('scene_description', ''))[:100]}"

        if tool_name == "crop_and_inspect" and isinstance(data, dict):
            return f"[crop_and_inspect] {str(data.get('answer', '') or data.get('description', ''))[:120]}"

        if tool_name == "crop_and_search" and isinstance(data, dict):
            return f"[crop_and_search] {str(data.get('summary', '') or data.get('evidence', ''))[:120]}"

        if tool_name == "count_objects" and isinstance(data, dict):
            return f"[count_objects] count={data.get('count', '?')} target={tool_args.get('target_object', '')}"

        if tool_name == "current_time" and isinstance(data, dict):
            return f"[current_time] {data.get('current_date', '')} {data.get('timezone', '')}".strip()

        return f"[{tool_name}] {str(result)[:120]}"

    async def _force_output(
        self,
        system_msg: Dict[str, Any],
        user_msg: Dict[str, Any],
        history: List[Dict[str, Any]],
        evidence_so_far: List[str],
        *,
        last_rejection_reason: str = "",
        lifecycle_kind: str = "standalone_request",
        parent_context_request_id: str = "",
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        evidence_block = ""
        if evidence_so_far:
            evidence_block = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(evidence_so_far[-12:]))
        if self.tools_list:
            prompt = (
                "You have no more tool turns. Produce one final <output> JSON now.\n"
                "Use the collected evidence. Be explicit about uncertainty.\n"
            )
        else:
            prompt = (
                "This is the last validation attempt for this structured output. "
                "Return one corrected JSON object now; this does not require a "
                "terminal verdict or the end of an investigation. Preserve "
                "uncertainty and choose the non-terminal option when the evidence "
                "does not justify a final verdict.\n"
            )
        if last_rejection_reason:
            prompt += (
                "Runtime validator feedback from the previous attempt:\n"
                f"{last_rejection_reason}\n"
            )
        if evidence_block:
            prompt += f"\nCollected evidence:\n{evidence_block}\n"

        messages = [system_msg, user_msg]
        recent = history[2:]
        if len(recent) > 8:
            recent = recent[-8:]
        messages.extend(recent)
        messages.append({"role": "user", "content": prompt})

        response, call_metadata = await self._call_llm(
            messages,
            max_tokens=self.final_output_max_tokens,
            generation_config=self.final_output_generation_config,
            lifecycle_kind=lifecycle_kind,
            parent_context_request_id=parent_context_request_id,
            suppress_tools=True,
        )
        metadata = {
            "forced_output": True,
            **call_metadata,
        }
        if response.text:
            output_json = self._extract_output(response.text)
            format_repair = ""
            if output_json is None:
                output_json, format_repair = (
                    self._try_parse_bare_json_with_repair(response.text)
                )
            if format_repair:
                metadata["format_repair"] = format_repair
            if output_json is not None:
                parsed, schema_error = self._validate_output_with_error(
                    output_json
                )
                if parsed is not None:
                    return parsed, metadata
                metadata["policy_action"] = deepcopy(output_json)
                metadata["rejection_reason"] = (
                    schema_error
                    or "output schema was invalid or incomplete"
                )
                metadata["error_class"] = "schema_error"
            preview_step = StageStep(metadata=metadata)
            self._attach_invalid_response_preview(preview_step, response.text)
        return None, metadata

    def _structured_output_correction_prompt(self, reason: str) -> str:
        """Return a bounded semantic retry instruction for a JSON stage.

        Guided decoding guarantees syntax, not that a proposed state update is
        admissible. Feed the exact validator feedback back to the model without
        silently rewriting its state transition or forcing a terminal answer.
        """

        message = (
            "The previous JSON object was rejected by the runtime validator. "
            "Return one corrected complete JSON object. Change only the fields "
            "needed to satisfy this feedback; do not invent IDs or state. "
        )
        if self.stage_name == "image_only_discrepancy_decision":
            message += (
                "A non-terminal continue proposal remains valid; do not force "
                "real or fake merely because this is a retry. Omit any Claim "
                "assessment that has no reviewed owned Evidence or listed "
                "directional Finding chain. Address every independent error in "
                "the validator feedback. "
            )
            if "resolved focused visual Evidence" in reason:
                message += (
                    "For each feedback mapping like 'evidence-X -> claim_ids "
                    "[claim-Y]', if you assess claim-Y or propose a discrepancy "
                    "for claim-Y, include evidence-X in "
                    "claim_assessments[].selected_evidence_ids or in "
                    "material_discrepancy.evidence_ids. Copy Evidence IDs "
                    "exactly from the feedback; do not shorten or alter any "
                    "character. Do not set visual_evidence_disposition for that "
                    "same claim. If validator feedback says supported/refuted "
                    "lacks a qualified directional Finding chain, change that "
                    "same Claim assessment to insufficient, still include the "
                    "mapped visual Evidence ID, and keep verdict_proposal "
                    "continue. "
                )
        return message + f"Runtime validator feedback: {reason}"

    def _has_duplicate_tool_call(self, steps: List[StageStep], tool_name: str, tool_args: Dict[str, Any]) -> bool:
        for step in [
            *self.prior_steps,
            *steps,
            *list(getattr(self, "_control_steps", [])),
        ]:
            if step.action_type != "tool_call":
                continue
            if routes_semantically_equivalent(
                step.tool_name,
                self._normalize_tool_args(step.tool_args),
                tool_name,
                self._normalize_tool_args(tool_args),
            ):
                return True
        return False

    def _tool_budget_reached(self, steps: List[StageStep], tool_name: str) -> bool:
        limit = self.tool_call_limits.get(tool_name)
        if limit is None:
            return False
        used = sum(
            1
            for step in [*self.prior_steps, *steps]
            if step.action_type == "tool_call" and step.tool_name == tool_name
        )
        return used >= limit

    @staticmethod
    def _normalize_tool_args(tool_args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in sorted(tool_args.items()):
            if key == "__question_id":
                normalized["question_id"] = value
                continue
            if key.startswith("__"):
                continue
            if isinstance(value, list):
                normalized[key] = value
            elif isinstance(value, str):
                normalized[key] = value.strip()
            else:
                normalized[key] = value
        return normalized

    def _unknown_tool_message(self, tool_name: str) -> str:
        available = ", ".join(self.tools.keys()) if self.tools else "(none)"
        return f"Tool '{tool_name}' is not available in this stage. Available tools: {available}"

    def _duplicate_tool_message(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        rejected_call = json.dumps(
            {
                "tool": tool_name,
                "arguments": self._normalize_tool_args(tool_args or {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            "The runtime rejected this duplicate call: "
            f"{rejected_call}. Do not repeat it in this correction chain. "
            "Use a different available tool or materially different arguments."
        )

    def _tool_budget_message(self, tool_name: str) -> str:
        return f"Tool budget for '{tool_name}' is exhausted. Use another tool or finalize the output."
