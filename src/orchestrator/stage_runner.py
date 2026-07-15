# -*- coding: utf-8 -*-
"""Single-path stage runner for multi-round ReAct execution."""
from __future__ import annotations

import json
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
from src.orchestrator.tool_cache import ToolResultCache
from src.orchestrator.tool_result import ToolResultContractError, parse_tool_result, serialize_tool_result
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool
from src.tools.vision_utils import image_to_data_url


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

    async def run(self, input_context: str) -> Tuple[Optional[BaseModel], List[StageStep]]:
        """Run the ReAct loop."""
        if str(getattr(self.llm, "provider", "")).lower() == "gemini" and str(
            getattr(self.llm, "wire_api", "")
        ).lower() != "interactions":
            raise RuntimeError("Gemini stages require wire_api='interactions'.")
        self.active_question_ids = list(
            dict.fromkeys(
                re.findall(r"\[((?:q[^\]]*)|(?:task-[^\]]+))\]", input_context)
            )
        )
        if self._uses_native_interactions():
            return await self._run_native_interactions(input_context)
        if self._uses_native_structured_output():
            return await self._run_native_structured_output(input_context)

        steps: List[StageStep] = []
        system_msg = {"role": "system", "content": self._build_system_content()}
        user_msg = self._build_user_message(input_context)
        history: List[Dict[str, Any]] = [system_msg, user_msg]
        evidence_so_far: List[str] = []

        for round_num in range(1, self.max_rounds + 1):
            messages = self._build_round_messages(system_msg, user_msg, history, evidence_so_far)
            response, llm_metadata = await self._call_llm(messages)
            step = StageStep(
                round=round_num,
                stage_name=self.stage_name,
                tokens={"prompt": response.prompt_tokens, "completion": response.completion_tokens},
                metadata={"stage": self.stage_name, **llm_metadata},
            )

            content = (response.text or "").strip()
            if not content:
                step.action_type = "format_error"
                step.thought = "(empty response)"
                steps.append(step)
                history.append({"role": "assistant", "content": ""})
                history.append({"role": "user", "content": "Response was empty. Produce one valid tool call or final output."})
                continue

            step.thought = self._extract_think(content)

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_name, tool_args = self._parse_tool_call(content)
                if tool_name not in self.tools:
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["invalid_tool_name"] = tool_name
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": self._unknown_tool_message(tool_name)})
                    continue

                step.action_type = "tool_call"
                step.tool_name = tool_name
                step.tool_args = self._prepare_tool_args(tool_name, dict(tool_args), input_context)
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
                    continue
                if self._has_duplicate_tool_call(steps, tool_name, step.tool_args):
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["duplicate_tool_call"] = True
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": self._duplicate_tool_message(tool_name)})
                    continue
                if self._tool_budget_reached(steps, tool_name):
                    step.action_type = "format_error"
                    step.metadata["error_class"] = "protocol_error"
                    step.metadata["tool_budget_reached"] = True
                    steps.append(step)
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": self._tool_budget_message(tool_name)})
                    continue
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
                    continue
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
                        continue
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
                    continue
                serialized, tool_metadata = await self._execute_tool(tool_name, dict(step.tool_args))
                step.tool_result = serialized
                step.metadata.update(tool_metadata)
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
                history.append({"role": "assistant", "content": content})
                history.append(
                    {
                        "role": "user",
                        "content": self._build_tool_response_message(
                            tool_name,
                            step.tool_args,
                            serialized,
                            function_call_id=str(step.metadata["function_call_id"]),
                            state_update=state_update,
                        ),
                    }
                )

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
                continue

            output_json = self._extract_output(content) or self._try_parse_bare_json(content)
            if output_json is not None:
                step.action_type = "output"
                step.output = output_json
                steps.append(step)
                parsed = self._validate_output(output_json)
                if parsed is not None:
                    accepted, reason = self._accept_output(parsed, steps)
                    if accepted:
                        return parsed, steps
                    step.action_type = "output_rejected"
                    step.metadata["rejection_reason"] = reason
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": f"Output rejected: {reason} Continue investigating."})
                    continue
                step.action_type = "output_rejected"
                step.metadata["rejection_reason"] = "output schema was invalid or incomplete"
                history.append({"role": "assistant", "content": content})
                history.append({"role": "user", "content": "Output schema was invalid. Try again with one valid JSON object."})
                continue

            step.action_type = "format_error"
            steps.append(step)
            history.append({"role": "assistant", "content": content})
            history.append({"role": "user", "content": "Use exactly one <tool_call> or one <output> block."})

        forced, forced_meta = await self._force_output(system_msg, user_msg, history, evidence_so_far)
        if forced is not None:
            accepted, reason = self._accept_output(forced, steps, final_attempt=True)
            if not accepted:
                forced_meta["rejection_reason"] = reason
                forced = None
        final_step = StageStep(
            round=len(steps) + 1,
            stage_name=self.stage_name,
            action_type="output" if forced is not None else "format_error",
            output=forced.model_dump() if forced is not None else None,
            metadata={"stage": self.stage_name, **forced_meta},
        )
        steps.append(final_step)
        return forced, steps

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
        previous_interaction_id: Optional[str] = None
        next_input: Any = self._build_native_input(input_context)

        for round_num in range(1, self.max_rounds + 2):
            request_previous_interaction_id = previous_interaction_id
            started = time.perf_counter()
            self.llm_api_calls += 1
            system_instruction = self.system_prompt
            response_format = self._native_response_format()
            payload = await self.llm.create_interaction(
                input_payload=next_input,
                system_instruction=system_instruction,
                previous_interaction_id=request_previous_interaction_id,
                response_format=response_format,
                store=True,
                max_tokens=self.max_output_tokens,
                generation_config=self.generation_config,
            )
            interaction_id, status = validate_interaction_response(payload)
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
                next_input = f"Output rejected: {reason}. Return a corrected JSON object."
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
        previous_interaction_id: Optional[str] = None
        next_input: Any = self._build_native_input(input_context)
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
                payload = await self.llm.create_interaction(
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
                        error_message = self._duplicate_tool_message(tool_name)
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

    def _build_native_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in self.tools_list:
            parameters = deepcopy(tool.parameters or {})
            parameters.setdefault("type", "object")
            properties = parameters.setdefault("properties", {})
            required = list(parameters.get("required", []) or [])
            properties.pop("image_input", None)
            required = [name for name in required if name != "image_input"]
            parameters = self._normalize_native_schema(parameters)
            properties = parameters.setdefault("properties", {})
            if self.stage_name == "verification":
                question_schema: Dict[str, Any] = {
                    "type": "string",
                    "description": "Planning question id advanced by this call, for example q0.",
                }
                if self.active_question_ids:
                    question_schema["enum"] = self.active_question_ids
                properties["question_id"] = question_schema
                if "question_id" not in required:
                    required.append("question_id")
                runtime_bound = self._runtime_bound_visual_fields(tool.name)
                if runtime_bound:
                    required = [name for name in required if name not in runtime_bound]
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
        image_url = image_to_data_url(self.image_path)
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
                payload = await self.llm.create_interaction(
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
        """Persist the exact model-visible request without transport-only fields."""

        return {
            "system_instruction": system_instruction,
            "input_payload": deepcopy(input_payload),
            "tools": deepcopy(tools),
            "response_format": deepcopy(response_format),
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
            suffix = " No more tool turns remain." if final_attempt else ""
            return False, required_question_error + suffix
        if self.output_validator:
            accepted, reason = self.output_validator(parsed, steps)
            if not accepted:
                suffix = " No more tool turns remain." if final_attempt else ""
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
            claim_text = self.question_claims.get(question_id, "").strip()
            if claim_text:
                tool_args["__claim_text"] = claim_text
            evidence_goal = self.question_evidence_goals.get(question_id, "").strip()
            if evidence_goal:
                tool_args["__evidence_goal"] = evidence_goal
            if tool_name in {"text_search", "visit", "crop_and_search"}:
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
            elif tool_name in {"ocr_with_position", "crop_and_search"}:
                bound.setdefault("goal", expected_property)
            elif tool_name == "count_objects":
                bound.setdefault("target_object", expected_property[:200])
            elif tool_name == "compare_with_reference":
                bound.setdefault("focus", expected_property[:400])
        return bound

    @staticmethod
    def _runtime_bound_visual_fields(tool_name: str) -> set[str]:
        if tool_name == "compare_with_reference":
            return {"reference_url"}
        if tool_name == "crop_and_inspect":
            return {"bbox", "focus_question"}
        if tool_name in {"ocr_with_position", "crop_and_search"}:
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
                parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(self.image_path)}})
            except Exception:
                pass
        parts.append({"type": "text", "text": input_context})
        if len(parts) == 1 and parts[0]["type"] == "text":
            return {"role": "user", "content": parts[0]["text"]}
        return {"role": "user", "content": parts}

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

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Tuple[LLMResponse, Dict[str, Any]]:
        started = time.perf_counter()
        self.llm_api_calls += 1
        response = await self.llm.get_response(messages)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return response, {"llm_duration_ms": duration_ms}

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
        claim_text = str(tool_args.pop("__claim_text", "")).strip()
        evidence_goal = str(tool_args.pop("__evidence_goal", "")).strip()
        if claim_text and tool_name in {"text_search", "visit", "crop_and_search"}:
            tool_args["goal"] = evidence_goal or claim_text
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
                result = await tool.call_async(tool_args)
            else:
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, tool_args)
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
        else:
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
            }
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
        args.pop("__claim_text", None)
        args.pop("__evidence_goal", None)
        if tool_name in {"compare_with_reference", "analyze_visual_anomalies"} and self.image_path:
            args["__image_input__"] = self.image_path
        return args

    def _build_tool_response_message(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        *,
        function_call_id: str,
        state_update: Optional[Dict[str, Any]] = None,
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
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) > self.tool_response_max_chars:
            text = text[: self.tool_response_max_chars] + "\n...<truncated>"
        return f"<tool_response>\n{text}\n</tool_response>"

    def _record_observation_update(
        self,
        step: StageStep,
        steps: List[StageStep],
    ) -> Optional[Dict[str, Any]]:
        if self.observation_callback is None or step.action_type not in {"tool_call", "format_error"}:
            return None
        update = self.observation_callback(step, list(self.prior_steps) + list(steps))
        if update:
            step.metadata["investigation_state_update"] = update
        return update

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

        if tool_name == "text_search":
            return self._compact_search_result(data)
        if tool_name == "visit":
            return self._compact_visit_result(data)
        if tool_name == "reverse_image_search":
            return self._compact_reverse_image_result(data)
        if tool_name == "crop_and_search":
            return self._compact_crop_and_search_result(data)
        if isinstance(data, (dict, list)):
            raw = json.dumps(data, ensure_ascii=False)
            if len(raw) <= self.tool_response_max_chars:
                return data
            return {"preview": raw[: self.tool_response_max_chars - 32] + "...<truncated>"}
        return str(result)[: self.tool_response_max_chars]

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
        visits = []
        for item in (data.get("visits", []) or [])[:3]:
            if isinstance(item, dict):
                unsafe = bool(item.get("injection_flags"))
                visits.append(
                    {
                        "url": item.get("url", ""),
                        "summary": "" if unsafe else str(item.get("summary", ""))[:180],
                        "evidence": "" if unsafe else str(item.get("evidence", ""))[:180],
                        "relevance": item.get("relevance", "low"),
                        "stance": item.get("stance", "unclear"),
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
            "evidence": "" if unsafe else str(data.get("evidence", ""))[:320],
            "stance": data.get("stance", "unclear"),
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
            "evidence": str(data.get("evidence", ""))[:320],
            "temporal_alignment": data.get(
                "temporal_alignment",
                "not_applicable",
            ),
        }

    def _validate_output(self, output_json: Dict[str, Any]) -> Optional[BaseModel]:
        if self.output_schema is None:
            return None
        if not isinstance(output_json, dict):
            return None
        if missing_required_paths(output_json, self._normalized_output_schema()):
            return None
        try:
            return self.output_schema.model_validate(output_json)
        except Exception:
            return None

    @staticmethod
    def _try_parse_bare_json(content: str) -> Optional[Dict[str, Any]]:
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        text = StageRunner._strip_markdown_fence(text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

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
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        evidence_block = ""
        if evidence_so_far:
            evidence_block = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(evidence_so_far[-12:]))
        prompt = (
            "You have no more tool turns. Produce one final <output> JSON now.\n"
            "Use the collected evidence. Be explicit about uncertainty.\n"
        )
        if evidence_block:
            prompt += f"\nCollected evidence:\n{evidence_block}\n"

        messages = [system_msg, user_msg]
        recent = history[2:]
        if len(recent) > 8:
            recent = recent[-8:]
        messages.extend(recent)
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        self.llm_api_calls += 1
        response = await self.llm.get_response(messages)
        metadata = {
            "forced_output": True,
            "llm_duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if response.text:
            output_json = self._extract_output(response.text) or self._try_parse_bare_json(response.text)
            if output_json:
                parsed = self._validate_output(output_json)
                if parsed is not None:
                    return parsed, metadata
        return None, metadata

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

    @staticmethod
    def _duplicate_tool_message(tool_name: str) -> str:
        return f"You already called '{tool_name}' with essentially the same target. Choose a meaningfully different next step."

    def _tool_budget_message(self, tool_name: str) -> str:
        return f"Tool budget for '{tool_name}' is exhausted. Use another tool or finalize the output."
