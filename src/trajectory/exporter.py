"""Pure canonical-trace to policy-example exporter."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence

from src.trajectory.schema import (
    ActionOnlyTrajectoryExample,
    PolicyExample,
    TrajectorySFTExample,
)
from src.orchestrator.evidence_semantics import (
    evidence_direction_is_coherent,
    evidence_is_qualified_for_stance,
    required_assessment_stances,
)
from src.orchestrator.investigation_models import target_fact_rows
from src.orchestrator.tool_result import parse_tool_result


FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "acceptable_evidence",
        "expected_status",
        "ground_truth",
        "teacher_score",
        "process_metrics",
        "source_access_policy",
        "excluded_domains",
        "excluded_urls",
    }
)


class TokenizerAdapter(Protocol):
    tokenizer_id: str

    def encode(self, text: str) -> Sequence[int]:
        """Encode one canonical JSON string without adding implicit tokens."""


class Utf8ByteTokenizer:
    """Stable default adapter; model-specific tokenizers remain injectable."""

    tokenizer_id = "utf8-byte-v1"

    def encode(self, text: str) -> Sequence[int]:
        return list(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _assert_no_private_data(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(
                    f"private/evaluator field is forbidden in policy input: {child_path}"
                )
            _assert_no_private_data(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_data(child, f"{path}[{index}]")


def _observation_refs(policy_input: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in (
                "call_id",
                "function_call_id",
                "evidence_id",
                "finding_id",
                "task_id",
                "fact_id",
                "interaction_id",
            ):
                rendered = str(value.get(key, "") or "").strip()
                if rendered:
                    refs.append(rendered)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(policy_input.get("input_payload"))
    return list(dict.fromkeys(refs))[:64]


def _example_type(stage: str) -> str | None:
    if stage == "unified_react":
        return "react"
    if stage == "unified_reflection":
        return "reflection"
    if stage == "unified_discrepancy_decision":
        return "discrepancy_decision"
    if stage == "unified_judgment":
        return "judgment"
    return None


def _strip_gemini_wire_instructions(value: str) -> str:
    marker = "Native Gemini Interactions protocol:"
    if marker in value:
        value = value.split(marker, 1)[0]
    return value.rstrip()


def _tool_schema_key(tool: Mapping[str, Any]) -> str:
    return canonical_json(tool)


def _normalize_tool_schema(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("function"), Mapping):
            function = dict(item["function"])
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(function.get("description", "")),
                        "parameters": function.get("parameters", {}),
                    },
                }
            )
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description", "")),
                    "parameters": item.get("parameters", {}),
                },
            }
        )
    return result


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def _qwen_parameter_value(value: Any) -> str:
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _qwen_think_block(thought: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>"


def _qwen_tool_call_block(
    *,
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for argument_name, argument_value in arguments.items():
        lines.extend(
            [
                f"<parameter={argument_name}>",
                _qwen_parameter_value(argument_value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _initial_observation_packet(trace: Mapping[str, Any]) -> str:
    state = _mapping(trace.get("state"))
    runtime_case = _mapping(state.get("runtime_case"))
    observations: list[dict[str, Any]] = []
    for step in _rows(state.get("all_steps")):
        if str(step.get("stage", "")) not in {"perception", "unified_react"}:
            continue
        if str(step.get("action_type", "")) != "tool_call":
            continue
        tool_name = str(step.get("tool_name", "")).strip()
        if tool_name not in {"perceive_scene", "ocr_with_position"}:
            continue
        observations.append(
            {
                "tool": tool_name,
                "result": _json_or_text(str(step.get("tool_result", ""))),
            }
        )
    payload = {
        "case_id": str(runtime_case.get("case_id") or trace.get("image_id") or ""),
        "image_sha256": str(runtime_case.get("image_sha256", "")),
        "input_mode": str(trace.get("input_mode") or state.get("input_mode") or ""),
        "initial_observations": observations,
    }
    return (
        "Image factual verification episode. Use the supplied observations and "
        "subsequent tool responses as context. Generate only the next policy "
        "message when it is your turn.\n\n"
        + canonical_json(payload)
    )


def _stage_user_packet(
    *,
    example_type: str,
    policy_input: Mapping[str, Any],
) -> str:
    input_payload = _compact_export_input_payload(policy_input.get("input_payload", ""))
    payload: dict[str, Any] = {
        "stage": example_type,
        "stage_instruction": _strip_gemini_wire_instructions(
            str(policy_input.get("system_instruction", ""))
        ),
        "input_payload": input_payload,
    }
    response_format = policy_input.get("response_format")
    if isinstance(response_format, Mapping) and example_type != "react":
        payload["required_structured_output_contract"] = dict(response_format)
    tools = _normalize_tool_schema(policy_input.get("tools"))
    if tools and example_type == "react":
        payload["authorized_tool_names"] = [
            str(tool["function"]["name"]) for tool in tools
        ]
    return canonical_json(payload)


def _stage_control_packet(
    *,
    example_type: str,
    policy_input: Mapping[str, Any],
) -> str:
    """Render a small transition control packet for a continued episode.

    The first stage needs its complete input projection because it introduces
    the case.  After that, the full event history already contains the prior
    assistant actions, tool observations, and reducer deltas.  Replaying each
    stage's cumulative runtime packet would therefore duplicate state dozens
    of times.  Keep only the current stage contract and active tool boundary.
    """

    payload: dict[str, Any] = {
        "stage": example_type,
        "output_mode": (
            "native_tool_call" if example_type == "react" else "structured_json"
        ),
    }
    instruction = _strip_gemini_wire_instructions(
        str(policy_input.get("system_instruction", ""))
    )
    if instruction:
        payload["stage_instruction"] = instruction
    tools = _normalize_tool_schema(policy_input.get("tools"))
    if example_type == "react":
        payload["authorized_tool_names"] = [
            str(tool["function"]["name"]) for tool in tools
        ]
    return canonical_json(payload)


def _compact_export_input_payload(value: Any) -> Any:
    """Remove archived workspace snapshots from model-visible SFT packets.

    Runtime stage requests intentionally retain the complete handoff in the
    canonical trace and context ledger.  The rendered stage-specific input
    already contains the bounded state projection needed by that stage, so
    exporting ``runtime_handoff.workspace`` again makes every policy turn carry
    a second copy of cumulative state.  Keep the handoff metadata and compact
    projection marker, but never copy the archived full workspace into SFT.
    """

    if isinstance(value, list):
        return [_compact_export_input_payload(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "runtime_handoff" and isinstance(child, Mapping):
                handoff = {
                    str(name): _compact_export_input_payload(item)
                    for name, item in child.items()
                    if name != "workspace"
                }
                result[str(key)] = handoff
            else:
                result[str(key)] = _compact_export_input_payload(child)
        return result
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
        compacted = _compact_export_input_payload(parsed)
        if compacted != parsed:
            return canonical_json(compacted)
    return value


def _trajectory_candidate_steps(
    trace: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any], str]]:
    state = _mapping(trace.get("state"))
    candidates: list[tuple[int, Mapping[str, Any], str]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if str(step.get("action_type", "")) in {
            "planning_revision",
            "format_error",
            "output_rejected",
            "policy_replan",
        }:
            continue
        stage = str(step.get("stage", "")).strip()
        example_type = _example_type(stage)
        metadata = _mapping(step.get("metadata"))
        if example_type is None:
            continue
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        candidates.append((index, step, example_type))
    return candidates


def trajectory_policy_step_ids(
    trace: Mapping[str, Any],
    *,
    episode_id: str | None = None,
) -> list[str]:
    """Return stable internal step IDs without exporting step-level SFT rows.

    RL reward ledgers still need to identify the policy turns in an episode.
    Those IDs are runtime bookkeeping only; they are deliberately not written
    to the SFT dataset as one-row-per-step examples.
    """

    resolved_episode_id = str(
        episode_id
        or trace.get("image_id")
        or _mapping(trace.get("state")).get("image_id")
        or ""
    ).strip()
    if not resolved_episode_id:
        return []
    result: list[str] = []
    for index, step, example_type in _trajectory_candidate_steps(trace):
        metadata = _mapping(step.get("metadata"))
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        result.append(
            f"{resolved_episode_id}:{example_type}:{interaction_id}"
            if interaction_id
            else f"{resolved_episode_id}:{example_type}:{index + 1}"
        )
    return result


def export_trajectory_sft_example(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    allow_incomplete_verdict_chain: bool = False,
    require_provider_thought: bool = True,
) -> TrajectorySFTExample | ActionOnlyTrajectoryExample:
    """Export one complete accepted episode as one prefix-preserving SFT row."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != (
        "image_only"
    ):
        raise ValueError("trajectory SFT exporter accepts image-only traces only")
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError("trajectory SFT exporter received an unsupported decision policy")
    _unified_react_quality_gate(
        trace,
        state,
        require_provider_thought=require_provider_thought,
    )

    tokenizer = tokenizer or Utf8ByteTokenizer()
    source_metadata = source_metadata or {}
    episode_id = str(
        trace.get("image_id") or state.get("image_id") or ""
    ).strip()
    runtime_case = _mapping(state.get("runtime_case"))
    case_id = str(runtime_case.get("case_id") or episode_id).strip()
    if not episode_id or not case_id:
        raise ValueError("canonical trace requires episode and case IDs")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "<ifv_training_mode>compact_full_trajectory_sft</ifv_training_mode>\n"
                "You are the Image Factual Verifier policy model. Follow each "
                "stage control and generate the next assistant message or tool "
                "call. Tool responses are observations, not text to imitate. "
                "The event history and recorded state deltas are the current "
                "investigation context."
            ),
        },
        {"role": "user", "content": _initial_observation_packet(trace)},
    ]
    tool_schemas: dict[str, dict[str, Any]] = {}
    tool_call_count = 0
    candidates = _trajectory_candidate_steps(trace)
    pending_tool_response: dict[str, Any] | None = None
    for position, (_, step, example_type) in enumerate(candidates):
        metadata = _mapping(step.get("metadata"))
        policy_input = dict(_mapping(metadata.get("policy_input")))
        policy_action = dict(_mapping(metadata.get("policy_action")))
        _assert_no_private_data(policy_input)
        _assert_no_private_data(policy_action)
        for tool_schema in _normalize_tool_schema(policy_input.get("tools")):
            tool_schemas.setdefault(_tool_schema_key(tool_schema), tool_schema)
        initial_stage_packet = _stage_user_packet(
            example_type=example_type,
            policy_input=policy_input,
        )
        stage_control = _stage_control_packet(
            example_type=example_type,
            policy_input=policy_input,
        )
        if position == 0:
            messages[1]["content"] += "\n\n" + initial_stage_packet
        elif pending_tool_response is not None:
            # Keep the role sequence expected by the Qwen chat template
            # (assistant -> tool -> assistant).  The stage boundary is still
            # carried once in the tool observation.  Do not replay the next
            # stage's cumulative runtime input: its relevant event deltas have
            # already been appended to the transcript.
            pending_tool_response["next_stage_control"] = stage_control
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_tool_response.get(
                        "function_call_id", ""
                    ),
                    "content": canonical_json(pending_tool_response),
                }
            )
            pending_tool_response = None
        else:
            messages.append({"role": "user", "content": stage_control})
        thought = str(step.get("thought", "") or "").strip()
        if thought:
            _assert_no_private_data(thought)
        if example_type == "react":
            if (
                policy_version == "unified-react-v1"
                and require_provider_thought
                and not thought
            ):
                raise ValueError(
                    "unified-react reasoning SFT requires provider-visible "
                    "thought for every ReAct action"
                )
            if str(policy_action.get("type", "")) != "tool_call":
                raise ValueError("react trajectory action must be a tool call")
            tool_call = {
                "name": str(policy_action.get("name", "")).strip(),
                "arguments": dict(_mapping(policy_action.get("arguments"))),
            }
            if not tool_call["name"]:
                raise ValueError("react trajectory action lacks tool name")
            function_call_id = str(
                metadata.get("function_call_id", "")
            ).strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        (
                            _qwen_think_block(thought) + "\n\n"
                            if thought
                            else ""
                        )
                        + _qwen_tool_call_block(
                            name=tool_call["name"],
                            arguments=tool_call["arguments"],
                        )
                    ),
                    "loss": True,
                }
            )
            tool_call_count += 1
            tool_result = str(step.get("tool_result", "") or "").strip()
            if tool_result:
                tool_response: dict[str, Any] = {
                    "function_call_id": function_call_id,
                    "tool": tool_call["name"],
                    "arguments": tool_call["arguments"],
                    "result": _json_or_text(tool_result),
                }
                state_update = metadata.get("investigation_state_update")
                if isinstance(state_update, Mapping):
                    _assert_no_private_data(state_update)
                    tool_response["investigation_state_update"] = dict(
                        state_update
                    )
                pending_tool_response = tool_response
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        (_qwen_think_block(thought) + "\n\n" if thought else "")
                        + canonical_json(policy_action)
                    ),
                    "loss": True,
                }
            )

    if pending_tool_response is not None:
        raise ValueError("trajectory ended after a tool call without a next policy turn")

    if len(messages) <= 2:
        raise ValueError("trajectory SFT export found no supervised policy turns")
    tools = (
        canonical_json(list(tool_schemas.values()))
        if tool_schemas
        else ""
    )
    token_payload = {"messages": messages}
    if tools:
        token_payload["tools"] = tools
    token_count = len(tokenizer.encode(canonical_json(token_payload)))
    example_payload = {
        "trajectory_version": (
            "ifv-trajectory-action-only-v1"
            if policy_version == "unified-react-v1"
            and not require_provider_thought
            else "ifv-trajectory-sft-v3"
        ),
        "episode_id": episode_id,
        "case_id": case_id,
        "source_run_id": str(source_metadata.get("source_run_id", "")),
        "runtime_commit": str(source_metadata.get("runtime_commit", "")),
        "release_id": str(source_metadata.get("release_id", "")),
        "runtime_contract_version": str(
            source_metadata.get("runtime_contract_version", "")
        ),
        "process_reference_protocol_version": str(
            source_metadata.get("process_reference_protocol_version", "")
        ),
        "messages": messages,
        "tools": tools,
        "token_count_estimate": token_count,
        "message_count": len(messages),
        "tool_call_count": tool_call_count,
    }
    if (
        policy_version == "unified-react-v1"
        and not require_provider_thought
    ):
        return ActionOnlyTrajectoryExample(**example_payload)
    return TrajectorySFTExample(**example_payload)


def export_trajectory_action_only_example(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    allow_incomplete_verdict_chain: bool = False,
) -> ActionOnlyTrajectoryExample:
    """Export a unified trace with executable actions but no thought targets."""

    exported = export_trajectory_sft_example(
        trace,
        tokenizer=tokenizer,
        source_metadata=source_metadata,
        allow_incomplete_verdict_chain=allow_incomplete_verdict_chain,
        require_provider_thought=False,
    )
    if not isinstance(exported, ActionOnlyTrajectoryExample):
        raise ValueError(
            "action-only export is reserved for unified-react-v1 traces"
        )
    return exported


def _v4_claim_has_directional_chain(
    *,
    claim_id: str,
    stance: str,
    selected_evidence_ids: set[str],
    claims: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    findings: Mapping[str, Mapping[str, Any]],
    successful_call_ids: set[str],
    selected_finding_ids: set[str] | None = None,
) -> bool:
    claim = claims.get(claim_id)
    if claim is None:
        return False
    claim_fact_id = str(claim.get("fact_id", "")).strip()
    for finding_id, finding in findings.items():
        if selected_finding_ids is not None and finding_id not in selected_finding_ids:
            continue
        if str(finding.get("stance", "")).strip() != stance:
            continue
        task_id = str(finding.get("task_id", "")).strip()
        task = tasks.get(task_id)
        if task is None or claim_id not in {
            str(item) for item in task.get("claim_ids", []) or []
        }:
            continue
        if claim_fact_id not in {
            str(item) for item in finding.get("fact_ids", []) or []
        }:
            continue
        for evidence_id in finding.get("evidence_ids", []) or []:
            evidence_id = str(evidence_id)
            row = evidence.get(evidence_id)
            if (
                evidence_id in selected_evidence_ids
                and row is not None
                and str(row.get("task_id", "")).strip() == task_id
                and evidence_is_qualified_for_stance(row, stance)
                and str(row.get("function_call_id", "")).strip()
                in successful_call_ids
            ):
                return True
    return False


def _v4_quality_gate(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    allow_incomplete_verdict_chain: bool = False,
) -> None:
    """Reject v4 episodes that cannot teach claim/discrepancy alignment."""

    if str(trace.get("termination", "")) != "success":
        raise ValueError("v4 policy export requires a successful trace")
    investigation = _mapping(state.get("investigation_state"))
    claims = {
        str(item.get("claim_id", "")): item
        for item in target_fact_rows(investigation)
        if str(item.get("claim_id", ""))
    }
    hypotheses = {
        str(item.get("hypothesis_id", "")): item
        for item in _rows(investigation.get("search_hypotheses"))
        if str(item.get("hypothesis_id", ""))
    }
    tasks = {
        str(item.get("task_id", "")): item
        for item in _rows(investigation.get("tasks"))
        if str(item.get("task_id", ""))
    }
    evidence = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    findings = {
        str(item.get("finding_id", "")): item
        for item in _rows(investigation.get("findings"))
        if str(item.get("finding_id", ""))
    }
    successful_call_ids: set[str] = set()
    for step in _rows(state.get("all_steps")):
        if str(step.get("action_type", "")) != "tool_call":
            continue
        metadata = _mapping(step.get("metadata"))
        call_id = str(metadata.get("function_call_id", "")).strip()
        try:
            _, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        except Exception:
            succeeded = False
        if call_id and succeeded and metadata.get("tool_success") is not False:
            successful_call_ids.add(call_id)
    if any(
        not str(item.get("function_call_id", "")).strip()
        or str(item.get("function_call_id", "")).strip() not in successful_call_ids
        for item in evidence.values()
    ):
        raise ValueError("v4 policy export rejects Evidence without a successful call")
    if any(
        str(item.get("stance", "")) in {"support", "refute"}
        and not evidence_direction_is_coherent(
            item,
            str(item.get("stance", "")),
        )
        for item in evidence.values()
    ):
        raise ValueError("v4 policy export rejects incoherent Evidence stance")
    facts = {
        str(item.get("fact_id", "")): item
        for item in _rows(investigation.get("facts"))
        if str(item.get("fact_id", ""))
    }
    if not claims or not any(
        str(item.get("salience", "")) == "high" for item in claims.values()
    ):
        raise ValueError("v4 policy export requires high-salience ImageClaims")
    if investigation.get("core_verdict_fact_id"):
        raise ValueError("v4 policy export rejects active legacy core ownership")
    for assessment in _rows(investigation.get("claim_assessments")):
        claim_id = str(assessment.get("claim_id", ""))
        selected_ids = {
            str(item) for item in assessment.get("evidence_ids", []) or []
        }
        if claim_id not in claims or not selected_ids <= set(evidence):
            raise ValueError("v4 policy export rejects invalid ClaimAssessment Evidence")
        if any(
            claim_id
            not in {
                str(item)
                for item in tasks.get(
                    str(evidence[evidence_id].get("task_id", "")),
                    {},
                ).get("claim_ids", [])
                or []
            }
            for evidence_id in selected_ids
        ):
            raise ValueError("v4 policy export rejects unowned ClaimAssessment Evidence")
        for stance in required_assessment_stances(
            str(assessment.get("assessment", ""))
        ):
            if not _v4_claim_has_directional_chain(
                claim_id=claim_id,
                stance=stance,
                selected_evidence_ids=selected_ids,
                claims=claims,
                tasks=tasks,
                evidence=evidence,
                findings=findings,
                successful_call_ids=successful_call_ids,
            ):
                raise ValueError(
                    "v4 policy export rejects ClaimAssessment/Evidence "
                    "direction mismatch"
                )
    for hypothesis_id, hypothesis in hypotheses.items():
        claim_ids = {str(item) for item in hypothesis.get("claim_ids", []) or []}
        task = tasks.get(str(hypothesis.get("task_id", "")))
        if (
            not claim_ids
            or not claim_ids <= set(claims)
            or task is None
            or str(task.get("hypothesis_id", "")) != hypothesis_id
            or {str(item) for item in task.get("claim_ids", []) or []}
            != claim_ids
        ):
            raise ValueError("v4 policy export rejects invalid hypothesis ownership")
    for discrepancy in _rows(investigation.get("material_discrepancies")):
        claim_ids = {
            str(item) for item in discrepancy.get("affected_claim_ids", []) or []
        }
        anchor_ids = {
            str(item)
            for item in discrepancy.get("visual_anchor_fact_ids", []) or []
        }
        evidence_ids = {
            str(item) for item in discrepancy.get("evidence_ids", []) or []
        }
        if not claim_ids or not claim_ids <= set(claims):
            raise ValueError("v4 policy export rejects unowned discrepancy claims")
        if not evidence_ids or not evidence_ids <= set(evidence):
            raise ValueError("v4 policy export rejects invalid discrepancy Evidence")
        if any(
            not anchor_ids
            & {str(item) for item in claims[claim_id].get("anchor_fact_ids", []) or []}
            for claim_id in claim_ids
        ):
            raise ValueError("v4 policy export rejects unanchored discrepancy")
        if any(
            anchor_id not in facts
            or str(
                _mapping(facts[anchor_id].get("origin")).get("type", "")
            )
            not in {"input_image", "ocr"}
            for anchor_id in anchor_ids
        ):
            raise ValueError("v4 policy export rejects non-visual discrepancy anchors")
        if any(
            not any(
                claim_id
                in {
                    str(item)
                    for item in tasks.get(
                        str(evidence[evidence_id].get("task_id", "")),
                        {},
                    ).get("claim_ids", [])
                    or []
                }
                for evidence_id in evidence_ids
            )
            for claim_id in claim_ids
        ):
            raise ValueError("v4 policy export rejects discrepancy Evidence misalignment")
        if (
            str(discrepancy.get("materiality", "")) == "decisive"
            and str(discrepancy.get("status", "")) == "established"
            and any(
                not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance="refute",
                    selected_evidence_ids=evidence_ids,
                    claims=claims,
                    tasks=tasks,
                    evidence=evidence,
                    findings=findings,
                    successful_call_ids=successful_call_ids,
                )
                for claim_id in claim_ids
            )
        ):
            raise ValueError(
                "v4 policy export rejects decisive discrepancy without "
                "qualified refute Evidence"
            )
    verdict = str(trace.get("verdict", ""))
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    if str(judgment.get("verdict", "")) != verdict or any(
        {str(item) for item in judgment.get(judgment_field, []) or []}
        != {str(item) for item in basis.get(basis_field, []) or []}
        for judgment_field, basis_field in (
            ("selected_claim_ids", "claim_ids"),
            ("selected_discrepancy_ids", "discrepancy_ids"),
            ("selected_visual_anchor_fact_ids", "visual_anchor_fact_ids"),
            ("selected_finding_ids", "finding_ids"),
            ("selected_evidence_ids", "evidence_ids"),
        )
    ):
        raise ValueError("v4 policy export rejects Judgment/verdict-basis mismatch")
    if verdict in {"fake", "real"}:
        expected_stance = "refute" if verdict == "fake" else "support"
        basis_claim_ids = {
            str(item) for item in basis.get("claim_ids", []) or []
        }
        basis_evidence_ids = {
            str(item) for item in basis.get("evidence_ids", []) or []
        }
        basis_finding_ids = {
            str(item) for item in basis.get("finding_ids", []) or []
        }
        linked_evidence_ids = {
            str(evidence_id)
            for finding_id in basis_finding_ids & set(findings)
            for evidence_id in findings[finding_id].get("evidence_ids", []) or []
        }
        if not allow_incomplete_verdict_chain and (
            not basis_claim_ids
            or not basis_evidence_ids
            or not basis_finding_ids
            or not basis_evidence_ids <= set(evidence)
            or not basis_evidence_ids <= linked_evidence_ids
            or any(
                not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance=expected_stance,
                    selected_evidence_ids=basis_evidence_ids,
                    selected_finding_ids=basis_finding_ids,
                    claims=claims,
                    tasks=tasks,
                    evidence=evidence,
                    findings=findings,
                    successful_call_ids=successful_call_ids,
                )
                for claim_id in basis_claim_ids
            )
        ):
            raise ValueError(
                "v4 policy export rejects incomplete verdict Finding/Evidence chain"
            )
    audits = _rows(investigation.get("discrepancy_coverage_audits"))
    investigation_stop = str(investigation.get("stop_reason", ""))
    terminal = [
        item
        for item in audits
        if (
            investigation_stop
            and str(item.get("stop_reason", "")) == investigation_stop
            or not investigation_stop
            and item.get("complete") is True
            and str(item.get("stop_reason", "")) == "verdict_determined"
        )
        and str(item.get("stop_reason", ""))
        in {
            "verdict_determined",
            "meaningful_routes_exhausted",
            "information_saturated",
            "hard_budget_exhausted",
        }
    ]
    if not terminal:
        raise ValueError("v4 policy export requires terminal discrepancy Coverage")
    terminal_actions = int(terminal[-1].get("action_count", 0) or 0)
    action_steps = [
        item
        for item in _rows(state.get("all_steps"))
        if str(item.get("stage", "")) == "unified_react"
        and str(item.get("action_type", "")) == "tool_call"
    ]
    if len(action_steps) > terminal_actions:
        raise ValueError("v4 policy export rejects post-verdict actions")


def _unified_react_quality_gate(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    require_provider_thought: bool = True,
) -> None:
    """Reject mixed or non-auditable unified-ReAct episodes before SFT export."""

    if str(trace.get("termination", "")) != "success":
        raise ValueError("unified-react policy export requires a successful trace")
    investigation = _mapping(state.get("investigation_state"))
    completed = {
        str(item)
        for item in investigation.get("unified_react_bootstrap_tools_completed", [])
        or []
    }
    if completed != {"perceive_scene", "ocr_with_position"}:
        raise ValueError(
            "unified-react policy export requires completed scene and OCR bootstrap"
        )
    retired_stages = {
        "perception",
        "image_account_planning",
        "image_only_planning",
        "image_only_investigation",
        "image_only_discrepancy_investigation",
        "image_only_query_concept_extraction",
        "image_only_query_replan",
        "image_only_route_local_replan",
        "image_only_evidence_decision",
        "image_only_discrepancy_judgment",
    }
    steps = _rows(state.get("all_steps"))
    if any(str(step.get("stage", "")) in retired_stages for step in steps):
        raise ValueError("unified-react trace contains a retired policy stage")
    actions = [
        step
        for step in steps
        if str(step.get("stage", "")) == "unified_react"
        and str(step.get("action_type", "")) == "tool_call"
    ]
    if len(actions) < 3:
        raise ValueError(
            "unified-react trace requires scene, OCR, and one investigation action"
        )
    action_names = [str(step.get("tool_name", "")).strip() for step in actions]
    if set(action_names[:2]) != {"perceive_scene", "ocr_with_position"}:
        raise ValueError(
            "unified-react trace must begin with model-selected scene/OCR actions"
        )
    if any(not _mapping(step.get("metadata")).get("unified_react_delta") for step in actions):
        raise ValueError(
            "unified-react trace requires one persisted reducer delta per action"
        )
    if require_provider_thought and any(
        not str(step.get("thought", "") or "").strip() for step in actions
    ):
        raise ValueError(
            "unified-react trace lacks provider-visible thought; route to action-only/RL"
        )
    if not target_fact_rows(investigation):
        raise ValueError("unified-react trace requires a reducer-created target fact")
    if not _rows(investigation.get("search_hypotheses")):
        raise ValueError("unified-react trace requires a reducer-created route")


def unified_react_training_buckets(
    trace: Mapping[str, Any],
) -> list[str]:
    """Return non-overlapping SFT buckets plus the RL-candidate membership.

    The structural audit is shared.  A trace with missing readable provider
    thought remains an executable action-only/RL candidate, but it must never
    enter the reasoning SFT corpus by accident.
    """

    state = _mapping(trace.get("state"))
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError(
            "unified training buckets accept unified-react-v1 traces only"
        )
    _unified_react_quality_gate(
        trace,
        state,
        require_provider_thought=False,
    )
    actions = [
        step
        for step in _rows(state.get("all_steps"))
        if str(step.get("stage", "")) == "unified_react"
        and str(step.get("action_type", "")) == "tool_call"
    ]
    reasoning_ready = all(
        str(step.get("thought", "") or "").strip() for step in actions
    )
    return [
        "reasoning_sft" if reasoning_ready else "action_only",
        "rl_candidate",
    ]


def export_policy_examples(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    allow_incomplete_verdict_chain: bool = False,
) -> List[PolicyExample]:
    """Export actual model-visible requests/actions from one canonical trace.

    ``allow_incomplete_verdict_chain`` is reserved for the post-rollout SFT
    release path after the frozen SFT judge has accepted the episode.  It only
    relaxes the final Finding/Evidence closure check; protocol validity,
    private-data isolation, judgment/basis consistency, and post-verdict
    action checks remain enforced.
    """

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != (
        "image_only"
    ):
        raise ValueError("policy exporter accepts image-only traces only")
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version != "unified-react-v1":
        raise ValueError("policy exporter received an unsupported decision policy")
    _unified_react_quality_gate(trace, state)

    tokenizer = tokenizer or Utf8ByteTokenizer()
    source_metadata = source_metadata or {}
    episode_id = str(
        trace.get("image_id") or state.get("image_id") or ""
    ).strip()
    if not episode_id:
        raise ValueError("canonical trace requires image_id")

    candidates: List[tuple[int, Mapping[str, Any], str]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if str(step.get("action_type", "")) in {
            "planning_revision",
            "format_error",
            "output_rejected",
            "policy_replan",
        }:
            continue
        stage = str(step.get("stage", "")).strip()
        example_type = _example_type(stage)
        metadata = _mapping(step.get("metadata"))
        if example_type is None:
            continue
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        candidates.append((index, step, example_type))

    trace_failed = str(trace.get("termination", "")).strip() == "error"
    examples: List[PolicyExample] = []
    for position, (index, step, example_type) in enumerate(candidates):
        metadata = _mapping(step.get("metadata"))
        policy_input = dict(_mapping(metadata.get("policy_input")))
        policy_action = dict(_mapping(metadata.get("policy_action")))
        _assert_no_private_data(policy_input)
        _assert_no_private_data(policy_action)
        input_ids = list(tokenizer.encode(canonical_json(policy_input)))
        action_ids = list(tokenizer.encode(canonical_json(policy_action)))
        action_valid = str(step.get("action_type", "")) not in {
            "format_error",
            "output_rejected",
            "policy_replan",
        }
        fatal_boundary = trace_failed and position == len(candidates) - 1
        trainable = action_valid and not fatal_boundary
        terminated = (
            example_type == "judgment"
            or position == len(candidates) - 1
            and str(trace.get("termination", "")) in {"success", "error"}
        )
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        step_id = (
            f"{episode_id}:{example_type}:{interaction_id}"
            if interaction_id
            else f"{episode_id}:{example_type}:{index + 1}"
        )
        examples.append(
            PolicyExample(
                trajectory_version=(
                    "ifv-policy-v3"
                ),
                tokenizer_id=tokenizer.tokenizer_id,
                episode_id=episode_id,
                step_id=step_id,
                source_run_id=str(source_metadata.get("source_run_id", "")),
                runtime_commit=str(source_metadata.get("runtime_commit", "")),
                release_id=str(source_metadata.get("release_id", "")),
                runtime_contract_version=str(
                    source_metadata.get("runtime_contract_version", "")
                ),
                process_reference_protocol_version=str(
                    source_metadata.get(
                        "process_reference_protocol_version",
                        "",
                    )
                ),
                example_type=example_type,
                runtime_observation_refs=_observation_refs(policy_input),
                policy_input=policy_input,
                policy_action=policy_action,
                policy_input_token_ids=input_ids,
                policy_action_token_ids=action_ids,
                policy_action_loss_mask=[
                    1 if trainable else 0 for _ in action_ids
                ],
                action_valid=action_valid,
                terminated=terminated,
                fatal_boundary=fatal_boundary,
            )
        )
    return examples


def examples_as_jsonl(examples: Iterable[PolicyExample]) -> str:
    lines = [
        item.model_dump_json(exclude_none=False)
        for item in examples
    ]
    return "\n".join(lines) + ("\n" if lines else "")
