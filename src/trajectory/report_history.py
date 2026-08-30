"""Lossless semantic history projection for post-hoc fact-check reports.

Canonical traces keep full per-turn provider packets for replay.  Those packets
repeat the same system prompt and cumulative workspace many times.  This module
preserves every chronological action, thought, tool result, failure, and reducer
delta while representing repeated runtime snapshots only once.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.trajectory.exporter import (
    _compact_export_input_payload,
    _normalize_tool_schema,
    _strip_gemini_wire_instructions,
)


REPORT_HISTORY_SCHEMA_VERSION = "ifv-full-event-history-v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _policy_contract(policy_input: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the active stage boundary without replaying its cumulative state."""

    result: dict[str, Any] = {}
    instruction = _strip_gemini_wire_instructions(
        _text(policy_input.get("system_instruction"))
    )
    if instruction:
        result["stage_instruction"] = instruction
    tools = _normalize_tool_schema(policy_input.get("tools"))
    if tools:
        result["authorized_tools"] = [
            str(item["function"]["name"])
            for item in tools
            if isinstance(item.get("function"), Mapping)
            and str(item["function"].get("name") or "").strip()
        ]
    response_format = policy_input.get("response_format")
    if isinstance(response_format, Mapping):
        result["response_format"] = dict(response_format)
    return result


def _runtime_event(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Select non-duplicated lifecycle fields from one recorded step."""

    keys = (
        "policy_action",
        "tool_success",
        "tool_execution_status",
        "tool_name",
        "retry_count",
        "protocol_corrections_used",
        "accepted_investigation_intent",
        "route_local_replan_trigger",
        "unified_react_delta",
        "investigation_state_update",
        "deterministic_segment_boundary",
        "stage",
        "interaction_status",
        "interaction_lifecycle_kind",
    )
    return {
        key: metadata[key]
        for key in keys
        if metadata.get(key) not in (None, "", [], {})
    }


def build_full_event_history(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete non-duplicated chronological report context.

    The projection preserves every row in ``state.all_steps``.  It deliberately
    excludes only the raw ``metadata.policy_input`` object from each event:
    its initial compact form is retained once, while its later cumulative
    contents are already represented by the preceding event stream and state
    deltas.
    """

    state = _mapping(trace.get("state"))
    steps = _rows(state.get("all_steps"))
    initial_input: dict[str, Any] = {}
    previous_contract: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    for index, step in enumerate(steps, start=1):
        metadata = _mapping(step.get("metadata"))
        policy_input = _mapping(metadata.get("policy_input"))
        contract = _policy_contract(policy_input) if policy_input else {}
        if not initial_input and policy_input:
            initial_input = {
                "stage": _text(step.get("stage")),
                "stage_instruction": contract.get("stage_instruction", ""),
                "input_payload": _compact_export_input_payload(
                    policy_input.get("input_payload", "")
                ),
                "authorized_tools": contract.get("authorized_tools", []),
                "response_format": contract.get("response_format"),
            }

        contract_delta = {
            key: value
            for key, value in contract.items()
            if previous_contract.get(key) != value
        }
        if contract:
            previous_contract = contract
        event = {
            "event_index": index,
            "round": step.get("round"),
            "stage": _text(step.get("stage")),
            "action_type": _text(step.get("action_type")),
            "thought": _text(step.get("thought")),
            "tool_name": _text(step.get("tool_name")),
            "tool_args": step.get("tool_args"),
            "tool_result": step.get("tool_result"),
            "output": step.get("output"),
            "tokens": step.get("tokens"),
            "runtime_event": _runtime_event(metadata),
        }
        if contract_delta:
            event["policy_contract_delta"] = contract_delta
        events.append(event)

    investigation = _mapping(state.get("investigation_state"))
    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    return {
        "schema_version": REPORT_HISTORY_SCHEMA_VERSION,
        "case_id": _text(
            _mapping(state.get("runtime_case")).get("case_id")
            or trace.get("case_id")
            or trace.get("image_id")
        ),
        "initial_context": initial_input,
        "chronological_events": events,
        "terminal_record": {
            "verdict": _text(judgment.get("verdict") or trace.get("verdict")),
            "overall_assessment": _text(
                judgment.get("overall_assessment")
                or trace.get("overall_assessment")
            ),
            "verdict_basis": basis,
            "judgment": judgment,
            "termination": _text(trace.get("termination")),
            "investigation_status": _text(trace.get("investigation_status")),
            "final_visual_audit": trace.get("final_visual_audit"),
        },
    }
