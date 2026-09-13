from __future__ import annotations

import json

import pytest

from scripts.trajectory.audit_canonical_trace_rebuild import (
    _answer,
    _candidate_steps,
    _canonical_result,
    _raw_json_prefix,
    _thought,
    _tool_call,
)


def test_audit_parses_provider_messages_without_copying_stage_control() -> None:
    response = (
        json.dumps(
            {
                "observation_locator": {
                    "observation_id": "call-1",
                    "tool_name": "text_search",
                    "tool_success": True,
                },
                "result": {"status": "success", "value": 1},
            }
        )
        + "\n\n<stage_control>{\"stage\":\"judgment\"}</stage_control>"
    )

    assert _raw_json_prefix(response)["observation_locator"]["observation_id"] == "call-1"
    assert _tool_call(
        json.dumps(
            {
                "name": "text_search",
                "arguments": json.dumps({"queries": "fact"}),
            }
        )
    ) == {"name": "text_search", "arguments": {"queries": "fact"}}
    assert _thought("<think>\nreason carefully\n</think>") == "reason carefully"
    assert _answer("<answer>{\"verdict\":\"real\"}</answer>") == {
        "verdict": "real"
    }


def test_audit_normalizes_only_the_tool_result_envelope() -> None:
    assert _canonical_result(
        json.dumps({"result": {"status": "success", "rows": [1]}})
    ) == {"status": "success", "rows": [1]}
    assert _canonical_result("plain tool output") == "plain tool output"


def test_audit_reconstructs_only_accepted_policy_steps() -> None:
    def step(stage: str, action_type: str, complete: bool = True) -> dict:
        metadata = (
            {"policy_input": {}, "policy_action": {"type": "tool_call"}}
            if complete
            else {}
        )
        return {
            "stage": stage,
            "action_type": action_type,
            "metadata": metadata,
        }

    trace = {
        "state": {
            "all_steps": [
                step("unified_react", "tool_call"),
                step("unified_react", "format_error"),
                step("unified_judgment", "final"),
                step("legacy_stage", "tool_call"),
                step("unified_react", "tool_call", complete=False),
            ]
        }
    }

    assert [item["stage"] for item in _candidate_steps(trace)] == [
        "unified_react",
        "unified_judgment",
    ]


def test_audit_reconstructs_bound_forced_judgment_correction() -> None:
    corrected = {
        "verdict": "real",
        "confidence": 0.7,
        "verdict_observation_ids": ["call-1"],
        "overall_assessment": "The evidence supports the claim.",
        "fact_check_report": {"headline": "Supported"},
    }
    trace = {
        "judgment": {**corrected, "policy_rule_id": "unified-react-v1"},
        "state": {
            "all_steps": [
                {
                    "stage": "unified_judgment",
                    "action_type": "output_rejected",
                    "output": {"verdict": "unknown"},
                    "metadata": {
                        "context_request_id": "req-1",
                        "policy_input": {"input_payload": ["history"]},
                        "policy_action": {"verdict": "unknown"},
                    },
                },
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": corrected,
                    "metadata": {
                        "forced_output": True,
                        "interaction_lifecycle_kind": "protocol_correction",
                        "parent_context_request_id": "req-1",
                    },
                },
            ]
        },
    }

    candidates = _candidate_steps(trace)

    assert len(candidates) == 1
    assert candidates[0]["metadata"]["policy_action"] == corrected
    assert candidates[0]["metadata"]["policy_input"] == {
        "input_payload": ["history"]
    }


def test_audit_rejects_non_object_tool_arguments() -> None:
    with pytest.raises(ValueError, match="arguments"):
        _tool_call(json.dumps({"name": "text_search", "arguments": "[]"}))
