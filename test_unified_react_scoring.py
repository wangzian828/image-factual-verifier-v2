from __future__ import annotations

import json

from src.trajectory.scoring import score_process_trace


def _trace_with_bootstrap_actions() -> dict:
    basis = {
        "schema_version": "ifv-raw-history-judgment-basis-v1",
        "decision_mode": "raw_history",
        "verdict_target": "The image depicts the Riverfest bridge.",
        "observation_ids": [
            "call-perceive_scene",
            "call-ocr_with_position",
            "call-text_search",
        ],
        "observations": [
            {
                "observation_id": "call-perceive_scene",
                "tool_name": "perceive_scene",
                "status": "success",
            },
            {
                "observation_id": "call-ocr_with_position",
                "tool_name": "ocr_with_position",
                "status": "success",
            },
            {
                "observation_id": "call-text_search",
                "tool_name": "text_search",
                "status": "success",
            },
        ],
        "action_count": 3,
        "unresolved_gaps": ["No decisive external evidence was available."],
    }
    judgment = {
        "policy_rule_id": "unified-react-v1",
        "verdict": "real",
        "confidence": 0.7,
        "selected_observation_ids": basis["observation_ids"],
        "verdict_observation_ids": ["call-perceive_scene"],
        "overall_assessment": "The bounded basis supports the depicted relation.",
        "unresolved_gaps": ["No decisive external evidence was available."],
        "fact_check_report": {"summary": "The image depicts the bridge."},
    }

    def tool_step(tool_name: str) -> dict:
        return {
            "stage": "unified_react",
            "action_type": "tool_call",
            "tool_name": tool_name,
            "tool_args": {},
            "tool_result": json.dumps({"status": "success"}),
            "metadata": {
                "tool_success": True,
                "function_call_id": f"call-{tool_name}",
            },
        }

    return {
        "image_id": "case-1",
        "input_mode": "image_only",
        "decision_policy_version": "unified-react-v1",
        "verdict": "real",
        "verdict_basis": basis,
        "judgment": judgment,
        "termination": "success",
        "state": {
            "input_mode": "image_only",
            "decision_policy_version": "unified-react-v1",
            "termination": "success",
            "all_steps": [
                tool_step("perceive_scene"),
                tool_step("ocr_with_position"),
                tool_step("text_search"),
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": judgment,
                },
            ],
            "investigation_state": {
                "schema_version": "ifv-unified-react-raw-history-v1",
                "case_id": "case-1",
                "image_sha256": "a" * 64,
                "objective": "Verify the factual content expressed by the image.",
                "action_count": 3,
                "stop_reason": "meaningful_routes_exhausted",
                "finish_rationale": "The available routes were exhausted.",
            },
        },
    }


def test_bootstrap_actions_are_not_counted_as_post_determination_actions() -> None:
    trace = _trace_with_bootstrap_actions()

    metrics, score = score_process_trace(
        trace,
        {
            "case_id": "case-1",
            "factual_status": "supported",
            "decisive_facts": [],
        },
    )

    assert metrics["post_determination_action_count"] == 0
    assert metrics["stop_quality"] == 1.0
    assert "stop_quality_invalid" not in score["training_exclusion_reasons"]
