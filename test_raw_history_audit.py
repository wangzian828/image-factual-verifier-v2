from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_real_trace import audit_trace


def _trace() -> dict:
    observation_id = "call-scene"
    judgment = {
        "policy_rule_id": "unified-react-v1",
        "verdict": "real",
        "confidence": 0.6,
        "selected_observation_ids": [observation_id],
        "verdict_observation_ids": [observation_id],
        "overall_assessment": "The retained observation supports the bounded label.",
        "fact_check_report": {
            "headline": "Bounded image fact check",
            "claim_under_review": "The image depicts a bridge over a river.",
            "verdict_summary": "The image is labeled real.",
            "key_findings": ["The scene observation records a bridge over water."],
            "evidence_summary": "The report uses the retained scene observation.",
            "remaining_uncertainties": [],
        },
    }
    basis = {
        "schema_version": "ifv-raw-history-judgment-basis-v1",
        "decision_mode": "bounded_binary_judgment",
        "objective": "Verify the factual content expressed by the image.",
        "observation_ids": [observation_id],
        "observations": [
            {
                "observation_id": observation_id,
                "function_call_id": observation_id,
                "tool_name": "perceive_scene",
                "status": "success",
            }
        ],
        "action_count": 1,
        "stop_reason": "meaningful_routes_exhausted",
        "finish_rationale": "The available routes were exhausted.",
    }
    action = {
        "stage": "unified_react",
        "action_type": "tool_call",
        "tool_name": "perceive_scene",
        "tool_args": {},
        "tool_result": json.dumps(
            {"status": "success", "scene_description": "A bridge crosses a river."}
        ),
        "thought": "The image needs a grounded scene observation.",
        "tokens": {"thought": 1},
        "metadata": {
            "native_interactions": True,
            "interaction_id": "interaction-scene",
            "previous_interaction_id": None,
            "interaction_lifecycle_kind": "tool_roundtrip",
            "function_call_id": observation_id,
            "tool_success": True,
            "policy_action": {
                "type": "tool_call",
                "name": "perceive_scene",
                "arguments": {},
            },
        },
    }
    return {
        "image_id": "case-raw-history",
        "input_mode": "image_only",
        "decision_policy_version": "unified-react-v1",
        "verdict": "real",
        "judgment": judgment,
        "verdict_basis": basis,
        "termination": "success",
        "token_usage": {"thought": 1},
        "state": {
            "image_id": "case-raw-history",
            "input_mode": "image_only",
            "decision_policy_version": "unified-react-v1",
            "termination": "success",
            "token_usage": {"thought": 1},
            "all_steps": [
                action,
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": judgment,
                    "tokens": {"thought": 0},
                    "metadata": {
                        "policy_action": {"type": "output", "value": judgment}
                    },
                },
            ],
            "investigation_state": {
                "schema_version": "ifv-unified-react-raw-history-v1",
                "case_id": "case-raw-history",
                "image_sha256": "a" * 64,
                "objective": basis["objective"],
                "action_count": 1,
                "stop_reason": basis["stop_reason"],
                "finish_rationale": basis["finish_rationale"],
            },
            "judgment": judgment,
        },
    }


def test_strict_audit_accepts_current_raw_history_trace(tmp_path: Path) -> None:
    path = tmp_path / "raw-history-trace.json"
    path.write_text(json.dumps(_trace()), encoding="utf-8")

    report = audit_trace(path)

    assert report.failures(strict_scheduler=True) == []
    assert report.stats["unified_react_actions"] == 1
    assert report.stats["react_runtime_successful_observations"] == 1
