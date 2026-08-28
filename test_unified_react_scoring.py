from __future__ import annotations

import json

from src.trajectory.scoring import score_process_trace


def _trace_with_bootstrap_actions() -> dict:
    claim_id = "claim-1"
    anchor_id = "vf-anchor-1"
    target_fact_id = "vf-target-1"
    task_id = "task-1"
    basis = {
        "policy_rule_id": "unified-react-v1",
        "decision_mode": "bounded_binary_judgment",
        "verdict_target": "The image depicts the Riverfest bridge.",
        "claim_ids": [claim_id],
        "discrepancy_ids": [],
        "visual_anchor_fact_ids": [anchor_id],
        "finding_ids": [],
        "evidence_ids": [],
        "unresolved_gaps": ["No decisive external evidence was available."],
    }
    judgment = {
        "policy_rule_id": "unified-react-v1",
        "verdict": "real",
        "confidence": 0.7,
        "selected_claim_ids": [claim_id],
        "selected_discrepancy_ids": [],
        "selected_visual_anchor_fact_ids": [anchor_id],
        "selected_finding_ids": [],
        "selected_evidence_ids": [],
        "overall_assessment": "The bounded basis supports the depicted relation.",
        "unresolved_gaps": ["No decisive external evidence was available."],
    }

    def tool_step(tool_name: str, *, investigation: bool) -> dict:
        update = (
            {"investigation_state_update": {"accepted": True}}
            if investigation
            else {}
        )
        return {
            "stage": "unified_react",
            "action_type": "tool_call",
            "tool_name": tool_name,
            "tool_args": {},
            "tool_result": json.dumps({"status": "success"}),
            "metadata": {
                "tool_success": True,
                "function_call_id": f"call-{tool_name}",
                **update,
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
                tool_step("perceive_scene", investigation=False),
                tool_step("ocr_with_position", investigation=False),
                tool_step("text_search", investigation=True),
            ],
            "investigation_state": {
                "target_facts": [
                    {
                        "claim_id": claim_id,
                        "fact_id": target_fact_id,
                        "anchor_fact_ids": [anchor_id],
                        "status": "unresolved",
                        "task_ids": [task_id],
                    }
                ],
                "tasks": [
                    {
                        "task_id": task_id,
                        "claim_ids": [claim_id],
                        "fact_ids": [target_fact_id],
                        "status": "exhausted",
                    }
                ],
                "evidence": [],
                "findings": [],
                "claim_assessments": [],
                "material_discrepancies": [],
                "discrepancy_coverage_audits": [
                    {
                        "action_count": 1,
                        "complete": False,
                        "stop_reason": "meaningful_routes_exhausted",
                    }
                ],
                "stop_reason": "meaningful_routes_exhausted",
                "action_count": 1,
                "discrepancy_verdict_basis": basis,
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
