from __future__ import annotations

from scripts.backfill_fact_check_reports import (
    _latest_records,
    _normalize_report_verdict,
)
from src.orchestrator.investigation_models import FactCheckReport
from src.trajectory.report_history import build_full_event_history


def _trace() -> dict[str, object]:
    system_instruction = "Agent system instruction"
    return {
        "case_id": "case-1",
        "image_id": "case-1",
        "verdict": "fake",
        "overall_assessment": "The named winner is incorrect.",
        "termination": "success",
        "verdict_basis": {
            "policy_rule_id": "unified-react-v1",
            "decision_mode": "evidence_determined",
            "verdict_target": "A won the final.",
            "claim_ids": ["claim-1"],
            "evidence_ids": ["evidence-1"],
        },
        "state": {
            "runtime_case": {"case_id": "case-1"},
            "investigation_state": {},
            "all_steps": [
                {
                    "round": 1,
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "thought": "Inspect the scene.",
                    "tool_name": "perceive_scene",
                    "tool_args": {},
                    "tool_result": '{"status":"success","scene":"podium"}',
                    "tokens": {"prompt": 10},
                    "metadata": {
                        "policy_input": {
                            "system_instruction": system_instruction,
                            "input_payload": (
                                '{"runtime_handoff":{"workspace":"duplicate",'
                                '"task_objective":"inspect"}}'
                            ),
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "perceive_scene",
                                    "parameters": {},
                                }
                            ],
                        },
                        "policy_action": {
                            "type": "tool_call",
                            "name": "perceive_scene",
                            "arguments": {},
                        },
                        "tool_success": True,
                        "unified_react_delta": {"accepted": True},
                    },
                },
                {
                    "round": 2,
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "thought": "Search the name.",
                    "tool_name": "text_search",
                    "tool_args": {"query": "A final winner"},
                    "tool_result": '{"status":"success","results":["B won"]}',
                    "tokens": {"prompt": 12},
                    "metadata": {
                        "policy_input": {
                            "system_instruction": system_instruction,
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "text_search",
                                    "parameters": {},
                                }
                            ],
                        },
                        "policy_action": {
                            "type": "tool_call",
                            "name": "text_search",
                            "arguments": {"query": "A final winner"},
                        },
                        "tool_success": True,
                        "investigation_state_update": {"new_evidence": "evidence-1"},
                    },
                },
            ],
        },
    }


def test_full_report_history_keeps_every_event_without_replaying_workspace() -> None:
    history = build_full_event_history(_trace())

    assert history["case_id"] == "case-1"
    assert len(history["chronological_events"]) == 2
    assert history["chronological_events"][0]["thought"] == "Inspect the scene."
    assert history["chronological_events"][1]["tool_args"] == {
        "query": "A final winner"
    }
    assert history["chronological_events"][1]["runtime_event"][
        "investigation_state_update"
    ] == {"new_evidence": "evidence-1"}
    assert history["initial_context"]["stage_instruction"] == "Agent system instruction"
    assert "workspace" not in str(history["initial_context"]["input_payload"])
    assert "policy_input" not in history["chronological_events"][0]


def test_posthoc_report_prefixes_the_immutable_verdict_when_missing() -> None:
    report = FactCheckReport(
        headline="A headline",
        claim_under_review="A claim",
        verdict_summary="The depicted claim is not authentic.",
        key_findings=["A supported finding."],
        evidence_summary="A supported evidence summary.",
    )

    normalized = _normalize_report_verdict(report, verdict="fake")

    assert normalized.verdict_summary == (
        "Fake: The depicted claim is not authentic."
    )


def test_latest_posthoc_report_record_wins_during_resume() -> None:
    rows = _latest_records(
        [
            {"case_id": "case-a", "status": "error"},
            {"case_id": "case-b", "status": "completed"},
            {"case_id": "case-a", "status": "completed"},
        ]
    )

    assert {row["case_id"]: row["status"] for row in rows} == {
        "case-a": "completed",
        "case-b": "completed",
    }
