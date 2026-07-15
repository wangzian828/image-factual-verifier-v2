from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_real_trace import audit_trace
from test_image_only_trajectory import (
    test_scripted_image_only_complete_trajectory,
)


def _scripted_trace(tmp_path: Path) -> Path:
    test_scripted_image_only_complete_trajectory(tmp_path)
    return tmp_path / "traces" / "case_scripted_v3.json"


def test_strict_audit_accepts_complete_image_only_trace(
    tmp_path: Path,
) -> None:
    report = audit_trace(_scripted_trace(tmp_path))

    assert not report.failures(strict_scheduler=True)
    assert report.stats["image_only_actions"] == 3
    assert report.stats["reflections"] == 0
    assert report.stats["decisive_facts"] == 1


def test_strict_audit_rejects_discovery_as_verdict_evidence(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    discovery_id = investigation["discoveries"][0]["discovery_id"]
    trace["verdict_basis"]["evidence_ids"].append(discovery_id)
    investigation["verdict_basis"]["evidence_ids"].append(discovery_id)
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=True)
    }

    assert "DISCOVERY_USED_AS_VERDICT_EVIDENCE" in codes
    assert "VERDICT_BASIS_EVIDENCE_UNKNOWN" in codes


def test_unverifiable_basis_does_not_require_finding_chain(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    decisive_ids = list(investigation["decisive_fact_ids"])
    for fact in investigation["facts"]:
        if fact["fact_id"] in decisive_ids:
            fact["status"] = "active"
    gaps = [f"Decisive evidence is absent for: {fact_id}" for fact_id in decisive_ids]
    basis = {
        "policy_rule_id": "reinspect-v2",
        "verdict_target": " | ".join(decisive_ids),
        "fact_ids": decisive_ids,
        "finding_ids": [],
        "evidence_ids": [],
        "mechanism": None,
        "unresolved_gaps": gaps,
    }
    judgment = {
        "verdict": "unverifiable",
        "confidence": 0.5,
        "policy_rule_id": "reinspect-v2",
        "selected_fact_ids": decisive_ids,
        "selected_finding_ids": [],
        "selected_evidence_ids": [],
        "overall_assessment": "The decisive facts remain unresolved.",
        "unresolved_gaps": gaps,
    }
    trace["verdict"] = "unverifiable"
    trace["verdict_basis"] = basis
    trace["judgment"] = judgment
    trace["state"]["judgment"] = judgment
    investigation["verdict_basis"] = basis
    investigation["judgment"] = judgment
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=False)
    }

    assert "VERDICT_FACT_WITHOUT_FINDING" not in codes
    assert "VERDICT_FINDING_WITHOUT_EVIDENCE" not in codes


def test_blocked_duplicate_route_is_a_warning_not_a_protocol_failure(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["state"]["all_steps"].append(
        {
            "round": 99,
            "stage": "image_only_investigation",
            "action_type": "format_error",
            "tool_name": "visit",
            "tool_args": {
                "__question_id": "task-duplicate",
                "url": ["https://example.org/already-visited"],
            },
            "tool_result": json.dumps(
                {
                    "status": "error",
                    "error": (
                        "You already called 'visit' with essentially the "
                        "same target."
                    ),
                }
            ),
            "metadata": {
                "stage": "image_only_investigation",
                "duplicate_tool_call": True,
                "function_call_id": "call-duplicate",
            },
        }
    )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    warnings = report.warnings(strict_scheduler=True)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["route_control_rejections"] == 1
    assert {item.code for item in warnings} == {
        "ROUTE_CONTROL_REJECTION"
    }


def test_successful_planning_revision_is_not_a_protocol_rejection(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["state"]["all_steps"].insert(
        2,
        {
            "round": 1,
            "stage": "image_only_planning",
            "action_type": "planning_revision",
            "tool_name": "",
            "tool_args": {},
            "tool_result": "",
            "output": {
                "proposals": [],
                "remaining_target_gaps": [],
            },
            "metadata": {
                "stage": "image_only_planning",
                "rejection_reason": (
                    "propose the depicted-world relation separately"
                ),
                "planning_revision_reason": (
                    "propose the depicted-world relation separately"
                ),
            },
        },
    )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["protocol_rejections"] == 0
