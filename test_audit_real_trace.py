from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_real_trace import audit_trace
from test_image_only_trajectory import (
    test_scripted_image_only_complete_trajectory,
)


def _v4_trace(tmp_path: Path) -> Path:
    claim_id = "claim-v4"
    hypothesis_id = "hypothesis-v4"
    task_id = "task-v4"
    evidence_id = "evidence-v4"
    discrepancy_id = "discrepancy-v4"
    anchor_id = "fact-anchor-v4"
    claim_fact_id = "fact-claim-v4"
    finding_id = "finding-v4"
    basis = {
        "policy_rule_id": "discrepancy-first-v4",
        "verdict_target": "The shown packet replaces the source microphone.",
        "claim_ids": [claim_id],
        "discrepancy_ids": [discrepancy_id],
        "visual_anchor_fact_ids": [anchor_id],
        "finding_ids": [finding_id],
        "evidence_ids": [evidence_id],
        "unresolved_gaps": [],
    }
    judgment = {
        "verdict": "fake",
        "confidence": 0.99,
        "policy_rule_id": "discrepancy-first-v4",
        "selected_claim_ids": [claim_id],
        "selected_discrepancy_ids": [discrepancy_id],
        "selected_visual_anchor_fact_ids": [anchor_id],
        "selected_finding_ids": [finding_id],
        "selected_evidence_ids": [evidence_id],
        "overall_assessment": "The selected source Evidence establishes the edit.",
        "unresolved_gaps": [],
    }
    investigation = {
        "brief": {"brief_id": "brief-v4", "case_id": "case-v4-audit"},
        "facts": [
            {
                "fact_id": anchor_id,
                "origin": {"type": "input_image", "origin_ids": ["image-v4"]},
            },
            {
                "fact_id": claim_fact_id,
                "origin": {"type": "input_image", "origin_ids": [anchor_id]},
            },
        ],
        "tasks": [
            {
                "task_id": task_id,
                "fact_ids": [claim_fact_id],
                "claim_ids": [claim_id],
                "hypothesis_id": hypothesis_id,
            }
        ],
        "evidence": [
            {
                "evidence_id": evidence_id,
                "task_id": task_id,
                "fact_ids": [claim_fact_id],
                "function_call_id": "call-visit-v4",
                "tool_name": "visit",
                "evidence_kind": "web_span",
                "stance": "refute",
                "quality": "strong",
                "directness": "direct",
                "claim_binding": "source_assertion",
                "relation_scope": "same_relation",
                "relation_stance": "contradicts",
                "risk_flags": [],
            }
        ],
        "findings": [
            {
                "finding_id": finding_id,
                "task_id": task_id,
                "fact_ids": [claim_fact_id],
                "evidence_ids": [evidence_id],
                "stance": "refute",
            }
        ],
        "image_claims": [
            {
                "claim_id": claim_id,
                "fact_id": claim_fact_id,
                "statement": "The person is holding the shown packet.",
                "anchor_fact_ids": [anchor_id],
                "salience": "high",
                "status": "refuted",
                "task_ids": [task_id],
            }
        ],
        "search_hypotheses": [
            {
                "hypothesis_id": hypothesis_id,
                "claim_ids": [claim_id],
                "statement": "A source image may show the original held object.",
                "status": "exhausted",
                "task_id": task_id,
            }
        ],
        "claim_assessments": [
            {
                "assessment_id": "assessment-v4",
                "claim_id": claim_id,
                "assessment": "refuted",
                "evidence_ids": [evidence_id],
            }
        ],
        "material_discrepancies": [
            {
                "discrepancy_id": discrepancy_id,
                "statement": "The packet replaces the source microphone.",
                "affected_claim_ids": [claim_id],
                "visual_anchor_fact_ids": [anchor_id],
                "evidence_ids": [evidence_id],
                "materiality": "decisive",
                "status": "established",
            }
        ],
        "discrepancy_decisions": [
            {
                "decision_id": "decision-v4",
                "reviewed_evidence_ids": [evidence_id],
                "output": {
                    "claim_assessments": [
                        {
                            "claim_id": claim_id,
                            "selected_evidence_ids": [evidence_id],
                        }
                    ],
                    "material_discrepancy": {
                        "evidence_ids": [evidence_id],
                    },
                },
            }
        ],
        "discrepancy_coverage_audits": [
            {
                "audit_id": "coverage-v4",
                "action_count": 2,
                "complete": True,
                "stop_reason": "verdict_determined",
            }
        ],
        "discrepancy_verdict_basis": basis,
        "discrepancy_judgment": judgment,
        "core_verdict_fact_id": None,
        "action_count": 2,
    }
    steps = [
        {
            "round": 1,
            "stage": "image_account_planning",
            "action_type": "output",
            "tokens": {"thought": 0},
            "metadata": {
                "native_interactions": True,
                "interaction_id": "interaction-plan-v4",
                "previous_interaction_id": None,
            },
        },
        {
            "round": 1,
            "stage": "image_only_discrepancy_investigation",
            "action_type": "tool_call",
            "tool_name": "text_search",
            "tool_result": json.dumps({"status": "success", "queries": []}),
            "tokens": {"thought": 0},
            "metadata": {
                "native_interactions": True,
                "interaction_id": "interaction-search-v4",
                "previous_interaction_id": "interaction-plan-v4",
                "function_call_id": "call-search-v4",
                "tool_success": True,
            },
        },
        {
            "round": 2,
            "stage": "image_only_discrepancy_investigation",
            "action_type": "tool_call",
            "tool_name": "visit",
            "tool_result": json.dumps({"status": "success", "evidence": "exact"}),
            "tokens": {"thought": 0},
            "metadata": {
                "native_interactions": True,
                "interaction_id": "interaction-visit-v4",
                "previous_interaction_id": "interaction-search-v4",
                "function_call_id": "call-visit-v4",
                "tool_success": True,
            },
        },
        {
            "round": 1,
            "stage": "image_only_discrepancy_decision",
            "action_type": "output",
            "tokens": {"thought": 0},
            "metadata": {
                "native_interactions": True,
                "interaction_id": "interaction-decision-v4",
                "previous_interaction_id": "interaction-visit-v4",
            },
        },
        {
            "round": 1,
            "stage": "image_only_discrepancy_judgment",
            "action_type": "output",
            "tokens": {"thought": 0},
            "metadata": {
                "native_interactions": True,
                "interaction_id": "interaction-judgment-v4",
                "previous_interaction_id": "interaction-decision-v4",
            },
        },
    ]
    trace = {
        "image_id": "case-v4-audit",
        "input_mode": "image_only",
        "decision_policy_version": "discrepancy-first-v4",
        "verdict": "fake",
        "verdict_basis": basis,
        "judgment": judgment,
        "termination": "success",
        "token_usage": {"thought": 0},
        "state": {
            "image_id": "case-v4-audit",
            "input_mode": "image_only",
            "decision_policy_version": "discrepancy-first-v4",
            "termination": "success",
            "token_usage": {"thought": 0},
            "all_steps": steps,
            "investigation_state": investigation,
            "judgment": judgment,
        },
    }
    path = tmp_path / "v4-trace.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_strict_audit_accepts_discrepancy_first_v4_trace(tmp_path: Path) -> None:
    report = audit_trace(_v4_trace(tmp_path))

    assert not report.failures(strict_scheduler=True)
    assert report.stats["image_claims"] == 1
    assert report.stats["material_discrepancies"] == 1
    assert report.stats["v4_actions"] == 2


def test_strict_audit_records_thought_tokens_for_all_v4_stages(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["token_usage"]["thought"] = 12
    trace["state"]["token_usage"]["thought"] = 12
    trace["state"]["all_steps"][0]["tokens"]["thought"] = 12
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert "THOUGHT_TOKENS_NONZERO" not in {
        issue.code for issue in report.failures(strict_scheduler=True)
    }

    trace["state"]["all_steps"][1]["tokens"]["thought"] = 1
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = audit_trace(trace_path)
    assert "THOUGHT_TOKENS_NONZERO" not in {
        issue.code for issue in report.failures(strict_scheduler=True)
    }


def test_strict_audit_accepts_bounded_binary_v4_trace(tmp_path: Path) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    basis = trace["verdict_basis"]
    basis.update(
        {
            "decision_mode": "bounded_binary_judgment",
            "discrepancy_ids": [],
            "finding_ids": [],
            "evidence_ids": [],
            "unresolved_gaps": ["The available material does not close the claim."],
        }
    )
    investigation["discrepancy_verdict_basis"] = dict(basis)
    investigation["material_discrepancies"] = []
    investigation["claim_assessments"] = []
    investigation["discrepancy_decisions"][0]["output"] = {
        "claim_assessments": [],
        "material_discrepancy": None,
    }
    investigation["stop_reason"] = "meaningful_routes_exhausted"
    investigation["discrepancy_coverage_audits"][0].update(
        {
            "complete": False,
            "stop_reason": "meaningful_routes_exhausted",
        }
    )
    for judgment in (
        trace["judgment"],
        trace["state"]["judgment"],
        investigation["discrepancy_judgment"],
    ):
        judgment.update(
            {
                "selected_discrepancy_ids": [],
                "selected_finding_ids": [],
                "selected_evidence_ids": [],
                "unresolved_gaps": [
                    "The available material does not close the claim."
                ],
            }
        )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert not report.failures(strict_scheduler=True)


def test_strict_audit_accepts_successful_v4_protocol_correction(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = trace["state"]["all_steps"]
    judgment_index = next(
        index
        for index, step in enumerate(steps)
        if step["stage"] == "image_only_discrepancy_judgment"
    )
    accepted = steps[judgment_index]
    rejected_interaction = "interaction-rejected-v4"
    rejected = {
        **accepted,
        "action_type": "output_rejected",
        "metadata": {
            **accepted["metadata"],
            "interaction_id": rejected_interaction,
            "previous_interaction_id": None,
            "interaction_lifecycle_kind": "standalone_request",
            "rejection_reason": "unresolved_gaps must match the compiled basis",
        },
    }
    accepted["metadata"].update(
        {
            "previous_interaction_id": rejected_interaction,
            "interaction_lifecycle_kind": "protocol_correction",
        }
    )
    steps.insert(judgment_index, rejected)
    for step in steps:
        metadata = step["metadata"]
        metadata.setdefault(
            "interaction_lifecycle_kind",
            (
                "tool_roundtrip"
                if step["stage"] == "image_only_discrepancy_investigation"
                else "standalone_request"
            ),
        )
        if metadata["interaction_lifecycle_kind"] != "protocol_correction":
            metadata["previous_interaction_id"] = None
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["successful_protocol_corrections"] == 1


def test_strict_audit_accepts_multihop_qwen_protocol_correction(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = trace["state"]["all_steps"]
    for index, step in enumerate(steps):
        metadata = step["metadata"]
        metadata["native_interactions"] = False
        metadata["native_chat_completions"] = True
        metadata["context_request_id"] = f"req-qwen-fixture-{index}"
        metadata["parent_context_request_id"] = None
        metadata["interaction_lifecycle_kind"] = (
            "tool_roundtrip"
            if step["stage"] == "image_only_discrepancy_investigation"
            else "standalone_request"
        )
    judgment_index = next(
        index
        for index, step in enumerate(steps)
        if step["stage"] == "image_only_discrepancy_judgment"
    )
    accepted = steps[judgment_index]
    rejected_root = {
        **accepted,
        "action_type": "output_rejected",
        "metadata": {
            **accepted["metadata"],
            "native_interactions": False,
            "native_chat_completions": True,
            "context_request_id": "req-qwen-root",
            "parent_context_request_id": None,
            "interaction_lifecycle_kind": "standalone_request",
            "rejection_reason": "assessment cites unknown Evidence",
        },
    }
    rejected_correction = {
        **accepted,
        "action_type": "output_rejected",
        "metadata": {
            **accepted["metadata"],
            "native_interactions": False,
            "native_chat_completions": True,
            "context_request_id": "req-qwen-correction-1",
            "parent_context_request_id": "req-qwen-root",
            "interaction_lifecycle_kind": "protocol_correction",
            "rejection_reason": "support Finding chain is missing",
        },
    }
    accepted["metadata"].update(
        {
            "native_interactions": False,
            "native_chat_completions": True,
            "context_request_id": "req-qwen-correction-2",
            "parent_context_request_id": "req-qwen-correction-1",
            "interaction_lifecycle_kind": "protocol_correction",
        }
    )
    steps[judgment_index:judgment_index] = [
        rejected_root,
        rejected_correction,
    ]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["successful_protocol_corrections"] == 2


def test_strict_audit_accepts_bounded_protocol_exhaustion_boundary(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = trace["state"]["all_steps"]
    for index, step in enumerate(steps):
        metadata = step["metadata"]
        metadata["native_interactions"] = False
        metadata["native_chat_completions"] = True
        metadata["context_request_id"] = f"req-bounded-fixture-{index}"
        metadata["parent_context_request_id"] = None
        metadata["interaction_lifecycle_kind"] = (
            "tool_roundtrip"
            if step["stage"] == "image_only_discrepancy_investigation"
            else "standalone_request"
        )
    judgment_index = next(
        index
        for index, step in enumerate(steps)
        if step["stage"] == "image_only_discrepancy_judgment"
    )
    source = next(
        step
        for step in steps
        if step["stage"] == "image_only_discrepancy_investigation"
    )
    rejected_root = {
        **source,
        "action_type": "format_error",
        "metadata": {
            **source["metadata"],
            "native_interactions": False,
            "native_chat_completions": True,
            "context_request_id": "req-route-root",
            "parent_context_request_id": None,
            "interaction_lifecycle_kind": "tool_roundtrip",
            "duplicate_tool_call": True,
            "rejection_reason": "the selected call duplicates an executed route",
        },
    }
    rejected_correction = {
        **source,
        "action_type": "format_error",
        "metadata": {
            **source["metadata"],
            "native_interactions": False,
            "native_chat_completions": True,
            "context_request_id": "req-route-correction",
            "parent_context_request_id": "req-route-root",
            "interaction_lifecycle_kind": "protocol_correction",
            "duplicate_tool_call": True,
            "correction_budget_exhausted": True,
            "rejection_reason": "the selected call still duplicates an executed route",
        },
    }
    boundary = {
        "round": 99,
        "stage": "image_only_discrepancy_investigation",
        "action_type": "output",
        "output": {
            "segment_summary": "Return to a discrepancy checkpoint.",
            "ready_for_reflection": True,
        },
        "metadata": {
            "stage": "image_only_discrepancy_investigation",
            "deterministic_segment_boundary": True,
            "protocol_correction_exhaustion_boundary": True,
            "resolved_rejection_request_ids": [
                "req-route-root",
                "req-route-correction",
            ],
        },
    }
    steps[judgment_index:judgment_index] = [
        rejected_root,
        rejected_correction,
        boundary,
    ]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)

    assert not report.failures(strict_scheduler=True)
    assert report.stats["bounded_protocol_fallbacks"] == 2
    assert report.stats["route_control_rejections"] == 0
    assert {
        issue.code
        for issue in report.warnings(strict_scheduler=True)
        if issue.code == "PROTOCOL_EXHAUSTION_BOUNDARY"
    } == {"PROTOCOL_EXHAUSTION_BOUNDARY"}


def test_strict_audit_rejects_v4_discrepancy_alignment_tampering(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    investigation["material_discrepancies"][0]["visual_anchor_fact_ids"] = [
        "fact-claim-v4"
    ]
    investigation["evidence"][0]["task_id"] = "unknown-task"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {issue.code for issue in report.failures(strict_scheduler=True)}

    assert "V4_DISCREPANCY_ANCHOR_MISALIGNED" in codes
    assert "V4_DISCREPANCY_EVIDENCE_OWNERSHIP_INVALID" in codes


def test_strict_audit_rejects_neutral_evidence_promoted_to_fake(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    evidence = trace["state"]["investigation_state"]["evidence"][0]
    evidence.update(
        {
            "stance": "neutral",
            "evidence_kind": "reference_comparison",
            "claim_binding": "same_subject",
            "same_capture_or_near_duplicate": False,
            "likely_different_original_capture": True,
            "edit_evidence_present": False,
        }
    )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {issue.code for issue in report.failures(strict_scheduler=True)}

    assert "V4_ASSESSMENT_EVIDENCE_DIRECTION_INVALID" in codes
    assert "V4_DISCREPANCY_EVIDENCE_DIRECTION_INVALID" in codes
    assert "V4_VERDICT_CHAIN_INVALID" in codes


def test_strict_audit_rejects_v4_verdict_without_finding_chain(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    investigation["findings"] = []
    for basis in (trace["verdict_basis"], investigation["discrepancy_verdict_basis"]):
        basis["finding_ids"] = []
    for judgment in (
        trace["judgment"],
        trace["state"]["judgment"],
        investigation["discrepancy_judgment"],
    ):
        judgment["selected_finding_ids"] = []
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {issue.code for issue in report.failures(strict_scheduler=True)}

    assert "V4_ASSESSMENT_EVIDENCE_DIRECTION_INVALID" in codes
    assert "V4_DISCREPANCY_EVIDENCE_DIRECTION_INVALID" in codes
    assert "V4_VERDICT_CHAIN_MISSING" in codes


def test_strict_audit_keeps_v4_discovery_separate_from_evidence(
    tmp_path: Path,
) -> None:
    trace_path = _v4_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    investigation["discoveries"] = [
        {
            "discovery_id": "evidence-v4",
            "task_id": "task-v4",
            "promoted_evidence_id": "evidence-v4",
        }
    ]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {issue.code for issue in report.failures(strict_scheduler=True)}

    assert "V4_DISCOVERY_PROMOTED_IN_PLACE" in codes
    assert "V4_DISCOVERY_USED_AS_VERDICT_EVIDENCE" in codes


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


def test_empty_text_search_query_is_a_format_warning_not_protocol_rejection(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["state"]["all_steps"].insert(
        2,
        {
            "round": 1,
            "stage": "image_only_investigation",
            "action_type": "format_error",
            "tool_name": "text_search",
            "tool_args": {"queries": []},
            "tool_result": json.dumps(
                {
                    "status": "error",
                    "error": (
                        "Search policy removed every query because it targeted a "
                        "ready-made fact-check verdict or an excluded source."
                    ),
                }
            ),
            "metadata": {
                "stage": "image_only_investigation",
                "search_policy_rejection": True,
                "function_call_id": "call-empty-query",
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
    assert report.stats["tool_argument_format_errors"] == 1
    assert "TOOL_ARGUMENT_FORMAT_ERROR" in {
        item.code for item in report.warnings(strict_scheduler=True)
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


def test_policy_rejected_query_is_warning_but_canonical_query_is_hard(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["state"]["all_steps"].insert(
        2,
        {
            "round": 1,
            "stage": "image_account_planning",
            "action_type": "planning_revision",
            "tool_name": "",
            "tool_args": {},
            "tool_result": "",
            "output": {
                "search_hypotheses": [
                    {"queries": ["viral image hoax visible person product"]}
                ]
            },
            "metadata": {
                "stage": "image_account_planning",
                "planning_revision_reason": "query policy rejected the proposal",
                "policy_action": {
                    "search_hypotheses": [
                        {"queries": ["viral image hoax visible person product"]}
                    ]
                },
            },
        },
    )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    corrected = audit_trace(trace_path)

    assert not corrected.failures(strict_scheduler=True)
    assert corrected.stats["fact_check_query_leaks"] == 0
    assert corrected.stats["fact_check_query_rejections"] == 2
    assert {
        item.code for item in corrected.warnings(strict_scheduler=True)
    } >= {"FACT_CHECK_QUERY_REJECTED"}

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    investigation["tasks"][0]["suggested_queries"] = [
        "viral image hoax visible person product"
    ]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    leaked = audit_trace(trace_path)

    assert "FACT_CHECK_QUERY_LEAK" in {
        item.code for item in leaked.failures(strict_scheduler=True)
    }


def test_fact_check_source_audit_requires_active_source_policy(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["state"]["investigation_state"]["discoveries"].append(
        {
            "discovery_id": "discovery-fact-check-source",
            "task_id": trace["state"]["investigation_state"]["tasks"][0][
                "task_id"
            ],
            "candidate_url": "https://www.politifact.com/factchecks/example/",
        }
    )
    trace["state"]["investigation_state"]["tasks"][0][
        "suggested_queries"
    ] = ["Sandra May Watkins fake image Snopes"]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    active = audit_trace(trace_path, enforce_source_access_policy=True)
    inactive = audit_trace(trace_path, enforce_source_access_policy=False)

    assert "FACT_CHECK_URL_LEAK" in {
        item.code for item in active.failures(strict_scheduler=True)
    }
    assert "FACT_CHECK_URL_LEAK" not in {
        item.code for item in inactive.failures(strict_scheduler=True)
    }
    assert "FACT_CHECK_QUERY_LEAK" in {
        item.code for item in active.failures(strict_scheduler=True)
    }
    assert "FACT_CHECK_QUERY_LEAK" not in {
        item.code for item in inactive.failures(strict_scheduler=True)
    }
    assert inactive.stats["fact_check_url_leaks"] == 0
    assert inactive.stats["fact_check_query_leaks"] == 0


def test_strict_audit_rejects_post_determination_action(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    terminal_action = investigation["coverage_audits"][-1]["action_count"]
    investigation["action_count"] = terminal_action + 1
    trace["state"]["all_steps"].append(
        {
            "round": 99,
            "stage": "image_only_investigation",
            "action_type": "tool_call",
            "tool_name": "text_search",
            "tool_args": {
                "__question_id": investigation["tasks"][0]["task_id"],
                "query": "redundant query after verdict",
            },
            "tool_result": json.dumps(
                {"status": "success", "results": []}
            ),
            "metadata": {
                "stage": "image_only_investigation",
                "function_call_id": "call-after-verdict",
            },
        }
    )
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=True)
    }

    assert "POST_DETERMINATION_ACTION" in codes


def test_strict_audit_rejects_unanchored_refinement(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    decision = investigation["evidence_decisions"][0]
    decision["accepted_refinement_fact_id"] = investigation["facts"][0]["fact_id"]
    decision["output"]["refinement"] = {
        "slot": "subject_identity",
        "statement": "The visible subject is Example.",
        "predicate": "identified_as",
        "anchor_fact_ids": ["missing-anchor"],
        "grounding_evidence_ids": decision["reviewed_evidence_ids"][:1],
        "question": "Is the visible subject Example?",
        "purpose": "Narrow the visible identity slot.",
        "suggested_tools": ["text_search"],
        "suggested_queries": ["Example"],
    }
    investigation["core_fact_refinement_count"] = 1
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=True)
    }

    assert "EVIDENCE_REFINEMENT_ANCHOR_INVALID" in codes


def test_query_replan_does_not_replace_interval_reflection_boundary(
    tmp_path: Path,
) -> None:
    trace_path = _scripted_trace(tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    investigation = trace["state"]["investigation_state"]
    template_step = next(
        step
        for step in trace["state"]["all_steps"]
        if step.get("stage") == "image_only_investigation"
        and step.get("action_type") == "tool_call"
    )
    for index in range(5):
        step = json.loads(json.dumps(template_step))
        step["round"] = 100 + index
        step["metadata"]["function_call_id"] = f"call-cadence-{index}"
        trace["state"]["all_steps"].append(step)
    investigation["action_count"] = 8
    investigation["stop_reason"] = "information_saturated"
    investigation["reflections"] = [
        {
            "reflection_id": "reflection-interval-4",
            "action_count": 4,
            "trigger": "interval",
            "output": {
                "task_updates": [],
                "new_tasks": [],
                "recommended_next_task_ids": [],
                "remaining_gaps": [],
                "ready_to_finish": False,
            },
            "accepted_task_update_ids": [],
            "accepted_new_task_ids": [],
            "rejected_reasons": [],
        },
        {
            "reflection_id": "reflection-interval-8",
            "action_count": 8,
            "trigger": "interval",
            "output": {
                "task_updates": [],
                "new_tasks": [],
                "recommended_next_task_ids": [],
                "remaining_gaps": [],
                "ready_to_finish": False,
            },
            "accepted_task_update_ids": [],
            "accepted_new_task_ids": [],
            "rejected_reasons": [],
        },
    ]
    task_id = investigation["tasks"][0]["task_id"]
    evidence_id = investigation["evidence"][0]["evidence_id"]
    evidence_text = investigation["evidence"][0]["exact_text"]
    evidence_phrase = evidence_text.split()[0]
    investigation["query_replans"] = [
        {
            "replan_id": "query-replan-between-boundaries",
            "action_count": 6,
            "trigger": "evidence_boundary",
            "new_evidence_ids": [evidence_id],
            "concept_extraction": {
                "task_id": task_id,
                "concepts": [
                    {
                        "concept_id": "concept-between-boundaries",
                        "evidence_id": evidence_id,
                        "evidence_phrase": evidence_phrase,
                        "role": "other",
                    }
                ],
            },
            "output": {
                "task_id": task_id,
                "selected_concept_id": "concept-between-boundaries",
                "concept_term": evidence_phrase,
                "replacement_query": f"example {evidence_phrase}",
                "rationale": "New Evidence changed the search direction.",
            },
            "accepted_queries": [f"example {evidence_phrase}"],
            "rejected_reason": "",
        }
    ]
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = audit_trace(trace_path)
    codes = {
        issue.code
        for issue in report.failures(strict_scheduler=True)
    }

    assert "IMAGE_ONLY_REFLECTION_CADENCE_INVALID" not in codes
