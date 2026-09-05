"""Read-only projections used to audit Agent fact-check reports.

The rollout never receives private construction gold.  This module only projects
an already-persisted trace into the evidence-bounded packet seen by a post-hoc
private-gold judge.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.orchestrator.react_runtime import REACT_RUNTIME_SCHEMA_VERSION
from src.orchestrator.tool_result import parse_tool_result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _text(value: Any, *, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _ids(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _successful_evidence(row: Mapping[str, Any]) -> bool:
    if "successful_call" in row:
        return bool(row.get("successful_call"))
    if "tool_success" in row:
        return bool(row.get("tool_success"))
    status = _text(row.get("status") or row.get("tool_status"), limit=100).lower()
    return status not in {"error", "failed", "failure"}


def _evidence_projection(
    row: Mapping[str, Any],
    *,
    selected_evidence_ids: set[str],
) -> dict[str, Any]:
    evidence_id = _text(row.get("evidence_id"), limit=100)
    return {
        "evidence_id": evidence_id,
        "selected_by_verdict_basis": evidence_id in selected_evidence_ids,
        "successful_call": _successful_evidence(row),
        "tool_name": _text(row.get("tool_name"), limit=100),
        "evidence_kind": _text(row.get("evidence_kind"), limit=100),
        "source_url": _text(
            row.get("source_url")
            or row.get("selected_url")
            or row.get("candidate_url"),
            limit=4000,
        ),
        "source_family": _text(row.get("source_family"), limit=300),
        "source_class": _text(row.get("source_class"), limit=100),
        "exact_text": _text(
            row.get("exact_text")
            or row.get("excerpt")
            or row.get("evidence")
            or row.get("summary")
            or row.get("details")
            or row.get("description"),
            limit=2400,
        ),
        "stance": _text(row.get("stance"), limit=100),
        "quality": _text(row.get("quality"), limit=100),
        "directness": _text(row.get("directness"), limit=100),
        "claim_binding": _text(row.get("claim_binding"), limit=100),
        "relation_scope": _text(row.get("relation_scope"), limit=100),
        "relation_stance": _text(row.get("relation_stance"), limit=100),
        "visual_answer_status": _text(row.get("visual_answer_status"), limit=100),
        "evidence_class": _text(row.get("evidence_class"), limit=100),
        "match_status": _text(row.get("match_status"), limit=100),
    }


def _compact_raw_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[nested content omitted]"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in list(value.items())[:80]:
            key = str(raw_key)
            if key.casefold() in {
                "image_input",
                "image",
                "image_url",
                "data_url",
                "base64",
                "content_bytes",
                "raw_html",
                "html",
            }:
                continue
            result[key] = _compact_raw_value(child, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_compact_raw_value(item, depth=depth + 1) for item in value[:40]]
    if isinstance(value, str):
        return value.strip()[:6000]
    return value


def _raw_action_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if (
            _text(step.get("stage"), limit=80) != "unified_react"
            or _text(step.get("action_type"), limit=80) != "tool_call"
            or _text(step.get("tool_name"), limit=120) == "finish_investigation"
        ):
            continue
        metadata = _mapping(step.get("metadata"))
        call_id = _text(metadata.get("function_call_id"), limit=160)
        raw_result = str(step.get("tool_result", ""))
        try:
            payload, succeeded = parse_tool_result(raw_result)
            observation = _compact_raw_value(payload)
            status = "success" if succeeded else "error"
        except Exception:
            observation = raw_result[:6000]
            status = "malformed"
            succeeded = False
        arguments = {
            str(key): _compact_raw_value(value)
            for key, value in _mapping(step.get("tool_args")).items()
            if str(key).casefold()
            not in {"image_input", "image", "image_url", "data_url", "base64"}
        }
        history.append(
            {
                "step_index": index,
                "turn": len(history) + 1,
                "observation_id": call_id,
                "tool": _text(step.get("tool_name"), limit=120),
                "thought": _text(step.get("thought"), limit=4000),
                "arguments": arguments,
                "observation": observation,
                "status": status,
                "tool_success": bool(succeeded),
            }
        )
        if len(history) >= 80:
            break
    return history


def build_agent_private_gold_candidate(
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a bounded, evidence-faithful terminal packet from one trace."""

    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    is_unified_react_runtime = (
        str(investigation.get("schema_version", "")).strip()
        == REACT_RUNTIME_SCHEMA_VERSION
    )
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    selected_evidence_ids = set(_ids(basis.get("evidence_ids"), limit=40))
    selected_finding_ids = set(_ids(basis.get("finding_ids"), limit=20))
    selected_claim_ids = set(_ids(basis.get("claim_ids"), limit=12))
    evidence = [
        _evidence_projection(item, selected_evidence_ids=selected_evidence_ids)
        for item in _rows(investigation.get("evidence"))
        if _text(item.get("evidence_id"), limit=100)
        and _successful_evidence(item)
    ]
    selected_evidence = [
        item for item in evidence if item["selected_by_verdict_basis"]
    ]
    findings = [
        {
            "finding_id": _text(item.get("finding_id"), limit=100),
            "selected_by_verdict_basis": _text(
                item.get("finding_id"), limit=100
            )
            in selected_finding_ids,
            "fact_ids": _ids(item.get("fact_ids"), limit=12),
            "evidence_ids": _ids(item.get("evidence_ids"), limit=20),
            "stance": _text(item.get("stance"), limit=100),
            "summary": _text(item.get("summary"), limit=2400),
        }
        for item in _rows(investigation.get("findings"))
        if _text(item.get("finding_id"), limit=100)
    ]
    target_facts = [
        {
            "claim_id": _text(item.get("claim_id"), limit=100),
            "selected_by_verdict_basis": _text(
                item.get("claim_id"), limit=100
            )
            in selected_claim_ids,
            "statement": _text(item.get("statement"), limit=2400),
            "status": _text(item.get("status"), limit=100),
            "anchor_fact_ids": _ids(item.get("anchor_fact_ids"), limit=12),
        }
        for item in _rows(investigation.get("target_facts"))
        if _text(item.get("claim_id"), limit=100)
    ]
    citations = [
        {
            "evidence_id": _text(item.get("evidence_id"), limit=100),
            "source_url": _text(item.get("source_url"), limit=4000),
            "source_family": _text(item.get("source_family"), limit=300),
            "evidence_kind": _text(item.get("evidence_kind"), limit=100),
            "relation_stance": _text(item.get("relation_stance"), limit=100),
            "excerpt": _text(item.get("excerpt"), limit=2400),
        }
        for item in _rows(judgment.get("evidence_citations"))
        if _text(item.get("evidence_id"), limit=100)
    ]
    result = {
        "candidate_kind": "agent_trace",
        "runtime_mode": (
            "image_grounded_react" if is_unified_react_runtime else "legacy_graph"
        ),
        "recorded_verdict": _text(
            judgment.get("verdict") or trace.get("verdict"),
            limit=100,
        ),
        "overall_assessment": _text(
            judgment.get("overall_assessment")
            or trace.get("overall_assessment"),
            limit=2400,
        ),
        "fact_check_report": _mapping(
            judgment.get("fact_check_report") or trace.get("fact_check_report")
        ),
        "termination": _text(trace.get("termination"), limit=100),
        "verdict_basis": {
            "decision_mode": _text(basis.get("decision_mode"), limit=100),
            "verdict_target": _text(basis.get("verdict_target"), limit=2400),
            "claim_ids": _ids(basis.get("claim_ids"), limit=12),
            "discrepancy_ids": _ids(basis.get("discrepancy_ids"), limit=12),
            "finding_ids": _ids(basis.get("finding_ids"), limit=20),
            "evidence_ids": _ids(basis.get("evidence_ids"), limit=40),
            "unresolved_gaps": _ids(
                basis.get("unresolved_gaps")
                or basis.get("open_questions"),
                limit=12,
            ),
        },
        "runtime_evidence_citations": citations,
        "selected_evidence": selected_evidence,
        "successful_evidence": evidence[:60],
        "selected_findings": [
            item for item in findings if item["selected_by_verdict_basis"]
        ],
        "selected_target_facts": [
            item for item in target_facts if item["selected_by_verdict_basis"]
        ],
        "rejected_policy_outputs": _rows(state.get("rejection_history"))[:16],
    }
    if is_unified_react_runtime:
        result["runtime_objective"] = _text(
            investigation.get("objective"),
            limit=1200,
        )
        raw_history = _raw_action_history(state)
        result["react_action_history"] = raw_history
        result["raw_observation_ids"] = [
            item["observation_id"]
            for item in raw_history
            if item["observation_id"]
        ]
        result["successful_observation_ids"] = [
            item["observation_id"]
            for item in raw_history
            if item["tool_success"] and item["observation_id"]
        ]
        result["unsuccessful_observation_ids"] = [
            item["observation_id"]
            for item in raw_history
            if not item["tool_success"] and item["observation_id"]
        ]
        result["action_count"] = int(investigation.get("action_count", 0) or 0)
        result["stop_reason"] = _text(
            investigation.get("stop_reason"),
            limit=100,
        )
        result["finish_rationale"] = _text(
            investigation.get("finish_rationale"),
            limit=1200,
        )
    return result


def agent_candidate_answer(packet: Mapping[str, Any]) -> dict[str, str]:
    """Project the Agent terminal packet into the common candidate-answer shape."""

    basis = _mapping(packet.get("verdict_basis"))
    report = _mapping(packet.get("fact_check_report"))
    target_facts = _rows(packet.get("selected_target_facts"))
    findings = _rows(packet.get("selected_findings"))
    report_findings = report.get("key_findings")
    if not isinstance(report_findings, list):
        report_findings = []
    core_fact = _text(report.get("claim_under_review"), limit=2400)
    if not core_fact:
        core_fact = _text(packet.get("overall_assessment"), limit=2400)
    if not core_fact:
        core_fact = _text(basis.get("verdict_target"), limit=2400)
    if not core_fact and target_facts:
        core_fact = _text(target_facts[0].get("statement"), limit=2400)
    reason_parts = [
        _text(report.get("verdict_summary"), limit=2400),
        _text(report.get("evidence_summary"), limit=2400),
        *[
            _text(item, limit=1200)
            for item in report_findings
            if isinstance(item, str)
        ],
        *[
            _text(item.get("summary"), limit=2400)
            for item in findings
        ],
        _text(packet.get("overall_assessment"), limit=2400),
    ]
    reason = "\n\n".join(dict.fromkeys(part for part in reason_parts if part))
    return {
        "core_fact": core_fact,
        "verdict": _text(packet.get("recorded_verdict"), limit=100),
        "reason": reason,
    }
