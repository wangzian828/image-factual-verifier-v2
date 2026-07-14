"""Deterministic process metrics and componentized teacher scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from src.orchestrator.tool_result import parse_tool_result


FACT_MATCH_THRESHOLD = 0.35
LABEL_TO_VERDICT = {
    "supported": "real",
    "refuted": "fake",
    "unverifiable": "unverifiable",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in {"the", "a", "an", "is", "are", "at", "in", "of", "to"}
    }


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def _fact_similarity(
    runtime_fact: Mapping[str, Any],
    gold_fact: Mapping[str, Any],
) -> float:
    score = _token_f1(
        runtime_fact.get("statement"),
        gold_fact.get("statement"),
    )
    if str(runtime_fact.get("kind", "")) == str(gold_fact.get("kind", "")):
        score = min(1.0, score + 0.1)
    return score


def _match_gold_facts(
    runtime_facts: Sequence[Mapping[str, Any]],
    gold_facts: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    available = {
        str(item.get("fact_id", "")).strip(): item
        for item in runtime_facts
        if str(item.get("fact_id", "")).strip()
    }
    matches: List[Dict[str, Any]] = []
    for gold in gold_facts:
        ranked = sorted(
            (
                (_fact_similarity(runtime, gold), fact_id, runtime)
                for fact_id, runtime in available.items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] < FACT_MATCH_THRESHOLD:
            matches.append(
                {
                    "gold_fact_id": str(gold.get("fact_id", "")),
                    "runtime_fact_id": None,
                    "similarity": round(ranked[0][0], 6) if ranked else 0.0,
                    "expected_status": str(gold.get("expected_status", "")),
                    "runtime_status": None,
                    "status_match": False,
                }
            )
            continue
        similarity, fact_id, runtime = ranked[0]
        available.pop(fact_id, None)
        expected_status = str(gold.get("expected_status", ""))
        runtime_status = str(runtime.get("status", ""))
        matches.append(
            {
                "gold_fact_id": str(gold.get("fact_id", "")),
                "runtime_fact_id": fact_id,
                "similarity": round(similarity, 6),
                "expected_status": expected_status,
                "runtime_status": runtime_status,
                "status_match": runtime_status == expected_status,
            }
        )
    return matches


def _canonical_url(value: Any) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        return ""
    try:
        parsed = urlsplit(rendered)
    except ValueError:
        return rendered.casefold().rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            parsed.query,
            "",
        )
    )


def _family_key(value: Any) -> str:
    rendered = str(value or "").casefold().strip()
    rendered = rendered.removeprefix("domain:")
    return rendered.split(".", 1)[0]


def _evidence_matches_reference(
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    evidence_text = " ".join(str(evidence.get("exact_text", "")).split())
    reference_text = " ".join(str(reference.get("exact_span", "")).split())
    text_match = (
        evidence_text == reference_text
        or reference_text in evidence_text
        or evidence_text in reference_text
    ) if evidence_text and reference_text else False
    return all(
        (
            _canonical_url(evidence.get("source_url"))
            == _canonical_url(reference.get("canonical_url")),
            text_match,
            str(evidence.get("stance", "")) == str(reference.get("stance", "")),
            str(evidence.get("artifact_sha256", "")).casefold()
            == str(reference.get("artifact_sha256", "")).casefold(),
            _family_key(evidence.get("source_family"))
            == _family_key(reference.get("source_family")),
        )
    )


def _successful_call_ids(
    steps: Sequence[Mapping[str, Any]],
) -> set[str]:
    result: set[str] = set()
    for step in steps:
        if str(step.get("action_type", "")) != "tool_call":
            continue
        call_id = str(
            _mapping(step.get("metadata")).get("function_call_id", "")
        ).strip()
        if not call_id:
            continue
        try:
            _, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        except Exception:
            succeeded = False
        if succeeded:
            result.add(call_id)
    return result


def _valid_findings(
    findings: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    successful_calls: set[str],
) -> set[str]:
    valid: set[str] = set()
    for finding in findings:
        finding_id = str(finding.get("finding_id", "")).strip()
        task_id = str(finding.get("task_id", "")).strip()
        task = tasks.get(task_id)
        if not finding_id or task is None:
            continue
        fact_ids = {str(item) for item in finding.get("fact_ids", []) or []}
        if not fact_ids or not fact_ids <= {
            str(item) for item in task.get("fact_ids", []) or []
        }:
            continue
        owned_evidence = [
            evidence.get(str(evidence_id))
            for evidence_id in finding.get("evidence_ids", []) or []
        ]
        if not owned_evidence or any(item is None for item in owned_evidence):
            continue
        if any(
            str(item.get("task_id", "")).strip() != task_id
            or str(item.get("function_call_id", "")).strip()
            not in successful_calls
            for item in owned_evidence
            if item is not None
        ):
            continue
        valid.add(finding_id)
    return valid


def _route_key(step: Mapping[str, Any]) -> str:
    args = {
        key: value
        for key, value in sorted(_mapping(step.get("tool_args")).items())
        if key not in {"image_input", "__claim_text", "__evidence_goal"}
    }
    return json.dumps(
        {
            "tool": str(step.get("tool_name", "")),
            "args": args,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _duplicate_action_rate(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[float, int]:
    routes: List[str] = []
    for step in steps:
        if (
            str(step.get("stage", "")) == "image_only_investigation"
            and str(step.get("action_type", "")) == "tool_call"
        ):
            routes.append(_route_key(step))
    duplicate_count = len(routes) - len(set(routes))
    return (
        duplicate_count / len(routes) if routes else 0.0,
        duplicate_count,
    )


def _first_error(
    trace: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    for index, step in enumerate(steps):
        if str(step.get("action_type", "")) in {
            "format_error",
            "output_rejected",
        }:
            return {
                "step_index": index,
                "stage": step.get("stage"),
                "tool_name": step.get("tool_name"),
                "error": str(
                    _mapping(step.get("metadata")).get(
                        "rejection_reason",
                        step.get("tool_result", ""),
                    )
                )[:1000],
            }
        if str(step.get("action_type", "")) == "tool_call":
            try:
                payload, succeeded = parse_tool_result(
                    str(step.get("tool_result", ""))
                )
            except Exception as exc:
                return {
                    "step_index": index,
                    "stage": step.get("stage"),
                    "tool_name": step.get("tool_name"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if not succeeded:
                return {
                    "step_index": index,
                    "stage": step.get("stage"),
                    "tool_name": step.get("tool_name"),
                    "error": str(payload.get("error", ""))[:1000],
                }
    if str(trace.get("termination", "")) == "error":
        return {
            "step_index": None,
            "stage": None,
            "tool_name": None,
            "error": str(trace.get("error", ""))[:1000],
        }
    return None


def score_process_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Score one post-rollout trace against evaluator-private references."""

    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    facts = _rows(investigation.get("facts"))
    decisive_ids = {
        str(item)
        for item in investigation.get("decisive_fact_ids", []) or []
    }
    decisive_facts = [
        fact for fact in facts if str(fact.get("fact_id", "")) in decisive_ids
    ]
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
    findings = _rows(investigation.get("findings"))
    findings_by_id = {
        str(item.get("finding_id", "")): item
        for item in findings
        if str(item.get("finding_id", ""))
    }
    steps = _rows(state.get("all_steps"))
    successful_calls = _successful_call_ids(steps)
    valid_finding_ids = _valid_findings(
        findings,
        tasks,
        evidence,
        successful_calls,
    )
    gold_facts = _rows(gold.get("decisive_facts"))
    fact_matches = _match_gold_facts(decisive_facts, gold_facts)
    matched_count = sum(
        item["runtime_fact_id"] is not None for item in fact_matches
    )
    status_match_count = sum(bool(item["status_match"]) for item in fact_matches)
    discovery_rate = matched_count / len(gold_facts) if gold_facts else 1.0
    status_accuracy = (
        status_match_count / len(gold_facts) if gold_facts else 1.0
    )
    decisive_fact_alignment = (
        discovery_rate + status_accuracy
    ) / 2

    matched_runtime_by_gold = {
        str(item["gold_fact_id"]): str(item["runtime_fact_id"])
        for item in fact_matches
        if item["runtime_fact_id"] is not None
    }
    acceptable_hits = 0
    acceptable_total = 0
    acceptable_evidence_ids: set[str] = set()
    bridge_hits = 0
    runtime_fact_by_id = {
        str(item.get("fact_id", "")): item for item in decisive_facts
    }
    for gold_fact in gold_facts:
        gold_fact_id = str(gold_fact.get("fact_id", ""))
        runtime_fact_id = matched_runtime_by_gold.get(gold_fact_id)
        references = _rows(gold_fact.get("acceptable_evidence"))
        acceptable_total += 1
        if runtime_fact_id is None:
            continue
        related_findings = [
            finding
            for finding in findings
            if runtime_fact_id
            in {str(item) for item in finding.get("fact_ids", []) or []}
            and str(finding.get("finding_id", "")) in valid_finding_ids
        ]
        related_evidence = [
            evidence[str(evidence_id)]
            for finding in related_findings
            for evidence_id in finding.get("evidence_ids", []) or []
            if str(evidence_id) in evidence
        ]
        hit_ids = {
            str(item.get("evidence_id", ""))
            for item in related_evidence
            if any(
                _evidence_matches_reference(item, reference)
                for reference in references
            )
        }
        if hit_ids:
            acceptable_hits += 1
            acceptable_evidence_ids.update(hit_ids)
        runtime_fact = runtime_fact_by_id.get(runtime_fact_id, {})
        origin_type = str(_mapping(runtime_fact.get("origin")).get("type", ""))
        if (
            gold_fact.get("visual_anchor")
            and origin_type in {"input_image", "ocr"}
            and related_findings
            and related_evidence
        ):
            bridge_hits += 1

    acceptable_evidence_hit_rate = (
        acceptable_hits / acceptable_total if acceptable_total else 1.0
    )
    bridge_completion = (
        bridge_hits / len(gold_facts) if gold_facts else 1.0
    )

    basis = _mapping(
        trace.get("verdict_basis") or investigation.get("verdict_basis")
    )
    basis_fact_ids = {
        str(item) for item in basis.get("fact_ids", []) or []
    }
    basis_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    expected_basis_ids = set(matched_runtime_by_gold.values())
    if str(gold.get("factual_status", "")) == "refuted":
        expected_basis_ids = {
            str(item["runtime_fact_id"])
            for item in fact_matches
            if item["runtime_fact_id"] is not None
            and item["expected_status"] == "refuted"
        }
    basis_overlap = len(basis_fact_ids & expected_basis_ids)
    basis_precision = (
        basis_overlap / len(basis_fact_ids) if basis_fact_ids else 0.0
    )
    basis_recall = (
        basis_overlap / len(expected_basis_ids)
        if expected_basis_ids
        else 1.0
    )
    verdict_basis_alignment = (
        2 * basis_precision * basis_recall / (basis_precision + basis_recall)
        if basis_precision + basis_recall
        else 0.0
    )
    citation_precision = (
        len(basis_evidence_ids & acceptable_evidence_ids)
        / len(basis_evidence_ids)
        if basis_evidence_ids
        else 0.0
    )

    valid_finding_precision = (
        len(valid_finding_ids) / len(findings) if findings else 1.0
    )
    invalid_tasks = sum(
        not {
            str(item) for item in task.get("fact_ids", []) or []
        } <= {
            str(item.get("fact_id", "")) for item in facts
        }
        for task in tasks.values()
    )
    false_task_rate = invalid_tasks / len(tasks) if tasks else 0.0
    invalid_activations = sum(
        fact_id not in runtime_fact_by_id
        or not any(
            fact_id in {
                str(item) for item in task.get("fact_ids", []) or []
            }
            and bool(task.get("suggested_tools"))
            for task in tasks.values()
        )
        for fact_id in decisive_ids
    )
    false_activation_rate = (
        invalid_activations / len(decisive_ids) if decisive_ids else 1.0
    )
    duplicate_action_rate, duplicate_action_count = _duplicate_action_rate(steps)
    unresolved_decisive = [
        fact_id
        for fact_id in decisive_ids
        if str(runtime_fact_by_id.get(fact_id, {}).get("status", ""))
        not in {"supported", "refuted"}
    ]
    premature_finish = bool(
        str(trace.get("termination", "")) == "success"
        and (
            str(trace.get("verdict", "")) in {"real", "fake"}
            and unresolved_decisive
        )
    )
    action_count = int(investigation.get("action_count", 0) or 0)
    decisive_evidence_per_tool_action = (
        len(basis_evidence_ids) / action_count if action_count else 0.0
    )
    engineering_error = (
        str(trace.get("termination", "")) == "error"
        or str(trace.get("verdict", "")) == "error"
    )
    expected_verdict = LABEL_TO_VERDICT.get(
        str(gold.get("factual_status", "")),
        "unverifiable",
    )
    result_correct = (
        not engineering_error
        and str(trace.get("verdict", "")) == expected_verdict
    )
    first_error = _first_error(trace, steps)

    process_metrics = {
        "schema_version": "ifv-process-metrics-v1",
        "case_id": str(gold.get("case_id") or trace.get("image_id") or ""),
        "engineering_error": engineering_error,
        "expected_verdict": expected_verdict,
        "runtime_verdict": trace.get("verdict"),
        "result_correct": result_correct,
        "fact_match_threshold": FACT_MATCH_THRESHOLD,
        "fact_matches": fact_matches,
        "decisive_fact_discovery_rate": round(discovery_rate, 6),
        "decisive_fact_status_accuracy": round(status_accuracy, 6),
        "decisive_fact_alignment": round(decisive_fact_alignment, 6),
        "acceptable_evidence_hit_rate": round(
            acceptable_evidence_hit_rate, 6
        ),
        "citation_precision": round(citation_precision, 6),
        "evidence_to_vision_bridge_completion": round(
            bridge_completion, 6
        ),
        "verdict_basis_alignment": round(verdict_basis_alignment, 6),
        "valid_finding_precision": round(valid_finding_precision, 6),
        "false_task_rate": round(false_task_rate, 6),
        "false_activation_rate": round(false_activation_rate, 6),
        "duplicate_action_rate": round(duplicate_action_rate, 6),
        "duplicate_action_count": duplicate_action_count,
        "premature_finish": premature_finish,
        "decisive_evidence_per_tool_action": round(
            decisive_evidence_per_tool_action, 6
        ),
        "reinspect_resolution_rate": round(
            (
                (len(decisive_ids) - len(unresolved_decisive))
                / len(decisive_ids)
            )
            if decisive_ids
            else 0.0,
            6,
        ),
        "tool_actions": action_count,
        "llm_api_calls": int(trace.get("llm_api_calls", 0) or 0),
        "token_usage": dict(_mapping(trace.get("token_usage"))),
        "latency_seconds": float(trace.get("time_taken", 0.0) or 0.0),
        "first_error": first_error,
    }

    normalized_cost = min(1.0, action_count / 24.0)
    components = {
        "result_reward": 1.0 if result_correct else 0.0,
        "grounded_finding_reward": round(valid_finding_precision, 6),
        "gap_coverage_reward": round(decisive_fact_alignment, 6),
        "bridge_reward": round(bridge_completion, 6),
        "stop_calibration_reward": (
            1.0
            if not engineering_error and not premature_finish
            else 0.0
        ),
        "duplicate_action_penalty": round(-duplicate_action_rate, 6),
        "invalid_task_penalty": round(
            -max(false_task_rate, false_activation_rate),
            6,
        ),
        "normalized_cost_penalty": round(-0.25 * normalized_cost, 6),
    }
    teacher_score = {
        "schema_version": "ifv-trajectory-score-v1",
        "case_id": process_metrics["case_id"],
        "components": components,
        "total": round(sum(components.values()), 6),
    }
    return process_metrics, teacher_score
