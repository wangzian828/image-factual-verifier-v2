"""Deterministic process metrics and componentized teacher scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from src.orchestrator.evidence_semantics import (
    evidence_is_qualified_for_stance,
    required_assessment_stances,
    same_capture_can_support_visual_claim,
)
from src.orchestrator.route_policy import semantic_duplicate_count
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
    result: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if token in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "at",
            "in",
            "of",
            "to",
            "ve",
            "vf",
            "rf",
        }:
            continue
        if re.fullmatch(r"[0-9a-f]{12,}", token):
            continue
        normalized = _light_fact_stem(token)
        if normalized:
            result.add(normalized)
    return result


def _light_fact_stem(token: str) -> str:
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stem = token[: -len(suffix)]
            return stem + "y" if suffix == "ies" else stem
    return token


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    unmatched_right = set(right_tokens)
    overlap = 0
    for left_token in sorted(left_tokens):
        matched = next(
            (
                right_token
                for right_token in sorted(unmatched_right)
                if _fact_tokens_equivalent(left_token, right_token)
            ),
            None,
        )
        if matched is not None:
            overlap += 1
            unmatched_right.remove(matched)
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def _fact_tokens_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return (
        len(shorter) >= 6
        and len(longer) - len(shorter) <= 2
        and longer.startswith(shorter)
    )


def _fact_similarity(
    runtime_fact: Mapping[str, Any],
    gold_fact: Mapping[str, Any],
    *,
    recovered_source_context: str = "",
) -> float:
    statement = runtime_fact.get("statement")
    score = max(
        _token_f1(statement, gold_fact.get("statement")),
        _token_f1(
            " ".join(
                item
                for item in (
                    str(statement or "").strip(),
                    str(recovered_source_context or "").strip(),
                )
                if item
            ),
            gold_fact.get("statement"),
        ),
    )
    if str(runtime_fact.get("kind", "")) == str(gold_fact.get("kind", "")):
        score = min(1.0, score + 0.1)
    return score


def _match_gold_facts(
    runtime_facts: Sequence[Mapping[str, Any]],
    gold_facts: Sequence[Mapping[str, Any]],
    *,
    recovered_source_context_by_fact: Mapping[str, str] | None = None,
) -> List[Dict[str, Any]]:
    context_by_fact = recovered_source_context_by_fact or {}
    available = {
        str(item.get("fact_id", "")).strip(): item
        for item in runtime_facts
        if str(item.get("fact_id", "")).strip()
    }
    matches: List[Dict[str, Any]] = []
    for gold in gold_facts:
        ranked = sorted(
            (
                (
                    _fact_similarity(
                        runtime,
                        gold,
                        recovered_source_context=context_by_fact.get(
                            fact_id,
                            "",
                        ),
                    ),
                    fact_id,
                    runtime,
                )
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


def _basis_same_capture_source_context(
    investigation: Mapping[str, Any],
    basis: Mapping[str, Any],
) -> Dict[str, str]:
    """Recover source-page context only from selected same-capture Evidence."""

    basis_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    evidence = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    discoveries = _rows(investigation.get("discoveries"))
    context_by_fact: Dict[str, List[str]] = {}
    for evidence_id in basis_evidence_ids:
        item = evidence.get(evidence_id)
        if item is None or not (
            str(item.get("claim_binding", "")) == "same_capture"
            and item.get("same_capture_or_near_duplicate") is True
            and item.get("likely_different_original_capture") is not True
        ):
            continue
        reference_url = _canonical_url(item.get("source_url"))
        task_id = str(item.get("task_id", ""))
        fact_ids = {
            str(fact_id) for fact_id in item.get("fact_ids", []) or []
        }
        if not reference_url or not task_id or not fact_ids:
            continue
        for discovery in discoveries:
            if (
                str(discovery.get("task_id", "")) != task_id
                or _canonical_url(discovery.get("reference_image_url"))
                != reference_url
                or not fact_ids
                & {
                    str(fact_id)
                    for fact_id in discovery.get("fact_ids", []) or []
                }
            ):
                continue
            source_context = " ".join(
                str(discovery.get(key, "") or "").strip()
                for key in ("title", "candidate_url", "snippet")
                if str(discovery.get(key, "") or "").strip()
            )
            if not source_context:
                continue
            for fact_id in fact_ids:
                context_by_fact.setdefault(fact_id, []).append(
                    source_context
                )
    return {
        fact_id: " ".join(dict.fromkeys(contexts))
        for fact_id, contexts in context_by_fact.items()
    }


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


def _duplicate_action_rate(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[float, int]:
    routes: List[tuple[str, Mapping[str, Any]]] = []
    for step in steps:
        if (
            str(step.get("stage", ""))
            in {
                "image_only_investigation",
                "image_only_visual_reinspection",
            }
            and str(step.get("action_type", "")) == "tool_call"
        ):
            routes.append(
                (
                    str(step.get("tool_name", "")),
                    _mapping(step.get("tool_args")),
                )
            )
    duplicate_count = semantic_duplicate_count(routes)
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


def _has_actual_visual_bridge(
    fact: Mapping[str, Any],
    related_evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_decision: Mapping[str, Any] | None = None,
) -> bool:
    status = str(fact.get("status", ""))
    bindings = {
        str(item.get("claim_binding", ""))
        for item in related_evidence
        if not item.get("risk_flags")
    }
    if status == "supported" and same_capture_can_support_visual_claim(fact):
        return bool(bindings & {"pixel_observation", "same_capture"})
    if status == "refuted":
        decision_output = _mapping(
            _mapping(semantic_decision).get("output")
        )
        selected_ids = {
            str(item)
            for item in decision_output.get("selected_evidence_ids", []) or []
        }
        if (
            decision_output.get("assessment") == "refuted"
            and decision_output.get("binding_requirement") == "text_sufficient"
            and any(
                str(item.get("evidence_id", "")) in selected_ids
                and str(item.get("claim_binding", "")) == "source_assertion"
                and str(item.get("quality", "")) in {"strong", "moderate"}
                and not item.get("risk_flags")
                for item in related_evidence
            )
        ):
            return True
        return any(
            str(item.get("claim_binding", ""))
            in {"pixel_observation", "same_capture", "source_assertion"}
            and str(item.get("directness", "")) == "direct"
            and str(item.get("quality", "")) in {"strong", "moderate"}
            and not item.get("risk_flags")
            for item in related_evidence
        )
    return bool(bindings & {"pixel_observation", "same_capture"})


def _post_determination_actions(
    steps: Sequence[Mapping[str, Any]],
    *,
    verdict: str,
    basis_fact_ids: set[str],
) -> int:
    determination_index: int | None = None
    seen_supported: set[str] = set()
    for index, step in enumerate(steps):
        if str(step.get("action_type", "")) != "tool_call":
            continue
        update = _mapping(
            _mapping(step.get("metadata")).get(
                "investigation_state_update"
            )
        )
        statuses = _mapping(update.get("fact_statuses"))
        if verdict == "fake" and any(
            fact_id in basis_fact_ids and str(status) == "refuted"
            for fact_id, status in statuses.items()
        ):
            determination_index = index
            break
        if verdict == "real":
            seen_supported.update(
                str(fact_id)
                for fact_id, status in statuses.items()
                if fact_id in basis_fact_ids and str(status) == "supported"
            )
            if basis_fact_ids and basis_fact_ids <= seen_supported:
                determination_index = index
                break
    if determination_index is None:
        return 0
    return sum(
        str(step.get("action_type", "")) == "tool_call"
        for step in steps[determination_index + 1 :]
    )


def _low_value_action_count(
    steps: Sequence[Mapping[str, Any]],
) -> int:
    count = 0
    for step in steps:
        if (
            str(step.get("stage", ""))
            not in {
                "image_only_investigation",
                "image_only_visual_reinspection",
            }
            or str(step.get("action_type", "")) != "tool_call"
        ):
            continue
        if str(step.get("tool_name", "")) == "current_time":
            count += 1
            continue
        update = _mapping(
            _mapping(step.get("metadata")).get(
                "investigation_state_update"
            )
        )
        if not any(
            update.get(key)
            for key in (
                "created_discovery_ids",
                "created_evidence_ids",
                "created_finding_ids",
            )
        ):
            count += 1
    return count


def score_process_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    score_metadata: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Score one post-rollout trace against evaluator-private references."""

    state = _mapping(trace.get("state"))
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version == "discrepancy-first-v4":
        return _score_discrepancy_trace(
            trace,
            gold,
            score_metadata=score_metadata,
        )

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
    evidence_decisions = _rows(investigation.get("evidence_decisions"))
    latest_decision_by_fact = {
        str(_mapping(item.get("output")).get("active_fact_id", "")): item
        for item in evidence_decisions
        if str(_mapping(item.get("output")).get("active_fact_id", ""))
    }
    findings_by_id = {
        str(item.get("finding_id", "")): item
        for item in findings
        if str(item.get("finding_id", ""))
    }
    coverage_audits = _rows(investigation.get("coverage_audits"))
    final_coverage = (
        _mapping(coverage_audits[-1]) if coverage_audits else {}
    )
    coverage_by_fact = {
        str(item.get("fact_id", "")): item
        for item in _rows(final_coverage.get("facts"))
        if str(item.get("fact_id", ""))
    }
    steps = _rows(state.get("all_steps"))
    successful_calls = _successful_call_ids(steps)
    valid_finding_ids = _valid_findings(
        findings,
        tasks,
        evidence,
        successful_calls,
    )
    basis = _mapping(
        trace.get("verdict_basis") or investigation.get("verdict_basis")
    )
    recovered_source_context_by_fact = _basis_same_capture_source_context(
        investigation,
        basis,
    )
    gold_facts = _rows(gold.get("decisive_facts"))
    fact_matches = _match_gold_facts(
        decisive_facts,
        gold_facts,
        recovered_source_context_by_fact=recovered_source_context_by_fact,
    )
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
    bridge_hits = 0
    runtime_fact_by_id = {
        str(item.get("fact_id", "")): item for item in decisive_facts
    }
    for gold_fact in gold_facts:
        gold_fact_id = str(gold_fact.get("fact_id", ""))
        runtime_fact_id = matched_runtime_by_gold.get(gold_fact_id)
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
        runtime_fact = runtime_fact_by_id.get(runtime_fact_id, {})
        if (
            gold_fact.get("visual_anchor")
            and _has_actual_visual_bridge(
                runtime_fact,
                related_evidence,
                semantic_decision=latest_decision_by_fact.get(
                    runtime_fact_id
                ),
            )
        ):
            bridge_hits += 1

    bridge_completion = (
        bridge_hits / len(gold_facts) if gold_facts else 1.0
    )

    basis_fact_ids = {
        str(item) for item in basis.get("fact_ids", []) or []
    }
    basis_finding_ids = {
        str(item) for item in basis.get("finding_ids", []) or []
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
    expected_minimal_evidence_ids = {
        str(evidence_id)
        for fact_id in basis_fact_ids
        for evidence_id in _mapping(
            coverage_by_fact.get(fact_id)
        ).get("winning_evidence_ids", [])
        or []
    }
    minimal_basis_overlap = len(
        basis_evidence_ids & expected_minimal_evidence_ids
    )
    basis_minimality_precision = (
        minimal_basis_overlap / len(basis_evidence_ids)
        if basis_evidence_ids
        else 0.0
    )
    basis_minimality_recall = (
        minimal_basis_overlap / len(expected_minimal_evidence_ids)
        if expected_minimal_evidence_ids
        else (1.0 if not basis_evidence_ids else 0.0)
    )
    basis_minimality = (
        2
        * basis_minimality_precision
        * basis_minimality_recall
        / (basis_minimality_precision + basis_minimality_recall)
        if basis_minimality_precision + basis_minimality_recall
        else 0.0
    )
    valid_finding_precision = (
        len(valid_finding_ids) / len(findings) if findings else 0.0
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
    blocked_duplicate_route_count = sum(
        bool(_mapping(step.get("metadata")).get("duplicate_tool_call"))
        for step in steps
    )
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
    conflict_resolutions = [
        str(item.get("conflict_resolution", "not_applicable"))
        for item in coverage_by_fact.values()
    ]
    resolved_conflict_count = sum(
        item in {"support_wins", "refute_wins"}
        for item in conflict_resolutions
    )
    unresolved_conflict_count = sum(
        item == "needs_discriminating_evidence"
        for item in conflict_resolutions
    )
    post_determination_action_count = _post_determination_actions(
        steps,
        verdict=str(trace.get("verdict", "")),
        basis_fact_ids=basis_fact_ids,
    )
    low_value_action_count = _low_value_action_count(steps)
    low_value_action_rate = (
        low_value_action_count / action_count if action_count else 0.0
    )

    process_metrics = {
        "schema_version": "ifv-process-metrics-v2",
        "case_id": str(gold.get("case_id") or trace.get("image_id") or ""),
        "score_metadata": dict(score_metadata or {}),
        "engineering_error": engineering_error,
        "expected_verdict": expected_verdict,
        "runtime_verdict": trace.get("verdict"),
        "result_correct": result_correct,
        "fact_match_threshold": FACT_MATCH_THRESHOLD,
        "fact_matches": fact_matches,
        "decisive_fact_discovery_rate": round(discovery_rate, 6),
        "decisive_fact_status_accuracy": round(status_accuracy, 6),
        "decisive_fact_alignment": round(decisive_fact_alignment, 6),
        "evidence_to_vision_bridge_completion": round(
            bridge_completion, 6
        ),
        "verdict_basis_alignment": round(verdict_basis_alignment, 6),
        "basis_minimality": round(basis_minimality, 6),
        "expected_minimal_evidence_ids": sorted(
            expected_minimal_evidence_ids
        ),
        "valid_finding_precision": round(valid_finding_precision, 6),
        "false_task_rate": round(false_task_rate, 6),
        "false_activation_rate": round(false_activation_rate, 6),
        "duplicate_action_rate": round(duplicate_action_rate, 6),
        "duplicate_action_count": duplicate_action_count,
        "blocked_duplicate_route_count": blocked_duplicate_route_count,
        "resolved_conflict_count": resolved_conflict_count,
        "unresolved_conflict_count": unresolved_conflict_count,
        "post_determination_action_count": post_determination_action_count,
        "low_value_action_count": low_value_action_count,
        "low_value_action_rate": round(low_value_action_rate, 6),
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
    training_exclusion_reasons: List[str] = []
    if not result_correct:
        training_exclusion_reasons.append("incorrect_result")
    if engineering_error:
        training_exclusion_reasons.append("engineering_error")
    if premature_finish:
        training_exclusion_reasons.append("premature_finish")
    if bridge_completion < 1.0:
        training_exclusion_reasons.append("visual_bridge_incomplete")
    if verdict_basis_alignment < 0.8:
        training_exclusion_reasons.append("verdict_basis_misaligned")
    if basis_minimality < 1.0:
        training_exclusion_reasons.append("verdict_basis_not_minimal")
    if duplicate_action_count:
        training_exclusion_reasons.append("semantic_duplicate_actions")
    if blocked_duplicate_route_count:
        training_exclusion_reasons.append(
            "blocked_semantic_duplicate_routes"
        )
    if post_determination_action_count:
        training_exclusion_reasons.append(
            "actions_after_verdict_determined"
        )
    if low_value_action_count > 1:
        training_exclusion_reasons.append("excess_low_value_actions")
    if valid_finding_precision < 1.0:
        training_exclusion_reasons.append("invalid_findings")
    if unresolved_conflict_count:
        training_exclusion_reasons.append("unresolved_evidence_conflict")
    training_eligible = not training_exclusion_reasons
    process_metrics["training_eligible"] = training_eligible
    process_metrics["training_exclusion_reasons"] = (
        training_exclusion_reasons
    )

    components = {
        "result_reward": 1.0 if result_correct else 0.0,
        "grounded_finding_reward": round(valid_finding_precision, 6),
        "gap_coverage_reward": round(decisive_fact_alignment, 6),
        "bridge_reward": round(bridge_completion, 6),
        "basis_minimality_reward": round(basis_minimality, 6),
        "stop_calibration_reward": (
            1.0
            if (
                not engineering_error
                and not premature_finish
                and post_determination_action_count == 0
            )
            else 0.0
        ),
        "duplicate_action_penalty": round(-duplicate_action_rate, 6),
        "blocked_duplicate_route_penalty": round(
            -min(1.0, blocked_duplicate_route_count / 4.0),
            6,
        ),
        "low_value_action_penalty": round(
            -low_value_action_rate,
            6,
        ),
        "invalid_task_penalty": round(
            -max(false_task_rate, false_activation_rate),
            6,
        ),
        "normalized_cost_penalty": round(-0.25 * normalized_cost, 6),
    }
    teacher_score = {
        "schema_version": "ifv-trajectory-score-v2",
        "case_id": process_metrics["case_id"],
        "score_metadata": dict(score_metadata or {}),
        "components": components,
        "total": round(sum(components.values()), 6),
        "training_eligible": training_eligible,
        "training_exclusion_reasons": training_exclusion_reasons,
        "diagnostics": {
            "fact_matches": fact_matches,
            "decisive_fact_ids": sorted(decisive_ids),
            "basis_fact_ids": sorted(basis_fact_ids),
            "basis_finding_ids": sorted(basis_finding_ids),
            "basis_evidence_ids": sorted(basis_evidence_ids),
            "valid_finding_ids": sorted(valid_finding_ids),
            "expected_minimal_evidence_ids": sorted(
                expected_minimal_evidence_ids
            ),
            "resolved_conflict_count": resolved_conflict_count,
            "unresolved_conflict_count": unresolved_conflict_count,
            "post_determination_action_count": (
                post_determination_action_count
            ),
            "low_value_action_count": low_value_action_count,
            "blocked_duplicate_route_count": (
                blocked_duplicate_route_count
            ),
            "engineering_error": engineering_error,
            "first_error": first_error,
        },
    }
    return process_metrics, teacher_score


def _v4_claim_directional_chain_ids(
    *,
    claim_id: str,
    stance: str,
    selected_evidence_ids: set[str],
    claims: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    findings: Mapping[str, Mapping[str, Any]],
    successful_calls: set[str],
    selected_finding_ids: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    claim = claims.get(claim_id)
    if claim is None:
        return set(), set()
    claim_fact_id = str(claim.get("fact_id", "")).strip()
    matched_evidence_ids: set[str] = set()
    matched_finding_ids: set[str] = set()
    for finding_id, finding in findings.items():
        if selected_finding_ids is not None and finding_id not in selected_finding_ids:
            continue
        if str(finding.get("stance", "")).strip() != stance:
            continue
        task_id = str(finding.get("task_id", "")).strip()
        task = tasks.get(task_id)
        if task is None or claim_id not in {
            str(item) for item in task.get("claim_ids", []) or []
        }:
            continue
        if claim_fact_id not in {
            str(item) for item in finding.get("fact_ids", []) or []
        }:
            continue
        for evidence_id in finding.get("evidence_ids", []) or []:
            evidence_id = str(evidence_id)
            row = evidence.get(evidence_id)
            if (
                evidence_id in selected_evidence_ids
                and row is not None
                and str(row.get("task_id", "")).strip() == task_id
                and evidence_is_qualified_for_stance(row, stance)
                and str(row.get("function_call_id", "")).strip()
                in successful_calls
            ):
                matched_evidence_ids.add(evidence_id)
                matched_finding_ids.add(finding_id)
    return matched_evidence_ids, matched_finding_ids


def _score_discrepancy_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    score_metadata: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Score v4 alignment and stop quality without legacy core-fact ownership."""

    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    claims = {
        str(item.get("claim_id", "")): item
        for item in _rows(investigation.get("image_claims"))
        if str(item.get("claim_id", ""))
    }
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
    findings = {
        str(item.get("finding_id", "")): item
        for item in _rows(investigation.get("findings"))
        if str(item.get("finding_id", ""))
    }
    assessments = _rows(investigation.get("claim_assessments"))
    discrepancies = _rows(investigation.get("material_discrepancies"))
    steps = _rows(state.get("all_steps"))
    successful_calls = _successful_call_ids(steps)
    decision_evidence_consistent = True
    latest_assessment_by_claim: Dict[str, Mapping[str, Any]] = {}
    for assessment in assessments:
        claim_id = str(assessment.get("claim_id", ""))
        selected_ids = {
            str(item) for item in assessment.get("evidence_ids", []) or []
        }
        if claim_id not in claims or not selected_ids <= set(evidence):
            decision_evidence_consistent = False
            continue
        latest_assessment_by_claim[claim_id] = assessment
        if any(
            claim_id
            not in {
                str(item)
                for item in tasks.get(
                    str(evidence[evidence_id].get("task_id", "")),
                    {},
                ).get("claim_ids", [])
                or []
            }
            for evidence_id in selected_ids
        ):
            decision_evidence_consistent = False
        for stance in required_assessment_stances(
            str(assessment.get("assessment", ""))
        ):
            if not _v4_claim_directional_chain_ids(
                claim_id=claim_id,
                stance=stance,
                selected_evidence_ids=selected_ids,
                claims=claims,
                tasks=tasks,
                evidence=evidence,
                findings=findings,
                successful_calls=successful_calls,
            )[1]:
                decision_evidence_consistent = False
    for claim_id, assessment in latest_assessment_by_claim.items():
        expected_status = {
            "supported": "supported",
            "refuted": "refuted",
            "conflicted": "conflicted",
            "insufficient": "unresolved",
        }.get(str(assessment.get("assessment", "")), "")
        if str(claims[claim_id].get("status", "")) != expected_status:
            decision_evidence_consistent = False
    decisive = [
        item
        for item in discrepancies
        if str(item.get("materiality", "")) == "decisive"
        and str(item.get("status", "")) == "established"
    ]
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    expected_verdict = LABEL_TO_VERDICT.get(
        str(gold.get("factual_status", "")),
        "",
    )
    engineering_error = (
        str(trace.get("termination", "")) != "success"
        or str(state.get("termination", "")) != "success"
    )
    result_correct = (
        not engineering_error
        and bool(expected_verdict)
        and str(trace.get("verdict", "")) == expected_verdict
    )

    aligned_discrepancy_ids: List[str] = []
    invalid_discrepancy_ids: List[str] = []
    for item in discrepancies:
        discrepancy_id = str(item.get("discrepancy_id", ""))
        claim_ids = {
            str(value) for value in item.get("affected_claim_ids", []) or []
        }
        anchor_ids = {
            str(value)
            for value in item.get("visual_anchor_fact_ids", []) or []
        }
        evidence_ids = {
            str(value) for value in item.get("evidence_ids", []) or []
        }
        aligned = bool(
            claim_ids
            and claim_ids <= set(claims)
            and evidence_ids
            and evidence_ids <= set(evidence)
            and all(
                anchor_ids
                & {
                    str(value)
                    for value in claims[claim_id].get("anchor_fact_ids", []) or []
                }
                for claim_id in claim_ids
            )
            and all(
                any(
                    claim_id
                    in {
                        str(value)
                        for value in tasks.get(
                            str(evidence[evidence_id].get("task_id", "")),
                            {},
                        ).get("claim_ids", [])
                        or []
                    }
                    for evidence_id in evidence_ids
                )
                for claim_id in claim_ids
            )
            and (
                str(item.get("materiality", "")) != "decisive"
                or str(item.get("status", "")) != "established"
                or all(
                    _v4_claim_directional_chain_ids(
                        claim_id=claim_id,
                        stance="refute",
                        selected_evidence_ids=evidence_ids,
                        claims=claims,
                        tasks=tasks,
                        evidence=evidence,
                        findings=findings,
                        successful_calls=successful_calls,
                    )[1]
                    and str(
                        latest_assessment_by_claim.get(claim_id, {}).get(
                            "assessment",
                            "",
                        )
                    )
                    == "refuted"
                    for claim_id in claim_ids
                )
            )
        )
        (aligned_discrepancy_ids if aligned else invalid_discrepancy_ids).append(
            discrepancy_id
        )

    selected_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    selected_discrepancy_ids = {
        str(item) for item in basis.get("discrepancy_ids", []) or []
    }
    selected_claim_ids = {str(item) for item in basis.get("claim_ids", []) or []}
    selected_finding_ids = {
        str(item) for item in basis.get("finding_ids", []) or []
    }
    verdict = str(trace.get("verdict", ""))
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    judgment_basis_consistent = bool(
        str(judgment.get("verdict", "")) == verdict
        and all(
            {
                str(item) for item in judgment.get(judgment_field, []) or []
            }
            == {str(item) for item in basis.get(basis_field, []) or []}
            for judgment_field, basis_field in (
                ("selected_claim_ids", "claim_ids"),
                ("selected_discrepancy_ids", "discrepancy_ids"),
                ("selected_visual_anchor_fact_ids", "visual_anchor_fact_ids"),
                ("selected_finding_ids", "finding_ids"),
                ("selected_evidence_ids", "evidence_ids"),
            )
        )
    )
    if verdict in {"fake", "real"}:
        expected_stance = "refute" if verdict == "fake" else "support"
        linked_evidence_ids = {
            str(evidence_id)
            for finding_id in selected_finding_ids & set(findings)
            for evidence_id in findings[finding_id].get("evidence_ids", []) or []
        }
        evidence_chain_complete = bool(
            selected_claim_ids
            and selected_evidence_ids
            and selected_finding_ids
            and selected_evidence_ids <= set(evidence)
            and selected_evidence_ids <= linked_evidence_ids
            and all(
                _v4_claim_directional_chain_ids(
                    claim_id=claim_id,
                    stance=expected_stance,
                    selected_evidence_ids=selected_evidence_ids,
                    selected_finding_ids=selected_finding_ids,
                    claims=claims,
                    tasks=tasks,
                    evidence=evidence,
                    findings=findings,
                    successful_calls=successful_calls,
                )[1]
                for claim_id in selected_claim_ids
            )
        )
    else:
        evidence_chain_complete = bool(
            selected_evidence_ids
            and selected_evidence_ids <= set(evidence)
            and all(
                str(evidence[evidence_id].get("function_call_id", ""))
                in successful_calls
                for evidence_id in selected_evidence_ids
            )
        )
    evidence_chain_recovery = 1.0 if evidence_chain_complete else 0.0
    discrepancy_alignment = (
        len(aligned_discrepancy_ids) / len(discrepancies)
        if discrepancies
        else 1.0 if str(trace.get("verdict", "")) != "fake" else 0.0
    )
    if str(trace.get("verdict", "")) == "fake":
        discrepancy_alignment = min(
            discrepancy_alignment,
            1.0
            if selected_discrepancy_ids
            and selected_discrepancy_ids <= set(aligned_discrepancy_ids)
            else 0.0,
        )
    audits = _rows(investigation.get("discrepancy_coverage_audits"))
    investigation_stop_reason = str(
        investigation.get("stop_reason", "")
    ).strip()
    if investigation_stop_reason:
        terminal = [
            item
            for item in audits
            if str(item.get("stop_reason", "")) == investigation_stop_reason
            and (
                investigation_stop_reason != "verdict_determined"
                or item.get("complete") is True
            )
        ]
    else:
        terminal = [
            item
            for item in audits
            if item.get("complete") is True
            and str(item.get("stop_reason", "")) == "verdict_determined"
        ]
    terminal_action_count = (
        int(terminal[-1].get("action_count", 0) or 0) if terminal else -1
    )
    action_steps = [
        item
        for item in steps
        if str(item.get("stage", ""))
        in {
            "image_only_discrepancy_investigation",
            "image_only_visual_reinspection",
        }
        and str(item.get("action_type", "")) == "tool_call"
    ]
    post_verdict_actions = max(0, len(action_steps) - terminal_action_count)
    stop_quality = (
        1.0
        if terminal and not post_verdict_actions
        else 0.0
    )
    protocol_rejections = sum(
        str(item.get("action_type", "")) in {"format_error", "output_rejected"}
        for item in steps
    )
    training_exclusion_reasons: List[str] = []
    if not result_correct:
        training_exclusion_reasons.append("incorrect_result")
    if engineering_error:
        training_exclusion_reasons.append("engineering_error")
    if evidence_chain_recovery < 1.0:
        training_exclusion_reasons.append("evidence_chain_incomplete")
    if discrepancy_alignment < 1.0:
        training_exclusion_reasons.append("discrepancy_misaligned")
    if not decision_evidence_consistent:
        training_exclusion_reasons.append("decision_evidence_inconsistent")
    if not judgment_basis_consistent:
        training_exclusion_reasons.append("judgment_basis_mismatch")
    if stop_quality < 1.0:
        training_exclusion_reasons.append("stop_quality_invalid")
    if protocol_rejections:
        training_exclusion_reasons.append("protocol_rejections")
    if investigation.get("core_verdict_fact_id"):
        training_exclusion_reasons.append("legacy_core_ownership")
    training_eligible = not training_exclusion_reasons
    process_metrics = {
        "schema_version": "ifv-process-metrics-v4",
        "case_id": str(gold.get("case_id") or trace.get("image_id") or ""),
        "expected_verdict": expected_verdict,
        "predicted_verdict": str(trace.get("verdict", "")),
        "result_correct": result_correct,
        "engineering_error": engineering_error,
        "evidence_chain_recovery": round(evidence_chain_recovery, 6),
        "discrepancy_alignment": round(discrepancy_alignment, 6),
        "decision_evidence_consistency": (
            1.0 if decision_evidence_consistent else 0.0
        ),
        "judgment_basis_consistency": 1.0 if judgment_basis_consistent else 0.0,
        "stop_quality": round(stop_quality, 6),
        "image_claim_count": len(claims),
        "material_discrepancy_count": len(discrepancies),
        "aligned_discrepancy_count": len(aligned_discrepancy_ids),
        "invalid_discrepancy_count": len(invalid_discrepancy_ids),
        "post_determination_action_count": post_verdict_actions,
        "tool_actions": int(investigation.get("action_count", 0) or 0),
        "training_eligible": training_eligible,
        "training_exclusion_reasons": training_exclusion_reasons,
    }
    components = {
        "result_reward": 1.0 if result_correct else 0.0,
        "evidence_chain_reward": round(evidence_chain_recovery, 6),
        "discrepancy_alignment_reward": round(discrepancy_alignment, 6),
        "stop_quality_reward": round(stop_quality, 6),
        "protocol_penalty": -1.0 if protocol_rejections else 0.0,
        "normalized_cost_penalty": round(
            -0.25 * min(1.0, len(action_steps) / 24.0),
            6,
        ),
    }
    teacher_score = {
        "schema_version": "ifv-trajectory-score-v4",
        "case_id": process_metrics["case_id"],
        "score_metadata": dict(score_metadata or {}),
        "components": components,
        "total": round(sum(components.values()), 6),
        "training_eligible": training_eligible,
        "training_exclusion_reasons": training_exclusion_reasons,
        "diagnostics": {
            "aligned_discrepancy_ids": aligned_discrepancy_ids,
            "invalid_discrepancy_ids": invalid_discrepancy_ids,
            "selected_discrepancy_ids": sorted(selected_discrepancy_ids),
            "selected_evidence_ids": sorted(selected_evidence_ids),
            "selected_finding_ids": sorted(selected_finding_ids),
            "decision_evidence_consistent": decision_evidence_consistent,
            "judgment_basis_consistent": judgment_basis_consistent,
            "post_determination_action_count": post_verdict_actions,
            "protocol_rejections": protocol_rejections,
        },
    }
    return process_metrics, teacher_score
