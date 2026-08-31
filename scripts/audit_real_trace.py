"""Strictly audit canonical v3 image-only traces."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.source_access import (  # noqa: E402
    FACT_CHECK_DOMAIN_MARKERS,
    SourceAccessPolicy,
    benchmark_source_access_policy,
    url_variants,
)
from src.orchestrator.source_provenance import domain_matches  # noqa: E402
from src.orchestrator.evidence_policy import (  # noqa: E402
    query_targets_fact_check_answer,
)
from src.orchestrator.investigation_models import target_fact_rows  # noqa: E402
from src.orchestrator.unified_react import (  # noqa: E402
    is_unified_react_budget_action,
)
from src.orchestrator.react_runtime import (  # noqa: E402
    REACT_RUNTIME_SCHEMA_VERSION,
    REACT_RUNTIME_TOOLS,
    is_unified_react_runtime_budget_action,
)
from src.orchestrator.evidence_semantics import (  # noqa: E402
    evidence_direction_is_coherent,
    evidence_is_qualified_for_stance,
    required_assessment_stances,
)
from src.orchestrator.tool_result import parse_tool_result  # noqa: E402


HARD = "hard"
SCHEDULER = "scheduler"
PROTOCOL = "protocol"
ROUTE_CONTROL = "route_control"
CORRECTION = "correction"
BOUNDED_FALLBACK = "bounded_fallback"
FORMAT = "format"
REJECTION_ACTIONS = frozenset({"format_error", "output_rejected", "policy_replan"})
UNIFIED_REACT_POLICY_VERSION = "unified-react-v1"
WEB_EVIDENCE_TOOLS = frozenset({"visit", "crop_and_search"})
COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY = (
    "composite:source_visual_discrepancy"
)
KNOWN_FACT_CHECK_QUERY_POLICY = SourceAccessPolicy(
    policy_id="canonical-trace-known-fact-check-sites",
    excluded_domains=frozenset(FACT_CHECK_DOMAIN_MARKERS),
)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    category: str = HARD
    location: str = ""


@dataclass
class TraceReport:
    path: str
    image_id: str = ""
    termination: str = ""
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def failures(self, *, strict_scheduler: bool) -> list[Issue]:
        return [
            issue
            for issue in self.issues
            if issue.category == HARD
            or (strict_scheduler and issue.category in {SCHEDULER, PROTOCOL})
        ]

    def warnings(self, *, strict_scheduler: bool) -> list[Issue]:
        failed = set(self.failures(strict_scheduler=strict_scheduler))
        return [issue for issue in self.issues if issue not in failed]

    def to_dict(self, *, strict_scheduler: bool) -> dict[str, Any]:
        failures = self.failures(strict_scheduler=strict_scheduler)
        warnings = self.warnings(strict_scheduler=strict_scheduler)
        return {
            "path": self.path,
            "image_id": self.image_id,
            "termination": self.termination,
            "passed": not failures,
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "stats": self.stats,
            "issues": [
                {
                    **asdict(issue),
                    "severity": "error" if issue in failures else "warning",
                }
                for issue in self.issues
            ],
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _task_descends_from(
    task_id: str,
    ancestor_task_id: str,
    task_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    current = str(task_id or "").strip()
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        task = task_by_id.get(current)
        if task is None:
            return False
        parent = str(task.get("parent_task_id", "") or "").strip()
        if parent == ancestor_task_id:
            return True
        current = parent
    return False


def _composite_visual_evidence_is_eligible(
    evidence: Mapping[str, Any],
    *,
    finding: Mapping[str, Any],
    task_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Accept claim-owned pixel Evidence without depending on its tool name."""

    if (
        str(evidence.get("evidence_kind", "")).strip() != "image_region"
        or str(evidence.get("claim_binding", "")).strip() != "pixel_observation"
        or str(evidence.get("stance", "")).strip() != "neutral"
        or str(evidence.get("directness", "direct")).strip() != "direct"
        or str(evidence.get("quality", "")).strip() not in {"strong", "moderate"}
        or str(evidence.get("visual_answer_status", "")).strip() == "ambiguous"
    ):
        return False
    finding_task = task_by_id.get(str(finding.get("task_id", "")).strip())
    evidence_task = task_by_id.get(str(evidence.get("task_id", "")).strip())
    if finding_task is None or evidence_task is None:
        return False
    finding_claims = {
        str(item) for item in finding_task.get("claim_ids", []) or []
    }
    evidence_claims = {
        str(item) for item in evidence_task.get("claim_ids", []) or []
    }
    finding_facts = {
        str(item) for item in finding.get("fact_ids", []) or []
    }
    evidence_facts = {
        str(item) for item in evidence.get("fact_ids", []) or []
    }
    return bool(
        finding_claims & evidence_claims
        or finding_facts & evidence_facts
    )


def _state(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    state = trace.get("state")
    return state if isinstance(state, Mapping) else trace


def _location(prefix: str, identifier: Any = "") -> str:
    value = str(identifier or "").strip()
    return f"{prefix}[{value}]" if value else prefix


def _issue(
    report: TraceReport,
    code: str,
    message: str,
    *,
    category: str = HARD,
    location: str = "",
) -> None:
    report.issues.append(
        Issue(code=code, message=message, category=category, location=location)
    )


def _step_label(index: int, step: Mapping[str, Any]) -> str:
    call_id = str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
    suffix = f",call={call_id}" if call_id else ""
    return f"state.all_steps[{index}]({step.get('stage', '')}:{step.get('round', '')}{suffix})"


def _parse_successful_tool_step(step: Mapping[str, Any]) -> bool:
    if step.get("action_type") != "tool_call":
        return False
    metadata = _mapping(step.get("metadata"))
    if metadata.get("tool_exception") or metadata.get("tool_success") is False:
        return False
    try:
        _, succeeded = parse_tool_result(str(step.get("tool_result", "")))
    except Exception:
        return False
    return succeeded


def _audit_termination(
    trace: Mapping[str, Any], state: Mapping[str, Any], report: TraceReport
) -> None:
    outer = str(trace.get("termination", "")).strip()
    inner = str(state.get("termination", "")).strip()
    report.termination = outer or inner
    if outer != "success" or inner != "success":
        _issue(
            report,
            "TERMINATION_NOT_SUCCESS",
            f"termination must be success at trace and state levels (trace={outer!r}, state={inner!r})",
            location="termination",
        )


def _numeric_token(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return numeric if math.isfinite(numeric) else None


def _audit_thought_tokens(
    trace: Mapping[str, Any], state: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    locations: list[tuple[str, Any, bool]] = []
    for prefix, usage in (
        ("token_usage", trace.get("token_usage")),
        ("state.token_usage", state.get("token_usage")),
    ):
        row = _mapping(usage)
        if "thought" in row:
            locations.append((f"{prefix}.thought", row.get("thought"), True))
        if "thought_tokens" in row:
            locations.append(
                (f"{prefix}.thought_tokens", row.get("thought_tokens"), True)
            )
        if "total_thought_tokens" in row:
            locations.append(
                (
                    f"{prefix}.total_thought_tokens",
                    row.get("total_thought_tokens"),
                    True,
                )
            )

    native_steps = 0
    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        tokens = _mapping(step.get("tokens"))
        if metadata.get("native_interactions"):
            native_steps += 1
            if not any(
                key in tokens
                for key in ("thought", "thought_tokens", "total_thought_tokens")
            ):
                _issue(
                    report,
                    "THOUGHT_TOKENS_UNRECORDED",
                    "Gemini Interactions step does not record a thought-token field",
                    location=_step_label(index, step),
                )
        for key in ("thought", "thought_tokens", "total_thought_tokens"):
            if key in tokens:
                locations.append(
                    (
                        f"state.all_steps[{index}].tokens.{key}",
                        tokens[key],
                        True,
                    )
                )

    recorded_paths = {path for path, _, _ in locations}
    for key, raw_value, path in _iter_named_values(trace):
        token_key = key.casefold()
        parent_segments = {
            segment.casefold()
            for segment in re.split(r"[.\[\]<>]+", path.rsplit(".", 1)[0])
            if segment
        }
        is_token_field = token_key in {"thought_tokens", "total_thought_tokens"}
        is_usage_total = (
            token_key == "thought"
            and bool(parent_segments & {"usage", "tokens", "token_usage"})
        )
        if (is_token_field or is_usage_total) and path not in recorded_paths:
            locations.append((path, raw_value, True))
            recorded_paths.add(path)

    report.stats["native_interaction_steps"] = native_steps
    report.stats["thought_token_fields"] = len(locations)
    for path, value, nonzero_allowed in locations:
        numeric = _numeric_token(value)
        if numeric is None:
            _issue(
                report,
                "THOUGHT_TOKENS_INVALID",
                f"thought-token value must be numeric, got {value!r}",
                location=path,
            )


def _audit_evidence_calls(
    evidence: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
    *,
    location_prefix: str,
    tool_field: str,
    stat_key: str,
) -> None:
    by_call: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, step in enumerate(steps):
        call_id = str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
        if call_id:
            by_call.setdefault(call_id, []).append((index, step))

    successful = 0
    for evidence_index, item in enumerate(evidence):
        evidence_id = str(item.get("evidence_id", "")).strip()
        location = (
            _location(location_prefix, evidence_id)
            if evidence_id
            else f"{location_prefix}[{evidence_index}]"
        )
        call_id = str(item.get("function_call_id", "")).strip()
        matches = by_call.get(call_id, []) if call_id else []
        successful_matches = [row for row in matches if _parse_successful_tool_step(row[1])]
        if len(matches) != 1 or len(successful_matches) != 1:
            _issue(
                report,
                "EVIDENCE_CALL_NOT_SUCCESSFUL",
                f"function_call_id {call_id!r} must resolve to exactly one successful tool step; found {len(matches)} step(s), {len(successful_matches)} successful",
                location=location,
            )
            continue
        successful += 1
        _, step = successful_matches[0]
        step_tool = str(step.get("tool_name", "")).strip()
        evidence_tool = str(item.get(tool_field, "")).strip()
        if evidence_tool != step_tool:
            _issue(
                report,
                "EVIDENCE_TOOL_MISMATCH",
                f"evidence tool {evidence_tool!r} does not match successful step tool {step_tool!r}",
                location=location,
            )
    report.stats[stat_key] = successful








def _fact_check_domain(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    if source_access_policy is not None:
        for variant in url_variants(value):
            hostname = (urlsplit(variant).hostname or "").lower().rstrip(".")
            for domain in sorted(
                source_access_policy.excluded_domains,
                key=len,
                reverse=True,
            ):
                if domain_matches(hostname, domain):
                    return domain
        return ""
    policy = benchmark_source_access_policy(
        [value],
        policy_id="canonical-trace-audit",
    )
    return next(iter(sorted(policy.excluded_domains)), "")


def _fact_check_query_domain(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    text = unquote(str(value or ""))
    candidates = re.findall(
        r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s\"'<>]*)?",
        text,
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        if domain := _fact_check_domain(
            candidate.rstrip(".,;:!?)]}"),
            source_access_policy=source_access_policy,
        ):
            return domain
    return ""


def _fact_check_embedded_url(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    urls = re.findall(
        r"https?://[^\s\"'<>]+",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    for url in urls:
        if domain := _fact_check_domain(
            url.rstrip(".,;:!?)]}"),
            source_access_policy=source_access_policy,
        ):
            return domain
    return ""


def _fact_check_url_query(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    for variant in url_variants(value):
        try:
            query = " ".join(
                f"{key} {item}" for key, item in parse_qsl(urlsplit(variant).query)
            )
        except ValueError:
            continue
        if domain := _fact_check_query_reference(
            query,
            source_access_policy=source_access_policy,
        ):
            return f"known fact-check domain {domain!r} in URL query"
        if source_access_policy is None and query_targets_fact_check_answer(query):
            return "fact-check-oriented URL query"
    return ""


def _fact_check_query_reference(
    value: str,
    *,
    source_access_policy: SourceAccessPolicy | None = None,
) -> str:
    policy = source_access_policy or KNOWN_FACT_CHECK_QUERY_POLICY
    return (
        policy.blocked_query_reference(value)
        or _fact_check_query_domain(
            value,
            source_access_policy=source_access_policy,
        )
    )


def _iter_named_values(
    value: Any,
    path: str = "",
    inherited_key: str = "",
) -> Iterable[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                if key == "tool_result" and isinstance(child, str):
                    try:
                        parsed = json.loads(child)
                    except (TypeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, (Mapping, list)):
                        yield from _iter_named_values(
                            parsed,
                            f"{child_path}<json>",
                        )
                        continue
                yield key, str(child), child_path
            else:
                yield from _iter_named_values(child, child_path, key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                yield inherited_key, str(child), child_path
            else:
                yield from _iter_named_values(child, child_path, inherited_key)


def _looks_like_url_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in {
        "source",
        "hostname",
        "registered_domain",
        "domain",
        "link",
        "href",
        "reference_image_candidates",
    } or "url" in lowered


def _looks_like_query_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in {"query", "queries", "suggested_query", "suggested_queries", "crop_query", "vlm_query"}


def _policy_rejected_step_for_path(
    trace: Mapping[str, Any],
    path: str,
) -> Mapping[str, Any] | None:
    """Return the step when a forbidden proposal was blocked before execution."""

    match = re.search(r"(?:^|\.)all_steps\[(\d+)\](?:\.|$)", path)
    if match is None:
        return None
    steps = _rows(_state(trace).get("all_steps"))
    index = int(match.group(1))
    if index >= len(steps):
        return None
    step = steps[index]
    metadata = _mapping(step.get("metadata"))
    if (
        str(step.get("action_type", ""))
        in {*REJECTION_ACTIONS, "planning_revision"}
        or metadata.get("search_policy_rejection") is True
        or int(metadata.get("policy_filtered_query_count", 0) or 0) > 0
    ):
        return step
    return None


def _audit_leaks(
    trace: Mapping[str, Any],
    report: TraceReport,
    *,
    enforce_source_access_policy: bool,
    source_access_policy: SourceAccessPolicy | None = None,
) -> None:
    seen: set[tuple[str, str]] = set()
    url_count = 0
    query_count = 0
    rejected_query_count = 0
    for key, raw_value, path in _iter_named_values(trace):
        query_key = _looks_like_query_key(key)
        if enforce_source_access_policy and _looks_like_url_key(key):
            domain = _fact_check_domain(
                raw_value,
                source_access_policy=source_access_policy,
            ) or _fact_check_query_domain(
                raw_value,
                source_access_policy=source_access_policy,
            )
        elif enforce_source_access_policy and not query_key:
            domain = _fact_check_embedded_url(
                raw_value,
                source_access_policy=source_access_policy,
            )
        else:
            domain = ""
        if domain:
            signature = ("url", path)
            if signature not in seen:
                seen.add(signature)
                url_count += 1
                _issue(
                    report,
                    "FACT_CHECK_URL_LEAK",
                    f"known fact-check domain {domain!r} appears in URL-bearing trace data",
                    location=path,
                )
        if enforce_source_access_policy and _looks_like_url_key(key) and (
            url_query_leak := _fact_check_url_query(
                raw_value,
                source_access_policy=source_access_policy,
            )
        ):
            signature = ("query", path)
            if signature not in seen:
                seen.add(signature)
                if _policy_rejected_step_for_path(trace, path) is not None:
                    rejected_query_count += 1
                    _issue(
                        report,
                        "FACT_CHECK_QUERY_REJECTED",
                        f"{url_query_leak} was rejected before execution: {raw_value!r}",
                        category=CORRECTION,
                        location=path,
                    )
                else:
                    query_count += 1
                    _issue(
                        report,
                        "FACT_CHECK_QUERY_LEAK",
                        f"{url_query_leak} leaked into the trace: {raw_value!r}",
                        location=path,
                    )
        if query_key:
            query_domain = (
                _fact_check_query_reference(
                    raw_value,
                    source_access_policy=source_access_policy,
                )
                if enforce_source_access_policy
                else ""
            )
            oriented = (
                query_targets_fact_check_answer(raw_value)
                if enforce_source_access_policy and source_access_policy is None
                else False
            )
            if query_domain or oriented:
                signature = ("query", path)
                if signature not in seen:
                    seen.add(signature)
                    detail = (
                        f"known fact-check domain {query_domain!r} is named in query"
                        if query_domain
                        else "fact-check-oriented query"
                    )
                    if _policy_rejected_step_for_path(trace, path) is not None:
                        rejected_query_count += 1
                        _issue(
                            report,
                            "FACT_CHECK_QUERY_REJECTED",
                            f"{detail} was rejected before execution: {raw_value!r}",
                            category=CORRECTION,
                            location=path,
                        )
                    else:
                        query_count += 1
                        _issue(
                            report,
                            "FACT_CHECK_QUERY_LEAK",
                            f"{detail} leaked into the trace: {raw_value!r}",
                            location=path,
                        )
    report.stats["fact_check_url_leaks"] = url_count
    report.stats["fact_check_query_leaks"] = query_count
    report.stats["fact_check_query_rejections"] = rejected_query_count




def _rejection_category(step: Mapping[str, Any]) -> str:
    metadata = _mapping(step.get("metadata"))
    if metadata.get("search_query_format_error"):
        return FORMAT
    if metadata.get("search_policy_rejection"):
        return CORRECTION
    if _empty_text_search_query(step):
        return FORMAT
    if metadata.get("unbalanced_priority_coverage") or metadata.get("tool_budget_reached"):
        return SCHEDULER
    reason = str(metadata.get("rejection_reason", "")).casefold()
    if not reason:
        try:
            reason = str(
                _mapping(json.loads(str(step.get("tool_result", "")))).get("error", "")
            ).casefold()
        except (TypeError, json.JSONDecodeError):
            reason = ""
    scheduler_markers = (
        "coverage requires",
        "required investigation questions",
        "tool budget",
        "budget ended",
        "before resampling",
        "untouched p1",
        "untouched p2",
        "at least 1 tool call",
    )
    if any(marker in reason for marker in scheduler_markers):
        return SCHEDULER
    return PROTOCOL


def _empty_text_search_query(step: Mapping[str, Any]) -> bool:
    if str(step.get("tool_name", "")).strip() != "text_search":
        return False
    tool_args = _mapping(step.get("tool_args"))
    queries = tool_args.get("queries", tool_args.get("query", []))
    if isinstance(queries, str):
        return not queries.strip()
    if not isinstance(queries, list):
        return True
    return not any(str(query).strip() for query in queries)


def _audit_rejections(
    steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    scheduler_count = 0
    protocol_count = 0
    route_control_count = 0
    corrected_count = 0
    bounded_fallback_count = 0
    format_count = 0

    def rejected(candidate: Mapping[str, Any]) -> bool:
        candidate_metadata = _mapping(candidate.get("metadata"))
        return (
            str(candidate.get("action_type", "")) in REJECTION_ACTIONS
            or bool(candidate_metadata.get("rejection_reason"))
        )

    def attempt_id(candidate: Mapping[str, Any]) -> str:
        candidate_metadata = _mapping(candidate.get("metadata"))
        if candidate_metadata.get("native_chat_completions"):
            return str(
                candidate_metadata.get("context_request_id", "")
                or candidate_metadata.get("interaction_id", "")
            ).strip()
        return str(
            candidate_metadata.get("interaction_id", "")
            or candidate_metadata.get("context_request_id", "")
        ).strip()

    def correction_parent_id(candidate: Mapping[str, Any]) -> str:
        candidate_metadata = _mapping(candidate.get("metadata"))
        if candidate_metadata.get("native_chat_completions"):
            return str(
                candidate_metadata.get("parent_context_request_id", "")
                or candidate_metadata.get("previous_interaction_id", "")
            ).strip()
        return str(
            candidate_metadata.get("previous_interaction_id", "")
            or candidate_metadata.get("parent_context_request_id", "")
        ).strip()

    def eventually_corrected(
        rejected_index: int,
        rejected_step: Mapping[str, Any],
    ) -> bool:
        root_id = attempt_id(rejected_step)
        if not root_id:
            return False
        reachable_rejected_ids = {root_id}
        stage = str(rejected_step.get("stage", ""))
        for candidate in steps[rejected_index + 1 :]:
            if str(candidate.get("stage", "")) != stage:
                continue
            candidate_metadata = _mapping(candidate.get("metadata"))
            lifecycle = str(
                candidate_metadata.get("interaction_lifecycle_kind", "")
            ).strip()
            is_unified_outer_correction = (
                stage == "unified_react"
                and lifecycle == "tool_roundtrip"
                and str(candidate.get("action_type", "")) == "tool_call"
                and bool(
                    _mapping(
                        candidate_metadata.get("unified_react_delta")
                    ).get("state_update", {})
                )
            )
            if lifecycle != "protocol_correction" and not is_unified_outer_correction:
                continue
            if correction_parent_id(candidate) not in reachable_rejected_ids:
                continue
            if not rejected(candidate):
                return True
            candidate_id = attempt_id(candidate)
            if candidate_id:
                reachable_rejected_ids.add(candidate_id)
        return False

    def eventually_reaches_bounded_fallback(
        rejected_index: int,
        rejected_step: Mapping[str, Any],
    ) -> bool:
        root_id = attempt_id(rejected_step)
        if not root_id:
            return False
        stage = str(rejected_step.get("stage", ""))
        for candidate in steps[rejected_index + 1 :]:
            if str(candidate.get("stage", "")) != stage:
                continue
            candidate_metadata = _mapping(candidate.get("metadata"))
            if not candidate_metadata.get(
                "protocol_correction_exhaustion_boundary"
            ):
                continue
            resolved = {
                str(item).strip()
                for item in candidate_metadata.get(
                    "resolved_rejection_request_ids", []
                )
                or []
                if str(item).strip()
            }
            if root_id in resolved:
                return True
        return False

    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        if str(step.get("action_type", "")) in {
            "planning_revision",
            "evidence_decision_revision",
        }:
            continue
        if not rejected(step):
            continue
        corrected = eventually_corrected(index, step)
        bounded_fallback = eventually_reaches_bounded_fallback(index, step)
        reason = str(metadata.get("rejection_reason", "")).strip()
        if not reason:
            try:
                payload = json.loads(str(step.get("tool_result", "")))
                reason = str(_mapping(payload).get("error", "")).strip()
            except (TypeError, json.JSONDecodeError):
                reason = ""
        route_control = bool(metadata.get("duplicate_tool_call")) or (
            bool(metadata.get("invalid_question_id"))
            and "already resolved" in reason.casefold()
        )
        category = (
            CORRECTION
            if corrected
            else BOUNDED_FALLBACK
            if bounded_fallback
            else ROUTE_CONTROL
            if route_control
            else _rejection_category(step)
        )
        if category == CORRECTION:
            corrected_count += 1
        elif category == BOUNDED_FALLBACK:
            bounded_fallback_count += 1
        elif category == ROUTE_CONTROL:
            route_control_count += 1
        elif category == FORMAT:
            format_count += 1
        elif category == SCHEDULER:
            scheduler_count += 1
        else:
            protocol_count += 1
        _issue(
            report,
            (
                "PROTOCOL_CORRECTION"
                if category == CORRECTION
                else "PROTOCOL_EXHAUSTION_BOUNDARY"
                if category == BOUNDED_FALLBACK
                else "ROUTE_CONTROL_REJECTION"
                if category == ROUTE_CONTROL
                else "TOOL_ARGUMENT_FORMAT_ERROR"
                if category == FORMAT
                else "SCHEDULER_REJECTION"
                if category == SCHEDULER
                else "PROTOCOL_REJECTION"
            ),
            reason or f"{step.get('action_type', 'rejected')} step",
            category=category,
            location=_step_label(index, step),
        )
    report.stats["scheduler_rejections"] = scheduler_count
    report.stats["protocol_rejections"] = protocol_count
    report.stats["route_control_rejections"] = route_control_count
    report.stats["successful_protocol_corrections"] = corrected_count
    report.stats["bounded_protocol_fallbacks"] = bounded_fallback_count
    report.stats["tool_argument_format_errors"] = format_count




def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    location_prefix: str,
    report: TraceReport,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identifier = str(row.get(id_field, "")).strip()
        location = (
            _location(location_prefix, identifier)
            if identifier
            else f"{location_prefix}[{index}]"
        )
        if not identifier:
            _issue(
                report,
                "IMAGE_ONLY_ID_MISSING",
                f"{id_field} must be non-empty",
                location=location,
            )
            continue
        if identifier in indexed:
            _issue(
                report,
                "IMAGE_ONLY_ID_DUPLICATE",
                f"duplicate {id_field} {identifier!r}",
                location=location,
            )
            continue
        indexed[identifier] = row
    return indexed


def _audit_image_only_interaction_chains(
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    main_chain_stages = {
        "image_only_planning",
        "image_only_investigation",
        "image_only_evidence_decision",
        "image_only_reflection",
        "image_only_judgment",
    }
    stage_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")) in main_chain_stages
    ]
    native_step_count = sum(
        bool(_mapping(step.get("metadata")).get("native_interactions"))
        for _, step in stage_steps
    )
    if not native_step_count:
        _issue(
            report,
            "IMAGE_ONLY_INTERACTIONS_MISSING",
            "image-only trace contains no native main-chain interactions",
            location="state.all_steps",
        )
        return

    previous: str | None = None
    root_count = 0
    for index, step in stage_steps:
        metadata = _mapping(step.get("metadata"))
        if not metadata.get("native_interactions"):
            continue
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        parent_recorded = "previous_interaction_id" in metadata
        raw_parent = metadata.get("previous_interaction_id")
        parent = "" if raw_parent is None else str(raw_parent).strip()
        location = _step_label(index, step)
        if not interaction_id:
            _issue(
                report,
                "INTERACTION_ID_MISSING",
                "image-only investigation interaction_id is missing",
                location=location,
            )
            continue
        if not parent_recorded:
            _issue(
                report,
                "INTERACTION_PARENT_UNRECORDED",
                "previous_interaction_id is not recorded",
                location=location,
            )
        if previous is None:
            root_count += 1
            if parent:
                _issue(
                    report,
                    "INTERACTION_CHAIN_ROOT_INVALID",
                    f"image-only main-chain root must have null parent, got {parent!r}",
                    location=location,
                )
            if str(step.get("stage", "")) != "image_only_planning":
                _issue(
                    report,
                    "INTERACTION_CHAIN_ROOT_STAGE_INVALID",
                    "image-only main chain must begin at Target Planning",
                    location=location,
                )
        elif parent != previous:
            _issue(
                report,
                "INTERACTION_CHAIN_BROKEN",
                f"expected previous_interaction_id {previous!r}, got {parent!r}",
                location=location,
            )
        previous = interaction_id

    report.stats["image_only_interaction_steps"] = native_step_count
    report.stats["image_only_interaction_segments"] = root_count


def _audit_discrepancy_interaction_chains(
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    stages = {
        "image_account_planning",
        "image_only_discrepancy_investigation",
        "image_only_discrepancy_decision",
        "image_only_discrepancy_judgment",
    }
    native = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")) in stages
        and (
            _mapping(step.get("metadata")).get("native_interactions")
            or _mapping(step.get("metadata")).get("native_chat_completions")
        )
    ]
    if not native:
        _issue(
            report,
            "V4_INTERACTIONS_MISSING",
            "v4 trace contains no native main-chain interactions",
            location="state.all_steps",
        )
        return
    # Qwen's local OpenAI-compatible path uses independent Chat Completions
    # requests instead of Gemini Interaction IDs.  These are intentionally
    # standalone stage requests; audit their durable request identity and do not
    # demand an Interaction parent chain that the provider cannot supply.
    if not any(
        _mapping(step.get("metadata")).get("native_interactions")
        for _, step in native
    ):
        missing = [
            (index, step)
            for index, step in native
            if not str(
                _mapping(step.get("metadata")).get("context_request_id", "")
            ).strip()
        ]
        for index, step in missing:
            _issue(
                report,
                "CONTEXT_REQUEST_ID_MISSING",
                "Chat Completions stage step lacks context_request_id",
                location=_step_label(index, step),
            )
        report.stats["v4_interaction_steps"] = len(native)
        report.stats["v4_interaction_segments"] = len(
            {
                str(
                    _mapping(step.get("metadata")).get(
                        "context_request_id", ""
                    )
                ).strip()
                for _, step in native
                if str(
                    _mapping(step.get("metadata")).get(
                        "context_request_id", ""
                    )
                ).strip()
            }
        )
        return
    previous = ""
    previous_stage = ""
    root_count = 0
    uses_explicit_lifecycles = any(
        str(
            _mapping(step.get("metadata")).get(
                "interaction_lifecycle_kind", ""
            )
        ).strip()
        for _, step in native
    )
    for position, (index, step) in enumerate(native):
        metadata = _mapping(step.get("metadata"))
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        raw_parent = metadata.get("previous_interaction_id")
        parent = "" if raw_parent is None else str(raw_parent).strip()
        location = _step_label(index, step)
        if not interaction_id:
            _issue(
                report,
                "INTERACTION_ID_MISSING",
                "v4 main-chain interaction_id is missing",
                location=location,
            )
            continue
        stage = str(step.get("stage", ""))
        lifecycle = str(
            metadata.get("interaction_lifecycle_kind", "")
        ).strip()
        if not parent:
            root_count += 1
        if position == 0:
            if parent:
                _issue(
                    report,
                    "INTERACTION_CHAIN_ROOT_INVALID",
                    "v4 Image Account Planning root must have a null parent",
                    location=location,
                )
            if str(step.get("stage", "")) != "image_account_planning":
                _issue(
                    report,
                    "INTERACTION_CHAIN_ROOT_STAGE_INVALID",
                    "v4 main chain must begin at Image Account Planning",
                    location=location,
                )
        elif not uses_explicit_lifecycles:
            if parent != previous:
                _issue(
                    report,
                    "INTERACTION_CHAIN_BROKEN",
                    f"expected previous_interaction_id {previous!r}, got {parent!r}",
                    location=location,
                )
        elif stage == previous_stage:
            if parent and parent != previous:
                _issue(
                    report,
                    "INTERACTION_CHAIN_BROKEN",
                    f"same-stage short chain expected parent {previous!r}, got {parent!r}",
                    location=location,
                )
        elif parent or lifecycle not in {
            "standalone_request",
            "tool_roundtrip",
        }:
            _issue(
                report,
                "INTERACTION_STAGE_LEAK",
                "cross-stage request must be a standalone interaction root",
                location=location,
            )
        previous = interaction_id
        previous_stage = stage
    report.stats["v4_interaction_steps"] = len(native)
    report.stats["v4_interaction_segments"] = root_count


def _v4_claim_has_directional_chain(
    *,
    claim_id: str,
    stance: str,
    selected_evidence_ids: set[str],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    task_by_id: Mapping[str, Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    finding_by_id: Mapping[str, Mapping[str, Any]],
    successful_calls: set[str],
    selected_finding_ids: set[str] | None = None,
) -> bool:
    claim = claim_by_id.get(claim_id)
    if claim is None:
        return False
    claim_fact_id = str(claim.get("fact_id", "")).strip()
    for finding_id, finding in finding_by_id.items():
        if selected_finding_ids is not None and finding_id not in selected_finding_ids:
            continue
        if str(finding.get("stance", "")).strip() != stance:
            continue
        task_id = str(finding.get("task_id", "")).strip()
        task = task_by_id.get(task_id)
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
            evidence = evidence_by_id.get(evidence_id)
            if (
                evidence_id in selected_evidence_ids
                and evidence is not None
                and str(evidence.get("task_id", "")).strip() == task_id
                and evidence_is_qualified_for_stance(evidence, stance)
                and str(evidence.get("function_call_id", "")).strip()
                in successful_calls
            ):
                return True
    return False


def _audit_discrepancy_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    investigation = _mapping(state.get("investigation_state"))
    if not investigation:
        _issue(
            report,
            "V4_STATE_MISSING",
            "v4 trace must contain state.investigation_state",
            location="state.investigation_state",
        )
        return
    if investigation.get("core_verdict_fact_id"):
        _issue(
            report,
            "V4_LEGACY_CORE_ACTIVE",
            "v4 main state must not select core_verdict_fact_id",
            location="state.investigation_state.core_verdict_fact_id",
        )

    facts = _rows(investigation.get("facts"))
    tasks = _rows(investigation.get("tasks"))
    discoveries = _rows(investigation.get("discoveries"))
    evidence = _rows(investigation.get("evidence"))
    findings = _rows(investigation.get("findings"))
    claims = target_fact_rows(investigation)
    hypotheses = _rows(investigation.get("search_hypotheses"))
    assessments = _rows(investigation.get("claim_assessments"))
    discrepancies = _rows(investigation.get("material_discrepancies"))
    decisions = _rows(investigation.get("discrepancy_decisions"))
    audits = _rows(investigation.get("discrepancy_coverage_audits"))

    fact_by_id = _unique_index(
        facts,
        id_field="fact_id",
        location_prefix="state.investigation_state.facts",
        report=report,
    )
    task_by_id = _unique_index(
        tasks,
        id_field="task_id",
        location_prefix="state.investigation_state.tasks",
        report=report,
    )
    discovery_by_id = _unique_index(
        discoveries,
        id_field="discovery_id",
        location_prefix="state.investigation_state.discoveries",
        report=report,
    )
    evidence_by_id = _unique_index(
        evidence,
        id_field="evidence_id",
        location_prefix="state.investigation_state.evidence",
        report=report,
    )
    finding_by_id = _unique_index(
        findings,
        id_field="finding_id",
        location_prefix="state.investigation_state.findings",
        report=report,
    )
    claim_by_id = _unique_index(
        claims,
        id_field="claim_id",
        location_prefix="state.investigation_state.target_facts",
        report=report,
    )
    hypothesis_by_id = _unique_index(
        hypotheses,
        id_field="hypothesis_id",
        location_prefix="state.investigation_state.search_hypotheses",
        report=report,
    )
    assessment_by_id = _unique_index(
        assessments,
        id_field="assessment_id",
        location_prefix="state.investigation_state.claim_assessments",
        report=report,
    )
    discrepancy_by_id = _unique_index(
        discrepancies,
        id_field="discrepancy_id",
        location_prefix="state.investigation_state.material_discrepancies",
        report=report,
    )
    decision_by_id = _unique_index(
        decisions,
        id_field="decision_id",
        location_prefix="state.investigation_state.discrepancy_decisions",
        report=report,
    )
    _unique_index(
        audits,
        id_field="audit_id",
        location_prefix="state.investigation_state.discrepancy_coverage_audits",
        report=report,
    )

    if not claim_by_id:
        _issue(report, "V4_CLAIMS_MISSING", "v4 trace requires ImageClaims")
    high_claim_ids = {
        claim_id
        for claim_id, claim in claim_by_id.items()
        if str(claim.get("salience", "")) == "high"
    }
    if not high_claim_ids:
        _issue(
            report,
            "V4_HIGH_SALIENCE_CLAIM_MISSING",
            "v4 Image Account requires a high-salience ImageClaim",
        )
    for claim_id, claim in claim_by_id.items():
        location = _location("state.investigation_state.target_facts", claim_id)
        claim_fact_id = str(claim.get("fact_id", "")).strip()
        if claim_fact_id not in fact_by_id:
            _issue(
                report,
                "V4_CLAIM_FACT_UNKNOWN",
                f"ImageClaim cites unknown fact {claim_fact_id!r}",
                location=location,
            )
        anchor_ids = {str(item) for item in claim.get("anchor_fact_ids", []) or []}
        invalid = sorted(
            item
            for item in anchor_ids
            if item not in fact_by_id
            or str(_mapping(fact_by_id[item].get("origin")).get("type", ""))
            not in {"input_image", "ocr"}
        )
        if not anchor_ids or invalid:
            _issue(
                report,
                "V4_CLAIM_ANCHOR_INVALID",
                "ImageClaim anchors must be existing pixel/OCR VisualFacts",
                location=location,
            )

    for hypothesis_id, hypothesis in hypothesis_by_id.items():
        location = _location(
            "state.investigation_state.search_hypotheses",
            hypothesis_id,
        )
        owned_claim_ids = {
            str(item) for item in hypothesis.get("claim_ids", []) or []
        }
        if not owned_claim_ids or not owned_claim_ids <= set(claim_by_id):
            _issue(
                report,
                "V4_HYPOTHESIS_CLAIM_INVALID",
                "SearchHypothesis must reference existing ImageClaims",
                location=location,
            )

    for discovery_id, discovery in discovery_by_id.items():
        location = _location(
            "state.investigation_state.discoveries",
            discovery_id,
        )
        task_id = str(discovery.get("task_id", "")).strip()
        if task_id not in task_by_id:
            _issue(
                report,
                "V4_DISCOVERY_TASK_UNKNOWN",
                f"Discovery references unknown task {task_id!r}",
                location=location,
            )
        if str(discovery.get("promoted_evidence_id", "") or "").strip():
            _issue(
                report,
                "V4_DISCOVERY_PROMOTED_IN_PLACE",
                "Discovery must remain separate from Evidence",
                location=location,
            )
        task_id = str(hypothesis.get("task_id", "")).strip()
        task = task_by_id.get(task_id)
        if (
            task is None
            or str(task.get("hypothesis_id", "")).strip() != hypothesis_id
            or {str(item) for item in task.get("claim_ids", []) or []}
            != owned_claim_ids
        ):
            _issue(
                report,
                "V4_HYPOTHESIS_TASK_OWNERSHIP_INVALID",
                "SearchHypothesis task must preserve claim/hypothesis ownership",
                location=location,
            )

    _audit_evidence_calls(
        evidence,
        steps,
        report,
        location_prefix="state.investigation_state.evidence",
        tool_field="tool_name",
        stat_key="v4_successful_evidence_calls",
    )
    successful_calls = {
        str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
        for step in steps
        if _parse_successful_tool_step(step)
        and str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
    }
    for evidence_id, item in evidence_by_id.items():
        location = _location("state.investigation_state.evidence", evidence_id)
        task_id = str(item.get("task_id", "")).strip()
        task = task_by_id.get(task_id)
        evidence_fact_ids = {
            str(value) for value in item.get("fact_ids", []) or []
        }
        if task is None:
            _issue(
                report,
                "V4_EVIDENCE_TASK_UNKNOWN",
                f"Evidence references unknown task {task_id!r}",
                location=location,
            )
        elif not evidence_fact_ids or not evidence_fact_ids <= {
            str(value) for value in task.get("fact_ids", []) or []
        }:
            _issue(
                report,
                "V4_EVIDENCE_FACT_OWNERSHIP_INVALID",
                "Evidence fact_ids must be owned by its ResearchTask",
                location=location,
            )
        stance = str(item.get("stance", "")).strip()
        if stance in {"support", "refute"} and not evidence_direction_is_coherent(
            item,
            stance,
        ):
            _issue(
                report,
                "V4_EVIDENCE_DIRECTION_INCOHERENT",
                "Evidence stance conflicts with its recorded comparison metadata",
                location=location,
            )

    for finding_id, finding in finding_by_id.items():
        location = _location("state.investigation_state.findings", finding_id)
        task_id = str(finding.get("task_id", "")).strip()
        task = task_by_id.get(task_id)
        finding_fact_ids = {
            str(value) for value in finding.get("fact_ids", []) or []
        }
        if task is None:
            _issue(
                report,
                "V4_FINDING_TASK_UNKNOWN",
                f"Finding references unknown task {task_id!r}",
                location=location,
            )
            continue
        if not finding_fact_ids or not finding_fact_ids <= {
            str(value) for value in task.get("fact_ids", []) or []
        }:
            _issue(
                report,
                "V4_FINDING_FACT_OWNERSHIP_INVALID",
                "Finding fact_ids must be owned by its ResearchTask",
                location=location,
            )
        finding_stance = str(finding.get("stance", "")).strip()
        finding_evidence_ids = {
            str(value) for value in finding.get("evidence_ids", []) or []
        }
        is_source_visual_composite = (
            COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY
            in {
                str(value)
                for value in finding.get("source_family_ids", []) or []
            }
        )
        composite_source_seen = False
        composite_visual_seen = False
        if not finding_evidence_ids:
            _issue(
                report,
                "V4_FINDING_EVIDENCE_MISSING",
                "Finding must cite Evidence",
                location=location,
            )
        for evidence_id in finding_evidence_ids:
            evidence_row = evidence_by_id.get(evidence_id)
            if evidence_row is None:
                _issue(
                    report,
                    "V4_FINDING_EVIDENCE_UNKNOWN",
                    f"Finding references unknown Evidence {evidence_id!r}",
                    location=location,
                )
                continue
            evidence_task_id = str(evidence_row.get("task_id", "")).strip()
            composite_visual_evidence = (
                is_source_visual_composite
                and _composite_visual_evidence_is_eligible(
                    evidence_row,
                    finding=finding,
                    task_by_id=task_by_id,
                )
            )
            if evidence_task_id != task_id and not (
                composite_visual_evidence
            ):
                _issue(
                    report,
                    "V4_FINDING_EVIDENCE_OWNERSHIP_INVALID",
                    "Finding Evidence must belong to its ResearchTask or an "
                    "explicit claim-owned visual Evidence",
                    location=location,
                )
                continue
            if is_source_visual_composite:
                if (
                    evidence_task_id == task_id
                    and str(evidence_row.get("evidence_kind", "")).strip()
                    == "web_span"
                    and str(evidence_row.get("claim_binding", "")).strip()
                    == "source_assertion"
                ):
                    composite_source_seen = True
                elif (
                    composite_visual_evidence
                ):
                    composite_visual_seen = True
                else:
                    _issue(
                        report,
                        "V4_COMPOSITE_FINDING_EVIDENCE_INVALID",
                        "Source-visual composite Finding must pair source "
                        "Evidence with claim-owned neutral pixel Evidence",
                        location=location,
                    )
            elif str(evidence_row.get("stance", "")).strip() != finding_stance:
                _issue(
                    report,
                    "V4_FINDING_EVIDENCE_STANCE_MISMATCH",
                    "Finding stance must match its cited Evidence",
                    location=location,
                )
        if is_source_visual_composite and (
            finding_stance != "refute"
            or not composite_source_seen
            or not composite_visual_seen
        ):
            _issue(
                report,
                "V4_COMPOSITE_FINDING_CHAIN_INCOMPLETE",
                "Source-visual composite Finding requires a refute stance, "
                "source Evidence, and claim-owned neutral pixel Evidence",
                location=location,
            )

    latest_assessment_by_claim: dict[str, Mapping[str, Any]] = {}
    for assessment_id, assessment in assessment_by_id.items():
        location = _location(
            "state.investigation_state.claim_assessments",
            assessment_id,
        )
        claim_id = str(assessment.get("claim_id", "")).strip()
        selected = {str(item) for item in assessment.get("evidence_ids", []) or []}
        if claim_id not in claim_by_id:
            _issue(
                report,
                "V4_ASSESSMENT_CLAIM_UNKNOWN",
                f"ClaimAssessment cites unknown claim {claim_id!r}",
                location=location,
            )
        if not selected <= set(evidence_by_id):
            _issue(
                report,
                "V4_ASSESSMENT_EVIDENCE_UNKNOWN",
                "ClaimAssessment cites unknown Evidence",
                location=location,
            )
        if claim_id in claim_by_id:
            latest_assessment_by_claim[claim_id] = assessment
            for evidence_id in selected & set(evidence_by_id):
                task = task_by_id.get(
                    str(evidence_by_id[evidence_id].get("task_id", "")).strip()
                )
                if task is None or claim_id not in {
                    str(item) for item in task.get("claim_ids", []) or []
                }:
                    _issue(
                        report,
                        "V4_ASSESSMENT_EVIDENCE_OWNERSHIP_INVALID",
                        "ClaimAssessment Evidence is outside the claim's tasks",
                        location=location,
                    )
            for stance in required_assessment_stances(
                str(assessment.get("assessment", ""))
            ):
                if not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance=stance,
                    selected_evidence_ids=selected,
                    claim_by_id=claim_by_id,
                    task_by_id=task_by_id,
                    evidence_by_id=evidence_by_id,
                    finding_by_id=finding_by_id,
                    successful_calls=successful_calls,
                ):
                    _issue(
                        report,
                        "V4_ASSESSMENT_EVIDENCE_DIRECTION_INVALID",
                        f"{assessment.get('assessment')} assessment requires an "
                        f"owned qualified {stance} Finding -> Evidence chain",
                        location=location,
                    )

    for claim_id, claim in claim_by_id.items():
        latest = latest_assessment_by_claim.get(claim_id)
        if latest is None:
            continue
        expected_status = {
            "supported": "supported",
            "refuted": "refuted",
            "conflicted": "conflicted",
            "insufficient": "unresolved",
        }.get(str(latest.get("assessment", "")), "")
        if str(claim.get("status", "")) != expected_status:
            _issue(
                report,
                "V4_CLAIM_ASSESSMENT_STATUS_MISMATCH",
                "ImageClaim status must match its latest ClaimAssessment",
                location=_location(
                    "state.investigation_state.target_facts",
                    claim_id,
                ),
            )

    for discrepancy_id, discrepancy in discrepancy_by_id.items():
        location = _location(
            "state.investigation_state.material_discrepancies",
            discrepancy_id,
        )
        affected = {
            str(item) for item in discrepancy.get("affected_claim_ids", []) or []
        }
        anchors = {
            str(item)
            for item in discrepancy.get("visual_anchor_fact_ids", []) or []
        }
        evidence_ids = {
            str(item) for item in discrepancy.get("evidence_ids", []) or []
        }
        if not affected or not affected <= set(claim_by_id):
            _issue(
                report,
                "V4_DISCREPANCY_CLAIM_INVALID",
                "MaterialDiscrepancy must reference existing ImageClaims",
                location=location,
            )
        if not evidence_ids or not evidence_ids <= set(evidence_by_id):
            _issue(
                report,
                "V4_DISCREPANCY_EVIDENCE_INVALID",
                "MaterialDiscrepancy must cite existing Evidence",
                location=location,
            )
        for claim_id in affected & set(claim_by_id):
            claim_anchors = {
                str(item)
                for item in claim_by_id[claim_id].get("anchor_fact_ids", []) or []
            }
            if not anchors & claim_anchors:
                _issue(
                    report,
                    "V4_DISCREPANCY_ANCHOR_MISALIGNED",
                    f"Discrepancy is not visibly anchored to claim {claim_id!r}",
                    location=location,
                )
            if (
                str(discrepancy.get("materiality", "")) == "decisive"
                and str(discrepancy.get("status", "")) == "established"
                and not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance="refute",
                    selected_evidence_ids=evidence_ids,
                    claim_by_id=claim_by_id,
                    task_by_id=task_by_id,
                    evidence_by_id=evidence_by_id,
                    finding_by_id=finding_by_id,
                    successful_calls=successful_calls,
                )
            ):
                _issue(
                    report,
                    "V4_DISCREPANCY_EVIDENCE_DIRECTION_INVALID",
                    f"Established decisive discrepancy lacks an owned qualified "
                    f"refute chain for claim {claim_id!r}",
                    location=location,
                )
            if not any(
                claim_id
                in {
                    str(item)
                    for item in task_by_id.get(
                        str(evidence_by_id[evidence_id].get("task_id", "")),
                        {},
                    ).get("claim_ids", [])
                    or []
                }
                for evidence_id in evidence_ids & set(evidence_by_id)
            ):
                _issue(
                    report,
                    "V4_DISCREPANCY_EVIDENCE_OWNERSHIP_INVALID",
                    f"Discrepancy Evidence is not owned by claim {claim_id!r}",
                    location=location,
                )

    for decision_id, decision in decision_by_id.items():
        location = _location(
            "state.investigation_state.discrepancy_decisions",
            decision_id,
        )
        reviewed = {
            str(item) for item in decision.get("reviewed_evidence_ids", []) or []
        }
        output = _mapping(decision.get("output"))
        cited = {
            str(item)
            for row in _rows(output.get("claim_assessments"))
            for item in row.get("selected_evidence_ids", []) or []
        }
        cited.update(
            str(item)
            for item in _mapping(output.get("material_discrepancy")).get(
                "evidence_ids",
                [],
            )
            or []
        )
        if not reviewed <= set(evidence_by_id) or not cited <= reviewed:
            _issue(
                report,
                "V4_DECISION_EVIDENCE_SCOPE_INVALID",
                "Discrepancy Decision may cite only reviewed existing Evidence",
                location=location,
            )

    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    verdict = str(trace.get("verdict", judgment.get("verdict", ""))).strip()
    decision_mode = str(
        basis.get("decision_mode", "evidence_determined")
        or "evidence_determined"
    )
    if str(basis.get("policy_rule_id", "")) != "discrepancy-first-v4":
        _issue(
            report,
            "V4_VERDICT_BASIS_POLICY_INVALID",
            "v4 verdict_basis must use discrepancy-first-v4",
            location="verdict_basis.policy_rule_id",
        )
    if (
        decision_mode == "evidence_determined"
        and verdict == "fake"
        and not basis.get("discrepancy_ids")
    ):
        _issue(
            report,
            "V4_FAKE_BASIS_DISCREPANCY_MISSING",
            "fake verdict requires a selected MaterialDiscrepancy",
            location="verdict_basis.discrepancy_ids",
        )
    basis_checks = (
        ("claim", "claim_ids", claim_by_id),
        ("discrepancy", "discrepancy_ids", discrepancy_by_id),
        ("evidence", "evidence_ids", evidence_by_id),
        ("finding", "finding_ids", finding_by_id),
        ("visual anchor", "visual_anchor_fact_ids", fact_by_id),
    )
    for name, field, known in basis_checks:
        selected = {str(item) for item in basis.get(field, []) or []}
        if not selected <= set(known):
            _issue(
                report,
                f"V4_VERDICT_{field.upper()}_UNKNOWN",
                f"v4 verdict basis cites unknown {name} IDs",
                location=f"verdict_basis.{field}",
            )
        judgment_field = f"selected_{field}"
        if {str(item) for item in judgment.get(judgment_field, []) or []} != selected:
            _issue(
                report,
                "V4_JUDGMENT_BASIS_MISMATCH",
                f"Judgment {judgment_field} must match verdict_basis",
                location=f"judgment.{judgment_field}",
            )
    basis_claim_ids = {str(item) for item in basis.get("claim_ids", []) or []}
    basis_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    basis_finding_ids = {
        str(item) for item in basis.get("finding_ids", []) or []
    }
    if basis_evidence_ids & set(discovery_by_id):
        _issue(
            report,
            "V4_DISCOVERY_USED_AS_VERDICT_EVIDENCE",
            "Discovery IDs cannot appear in verdict_basis.evidence_ids",
            location="verdict_basis.evidence_ids",
        )
    if decision_mode == "evidence_determined" and verdict in {"fake", "real"}:
        expected_stance = "refute" if verdict == "fake" else "support"
        if not basis_evidence_ids or not basis_finding_ids:
            _issue(
                report,
                "V4_VERDICT_CHAIN_MISSING",
                f"{verdict} verdict requires Finding -> Evidence provenance",
                location="verdict_basis",
            )
        for claim_id in basis_claim_ids & set(claim_by_id):
            if not _v4_claim_has_directional_chain(
                claim_id=claim_id,
                stance=expected_stance,
                selected_evidence_ids=basis_evidence_ids,
                selected_finding_ids=basis_finding_ids,
                claim_by_id=claim_by_id,
                task_by_id=task_by_id,
                evidence_by_id=evidence_by_id,
                finding_by_id=finding_by_id,
                successful_calls=successful_calls,
            ):
                _issue(
                    report,
                    "V4_VERDICT_CHAIN_INVALID",
                    f"{verdict} basis lacks an owned qualified chain for "
                    f"ImageClaim {claim_id!r}",
                    location="verdict_basis",
                )
        linked_basis_evidence = {
            str(evidence_id)
            for finding_id in basis_finding_ids & set(finding_by_id)
            for evidence_id in finding_by_id[finding_id].get("evidence_ids", [])
            or []
        }
        if not basis_evidence_ids <= linked_basis_evidence:
            _issue(
                report,
                "V4_VERDICT_EVIDENCE_WITHOUT_FINDING",
                "Every selected verdict Evidence must be linked by a selected Finding",
                location="verdict_basis",
            )
    elif decision_mode == "bounded_binary_judgment":
        action_count = int(investigation.get("action_count", 0) or 0)
        stop_reason = str(investigation.get("stop_reason", ""))
        unresolved_gaps = [
            str(item).strip()
            for item in basis.get("unresolved_gaps", []) or []
            if str(item).strip()
        ]
        if action_count <= 0:
            _issue(
                report,
                "V4_BOUNDED_JUDGMENT_WITHOUT_INVESTIGATION",
                "bounded binary Judgment requires at least one accepted investigation action",
                location="state.investigation_state.action_count",
            )
        if stop_reason not in {
            "meaningful_routes_exhausted",
            "hard_budget_exhausted",
        }:
            _issue(
                report,
                "V4_BOUNDED_JUDGMENT_STOP_INVALID",
                "bounded binary Judgment requires a finite-information terminal stop",
                location="state.investigation_state.stop_reason",
            )
        if not basis_claim_ids or not unresolved_gaps:
            _issue(
                report,
                "V4_BOUNDED_JUDGMENT_GAPS_MISSING",
                "bounded binary Judgment must preserve its claims and unresolved gaps",
                location="verdict_basis",
            )
        linked_basis_evidence = {
            str(evidence_id)
            for finding_id in basis_finding_ids & set(finding_by_id)
            for evidence_id in finding_by_id[finding_id].get("evidence_ids", [])
            or []
        }
        if not basis_evidence_ids <= linked_basis_evidence:
            _issue(
                report,
                "V4_VERDICT_EVIDENCE_WITHOUT_FINDING",
                "Every selected bounded-judgment Evidence must be linked by a selected Finding",
                location="verdict_basis",
            )
    else:
        _issue(
            report,
            "V4_VERDICT_DECISION_MODE_INVALID",
            f"unknown v4 verdict decision_mode {decision_mode!r}",
            location="verdict_basis.decision_mode",
        )
    if str(judgment.get("verdict", "")) != verdict:
        _issue(
            report,
            "V4_JUDGMENT_VERDICT_MISMATCH",
            "Judgment verdict must match compiled verdict",
            location="judgment.verdict",
        )

    terminal_stop_reason = str(investigation.get("stop_reason", ""))
    terminal_audits = [
        item
        for item in audits
        if (
            item.get("complete") is True
            and str(item.get("stop_reason", "")) == "verdict_determined"
        )
        or str(item.get("stop_reason", ""))
        in {
            "meaningful_routes_exhausted",
            "hard_budget_exhausted",
        }
    ]
    if not terminal_audits:
        _issue(
            report,
            "V4_TERMINAL_COVERAGE_MISSING",
            "v4 successful trace requires an auditable terminal Coverage boundary",
        )
    else:
        terminal_action_count = int(terminal_audits[-1].get("action_count", 0) or 0)
        action_steps = [
            step
            for step in steps
            if str(step.get("stage", ""))
            in {
                "image_only_discrepancy_investigation",
                "image_only_visual_reinspection",
            }
            and step.get("action_type") == "tool_call"
        ]
        action_count = int(investigation.get("action_count", 0) or 0)
        if action_count != len(action_steps):
            _issue(
                report,
                "V4_ACTION_COUNT_MISMATCH",
                f"action_count={action_count}, but trace records "
                f"{len(action_steps)} investigation tool calls",
                location="state.investigation_state.action_count",
            )
        if action_count > 24:
            _issue(
                report,
                "V4_ACTION_BUDGET_EXCEEDED",
                f"v4 investigation used {action_count} actions; maximum is 24",
                location="state.investigation_state.action_count",
            )
        if terminal_action_count != action_count:
            _issue(
                report,
                "V4_TERMINAL_ACTION_COUNT_MISMATCH",
                "Terminal Coverage action_count must match final v4 action_count",
                location="state.investigation_state.discrepancy_coverage_audits",
            )
        if len(action_steps) > terminal_action_count:
            _issue(
                report,
                "POST_DETERMINATION_ACTION",
                "v4 trace contains a tool action after terminal Coverage",
                category=SCHEDULER,
            )

    report.stats.update(
        {
            "target_facts": len(claims),
            "search_hypotheses": len(hypotheses),
            "v4_discoveries": len(discoveries),
            "claim_assessments": len(assessments),
            "material_discrepancies": len(discrepancies),
            "discrepancy_decisions": len(decisions),
            "v4_actions": int(investigation.get("action_count", 0) or 0),
            "v4_stop_reason": terminal_stop_reason,
            "v4_progress_events": len(
                _rows(investigation.get("progress_events"))
            ),
        }
    )
    _audit_discrepancy_interaction_chains(steps, report)


def _audit_image_only_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    if str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    ) != "reinspect-v2":
        _issue(
            report,
            "IMAGE_ONLY_POLICY_MISMATCH",
            "image-only canonical trace must use decision_policy_version=reinspect-v2",
            location="decision_policy_version",
        )

    investigation = _mapping(state.get("investigation_state"))
    if not investigation:
        _issue(
            report,
            "IMAGE_ONLY_STATE_MISSING",
            "image-only trace must contain state.investigation_state",
            location="state.investigation_state",
        )
        return

    collection_names = (
        "entities",
        "facts",
        "tasks",
        "retrieval_anchors",
        "discoveries",
        "evidence",
        "findings",
        "failures",
        "reflections",
        "query_replans",
        "evidence_decisions",
        "visual_reinspections",
        "coverage_audits",
    )
    for name in collection_names:
        if not isinstance(investigation.get(name), list):
            _issue(
                report,
                "IMAGE_ONLY_COLLECTION_INVALID",
                f"state.investigation_state.{name} must be an array",
                location=f"state.investigation_state.{name}",
            )

    entities = _rows(investigation.get("entities"))
    facts = _rows(investigation.get("facts"))
    tasks = _rows(investigation.get("tasks"))
    anchors = _rows(investigation.get("retrieval_anchors"))
    discoveries = _rows(investigation.get("discoveries"))
    evidence = _rows(investigation.get("evidence"))
    findings = _rows(investigation.get("findings"))
    failures = _rows(investigation.get("failures"))
    reflections = _rows(investigation.get("reflections"))
    query_replans = _rows(investigation.get("query_replans"))
    evidence_decisions = _rows(investigation.get("evidence_decisions"))
    visual_reinspections = _rows(
        investigation.get("visual_reinspections")
    )
    coverage_audits = _rows(investigation.get("coverage_audits"))

    entity_by_id = _unique_index(
        entities,
        id_field="entity_id",
        location_prefix="state.investigation_state.entities",
        report=report,
    )
    fact_by_id = _unique_index(
        facts,
        id_field="fact_id",
        location_prefix="state.investigation_state.facts",
        report=report,
    )
    task_by_id = _unique_index(
        tasks,
        id_field="task_id",
        location_prefix="state.investigation_state.tasks",
        report=report,
    )
    anchor_by_id = _unique_index(
        anchors,
        id_field="anchor_id",
        location_prefix="state.investigation_state.retrieval_anchors",
        report=report,
    )
    discovery_by_id = _unique_index(
        discoveries,
        id_field="discovery_id",
        location_prefix="state.investigation_state.discoveries",
        report=report,
    )
    evidence_by_id = _unique_index(
        evidence,
        id_field="evidence_id",
        location_prefix="state.investigation_state.evidence",
        report=report,
    )
    finding_by_id = _unique_index(
        findings,
        id_field="finding_id",
        location_prefix="state.investigation_state.findings",
        report=report,
    )
    failure_by_id = _unique_index(
        failures,
        id_field="failure_id",
        location_prefix="state.investigation_state.failures",
        report=report,
    )
    _unique_index(
        reflections,
        id_field="reflection_id",
        location_prefix="state.investigation_state.reflections",
        report=report,
    )
    query_replan_by_id = _unique_index(
        query_replans,
        id_field="replan_id",
        location_prefix="state.investigation_state.query_replans",
        report=report,
    )
    decision_by_id = _unique_index(
        evidence_decisions,
        id_field="decision_id",
        location_prefix="state.investigation_state.evidence_decisions",
        report=report,
    )
    visual_reinspection_by_id = _unique_index(
        visual_reinspections,
        id_field="visual_question_id",
        location_prefix="state.investigation_state.visual_reinspections",
        report=report,
    )
    _unique_index(
        coverage_audits,
        id_field="audit_id",
        location_prefix="state.investigation_state.coverage_audits",
        report=report,
    )

    successful_calls = {
        str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
        for step in steps
        if _parse_successful_tool_step(step)
        and str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
    }
    all_known_origins = {
        *entity_by_id,
        *fact_by_id,
        *task_by_id,
        *anchor_by_id,
        *discovery_by_id,
        *evidence_by_id,
        *finding_by_id,
        *failure_by_id,
        str(_mapping(investigation.get("brief")).get("brief_id", "")).strip(),
        str(_mapping(investigation.get("brief")).get("case_id", "")).strip(),
    }
    all_known_origins.discard("")

    replanned_task_ids: set[str] = set()
    for replan_id, replan in query_replan_by_id.items():
        location = _location(
            "state.investigation_state.query_replans",
            replan_id,
        )
        output = _mapping(replan.get("output"))
        task_id = str(output.get("task_id", "")).strip()
        if task_id not in task_by_id:
            _issue(
                report,
                "QUERY_REPLAN_TASK_UNKNOWN",
                f"Query Replan targets unknown task {task_id!r}",
                location=location,
            )
        elif task_id in replanned_task_ids:
            _issue(
                report,
                "QUERY_REPLAN_REPEATED",
                f"Task {task_id!r} received more than one Query Replan",
                location=location,
            )
        replanned_task_ids.add(task_id)
        evidence_ids = {
            str(item)
            for item in replan.get("new_evidence_ids", []) or []
        }
        unknown_evidence = sorted(evidence_ids - set(evidence_by_id))
        if unknown_evidence:
            _issue(
                report,
                "QUERY_REPLAN_EVIDENCE_UNKNOWN",
                "Query Replan cites unknown Evidence: "
                + ", ".join(unknown_evidence),
                location=location,
            )
        if str(replan.get("trigger", "")).strip() == "evidence_boundary":
            if not evidence_ids:
                _issue(
                    report,
                    "QUERY_REPLAN_EVIDENCE_MISSING",
                    "Evidence-boundary Query Replan requires new Evidence",
                    location=location,
                )
            elif task_id in task_by_id:
                task_fact_ids = {
                    str(item)
                    for item in task_by_id[task_id].get("fact_ids", []) or []
                }
                if any(
                    not task_fact_ids
                    & {
                        str(item)
                        for item in evidence_by_id[evidence_id].get(
                            "fact_ids",
                            [],
                        )
                        or []
                    }
                    for evidence_id in evidence_ids & set(evidence_by_id)
                ):
                    _issue(
                        report,
                        "QUERY_REPLAN_EVIDENCE_OWNERSHIP_INVALID",
                        "Query Replan Evidence must belong to its task fact",
                        location=location,
                    )
        extraction = _mapping(replan.get("concept_extraction"))
        if str(extraction.get("task_id", "")).strip() != task_id:
            _issue(
                report,
                "QUERY_CONCEPT_TASK_MISMATCH",
                "Query Concept Extraction must belong to the replanned task",
                location=location,
            )
        concept_by_id: dict[str, Mapping[str, Any]] = {}
        for concept in _rows(extraction.get("concepts")):
            concept_id = str(concept.get("concept_id", "")).strip()
            if not concept_id:
                _issue(
                    report,
                    "QUERY_CONCEPT_ID_MISSING",
                    "Query Concept must have a non-empty concept_id",
                    location=location,
                )
                continue
            if concept_id in concept_by_id:
                _issue(
                    report,
                    "QUERY_CONCEPT_ID_DUPLICATE",
                    f"Duplicate Query Concept id {concept_id!r}",
                    location=location,
                )
                continue
            concept_by_id[concept_id] = concept
            evidence_id = str(concept.get("evidence_id", "")).strip()
            if evidence_id not in evidence_ids:
                _issue(
                    report,
                    "QUERY_CONCEPT_EVIDENCE_OUT_OF_SCOPE",
                    f"Query Concept {concept_id!r} does not cite new Evidence",
                    location=location,
                )
                continue
            evidence_row = evidence_by_id.get(evidence_id)
            if evidence_row is None:
                continue
            phrase = " ".join(
                re.findall(
                    r"[\w]+",
                    str(concept.get("evidence_phrase", "")).casefold(),
                    flags=re.UNICODE,
                )
            )
            exact_text = " ".join(
                re.findall(
                    r"[\w]+",
                    str(evidence_row.get("exact_text", "")).casefold(),
                    flags=re.UNICODE,
                )
            )
            if not phrase or phrase not in exact_text:
                _issue(
                    report,
                    "QUERY_CONCEPT_PHRASE_UNGROUNDED",
                    f"Query Concept {concept_id!r} lacks an exact Evidence phrase",
                    location=location,
                )
        selected_concept_id = str(
            output.get("selected_concept_id", "")
        ).strip()
        if selected_concept_id not in concept_by_id:
            _issue(
                report,
                "QUERY_REPLAN_CONCEPT_UNKNOWN",
                "Query Replan selected a concept absent from its extraction",
                location=location,
            )
        replacement_query = str(
            output.get("replacement_query", "")
        ).strip()
        if not replacement_query:
            _issue(
                report,
                "QUERY_REPLAN_QUERY_MISSING",
                "Query Replan must contain one replacement query",
                location=location,
            )
        elif selected_concept_id in concept_by_id:
            concept_term = " ".join(
                re.findall(
                    r"[\w]+",
                    str(output.get("concept_term", "")).casefold(),
                    flags=re.UNICODE,
                )
            )
            normalized_query = " ".join(
                re.findall(
                    r"[\w]+",
                    replacement_query.casefold(),
                    flags=re.UNICODE,
                )
            )
            if not concept_term or concept_term not in normalized_query:
                _issue(
                    report,
                    "QUERY_REPLAN_CONCEPT_UNUSED",
                    "Query Replan must use its declared concept term",
                    location=location,
                )
        accepted_queries = [
            str(item).strip()
            for item in replan.get("accepted_queries", []) or []
            if str(item).strip()
        ]
        rejected_reason = str(replan.get("rejected_reason", "")).strip()
        if rejected_reason and accepted_queries:
            _issue(
                report,
                "QUERY_REPLAN_REJECTED_WITH_QUERIES",
                "Rejected Query Replan cannot expose accepted queries",
                location=location,
            )

    accepted_refinement_ids: list[str] = []
    for decision_id, decision in decision_by_id.items():
        location = _location(
            "state.investigation_state.evidence_decisions",
            decision_id,
        )
        output = _mapping(decision.get("output"))
        active_fact_id = str(output.get("active_fact_id", "")).strip()
        if active_fact_id not in fact_by_id:
            _issue(
                report,
                "EVIDENCE_DECISION_FACT_UNKNOWN",
                f"Evidence decision targets unknown fact {active_fact_id!r}",
                location=location,
            )
        reviewed_ids = {
            str(item)
            for item in decision.get("reviewed_evidence_ids", []) or []
        }
        selected_ids = {
            str(item)
            for item in output.get("selected_evidence_ids", []) or []
        }
        unknown_evidence = sorted(
            (reviewed_ids | selected_ids) - set(evidence_by_id)
        )
        if unknown_evidence:
            _issue(
                report,
                "EVIDENCE_DECISION_EVIDENCE_UNKNOWN",
                "Evidence decision cites unknown Evidence: "
                + ", ".join(unknown_evidence),
                location=location,
            )
        if selected_ids and not selected_ids & reviewed_ids:
            _issue(
                report,
                "EVIDENCE_DECISION_NOT_INCREMENTAL",
                "Evidence decision must select at least one newly reviewed item",
                location=location,
            )
        assessment = str(output.get("assessment", "")).strip()
        if assessment in {"supported", "refuted", "conflicted"} and not selected_ids:
            _issue(
                report,
                "EVIDENCE_DECISION_BASIS_EMPTY",
                "A material Evidence decision must select Evidence",
                location=location,
            )
        for evidence_id in selected_ids & set(evidence_by_id):
            selected_evidence_row = evidence_by_id[evidence_id]
            task_id = str(selected_evidence_row.get("task_id", "")).strip()
            if active_fact_id not in {
                str(item)
                for item in selected_evidence_row.get("fact_ids", []) or []
            }:
                _issue(
                    report,
                    "EVIDENCE_DECISION_FACT_OWNERSHIP_INVALID",
                    (
                        f"Selected Evidence {evidence_id!r} does not belong to "
                        f"active fact {active_fact_id!r}"
                    ),
                    location=location,
                )
            task = task_by_id.get(task_id)
            if task is None or active_fact_id not in {
                str(item) for item in task.get("fact_ids", []) or []
            }:
                _issue(
                    report,
                    "EVIDENCE_DECISION_TASK_OWNERSHIP_INVALID",
                    (
                        f"Selected Evidence {evidence_id!r} is not owned by an "
                        "active-fact ResearchTask"
                    ),
                    location=location,
                )
        if (
            assessment in {"supported", "refuted"}
            and str(output.get("binding_requirement", "")).strip()
            == "same_capture_required"
            and not any(
                str(evidence_by_id[evidence_id].get("claim_binding", "")).strip()
                == "same_capture"
                or evidence_by_id[evidence_id].get(
                    "same_capture_or_near_duplicate"
                )
                is True
                for evidence_id in selected_ids & set(evidence_by_id)
            )
        ):
            _issue(
                report,
                "EVIDENCE_DECISION_CAPTURE_BINDING_MISSING",
                (
                    "Terminal decision requires same-capture binding but its "
                    "selected Evidence does not provide it"
                ),
                location=location,
            )
        refinement = _mapping(output.get("refinement"))
        accepted_refinement_id = str(
            decision.get("accepted_refinement_fact_id") or ""
        ).strip()
        if accepted_refinement_id:
            accepted_refinement_ids.append(accepted_refinement_id)
            if accepted_refinement_id not in fact_by_id:
                _issue(
                    report,
                    "EVIDENCE_REFINEMENT_FACT_UNKNOWN",
                    (
                        "Evidence decision accepted unknown refinement fact "
                        f"{accepted_refinement_id!r}"
                    ),
                    location=location,
                )
            if not refinement:
                _issue(
                    report,
                    "EVIDENCE_REFINEMENT_OUTPUT_MISSING",
                    "Accepted refinement requires a structured refinement proposal",
                    location=location,
                )
            anchor_ids = {
                str(item)
                for item in refinement.get("anchor_fact_ids", []) or []
            }
            grounding_ids = {
                str(item)
                for item in refinement.get("grounding_evidence_ids", []) or []
            }
            invalid_anchors = sorted(
                anchor_id
                for anchor_id in anchor_ids
                if anchor_id not in fact_by_id
                or str(
                    _mapping(fact_by_id[anchor_id].get("origin")).get("type", "")
                )
                not in {"input_image", "ocr"}
            )
            if invalid_anchors:
                _issue(
                    report,
                    "EVIDENCE_REFINEMENT_ANCHOR_INVALID",
                    (
                        "Refinement anchors must be existing pixel/OCR facts: "
                        + ", ".join(invalid_anchors)
                    ),
                    location=location,
                )
            if (
                not grounding_ids
                or not grounding_ids <= selected_ids
                or not grounding_ids <= reviewed_ids
            ):
                _issue(
                    report,
                    "EVIDENCE_REFINEMENT_GROUNDING_INVALID",
                    (
                        "Refinement grounding must use newly reviewed selected "
                        "Evidence"
                    ),
                    location=location,
                )
        accepted_visual_id = str(
            decision.get("accepted_visual_question_id") or ""
        ).strip()
        accepted_visual_task_id = str(
            decision.get("accepted_visual_task_id") or ""
        ).strip()
        visual_request = _mapping(output.get("visual_reinspection"))
        if accepted_visual_id:
            visual_record = visual_reinspection_by_id.get(accepted_visual_id)
            if visual_record is None:
                _issue(
                    report,
                    "VISUAL_REINSPECTION_RECORD_UNKNOWN",
                    (
                        "Evidence decision accepted unknown visual question "
                        f"{accepted_visual_id!r}"
                    ),
                    location=location,
                )
            if not visual_request:
                _issue(
                    report,
                    "VISUAL_REINSPECTION_OUTPUT_MISSING",
                    (
                        "Accepted visual reinspection requires a structured "
                        "visual_reinspection request"
                    ),
                    location=location,
                )
            if visual_record is not None and (
                str(visual_record.get("task_id", "")).strip()
                != accepted_visual_task_id
            ):
                _issue(
                    report,
                    "VISUAL_REINSPECTION_TASK_MISMATCH",
                    (
                        "Accepted visual task id does not match its persisted "
                        "visual reinspection record"
                    ),
                    location=location,
                )
        if accepted_visual_id and accepted_refinement_id:
            _issue(
                report,
                "VISUAL_REINSPECTION_REFINEMENT_CONFLICT",
                (
                    "One Evidence decision cannot accept both visual "
                    "reinspection and core refinement"
                ),
                location=location,
            )
        unknown_findings = sorted(
            {
                str(item)
                for item in decision.get("finding_ids", []) or []
            }
            - set(finding_by_id)
        )
        if unknown_findings:
            _issue(
                report,
                "EVIDENCE_DECISION_FINDING_UNKNOWN",
                "Evidence decision cites unknown Findings: "
                + ", ".join(unknown_findings),
                location=location,
            )

    visual_tool_steps = {
        str(_mapping(step.get("metadata")).get("visual_question_id", "")).strip(): step
        for step in steps
        if str(step.get("stage", "")) == "image_only_visual_reinspection"
        and str(step.get("action_type", "")) == "tool_call"
        and str(_mapping(step.get("metadata")).get("visual_question_id", "")).strip()
    }
    for visual_question_id, visual_record in visual_reinspection_by_id.items():
        location = _location(
            "state.investigation_state.visual_reinspections",
            visual_question_id,
        )
        task_id = str(visual_record.get("task_id", "")).strip()
        fact_id = str(visual_record.get("fact_id", "")).strip()
        request = _mapping(visual_record.get("request"))
        if task_id not in task_by_id:
            _issue(
                report,
                "VISUAL_REINSPECTION_TASK_UNKNOWN",
                f"Visual reinspection references unknown task {task_id!r}",
                location=location,
            )
        elif fact_id not in {
            str(item)
            for item in task_by_id[task_id].get("fact_ids", []) or []
        }:
            _issue(
                report,
                "VISUAL_REINSPECTION_FACT_OWNERSHIP_INVALID",
                "Visual reinspection fact must be owned by its ResearchTask",
                location=location,
            )
        invalid_anchors = sorted(
            str(item)
            for item in request.get("anchor_fact_ids", []) or []
            if str(item) not in fact_by_id
            or str(
                _mapping(fact_by_id[str(item)].get("origin")).get("type", "")
            )
            not in {"input_image", "ocr"}
        )
        if invalid_anchors:
            _issue(
                report,
                "VISUAL_REINSPECTION_ANCHOR_INVALID",
                (
                    "Visual reinspection anchors must be pixel/OCR facts: "
                    + ", ".join(invalid_anchors)
                ),
                location=location,
            )
        grounding_ids = {
            str(item)
            for item in request.get("grounding_evidence_ids", []) or []
        }
        if not grounding_ids or not grounding_ids <= set(evidence_by_id):
            _issue(
                report,
                "VISUAL_REINSPECTION_GROUNDING_INVALID",
                "Visual reinspection grounding Evidence is missing or unknown",
                location=location,
            )
        evidence_ids = {
            str(item)
            for item in visual_record.get("evidence_ids", []) or []
        }
        failure_ids = {
            str(item)
            for item in visual_record.get("failure_ids", []) or []
        }
        if not evidence_ids <= set(evidence_by_id):
            _issue(
                report,
                "VISUAL_REINSPECTION_EVIDENCE_UNKNOWN",
                "Visual reinspection cites unknown Evidence",
                location=location,
            )
        if not failure_ids <= set(failure_by_id):
            _issue(
                report,
                "VISUAL_REINSPECTION_FAILURE_UNKNOWN",
                "Visual reinspection cites unknown failures",
                location=location,
            )
        status = str(visual_record.get("status", "")).strip()
        if status == "resolved" and not evidence_ids:
            _issue(
                report,
                "VISUAL_REINSPECTION_RESOLUTION_EMPTY",
                "Resolved visual reinspection must produce Evidence",
                location=location,
            )
        if status == "failed" and not failure_ids:
            _issue(
                report,
                "VISUAL_REINSPECTION_FAILURE_EMPTY",
                "Failed visual reinspection must record a failure",
                location=location,
            )
        if status in {"resolved", "failed"} and visual_question_id not in visual_tool_steps:
            _issue(
                report,
                "VISUAL_REINSPECTION_TOOL_STEP_MISSING",
                "Completed visual reinspection lacks its dedicated tool step",
                location=location,
            )

    refinement_count = int(
        investigation.get("core_fact_refinement_count", 0) or 0
    )
    if refinement_count > 1 or len(accepted_refinement_ids) > 1:
        _issue(
            report,
            "CORE_REFINEMENT_BUDGET_EXCEEDED",
            "An image-only investigation may refine its active visual slot once",
            location="state.investigation_state.core_fact_refinement_count",
        )
    if refinement_count != len(accepted_refinement_ids):
        _issue(
            report,
            "CORE_REFINEMENT_COUNT_MISMATCH",
            (
                f"core_fact_refinement_count={refinement_count}, but "
                f"{len(accepted_refinement_ids)} accepted refinements are recorded"
            ),
            location="state.investigation_state.core_fact_refinement_count",
        )

    for fact_id, fact in fact_by_id.items():
        location = _location("state.investigation_state.facts", fact_id)
        subject = str(fact.get("subject_entity_id", "")).strip()
        object_id = str(fact.get("object_entity_id", "") or "").strip()
        if subject not in entity_by_id:
            _issue(
                report,
                "VISUAL_FACT_ENTITY_UNKNOWN",
                f"subject_entity_id {subject!r} does not exist",
                location=location,
            )
        if object_id and object_id not in entity_by_id:
            _issue(
                report,
                "VISUAL_FACT_ENTITY_UNKNOWN",
                f"object_entity_id {object_id!r} does not exist",
                location=location,
            )
        unknown_basis = sorted(
            set(str(item) for item in fact.get("basis_ids", []) or [])
            - set(all_known_origins)
        )
        if unknown_basis:
            _issue(
                report,
                "VISUAL_FACT_BASIS_UNKNOWN",
                "unknown fact basis ids: " + ", ".join(unknown_basis),
                location=location,
            )

    failures_by_task: dict[str, list[str]] = {}
    for failure_id, failure in failure_by_id.items():
        task_id = str(failure.get("task_id", "")).strip()
        failures_by_task.setdefault(task_id, []).append(failure_id)
        if task_id not in task_by_id:
            _issue(
                report,
                "FAILURE_TASK_UNKNOWN",
                f"failure references unknown task {task_id!r}",
                location=_location("state.investigation_state.failures", failure_id),
            )

    for task_id, task in task_by_id.items():
        location = _location("state.investigation_state.tasks", task_id)
        unknown_facts = sorted(
            set(str(item) for item in task.get("fact_ids", []) or [])
            - set(fact_by_id)
        )
        if unknown_facts:
            _issue(
                report,
                "RESEARCH_TASK_FACT_UNKNOWN",
                "task references unknown facts: " + ", ".join(unknown_facts),
                location=location,
            )
        parent_id = str(task.get("parent_task_id", "") or "").strip()
        if parent_id and parent_id not in task_by_id:
            _issue(
                report,
                "RESEARCH_TASK_PARENT_UNKNOWN",
                f"parent_task_id {parent_id!r} does not exist",
                location=location,
            )
        unknown_origins = sorted(
            set(str(item) for item in task.get("origin_ids", []) or [])
            - all_known_origins
        )
        if unknown_origins:
            _issue(
                report,
                "RESEARCH_TASK_ORIGIN_UNKNOWN",
                "task references unknown origins: " + ", ".join(unknown_origins),
                location=location,
            )
        task_findings = [
            str(item) for item in task.get("finding_ids", []) or []
        ]
        if str(task.get("status", "")) == "resolved" and not task_findings:
            _issue(
                report,
                "RESOLVED_TASK_WITHOUT_FINDING",
                "resolved ResearchTask must reference a Finding",
                location=location,
            )
        if str(task.get("status", "")) in {"blocked", "exhausted"} and not failures_by_task.get(task_id):
            _issue(
                report,
                "BLOCKED_TASK_WITHOUT_FAILURE",
                "blocked/exhausted ResearchTask must reference a recorded Failure",
                location=location,
            )
        for finding_id in task_findings:
            if finding_id not in finding_by_id:
                _issue(
                    report,
                    "RESEARCH_TASK_FINDING_UNKNOWN",
                    f"task references unknown Finding {finding_id!r}",
                    location=location,
                )

    for discovery_id, discovery in discovery_by_id.items():
        location = _location(
            "state.investigation_state.discoveries", discovery_id
        )
        task_id = str(discovery.get("task_id", "")).strip()
        if task_id not in task_by_id:
            _issue(
                report,
                "DISCOVERY_TASK_UNKNOWN",
                f"Discovery references unknown task {task_id!r}",
                location=location,
            )
        if str(discovery.get("promoted_evidence_id", "") or "").strip():
            _issue(
                report,
                "DISCOVERY_PROMOTED_IN_PLACE",
                "Discovery must remain separate from Evidence",
                location=location,
            )

    _audit_evidence_calls(
        evidence,
        steps,
        report,
        location_prefix="state.investigation_state.evidence",
        tool_field="tool_name",
        stat_key="image_only_evidence_with_successful_call",
    )
    for evidence_id, item in evidence_by_id.items():
        location = _location("state.investigation_state.evidence", evidence_id)
        task_id = str(item.get("task_id", "")).strip()
        if task_id not in task_by_id:
            _issue(
                report,
                "EVIDENCE_TASK_UNKNOWN",
                f"Evidence references unknown task {task_id!r}",
                location=location,
            )
        unknown_facts = sorted(
            set(str(value) for value in item.get("fact_ids", []) or [])
            - set(fact_by_id)
        )
        if unknown_facts:
            _issue(
                report,
                "EVIDENCE_FACT_UNKNOWN",
                "Evidence references unknown facts: " + ", ".join(unknown_facts),
                location=location,
            )

    findings_by_fact: dict[str, list[str]] = {}
    for finding_id, finding in finding_by_id.items():
        location = _location("state.investigation_state.findings", finding_id)
        task_id = str(finding.get("task_id", "")).strip()
        task = task_by_id.get(task_id)
        is_source_visual_composite = (
            COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY
            in {
                str(value)
                for value in finding.get("source_family_ids", []) or []
            }
        )
        if task is None:
            _issue(
                report,
                "FINDING_TASK_UNKNOWN",
                f"Finding references unknown task {task_id!r}",
                location=location,
            )
            continue
        finding_facts = {
            str(item) for item in finding.get("fact_ids", []) or []
        }
        if not finding_facts <= {
            str(item) for item in task.get("fact_ids", []) or []
        }:
            _issue(
                report,
                "FINDING_FACT_OWNERSHIP_INVALID",
                "Finding fact_ids must be owned by its ResearchTask",
                location=location,
            )
        for fact_id in finding_facts:
            findings_by_fact.setdefault(fact_id, []).append(finding_id)
        for evidence_id in finding.get("evidence_ids", []) or []:
            evidence_record = evidence_by_id.get(str(evidence_id))
            if evidence_record is None:
                _issue(
                    report,
                    "FINDING_EVIDENCE_UNKNOWN",
                    f"Finding references unknown Evidence {evidence_id!r}",
                    location=location,
                )
                continue
            evidence_task_id = str(evidence_record.get("task_id", "")).strip()
            composite_visual_evidence = (
                is_source_visual_composite
                and _composite_visual_evidence_is_eligible(
                    evidence_record,
                    finding=finding,
                    task_by_id=task_by_id,
                )
            )
            if evidence_task_id != task_id and not (
                composite_visual_evidence
            ):
                _issue(
                    report,
                    "FINDING_EVIDENCE_OWNERSHIP_INVALID",
                    "Finding Evidence must belong to its ResearchTask or an "
                    "explicit claim-owned visual Evidence",
                    location=location,
                )

    investigation_tool_steps = [
        step
        for step in steps
        if str(step.get("stage", ""))
        in {
            "image_only_investigation",
            "image_only_visual_reinspection",
        }
        and str(step.get("action_type", "")) == "tool_call"
    ]
    action_count = int(investigation.get("action_count", 0) or 0)
    if action_count != len(investigation_tool_steps):
        _issue(
            report,
            "IMAGE_ONLY_ACTION_COUNT_MISMATCH",
            f"action_count={action_count}, but trace records {len(investigation_tool_steps)} investigation tool calls",
            location="state.investigation_state.action_count",
        )
    if action_count > 24:
        _issue(
            report,
            "IMAGE_ONLY_ACTION_BUDGET_EXCEEDED",
            f"image-only investigation used {action_count} actions; maximum is 24",
            location="state.investigation_state.action_count",
        )
    terminal_audits = [
        item
        for item in coverage_audits
        if str(item.get("stop_reason", "")) == "verdict_determined"
    ]
    if terminal_audits:
        terminal_action = min(
            int(item.get("action_count", 0) or 0)
            for item in terminal_audits
        )
        if action_count > terminal_action:
            _issue(
                report,
                "POST_DETERMINATION_ACTION",
                (
                    f"Verdict was determined at action {terminal_action}, but "
                    f"the trace continued to action {action_count}"
                ),
                category=SCHEDULER,
                location="state.investigation_state.action_count",
            )
        late_reflections = [
            int(item.get("action_count", 0) or 0)
            for item in reflections
            if int(item.get("action_count", 0) or 0) >= terminal_action
        ]
        if late_reflections:
            _issue(
                report,
                "POST_DETERMINATION_REFLECTION",
                (
                    "Reflection ran after the semantic verdict boundary at "
                    f"action {terminal_action}"
                ),
                category=SCHEDULER,
                location="state.investigation_state.reflections",
            )
    for index, step in enumerate(investigation_tool_steps):
        count = _mapping(step.get("metadata")).get("function_call_count")
        if count is not None and int(count or 0) != 1:
            _issue(
                report,
                "IMAGE_ONLY_PARALLEL_TOOL_CALL",
                "each image-only action turn must contain exactly one tool call",
                location=f"image_only_tool_steps[{index}]",
            )

    interval_reflection_counts = [
        int(item.get("action_count", 0) or 0)
        for item in reflections
        if str(item.get("trigger", "interval")).strip() == "interval"
    ]
    saturation_reflection_counts = [
        int(item.get("action_count", 0) or 0)
        for item in reflections
        if str(item.get("trigger", "interval")).strip() == "saturation"
    ]
    unknown_reflection_triggers = [
        str(item.get("trigger", "")).strip()
        for item in reflections
        if str(item.get("trigger", "interval")).strip()
        not in {"interval", "saturation"}
    ]
    if unknown_reflection_triggers:
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_TRIGGER_INVALID",
            "unknown Reflection trigger(s): "
            + ", ".join(unknown_reflection_triggers),
            location="state.investigation_state.reflections",
        )
    if len(saturation_reflection_counts) > 1:
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_CADENCE_INVALID",
            "at most one saturation Reflection is allowed; "
            f"found {saturation_reflection_counts}",
            location="state.investigation_state.reflections",
        )
    if any(count <= 0 for count in saturation_reflection_counts):
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_CADENCE_INVALID",
            "saturation Reflection action counts must be positive; "
            f"found {saturation_reflection_counts}",
            location="state.investigation_state.reflections",
        )
    invalid_interval_counts = [
        count
        for count in interval_reflection_counts
        if count <= 0 or count % 4 != 0
    ]
    if invalid_interval_counts:
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_CADENCE_INVALID",
            "interval Reflection action counts must be positive multiples of 4; "
            f"found {invalid_interval_counts}",
            location="state.investigation_state.reflections",
        )
    terminal_stop = str(investigation.get("stop_reason", "")) in {
        "coverage_complete",
        "verdict_determined",
        "information_saturated",
        "hard_budget_exhausted",
    }
    required_reflection_limit = (
        action_count - 1
        if terminal_stop and action_count % 4 == 0
        else action_count
    )
    expected_boundaries = list(
        range(4, required_reflection_limit + 1, 4)
    )
    allowed_boundaries = set(expected_boundaries)
    if terminal_stop and action_count > 0 and action_count % 4 == 0:
        # The terminal condition may be accepted immediately after the action,
        # or after the scheduled Reflection at that same boundary. Both orders
        # are valid; earlier scheduled boundaries remain mandatory.
        allowed_boundaries.add(action_count)
    missing_boundaries = [
        boundary
        for boundary in expected_boundaries
        if boundary not in interval_reflection_counts
    ]
    unexpected_interval_counts = [
        count
        for count in interval_reflection_counts
        if count not in allowed_boundaries
    ]
    if missing_boundaries or unexpected_interval_counts:
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_CADENCE_INVALID",
            (
                "scheduled Reflection boundaries require an interval Reflection; "
                f"missing={missing_boundaries}, "
                f"unexpected_interval={unexpected_interval_counts}, "
                f"interval={interval_reflection_counts}"
            ),
            location="state.investigation_state.reflections",
        )

    decisive_ids = [
        str(item)
        for item in investigation.get("decisive_fact_ids", []) or []
    ]
    if not decisive_ids:
        _issue(
            report,
            "DECISIVE_FACTS_MISSING",
            "image-only judgment requires at least one decisive VisualFact",
            location="state.investigation_state.decisive_fact_ids",
        )
    unknown_decisive = sorted(set(decisive_ids) - set(fact_by_id))
    if unknown_decisive:
        _issue(
            report,
            "DECISIVE_FACT_UNKNOWN",
            "unknown decisive fact ids: " + ", ".join(unknown_decisive),
            location="state.investigation_state.decisive_fact_ids",
        )

    statuses = {
        fact_id: str(fact_by_id.get(fact_id, {}).get("status", ""))
        for fact_id in decisive_ids
    }
    if any(status == "refuted" for status in statuses.values()):
        expected_verdict = "fake"
        expected_basis_facts = {
            fact_id for fact_id, status in statuses.items() if status == "refuted"
        }
    elif statuses and all(status == "supported" for status in statuses.values()):
        expected_verdict = "real"
        expected_basis_facts = set(decisive_ids)
    else:
        expected_verdict = "unverifiable"
        expected_basis_facts = {
            fact_id
            for fact_id, status in statuses.items()
            if status not in {"supported", "refuted"}
        }

    verdict = str(trace.get("verdict", "")).strip()
    if verdict != expected_verdict:
        _issue(
            report,
            "IMAGE_ONLY_VERDICT_STATE_MISMATCH",
            f"decisive fact states require verdict={expected_verdict}, got {verdict!r}",
            location="verdict",
        )

    basis = _mapping(
        trace.get("verdict_basis") or investigation.get("verdict_basis")
    )
    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    if str(basis.get("policy_rule_id", "")) != "reinspect-v2":
        _issue(
            report,
            "VERDICT_BASIS_POLICY_INVALID",
            "verdict_basis.policy_rule_id must be reinspect-v2",
            location="verdict_basis.policy_rule_id",
        )
    basis_facts = {str(item) for item in basis.get("fact_ids", []) or []}
    basis_findings = {
        str(item) for item in basis.get("finding_ids", []) or []
    }
    basis_evidence = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }
    if expected_verdict == "fake":
        if len(basis_facts) != 1 or not basis_facts <= expected_basis_facts:
            _issue(
                report,
                "VERDICT_BASIS_FACT_SET_INVALID",
                (
                    "fake verdict basis must select exactly one decisive refuted "
                    f"fact from {sorted(expected_basis_facts)}, found "
                    f"{sorted(basis_facts)}"
                ),
                location="verdict_basis.fact_ids",
            )
    elif basis_facts != expected_basis_facts:
        _issue(
            report,
            "VERDICT_BASIS_FACT_SET_INVALID",
            f"expected basis facts {sorted(expected_basis_facts)}, found {sorted(basis_facts)}",
            location="verdict_basis.fact_ids",
        )
    if basis_evidence & set(discovery_by_id):
        _issue(
            report,
            "DISCOVERY_USED_AS_VERDICT_EVIDENCE",
            "Discovery ids cannot appear in verdict_basis.evidence_ids",
            location="verdict_basis.evidence_ids",
        )
    unknown_basis_findings = sorted(basis_findings - set(finding_by_id))
    unknown_basis_evidence = sorted(basis_evidence - set(evidence_by_id))
    if unknown_basis_findings:
        _issue(
            report,
            "VERDICT_BASIS_FINDING_UNKNOWN",
            "unknown basis Finding ids: " + ", ".join(unknown_basis_findings),
            location="verdict_basis.finding_ids",
        )
    if unknown_basis_evidence:
        _issue(
            report,
            "VERDICT_BASIS_EVIDENCE_UNKNOWN",
            "unknown basis Evidence ids: " + ", ".join(unknown_basis_evidence),
            location="verdict_basis.evidence_ids",
        )
    if expected_verdict != "unverifiable":
        for fact_id in basis_facts:
            selected_findings = (
                set(findings_by_fact.get(fact_id, [])) & basis_findings
            )
            if not selected_findings:
                _issue(
                    report,
                    "VERDICT_FACT_WITHOUT_FINDING",
                    f"basis fact {fact_id!r} has no selected Finding",
                    location="verdict_basis",
                )
                continue
            selected_evidence = {
                str(evidence_id)
                for finding_id in selected_findings
                for evidence_id in finding_by_id[finding_id].get(
                    "evidence_ids",
                    [],
                )
                or []
            } & basis_evidence
            if not selected_evidence:
                _issue(
                    report,
                    "VERDICT_FINDING_WITHOUT_EVIDENCE",
                    f"basis fact {fact_id!r} has no selected Evidence",
                    location="verdict_basis",
                )
            for evidence_id in selected_evidence:
                call_id = str(
                    evidence_by_id[evidence_id].get("function_call_id", "")
                ).strip()
                if call_id not in successful_calls:
                    _issue(
                        report,
                        "VERDICT_EVIDENCE_CALL_NOT_SUCCESSFUL",
                        f"basis Evidence {evidence_id!r} lacks a successful "
                        "tool call",
                        location="verdict_basis",
                    )

    judgment_sets = (
        (
            {str(item) for item in judgment.get("selected_fact_ids", []) or []},
            basis_facts,
            "fact",
        ),
        (
            {str(item) for item in judgment.get("selected_finding_ids", []) or []},
            basis_findings,
            "finding",
        ),
        (
            {str(item) for item in judgment.get("selected_evidence_ids", []) or []},
            basis_evidence,
            "evidence",
        ),
    )
    if str(judgment.get("verdict", "")) != verdict:
        _issue(
            report,
            "JUDGMENT_VERDICT_MISMATCH",
            "Judgment verdict must match the canonical trace verdict",
            location="judgment.verdict",
        )
    for selected, compiled, name in judgment_sets:
        if selected != compiled:
            _issue(
                report,
                "JUDGMENT_BASIS_MISMATCH",
                f"Judgment selected {name} ids do not match verdict_basis",
                location=f"judgment.selected_{name}_ids",
            )

    report.stats.update(
        {
            "visual_facts": len(facts),
            "research_tasks": len(tasks),
            "discoveries": len(discoveries),
            "image_only_evidence": len(evidence),
            "findings": len(findings),
            "reflections": len(reflections),
            "query_replans": len(query_replans),
            "evidence_decisions": len(evidence_decisions),
            "visual_reinspections": len(visual_reinspections),
            "core_refinements": refinement_count,
            "decisive_facts": len(decisive_ids),
            "image_only_actions": action_count,
        }
    )
    _audit_image_only_interaction_chains(steps, report)


def _audit_unified_react_interaction_chains(
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Audit the mixed tool-roundtrip/standalone lifecycle of unified ReAct."""

    stages = {
        "unified_react",
        "unified_reflection",
        "unified_discrepancy_decision",
        "unified_judgment",
    }
    native = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")) in stages
        and _mapping(step.get("metadata")).get("native_interactions")
    ]
    if not native:
        _issue(
            report,
            "UNIFIED_REACT_INTERACTIONS_MISSING",
            "unified-ReAct trace contains no native main-chain interactions",
            location="state.all_steps",
        )
        return

    previous_tool_interaction = ""
    saw_tool_roundtrip = False
    seen_ids: set[str] = set()
    for index, step in native:
        metadata = _mapping(step.get("metadata"))
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        parent = str(
            metadata.get("previous_interaction_id", "") or ""
        ).strip()
        lifecycle = str(
            metadata.get("interaction_lifecycle_kind", "")
        ).strip()
        location = _step_label(index, step)
        if not interaction_id:
            _issue(
                report,
                "INTERACTION_ID_MISSING",
                "unified-ReAct native interaction lacks interaction_id",
                location=location,
            )
            continue
        if interaction_id in seen_ids:
            _issue(
                report,
                "INTERACTION_ID_DUPLICATE",
                f"duplicate native interaction_id {interaction_id!r}",
                location=location,
            )
        seen_ids.add(interaction_id)
        if lifecycle == "tool_roundtrip":
            if not saw_tool_roundtrip:
                if parent:
                    _issue(
                        report,
                        "INTERACTION_CHAIN_ROOT_INVALID",
                        "first unified tool roundtrip must have a null parent",
                        location=location,
                    )
                if str(step.get("tool_name", "")).strip() not in {
                    "perceive_scene",
                    "ocr_with_position",
                }:
                    _issue(
                        report,
                        "UNIFIED_REACT_BOOTSTRAP_ROOT_INVALID",
                        "unified tool chain must begin with scene perception or OCR",
                        location=location,
                    )
            elif parent != previous_tool_interaction:
                _issue(
                    report,
                    "INTERACTION_CHAIN_BROKEN",
                    (
                        "unified tool roundtrip must continue from the prior "
                        f"tool interaction {previous_tool_interaction!r}, got "
                        f"{parent!r}"
                    ),
                    location=location,
                )
            saw_tool_roundtrip = True
            previous_tool_interaction = interaction_id
        elif lifecycle == "standalone_request":
            if parent:
                _issue(
                    report,
                    "UNIFIED_STANDALONE_PARENT_INVALID",
                    "Reflection, Decision, and Judgment must be standalone requests",
                    location=location,
                )
        elif lifecycle == "protocol_correction":
            if not parent or parent not in seen_ids:
                _issue(
                    report,
                    "UNIFIED_CORRECTION_PARENT_INVALID",
                    (
                        "unified protocol correction must name an earlier "
                        "native interaction in the same trace"
                    ),
                    location=location,
                )
        else:
            _issue(
                report,
                "UNIFIED_INTERACTION_LIFECYCLE_INVALID",
                f"unexpected unified interaction lifecycle {lifecycle!r}",
                location=location,
            )

    report.stats["unified_interaction_steps"] = len(native)
    report.stats["unified_tool_roundtrips"] = sum(
        str(
            _mapping(step.get("metadata")).get(
                "interaction_lifecycle_kind", ""
            )
        ).strip()
        == "tool_roundtrip"
        for _, step in native
    )


def _audit_unified_react_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Audit the clean tool-first ``unified-react-v1`` canonical contract."""

    if str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    ) != UNIFIED_REACT_POLICY_VERSION:
        _issue(
            report,
            "UNIFIED_REACT_POLICY_MISMATCH",
            "unified trace must use decision_policy_version=unified-react-v1",
            location="decision_policy_version",
        )

    investigation = _mapping(state.get("investigation_state"))
    if not investigation:
        _issue(
            report,
            "UNIFIED_REACT_STATE_MISSING",
            "unified trace must contain state.investigation_state",
            location="state.investigation_state",
        )
        return

    if (
        str(investigation.get("schema_version", "")).strip()
        == REACT_RUNTIME_SCHEMA_VERSION
    ):
        _audit_current_react_runtime_trace(trace, state, steps, report)
        return

    collection_names = (
        "entities",
        "facts",
        "target_facts",
        "search_hypotheses",
        "tasks",
        "retrieval_anchors",
        "discoveries",
        "evidence",
        "findings",
        "failures",
        "discrepancy_coverage_audits",
    )
    for name in collection_names:
        if not isinstance(investigation.get(name), list):
            _issue(
                report,
                "UNIFIED_REACT_COLLECTION_INVALID",
                f"state.investigation_state.{name} must be an array",
                location=f"state.investigation_state.{name}",
            )

    entities = _rows(investigation.get("entities"))
    facts = _rows(investigation.get("facts"))
    targets = _rows(investigation.get("target_facts"))
    hypotheses = _rows(investigation.get("search_hypotheses"))
    tasks = _rows(investigation.get("tasks"))
    anchors = _rows(investigation.get("retrieval_anchors"))
    discoveries = _rows(investigation.get("discoveries"))
    evidence = _rows(investigation.get("evidence"))
    findings = _rows(investigation.get("findings"))
    failures = _rows(investigation.get("failures"))
    coverage_audits = _rows(
        investigation.get("discrepancy_coverage_audits")
    )

    entity_by_id = _unique_index(
        entities,
        id_field="entity_id",
        location_prefix="state.investigation_state.entities",
        report=report,
    )
    fact_by_id = _unique_index(
        facts,
        id_field="fact_id",
        location_prefix="state.investigation_state.facts",
        report=report,
    )
    target_by_claim = _unique_index(
        targets,
        id_field="claim_id",
        location_prefix="state.investigation_state.target_facts",
        report=report,
    )
    hypothesis_by_id = _unique_index(
        hypotheses,
        id_field="hypothesis_id",
        location_prefix="state.investigation_state.search_hypotheses",
        report=report,
    )
    task_by_id = _unique_index(
        tasks,
        id_field="task_id",
        location_prefix="state.investigation_state.tasks",
        report=report,
    )
    anchor_by_id = _unique_index(
        anchors,
        id_field="anchor_id",
        location_prefix="state.investigation_state.retrieval_anchors",
        report=report,
    )
    discovery_by_id = _unique_index(
        discoveries,
        id_field="discovery_id",
        location_prefix="state.investigation_state.discoveries",
        report=report,
    )
    evidence_by_id = _unique_index(
        evidence,
        id_field="evidence_id",
        location_prefix="state.investigation_state.evidence",
        report=report,
    )
    finding_by_id = _unique_index(
        findings,
        id_field="finding_id",
        location_prefix="state.investigation_state.findings",
        report=report,
    )
    failure_by_id = _unique_index(
        failures,
        id_field="failure_id",
        location_prefix="state.investigation_state.failures",
        report=report,
    )

    all_known_origins = {
        *entity_by_id,
        *fact_by_id,
        *target_by_claim,
        *hypothesis_by_id,
        *task_by_id,
        *anchor_by_id,
        *discovery_by_id,
        *evidence_by_id,
        *finding_by_id,
        *failure_by_id,
        str(_mapping(investigation.get("brief")).get("brief_id", "")).strip(),
        str(_mapping(investigation.get("brief")).get("case_id", "")).strip(),
    }
    all_known_origins.discard("")

    for fact_id, fact in fact_by_id.items():
        location = _location("state.investigation_state.facts", fact_id)
        unknown_basis = sorted(
            {
                str(item)
                for item in fact.get("basis_ids", []) or []
                if str(item)
            }
            - all_known_origins
        )
        if unknown_basis:
            _issue(
                report,
                "UNIFIED_VISUAL_FACT_BASIS_UNKNOWN",
                "unknown fact basis ids: " + ", ".join(unknown_basis),
                location=location,
            )

    for claim_id, target in target_by_claim.items():
        location = _location(
            "state.investigation_state.target_facts", claim_id
        )
        fact_id = str(target.get("fact_id", "")).strip()
        if fact_id not in fact_by_id:
            _issue(
                report,
                "UNIFIED_TARGET_FACT_UNKNOWN",
                f"target claim references unknown fact {fact_id!r}",
                location=location,
            )
        unknown_anchors = sorted(
            {
                str(item)
                for item in target.get("anchor_fact_ids", []) or []
                if str(item)
            }
            - set(fact_by_id)
        )
        if unknown_anchors:
            _issue(
                report,
                "UNIFIED_TARGET_ANCHOR_UNKNOWN",
                "target claim references unknown visual facts: "
                + ", ".join(unknown_anchors),
                location=location,
            )
        unknown_tasks = sorted(
            {
                str(item)
                for item in target.get("task_ids", []) or []
                if str(item)
            }
            - set(task_by_id)
        )
        if unknown_tasks:
            _issue(
                report,
                "UNIFIED_TARGET_TASK_UNKNOWN",
                "target claim references unknown task ids: "
                + ", ".join(unknown_tasks),
                location=location,
            )

    for hypothesis_id, hypothesis in hypothesis_by_id.items():
        location = _location(
            "state.investigation_state.search_hypotheses", hypothesis_id
        )
        unknown_claims = sorted(
            {
                str(item)
                for item in hypothesis.get("claim_ids", []) or []
                if str(item)
            }
            - set(target_by_claim)
        )
        if unknown_claims:
            _issue(
                report,
                "UNIFIED_HYPOTHESIS_CLAIM_UNKNOWN",
                "route references unknown target claims: "
                + ", ".join(unknown_claims),
                location=location,
            )
        task_id = str(hypothesis.get("task_id", "")).strip()
        if task_id not in task_by_id:
            _issue(
                report,
                "UNIFIED_HYPOTHESIS_TASK_UNKNOWN",
                f"route references unknown task {task_id!r}",
                location=location,
            )

    failures_by_task: dict[str, list[str]] = {}
    for failure_id, failure in failure_by_id.items():
        task_id = str(failure.get("task_id", "")).strip()
        failures_by_task.setdefault(task_id, []).append(failure_id)
        if task_id not in task_by_id:
            _issue(
                report,
                "UNIFIED_FAILURE_TASK_UNKNOWN",
                f"failure references unknown task {task_id!r}",
                location=_location(
                    "state.investigation_state.failures", failure_id
                ),
            )

    for task_id, task in task_by_id.items():
        location = _location("state.investigation_state.tasks", task_id)
        unknown_facts = sorted(
            {
                str(item)
                for item in task.get("fact_ids", []) or []
                if str(item)
            }
            - set(fact_by_id)
        )
        unknown_claims = sorted(
            {
                str(item)
                for item in task.get("claim_ids", []) or []
                if str(item)
            }
            - set(target_by_claim)
        )
        hypothesis_id = str(task.get("hypothesis_id") or "").strip()
        if unknown_facts:
            _issue(
                report,
                "UNIFIED_TASK_FACT_UNKNOWN",
                "task references unknown facts: " + ", ".join(unknown_facts),
                location=location,
            )
        if unknown_claims:
            _issue(
                report,
                "UNIFIED_TASK_CLAIM_UNKNOWN",
                "task references unknown claims: " + ", ".join(unknown_claims),
                location=location,
            )
        if hypothesis_id and hypothesis_id not in hypothesis_by_id:
            _issue(
                report,
                "UNIFIED_TASK_HYPOTHESIS_UNKNOWN",
                f"task references unknown route {hypothesis_id!r}",
                location=location,
            )
        parent_id = str(task.get("parent_task_id", "") or "").strip()
        if not hypothesis_id and parent_id not in task_by_id:
            _issue(
                report,
                "UNIFIED_TASK_PARENT_UNKNOWN",
                (
                    "a route-independent unified task must be a child of an "
                    "existing route task"
                ),
                location=location,
            )
        unknown_origins = sorted(
            {
                str(item)
                for item in task.get("origin_ids", []) or []
                if str(item)
            }
            - all_known_origins
        )
        if unknown_origins:
            _issue(
                report,
                "UNIFIED_TASK_ORIGIN_UNKNOWN",
                "task references unknown origins: " + ", ".join(unknown_origins),
                location=location,
            )
        if (
            str(task.get("status", "")).strip() == "blocked"
            and not failures_by_task.get(task_id)
        ):
            _issue(
                report,
                "UNIFIED_BLOCKED_TASK_WITHOUT_FAILURE",
                "blocked task must reference a recorded failure",
                location=location,
            )

    legacy_stages = {
        "perception",
        "image_account_planning",
        "image_only_planning",
        "image_only_investigation",
        "image_only_discrepancy_investigation",
        "image_only_query_concept_extraction",
        "image_only_query_replan",
        "image_only_route_local_replan",
        "image_only_evidence_decision",
    }
    for index, step in enumerate(steps):
        if str(step.get("stage", "")).strip() in legacy_stages:
            _issue(
                report,
                "UNIFIED_REACT_LEGACY_STAGE",
                "unified trace must not include a legacy policy stage",
                location=_step_label(index, step),
            )

    actions = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_react"
        and str(step.get("action_type", "")).strip() == "tool_call"
    ]
    if len(actions) < 3:
        _issue(
            report,
            "UNIFIED_REACT_ACTIONS_MISSING",
            "unified trace requires scene, OCR, and one investigation action",
            location="state.all_steps",
        )
        _audit_unified_react_interaction_chains(steps, report)
        return

    bootstrap_actions = actions[:2]
    bootstrap_names = {
        str(step.get("tool_name", "")).strip()
        for _, step in bootstrap_actions
    }
    if bootstrap_names != {"perceive_scene", "ocr_with_position"}:
        _issue(
            report,
            "UNIFIED_REACT_BOOTSTRAP_INVALID",
            "the first two accepted actions must be perceive_scene and ocr_with_position",
            location="state.all_steps",
        )
    completed = {
        str(item)
        for item in investigation.get(
            "unified_react_bootstrap_tools_completed", []
        )
        or []
    }
    if completed != {"perceive_scene", "ocr_with_position"}:
        _issue(
            report,
            "UNIFIED_REACT_BOOTSTRAP_STATE_INVALID",
            "unified workspace must persist both completed visual bootstrap tools",
            location=(
                "state.investigation_state."
                "unified_react_bootstrap_tools_completed"
            ),
        )
    if investigation.get("unified_react_bootstrap_failures"):
        _issue(
            report,
            "UNIFIED_REACT_BOOTSTRAP_FAILURE_RECORDED",
            "successful unified trace cannot retain a bootstrap failure",
            location=(
                "state.investigation_state."
                "unified_react_bootstrap_failures"
            ),
        )

    action_ids: set[str] = set()
    for action_position, (index, step) in enumerate(actions):
        metadata = _mapping(step.get("metadata"))
        delta = _mapping(metadata.get("unified_react_delta"))
        location = _step_label(index, step)
        tool_name = str(step.get("tool_name", "")).strip()
        call_id = str(metadata.get("function_call_id", "")).strip()
        if not delta:
            _issue(
                report,
                "UNIFIED_REACT_DELTA_MISSING",
                "every accepted unified tool action requires a persisted reducer delta",
                location=location,
            )
            continue
        if str(delta.get("tool_name", "")).strip() != tool_name:
            _issue(
                report,
                "UNIFIED_REACT_DELTA_TOOL_MISMATCH",
                "reducer delta tool_name must match the executed tool",
                location=location,
            )
        if str(delta.get("function_call_id", "")).strip() != call_id:
            _issue(
                report,
                "UNIFIED_REACT_DELTA_CALL_MISMATCH",
                "reducer delta function_call_id must match the action",
                location=location,
            )
        if not _mapping(delta.get("state_update")).get("accepted", False):
            _issue(
                report,
                "UNIFIED_REACT_DELTA_REJECTED",
                "accepted tool action has no accepted reducer state update",
                location=location,
            )
        if not call_id:
            _issue(
                report,
                "UNIFIED_REACT_CALL_ID_MISSING",
                "accepted unified tool action lacks function_call_id",
                location=location,
            )
        elif call_id in action_ids:
            _issue(
                report,
                "UNIFIED_REACT_CALL_ID_DUPLICATE",
                f"duplicate unified function_call_id {call_id!r}",
                location=location,
            )
        action_ids.add(call_id)
        policy_action = _mapping(metadata.get("policy_action"))
        if (
            str(policy_action.get("type", "")).strip() != "tool_call"
            or str(policy_action.get("name", "")).strip() != tool_name
        ):
            _issue(
                report,
                "UNIFIED_REACT_POLICY_ACTION_MISMATCH",
                "native policy_action must record the same tool call",
                location=location,
            )
        if action_position == 2:
            intent = _mapping(delta.get("accepted_investigation_intent"))
            update = _mapping(delta.get("state_update"))
            if not intent or not _mapping(update.get("initial_intent")):
                _issue(
                    report,
                    "UNIFIED_REACT_INITIAL_INTENT_MISSING",
                    "first non-bootstrap action must persist its accepted intent",
                    location=location,
                )
            if not target_by_claim or not hypothesis_by_id or not task_by_id:
                _issue(
                    report,
                    "UNIFIED_REACT_INITIAL_GRAPH_MISSING",
                    "first non-bootstrap action must create target, route, and task",
                    location=location,
                )

    budget_actions = [
        step
        for _, step in actions
        if is_unified_react_budget_action(step)
    ]
    action_count = int(investigation.get("action_count", 0) or 0)
    if action_count != len(budget_actions):
        _issue(
            report,
            "UNIFIED_REACT_ACTION_COUNT_MISMATCH",
            (
                f"action_count={action_count}, but trace records "
                f"{len(budget_actions)} accepted investigation actions"
            ),
            location="state.investigation_state.action_count",
        )
    if action_count > 24:
        _issue(
            report,
            "UNIFIED_REACT_ACTION_BUDGET_EXCEEDED",
            f"unified investigation used {action_count} actions; maximum is 24",
            location="state.investigation_state.action_count",
        )

    current_action_count = 0
    reflection_counts: list[int] = []
    decision_steps = 0
    judgment_steps = 0
    for _index, step in enumerate(steps):
        stage = str(step.get("stage", "")).strip()
        if (
            stage == "unified_react"
            and str(step.get("action_type", "")).strip() == "tool_call"
            and is_unified_react_budget_action(step)
        ):
            current_action_count += 1
        if (
            stage == "unified_reflection"
            and _mapping(step.get("metadata")).get("native_interactions")
        ):
            reflection_counts.append(current_action_count)
        elif (
            stage == "unified_discrepancy_decision"
            and _mapping(step.get("metadata")).get("native_interactions")
        ):
            decision_steps += 1
        elif (
            stage == "unified_judgment"
            and _mapping(step.get("metadata")).get("native_interactions")
        ):
            judgment_steps += 1
    invalid_reflections = [
        count
        for count in reflection_counts
        if count <= 0 or count % 4 != 0
    ]
    terminal_stop = str(investigation.get("stop_reason", "")).strip()
    required_limit = (
        action_count - 1
        if terminal_stop and action_count > 0 and action_count % 4 == 0
        else action_count
    )
    expected_reflections = list(range(4, required_limit + 1, 4))
    allowed_reflection_counts = set(expected_reflections)
    if terminal_stop and action_count > 0 and action_count % 4 == 0:
        # A final Reflection at the same boundary may be what establishes the
        # terminal global state.  Earlier four-action checkpoints remain
        # mandatory, but this terminal one is optional.
        allowed_reflection_counts.add(action_count)
    missing_reflections = [
        count for count in expected_reflections if count not in reflection_counts
    ]
    unexpected_reflections = [
        count
        for count in reflection_counts
        if count not in allowed_reflection_counts
    ]
    if (
        invalid_reflections
        or len(reflection_counts) != len(set(reflection_counts))
        or missing_reflections
        or unexpected_reflections
    ):
        _issue(
            report,
            "UNIFIED_REACT_REFLECTION_CADENCE_INVALID",
            (
                "unified Reflection is only allowed at non-terminal four-action "
                "boundaries, with an optional terminal reflection; "
                f"expected={expected_reflections}, got={reflection_counts}"
            ),
            location="state.all_steps",
        )
    if not decision_steps:
        _issue(
            report,
            "UNIFIED_REACT_DECISION_MISSING",
            "unified trace must include at least one Discrepancy Decision",
            location="state.all_steps",
        )
    if judgment_steps != 1:
        _issue(
            report,
            "UNIFIED_REACT_JUDGMENT_COUNT_INVALID",
            f"unified trace must include exactly one Judgment, got {judgment_steps}",
            location="state.all_steps",
        )

    terminal_coverage = [
        item
        for item in coverage_audits
        if str(item.get("stop_reason", "")).strip() == terminal_stop
        and terminal_stop
    ]
    if not terminal_coverage:
        _issue(
            report,
            "UNIFIED_REACT_TERMINAL_COVERAGE_MISSING",
            "unified trace requires a terminal discrepancy Coverage record",
            location=(
                "state.investigation_state."
                "discrepancy_coverage_audits"
            ),
        )

    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    if str(basis.get("policy_rule_id", "")).strip() != UNIFIED_REACT_POLICY_VERSION:
        _issue(
            report,
            "UNIFIED_REACT_BASIS_POLICY_INVALID",
            "unified verdict basis must use unified-react-v1",
            location="verdict_basis.policy_rule_id",
        )
    if str(judgment.get("policy_rule_id", "")).strip() != UNIFIED_REACT_POLICY_VERSION:
        _issue(
            report,
            "UNIFIED_REACT_JUDGMENT_POLICY_INVALID",
            "unified Judgment must use unified-react-v1",
            location="judgment.policy_rule_id",
        )
    verdict = str(trace.get("verdict", "")).strip()
    if verdict not in {"real", "fake"}:
        _issue(
            report,
            "UNIFIED_REACT_VERDICT_INVALID",
            f"unified trace verdict must be real or fake, got {verdict!r}",
            location="verdict",
        )
    if str(judgment.get("verdict", "")).strip() != verdict:
        _issue(
            report,
            "UNIFIED_REACT_JUDGMENT_VERDICT_MISMATCH",
            "trace verdict and Judgment verdict must match",
            location="judgment.verdict",
        )
    for basis_field, judgment_field, known in (
        ("claim_ids", "selected_claim_ids", set(target_by_claim)),
        (
            "visual_anchor_fact_ids",
            "selected_visual_anchor_fact_ids",
            set(fact_by_id),
        ),
        ("finding_ids", "selected_finding_ids", set(finding_by_id)),
        ("evidence_ids", "selected_evidence_ids", set(evidence_by_id)),
    ):
        selected = {
            str(item)
            for item in basis.get(basis_field, []) or []
            if str(item)
        }
        unknown = sorted(selected - known)
        if unknown:
            _issue(
                report,
                "UNIFIED_REACT_BASIS_REFERENCE_UNKNOWN",
                f"unknown {basis_field}: " + ", ".join(unknown),
                location=f"verdict_basis.{basis_field}",
            )
        judgment_selected = {
            str(item)
            for item in judgment.get(judgment_field, []) or []
            if str(item)
        }
        if selected != judgment_selected:
            _issue(
                report,
                "UNIFIED_REACT_JUDGMENT_BASIS_MISMATCH",
                (
                    f"Judgment {judgment_field} must equal verdict basis "
                    f"{basis_field}"
                ),
                location=f"judgment.{judgment_field}",
            )
    if str(basis.get("decision_mode", "")).strip() not in {
        "evidence_determined",
        "bounded_binary_judgment",
    }:
        _issue(
            report,
            "UNIFIED_REACT_DECISION_MODE_INVALID",
            "unified verdict basis has an unsupported decision_mode",
            location="verdict_basis.decision_mode",
        )

    report.stats.update(
        {
            "unified_react_actions": action_count,
            "unified_react_target_facts": len(target_by_claim),
            "unified_react_routes": len(hypothesis_by_id),
            "unified_react_tasks": len(task_by_id),
            "unified_react_reflections": len(reflection_counts),
            "unified_react_decisions": decision_steps,
        }
    )
    _audit_unified_react_interaction_chains(steps, report)


def _audit_current_react_runtime_trace(
    trace: Mapping[str, Any],
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Audit the active compact ReAct runtime without legacy graph semantics."""

    investigation = _mapping(state.get("investigation_state"))
    required_lists = (
        "discoveries",
        "evidence",
        "failures",
        "attempted_queries",
        "visited_urls",
        "attempted_actions",
        "recent_actions",
        "open_questions",
    )
    for name in required_lists:
        if not isinstance(investigation.get(name), list):
            _issue(
                report,
                "REACT_RUNTIME_COLLECTION_INVALID",
                f"current ReAct state field {name!r} must be an array",
                location=f"state.investigation_state.{name}",
            )
    if not isinstance(investigation.get("visual_memory"), Mapping):
        _issue(
            report,
            "REACT_RUNTIME_VISUAL_MEMORY_INVALID",
            "current ReAct state must contain a visual_memory object",
            location="state.investigation_state.visual_memory",
        )

    retired_state_fields = {
        "target_facts",
        "search_hypotheses",
        "claim_assessments",
        "material_discrepancies",
        "tasks",
        "findings",
        "retrieval_anchors",
        "brief",
        "core_verdict_fact_id",
    }
    present_retired_fields = sorted(
        field_name for field_name in retired_state_fields if field_name in investigation
    )
    if present_retired_fields:
        _issue(
            report,
            "REACT_RUNTIME_LEGACY_STATE",
            "current ReAct state contains retired graph fields: "
            + ", ".join(present_retired_fields),
            location="state.investigation_state",
        )

    actions = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_react"
        and str(step.get("action_type", "")).strip() == "tool_call"
    ]
    if not actions:
        _issue(
            report,
            "REACT_RUNTIME_ACTIONS_MISSING",
            "current ReAct trace requires at least one accepted tool action",
            location="state.all_steps",
        )

    accepted_tool_names = set(REACT_RUNTIME_TOOLS) | {"finish_investigation"}
    call_ids: set[str] = set()
    native_interactions: list[tuple[int, Mapping[str, Any]]] = []
    for index, step in actions:
        location = _step_label(index, step)
        tool_name = str(step.get("tool_name", "")).strip()
        metadata = _mapping(step.get("metadata"))
        call_id = str(metadata.get("function_call_id", "")).strip()
        if tool_name not in accepted_tool_names:
            _issue(
                report,
                "REACT_RUNTIME_TOOL_UNKNOWN",
                f"unknown current ReAct tool {tool_name!r}",
                location=location,
            )
        if not call_id:
            _issue(
                report,
                "REACT_RUNTIME_CALL_ID_MISSING",
                "accepted current ReAct action lacks function_call_id",
                location=location,
            )
        elif call_id in call_ids:
            _issue(
                report,
                "REACT_RUNTIME_CALL_ID_DUPLICATE",
                f"duplicate current ReAct function_call_id {call_id!r}",
                location=location,
            )
        call_ids.add(call_id)

        delta = _mapping(
            metadata.get("react_state_delta")
            or metadata.get("investigation_state_update")
        )
        if not delta.get("accepted", False):
            _issue(
                report,
                "REACT_RUNTIME_DELTA_REJECTED",
                "accepted current ReAct action has no accepted reducer delta",
                location=location,
            )
        policy_action = _mapping(metadata.get("policy_action"))
        if (
            str(policy_action.get("type", "")).strip() != "tool_call"
            or str(policy_action.get("name", "")).strip() != tool_name
        ):
            _issue(
                report,
                "REACT_RUNTIME_POLICY_ACTION_MISMATCH",
                "policy_action must record the same current ReAct tool call",
                location=location,
            )

        try:
            _payload, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        except Exception as exc:
            succeeded = False
            _issue(
                report,
                "REACT_RUNTIME_TOOL_RESULT_INVALID",
                f"cannot parse tool result: {exc}",
                location=location,
            )
        if not succeeded and not delta.get("failure"):
            _issue(
                report,
                "REACT_RUNTIME_FAILURE_NOT_RECORDED",
                "a failed tool result must produce a reducer failure record",
                location=location,
            )

        if _mapping(metadata).get("native_interactions"):
            native_interactions.append((index, metadata))

    budget_actions = [
        step
        for _, step in actions
        if is_unified_react_runtime_budget_action(step)
    ]
    action_count = int(investigation.get("action_count", 0) or 0)
    if action_count != len(budget_actions):
        _issue(
            report,
            "REACT_RUNTIME_ACTION_COUNT_MISMATCH",
            (
                f"action_count={action_count}, but trace records "
                f"{len(budget_actions)} budgeted ReAct actions"
            ),
            location="state.investigation_state.action_count",
        )
    if action_count < 0 or action_count > 24:
        _issue(
            report,
            "REACT_RUNTIME_ACTION_BUDGET_INVALID",
            f"current ReAct action_count must be between 0 and 24, got {action_count}",
            location="state.investigation_state.action_count",
        )

    for index, metadata in native_interactions:
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        parent_id = str(metadata.get("previous_interaction_id", "") or "").strip()
        lifecycle = str(
            metadata.get("interaction_lifecycle_kind", "")
        ).strip()
        location = _step_label(index, steps[index])
        if not interaction_id:
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_ID_MISSING",
                "native current ReAct action lacks interaction_id",
                location=location,
            )
        if lifecycle != "tool_roundtrip":
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_LIFECYCLE_INVALID",
                "native current ReAct tool action must use tool_roundtrip lifecycle",
                location=location,
            )
        if parent_id:
            _issue(
                report,
                "REACT_RUNTIME_INTERACTION_PARENT_INVALID",
                (
                    "each current ReAct action starts a fresh compact request; "
                    "previous_interaction_id must be empty"
                ),
                location=location,
            )

    judgment_outputs = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")).strip() == "unified_judgment"
        and str(step.get("action_type", "")).strip() == "output"
    ]
    if len(judgment_outputs) != 1:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_COUNT_INVALID",
            f"current ReAct trace must contain exactly one final Judgment, got {len(judgment_outputs)}",
            location="state.all_steps",
        )

    basis = _mapping(trace.get("verdict_basis"))
    if str(basis.get("schema_version", "")).strip() != (
        "ifv-unified-judgment-basis-v1"
    ):
        _issue(
            report,
            "REACT_RUNTIME_BASIS_INVALID",
            "current ReAct verdict basis has an invalid schema_version",
            location="verdict_basis.schema_version",
        )
    if str(basis.get("decision_mode", "")).strip() != "bounded_binary_judgment":
        _issue(
            report,
            "REACT_RUNTIME_DECISION_MODE_INVALID",
            "current ReAct verdict basis must use bounded_binary_judgment",
            location="verdict_basis.decision_mode",
        )

    evidence_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", "")).strip()
    }
    selected_evidence_ids = {
        str(item).strip()
        for item in basis.get("evidence_ids", []) or []
        if str(item).strip()
    }
    unknown_basis_evidence = sorted(selected_evidence_ids - evidence_ids)
    if unknown_basis_evidence:
        _issue(
            report,
            "REACT_RUNTIME_BASIS_EVIDENCE_UNKNOWN",
            "verdict basis references unknown evidence: "
            + ", ".join(unknown_basis_evidence),
            location="verdict_basis.evidence_ids",
        )

    judgment = _mapping(trace.get("judgment") or state.get("judgment"))
    verdict = str(trace.get("verdict", "")).strip()
    if verdict not in {"real", "fake"}:
        _issue(
            report,
            "REACT_RUNTIME_VERDICT_INVALID",
            f"current ReAct verdict must be real or fake, got {verdict!r}",
            location="verdict",
        )
    if str(judgment.get("policy_rule_id", "")).strip() != (
        UNIFIED_REACT_POLICY_VERSION
    ):
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_POLICY_INVALID",
            "current ReAct Judgment must use unified-react-v1",
            location="judgment.policy_rule_id",
        )
    if str(judgment.get("verdict", "")).strip() != verdict:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_VERDICT_MISMATCH",
            "trace verdict and current ReAct Judgment verdict must match",
            location="judgment.verdict",
        )
    judgment_selected_evidence = {
        str(item).strip()
        for item in judgment.get("selected_evidence_ids", []) or []
        if str(item).strip()
    }
    if judgment_selected_evidence != selected_evidence_ids:
        _issue(
            report,
            "REACT_RUNTIME_JUDGMENT_BASIS_MISMATCH",
            "Judgment selected_evidence_ids must equal verdict_basis.evidence_ids",
            location="judgment.selected_evidence_ids",
        )
    if not isinstance(judgment.get("fact_check_report"), Mapping):
        _issue(
            report,
            "REACT_RUNTIME_REPORT_MISSING",
            "current ReAct Judgment must contain fact_check_report",
            location="judgment.fact_check_report",
        )

    report.stats.update(
        {
            "unified_react_actions": action_count,
            "unified_react_target_facts": 0,
            "unified_react_routes": 0,
            "unified_react_tasks": 0,
            "unified_react_reflections": 0,
            "unified_react_decisions": 0,
            "react_runtime_discoveries": len(
                _rows(investigation.get("discoveries"))
            ),
            "react_runtime_evidence": len(_rows(investigation.get("evidence"))),
            "react_runtime_failures": len(_rows(investigation.get("failures"))),
        }
    )


def audit_trace(
    path: Path,
    *,
    enforce_source_access_policy: bool = True,
    source_access_policy: SourceAccessPolicy | None = None,
) -> TraceReport:
    report = TraceReport(path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(report, "TRACE_JSON_INVALID", f"cannot read trace JSON: {exc}")
        return report
    if not isinstance(payload, Mapping):
        _issue(report, "TRACE_ROOT_INVALID", "trace JSON root must be an object")
        return report

    state = _state(payload)
    report.image_id = str(payload.get("image_id", state.get("image_id", path.stem)))
    steps = _rows(state.get("all_steps"))
    is_image_only = str(
        payload.get("input_mode") or state.get("input_mode") or ""
    ) == "image_only"
    report.stats.update(
        {
            "steps": len(steps),
        }
    )

    if "state" not in payload or not isinstance(payload.get("state"), Mapping):
        _issue(report, "CANONICAL_STATE_MISSING", "canonical trace must contain a state object")
    if not isinstance(state.get("all_steps"), list):
        _issue(report, "STEPS_MISSING", "canonical state must contain all_steps")
    if not is_image_only:
        _issue(
            report,
            "UNSUPPORTED_TRACE_MODE",
            "v3 strict audit accepts input_mode=image_only only",
            location="input_mode",
        )
        return report

    _audit_termination(payload, state, report)
    _audit_thought_tokens(payload, state, steps, report)
    policy_version = str(
        payload.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version == UNIFIED_REACT_POLICY_VERSION:
        _audit_unified_react_trace(payload, state, steps, report)
    else:
        _issue(
            report,
            "UNSUPPORTED_POLICY_VERSION",
            (
                "current strict audit accepts only "
                f"{UNIFIED_REACT_POLICY_VERSION}; legacy policy traces are archived"
            ),
            location="decision_policy_version",
        )
    _audit_leaks(
        payload,
        report,
        enforce_source_access_policy=enforce_source_access_policy,
        source_access_policy=source_access_policy,
    )
    _audit_rejections(steps, report)
    return report


def discover_trace_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [
            input_path
        ] if input_path.suffix.casefold() == ".json" and not _looks_like_benchmark_artifact(
            input_path
        ) else []
    if not input_path.is_dir():
        return []
    if input_path.name.casefold() == "traces":
        return sorted(
            path
            for path in input_path.glob("*.json")
            if path.is_file() and not _looks_like_benchmark_artifact(path)
        )
    traces_dirs = sorted(
        path for path in input_path.rglob("traces") if path.is_dir()
    )
    if traces_dirs:
        return sorted(
            {
                trace
                for directory in traces_dirs
                for trace in directory.glob("*.json")
                if trace.is_file() and not _looks_like_benchmark_artifact(trace)
            }
        )
    return sorted(
        path
        for path in input_path.glob("*.json")
        if path.is_file() and not _looks_like_benchmark_artifact(path)
    )


def _looks_like_benchmark_artifact(path: Path) -> bool:
    stem = path.stem.casefold()
    if stem in {"run_manifest", "summary", "predictions", "manifest"}:
        return True
    return bool(re.search(r"(?:^|[-_.])(benchmark|gold)(?:$|[-_.])", stem))


def _print_human(reports: Sequence[TraceReport], *, strict_scheduler: bool) -> None:
    for report in reports:
        failures = report.failures(strict_scheduler=strict_scheduler)
        warnings = report.warnings(strict_scheduler=strict_scheduler)
        status = "PASS" if not failures else "FAIL"
        print(
            f"[{status}] {report.path} "
            f"(termination={report.termination or 'unknown'}, failures={len(failures)}, warnings={len(warnings)})"
        )
        print(
            "  "
            f"steps={report.stats.get('steps', 0)} "
            f"claims={report.stats.get('claims', 0)} "
            f"evidence={report.stats.get('evidence', 0)} "
            f"scheduler_rejections={report.stats.get('scheduler_rejections', 0)} "
            f"protocol_rejections={report.stats.get('protocol_rejections', 0)} "
            f"route_control_rejections={report.stats.get('route_control_rejections', 0)}"
        )
        failure_set = set(failures)
        for issue in report.issues:
            severity = "ERROR" if issue in failure_set else "WARN"
            where = f" at {issue.location}" if issue.location else ""
            print(f"  {severity} {issue.code}{where}: {issue.message}")

    failed = sum(
        bool(report.failures(strict_scheduler=strict_scheduler)) for report in reports
    )
    warnings = sum(
        len(report.warnings(strict_scheduler=strict_scheduler)) for report in reports
    )
    scheduler_rejections = sum(
        report.stats.get("scheduler_rejections", 0) for report in reports
    )
    protocol_rejections = sum(
        report.stats.get("protocol_rejections", 0) for report in reports
    )
    route_control_rejections = sum(
        report.stats.get("route_control_rejections", 0)
        for report in reports
    )
    print(
        f"Audited {len(reports)} trace(s): {len(reports) - failed} passed, "
        f"{failed} failed, {warnings} warning(s), "
        f"{scheduler_rejections} scheduler rejection(s), "
        f"{protocol_rejections} protocol rejection(s), "
        f"{route_control_rejections} route-control rejection(s)."
    )


def _json_summary(
    reports: Sequence[TraceReport], *, strict_scheduler: bool
) -> dict[str, Any]:
    failed = sum(
        bool(report.failures(strict_scheduler=strict_scheduler)) for report in reports
    )
    return {
        "passed": failed == 0,
        "strict_scheduler": strict_scheduler,
        "trace_count": len(reports),
        "passed_count": len(reports) - failed,
        "failed_count": failed,
        "warning_count": sum(
            len(report.warnings(strict_scheduler=strict_scheduler)) for report in reports
        ),
        "scheduler_rejections": sum(
            report.stats.get("scheduler_rejections", 0) for report in reports
        ),
        "protocol_rejections": sum(
            report.stats.get("protocol_rejections", 0) for report in reports
        ),
        "route_control_rejections": sum(
            report.stats.get("route_control_rejections", 0)
            for report in reports
        ),
        "traces": [
            report.to_dict(strict_scheduler=strict_scheduler) for report in reports
        ],
    }


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Strictly audit canonical v3 image-only traces."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A canonical trace JSON file or a directory containing traces.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one machine-readable JSON summary.",
    )
    parser.add_argument(
        "--strict-scheduler",
        action="store_true",
        help="Treat runtime action/output rejections as failures instead of warnings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_trace_files(args.input_path)
    if not files:
        message = f"No trace JSON files found at: {args.input_path}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "strict_scheduler": args.strict_scheduler,
                        "trace_count": 0,
                        "passed_count": 0,
                        "failed_count": 0,
                        "warning_count": 0,
                        "error": message,
                        "traces": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(message, file=sys.stderr)
        return 1

    reports = [audit_trace(path) for path in files]
    if args.json_output:
        print(
            json.dumps(
                _json_summary(reports, strict_scheduler=args.strict_scheduler),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_human(reports, strict_scheduler=args.strict_scheduler)
    return int(
        any(
            report.failures(strict_scheduler=args.strict_scheduler)
            for report in reports
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
