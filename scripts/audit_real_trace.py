"""Audit canonical traces from a small set of real verification runs."""
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

from src.orchestrator.evidence_policy import (  # noqa: E402
    VISUAL_OBSERVATION_TOOLS,
    query_targets_fact_check_answer,
    tool_can_decide_claim,
)
from src.orchestrator.source_access import (  # noqa: E402
    FACT_CHECK_DOMAIN_MARKERS,
    SourceAccessPolicy,
    benchmark_source_access_policy,
    url_variants,
)
from src.orchestrator.state import EvidenceRecord  # noqa: E402
from src.orchestrator.tool_result import parse_tool_result  # noqa: E402


HARD = "hard"
SCHEDULER = "scheduler"
PROTOCOL = "protocol"
REJECTION_ACTIONS = frozenset({"format_error", "output_rejected"})
WEB_EVIDENCE_TOOLS = frozenset({"visit", "text_search", "crop_and_search"})
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
    locations: list[tuple[str, Any]] = []
    for prefix, usage in (
        ("token_usage", trace.get("token_usage")),
        ("state.token_usage", state.get("token_usage")),
    ):
        row = _mapping(usage)
        if "thought" in row:
            locations.append((f"{prefix}.thought", row.get("thought")))
        if "thought_tokens" in row:
            locations.append((f"{prefix}.thought_tokens", row.get("thought_tokens")))
        if "total_thought_tokens" in row:
            locations.append(
                (f"{prefix}.total_thought_tokens", row.get("total_thought_tokens"))
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
                locations.append((f"state.all_steps[{index}].tokens.{key}", tokens[key]))

    recorded_paths = {path for path, _ in locations}
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
            locations.append((path, raw_value))
            recorded_paths.add(path)

    report.stats["native_interaction_steps"] = native_steps
    report.stats["thought_token_fields"] = len(locations)
    for path, value in locations:
        numeric = _numeric_token(value)
        if numeric is None:
            _issue(
                report,
                "THOUGHT_TOKENS_INVALID",
                f"thought-token value must be numeric zero, got {value!r}",
                location=path,
            )
        elif numeric != 0:
            _issue(
                report,
                "THOUGHT_TOKENS_NONZERO",
                f"Gemini thought tokens must be zero, got {value!r}",
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


def _audit_external_claims(
    claims: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence:
        by_claim.setdefault(str(item.get("claim_id", "")).strip(), []).append(item)
    source_records = {
        str(source.get("source_id", "")).strip(): source
        for source in sources
        if str(source.get("source_id", "")).strip()
    }
    successful_steps: dict[str, Mapping[str, Any]] = {}
    duplicate_calls: set[str] = set()
    for step in steps:
        call_id = str(_mapping(step.get("metadata")).get("function_call_id", "")).strip()
        if not call_id or not _parse_successful_tool_step(step):
            continue
        if call_id in successful_steps:
            duplicate_calls.add(call_id)
        successful_steps[call_id] = step

    decided = 0
    for claim_index, claim in enumerate(claims):
        if str(claim.get("claim_scope", "external_fact")) != "external_fact":
            continue
        status = str(claim.get("status", "")).strip()
        if status not in {"supported", "refuted"}:
            continue
        decided += 1
        claim_id = str(claim.get("claim_id", claim_index)).strip()
        expected_stance = "support" if status == "supported" else "refute"
        eligible: list[Mapping[str, Any]] = []
        for item in by_claim.get(claim_id, []):
            try:
                validated = EvidenceRecord.model_validate(dict(item))
            except Exception:
                continue
            step = successful_steps.get(validated.function_call_id)
            source = source_records.get(validated.source_id, {})
            source_url = str(source.get("canonical_url", "")).strip()
            if (
                validated.evidence_kind == "web_span"
                and validated.directness == "direct"
                and validated.stance == expected_stance
                and validated.tool_name in WEB_EVIDENCE_TOOLS
                and validated.function_call_id not in duplicate_calls
                and step is not None
                and str(step.get("tool_name", "")).strip() == validated.tool_name
                and url_variants(source_url)
                and str(source.get("artifact_sha256", "")).strip().casefold()
                == validated.artifact_sha256
                and tool_can_decide_claim(validated.tool_name, "external_fact")
                and _tool_result_has_eligible_web_record(
                    step,
                    evidence=validated,
                    source_url=source_url,
                    claim_text=str(claim.get("text", "")).strip(),
                )
            ):
                eligible.append(item)
        if not eligible:
            _issue(
                report,
                "EXTERNAL_FACT_MISSING_DIRECT_WEB_EVIDENCE",
                f"{status} external_fact requires matching-stance direct eligible Web evidence",
                location=_location("state.ledgers.claims", claim_id),
            )
    report.stats["decided_external_fact_claims"] = decided


def _tool_result_has_eligible_web_record(
    step: Mapping[str, Any],
    *,
    evidence: EvidenceRecord,
    source_url: str,
    claim_text: str,
) -> bool:
    try:
        payload, succeeded = parse_tool_result(str(step.get("tool_result", "")))
    except Exception:
        return False
    if not succeeded:
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, Mapping):
            span = _mapping(value.get("evidence_span"))
            record_url = str(
                value.get("selected_url", "") or value.get("url", "")
            ).strip()
            record_stance = str(value.get("stance", "")).strip().casefold()
            matches = (
                str(value.get("evidence", "")) == evidence.exact_text
                and bool(value.get("evidence_eligible"))
                and not value.get("injection_flags")
                and str(value.get("directness", "")).strip().casefold() == "direct"
                and record_stance == evidence.stance
                and str(value.get("relevance", "")).strip().casefold()
                in {"high", "medium", "low"}
                and str(value.get("goal", "")).strip() == claim_text
                and str(value.get("artifact_sha256", "")).strip().casefold()
                == evidence.artifact_sha256
                and str(value.get("retrieved_at", "")).strip() == evidence.retrieved_at
                and span.get("start") == evidence.span_start
                and span.get("end") == evidence.span_end
                and bool(set(url_variants(record_url)) & set(url_variants(source_url)))
            )
            if matches:
                return True
            return any(walk(child) for child in value.values())
        if isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(payload)


def _audit_visual_external_evidence(
    claims: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    scopes = {
        str(claim.get("claim_id", "")).strip(): str(
            claim.get("claim_scope", "external_fact")
        ).strip()
        for claim in claims
    }
    for evidence_index, item in enumerate(evidence):
        claim_id = str(item.get("claim_id", "")).strip()
        tool_name = str(item.get("tool_name", "")).strip()
        if scopes.get(claim_id) != "external_fact":
            continue
        if tool_name not in VISUAL_OBSERVATION_TOOLS and item.get("evidence_kind") != "image_region":
            continue
        if item.get("stance") != "neutral":
            _issue(
                report,
                "VISUAL_EXTERNAL_FACT_NOT_NEUTRAL",
                f"visual tool {tool_name!r} must record neutral ledger evidence for external_fact",
                location=_location(
                    "state.ledgers.evidence", item.get("evidence_id", evidence_index)
                ),
            )


def _fact_check_domain(value: str) -> str:
    policy = benchmark_source_access_policy(
        [value],
        policy_id="canonical-trace-audit",
    )
    return next(iter(sorted(policy.excluded_domains)), "")


def _fact_check_query_domain(value: str) -> str:
    text = unquote(str(value or ""))
    candidates = re.findall(
        r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s\"'<>]*)?",
        text,
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        if domain := _fact_check_domain(candidate.rstrip(".,;:!?)]}")):
            return domain
    return ""


def _fact_check_embedded_url(value: str) -> str:
    urls = re.findall(
        r"https?://[^\s\"'<>]+",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    for url in urls:
        if domain := _fact_check_domain(url.rstrip(".,;:!?)]}")):
            return domain
    return ""


def _fact_check_url_query(value: str) -> str:
    for variant in url_variants(value):
        try:
            query = " ".join(
                f"{key} {item}" for key, item in parse_qsl(urlsplit(variant).query)
            )
        except ValueError:
            continue
        if domain := _fact_check_query_reference(query):
            return f"known fact-check domain {domain!r} in URL query"
        if query_targets_fact_check_answer(query):
            return "fact-check-oriented URL query"
    return ""


def _fact_check_query_reference(value: str) -> str:
    return (
        KNOWN_FACT_CHECK_QUERY_POLICY.blocked_query_reference(value)
        or _fact_check_query_domain(value)
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


def _audit_leaks(trace: Mapping[str, Any], report: TraceReport) -> None:
    seen: set[tuple[str, str]] = set()
    url_count = 0
    query_count = 0
    for key, raw_value, path in _iter_named_values(trace):
        query_key = _looks_like_query_key(key)
        if _looks_like_url_key(key):
            domain = _fact_check_domain(raw_value) or _fact_check_query_domain(raw_value)
        elif not query_key:
            domain = _fact_check_embedded_url(raw_value)
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
        if _looks_like_url_key(key) and (
            url_query_leak := _fact_check_url_query(raw_value)
        ):
            signature = ("query", path)
            if signature not in seen:
                seen.add(signature)
                query_count += 1
                _issue(
                    report,
                    "FACT_CHECK_QUERY_LEAK",
                    f"{url_query_leak} leaked into the trace: {raw_value!r}",
                    location=path,
                )
        if query_key:
            query_domain = _fact_check_query_reference(raw_value)
            oriented = query_targets_fact_check_answer(raw_value)
            if query_domain or oriented:
                signature = ("query", path)
                if signature not in seen:
                    seen.add(signature)
                    query_count += 1
                    detail = (
                        f"known fact-check domain {query_domain!r} is named in query"
                        if query_domain
                        else "fact-check-oriented query"
                    )
                    _issue(
                        report,
                        "FACT_CHECK_QUERY_LEAK",
                        f"{detail} leaked into the trace: {raw_value!r}",
                        location=path,
                    )
    report.stats["fact_check_url_leaks"] = url_count
    report.stats["fact_check_query_leaks"] = query_count


def _audit_interaction_chains(
    steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    groups: dict[tuple[str, int], list[tuple[int, Mapping[str, Any]]]] = {}
    verification_steps = 0
    native_verification_steps = 0
    iteration_order: list[int] = []
    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        if str(step.get("stage", "")) != "verification":
            continue
        verification_steps += 1
        if not metadata.get("native_interactions"):
            _issue(
                report,
                "INTERACTION_METADATA_MISSING",
                "verification step is not recorded as Gemini Interactions",
                location=_step_label(index, step),
            )
            continue
        native_verification_steps += 1
        raw_iteration = metadata.get("verification_iteration")
        try:
            if isinstance(raw_iteration, bool):
                raise ValueError
            iteration = int(raw_iteration)
        except (TypeError, ValueError):
            _issue(
                report,
                "INTERACTION_ITERATION_MISSING",
                "verification Interactions step needs an integer verification_iteration",
                location=_step_label(index, step),
            )
            continue
        if iteration < 1:
            _issue(
                report,
                "INTERACTION_ITERATION_INVALID",
                "verification_iteration must be a positive integer",
                location=_step_label(index, step),
            )
            continue
        groups.setdefault(("verification", iteration), []).append((index, step))
        if not iteration_order or iteration_order[-1] != iteration:
            iteration_order.append(iteration)

    if not verification_steps:
        _issue(
            report,
            "VERIFICATION_STEPS_MISSING",
            "canonical trace contains no verification steps",
            location="state.all_steps",
        )
    elif not native_verification_steps:
        _issue(
            report,
            "INTERACTION_STEPS_MISSING",
            "canonical trace contains no auditable verification Interactions steps",
            location="state.all_steps",
        )

    iterations = sorted(iteration for _, iteration in groups)
    expected_iterations = list(range(1, max(iterations, default=0) + 1))
    if iterations != expected_iterations:
        _issue(
            report,
            "INTERACTION_ITERATION_GAP",
            f"verification interaction iterations must be contiguous from 1; found {iterations}",
            location="state.all_steps",
        )
    if iteration_order != iterations:
        _issue(
            report,
            "INTERACTION_ITERATION_INTERLEAVED",
            f"verification interaction iterations must not interleave; observed {iteration_order}",
            location="state.all_steps",
        )

    for (_, iteration), members in sorted(groups.items()):
        previous_interaction: str | None = None
        seen_interactions: dict[str, str] = {}
        for index, step in members:
            metadata = _mapping(step.get("metadata"))
            interaction_id = str(metadata.get("interaction_id", "")).strip()
            parent_recorded = "previous_interaction_id" in metadata
            raw_parent = metadata.get("previous_interaction_id")
            parent = "" if raw_parent is None else str(raw_parent).strip()
            location = _step_label(index, step)
            if not interaction_id:
                _issue(
                    report,
                    "INTERACTION_ID_MISSING",
                    f"verification iteration {iteration} has no interaction_id",
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

            if interaction_id in seen_interactions:
                if parent != seen_interactions[interaction_id]:
                    _issue(
                        report,
                        "INTERACTION_CHAIN_BROKEN",
                        f"interaction_id {interaction_id!r} was recorded with inconsistent parents",
                        location=location,
                    )
                if previous_interaction != interaction_id:
                    _issue(
                        report,
                        "INTERACTION_ID_REUSED",
                        f"interaction_id {interaction_id!r} is reused non-contiguously",
                        location=location,
                    )
                continue

            expected = previous_interaction
            if expected is None:
                if parent:
                    _issue(
                        report,
                        "INTERACTION_CHAIN_ROOT_INVALID",
                        f"iteration {iteration} root must have null previous_interaction_id, got {parent!r}",
                        location=location,
                    )
            elif parent != expected:
                _issue(
                    report,
                    "INTERACTION_CHAIN_BROKEN",
                    f"expected previous_interaction_id {expected!r}, got {parent!r}",
                    location=location,
                )
            seen_interactions[interaction_id] = parent
            previous_interaction = interaction_id

    report.stats["verification_interaction_iterations"] = len(groups)
    report.stats["verification_interaction_steps"] = native_verification_steps


def _rejection_category(step: Mapping[str, Any]) -> str:
    metadata = _mapping(step.get("metadata"))
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


def _audit_rejections(
    steps: Sequence[Mapping[str, Any]], report: TraceReport
) -> None:
    scheduler_count = 0
    protocol_count = 0
    for index, step in enumerate(steps):
        metadata = _mapping(step.get("metadata"))
        rejected = (
            str(step.get("action_type", "")) in REJECTION_ACTIONS
            or bool(metadata.get("rejection_reason"))
        )
        if not rejected:
            continue
        category = _rejection_category(step)
        if category == SCHEDULER:
            scheduler_count += 1
        else:
            protocol_count += 1
        reason = str(metadata.get("rejection_reason", "")).strip()
        if not reason:
            try:
                payload = json.loads(str(step.get("tool_result", "")))
                reason = str(_mapping(payload).get("error", "")).strip()
            except (TypeError, json.JSONDecodeError):
                reason = ""
        _issue(
            report,
            "SCHEDULER_REJECTION" if category == SCHEDULER else "PROTOCOL_REJECTION",
            reason or f"{step.get('action_type', 'rejected')} step",
            category=category,
            location=_step_label(index, step),
        )
    report.stats["scheduler_rejections"] = scheduler_count
    report.stats["protocol_rejections"] = protocol_count


def audit_trace(path: Path) -> TraceReport:
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
    ledgers = _mapping(state.get("ledgers"))
    verification = _mapping(state.get("verification"))
    claims = _rows(ledgers.get("claims"))
    sources = _rows(ledgers.get("sources"))
    evidence = _rows(ledgers.get("evidence"))
    accepted_evidence = _rows(verification.get("evidence"))
    report.stats.update(
        {
            "steps": len(steps),
            "claims": len(claims),
            "sources": len(sources),
            "evidence": len(evidence),
            "accepted_evidence": len(accepted_evidence),
        }
    )

    if "state" not in payload or not isinstance(payload.get("state"), Mapping):
        _issue(report, "CANONICAL_STATE_MISSING", "canonical trace must contain a state object")
    if not isinstance(state.get("ledgers"), Mapping):
        _issue(report, "LEDGERS_MISSING", "canonical state must contain ledgers")
    else:
        for collection in ("claims", "sources", "evidence", "discoveries", "failures"):
            if not isinstance(ledgers.get(collection), list):
                _issue(
                    report,
                    "LEDGER_COLLECTION_INVALID",
                    f"state.ledgers.{collection} must be an array",
                    location=f"state.ledgers.{collection}",
                )
    if not isinstance(state.get("all_steps"), list):
        _issue(report, "STEPS_MISSING", "canonical state must contain all_steps")

    _audit_termination(payload, state, report)
    _audit_thought_tokens(payload, state, steps, report)
    _audit_evidence_calls(
        evidence,
        steps,
        report,
        location_prefix="state.ledgers.evidence",
        tool_field="tool_name",
        stat_key="evidence_with_successful_call",
    )
    _audit_evidence_calls(
        accepted_evidence,
        steps,
        report,
        location_prefix="state.verification.evidence",
        tool_field="tool_used",
        stat_key="accepted_evidence_with_successful_call",
    )
    _audit_external_claims(claims, sources, evidence, steps, report)
    _audit_visual_external_evidence(claims, evidence, report)
    _audit_leaks(payload, report)
    _audit_interaction_chains(steps, report)
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
            for path in input_path.rglob("*.json")
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
                for trace in directory.rglob("*.json")
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
            f"protocol_rejections={report.stats.get('protocol_rejections', 0)}"
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
    print(
        f"Audited {len(reports)} trace(s): {len(reports) - failed} passed, "
        f"{failed} failed, {warnings} warning(s), "
        f"{scheduler_rejections} scheduler rejection(s), "
        f"{protocol_rejections} protocol rejection(s)."
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
        description="Audit canonical traces from a small number of real verification runs."
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
        help="Treat scheduler and protocol rejections as failures instead of warnings.",
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
