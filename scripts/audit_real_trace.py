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
    web_record_is_temporally_eligible,
)
from src.orchestrator.ledger import evidence_goal_for_case  # noqa: E402
from src.orchestrator.source_access import (  # noqa: E402
    FACT_CHECK_DOMAIN_MARKERS,
    SourceAccessPolicy,
    benchmark_source_access_policy,
    url_variants,
)
from src.orchestrator.state import EvidenceRecord, VerificationCase  # noqa: E402
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
    verification_case: VerificationCase | None = None,
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
        expected_goal = (
            evidence_goal_for_case(
                str(claim.get("text", "")).strip(),
                verification_case,
            )
            if verification_case is not None
            else str(claim.get("text", "")).strip()
        )
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
                    claim_text=expected_goal,
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
                and web_record_is_temporally_eligible(value, claim_text)
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


def _audit_initial_required_question_service(
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    report: TraceReport,
) -> None:
    """Mirror the first-output P1/P2 attempt gate enforced by StageRunner."""

    plan_history = _rows(state.get("plan_history"))
    if not plan_history:
        _issue(
            report,
            "INITIAL_PLAN_MISSING",
            "canonical trace must retain the initial verification plan",
            location="state.plan_history",
        )
        return
    initial_questions = _rows(plan_history[0].get("questions"))
    required_ids = [
        str(question.get("question_id", "")).strip()
        for question in initial_questions
        if str(question.get("question_id", "")).strip()
        and int(question.get("priority", 1) or 1) <= 2
    ]
    if not required_ids:
        _issue(
            report,
            "INITIAL_REQUIRED_QUESTIONS_MISSING",
            "initial verification plan must contain at least one priority-1/2 question",
            location="state.plan_history[0].questions",
        )
        return

    first_output_index = next(
        (
            index
            for index, step in enumerate(steps)
            if str(step.get("stage", "")) == "verification"
            and str(step.get("action_type", "")) == "output"
        ),
        None,
    )
    if first_output_index is None:
        _issue(
            report,
            "VERIFICATION_ACCEPTED_OUTPUT_MISSING",
            "successful trace must contain an accepted verification output",
            location="state.all_steps",
        )
        return

    attempted: set[str] = set()
    resolved: set[str] = set()
    for step in steps[:first_output_index]:
        if (
            str(step.get("stage", "")) == "verification"
            and str(step.get("action_type", "")) == "tool_call"
        ):
            question_id = str(
                _mapping(step.get("tool_args")).get("__question_id", "")
            ).strip()
            if question_id:
                attempted.add(question_id)
        update = _mapping(_mapping(step.get("metadata")).get("investigation_state_update"))
        belief_delta = _mapping(update.get("belief_delta"))
        claim_id = str(belief_delta.get("claim_id", "")).strip()
        if (
            claim_id.startswith("claim-")
            and str(belief_delta.get("new_status", "")) in {"supported", "refuted"}
        ):
            resolved.add(claim_id.removeprefix("claim-"))

    missing = [
        question_id
        for question_id in required_ids
        if question_id not in attempted and question_id not in resolved
    ]
    report.stats["initial_required_questions"] = len(required_ids)
    report.stats["initial_required_questions_attempted"] = len(
        set(required_ids) & attempted
    )
    if missing:
        _issue(
            report,
            "INITIAL_REQUIRED_QUESTION_UNTOUCHED",
            "first accepted verification output skipped required question ids: "
            + ", ".join(missing),
            location=_step_label(first_output_index, steps[first_output_index]),
        )


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
    investigation_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if str(step.get("stage", "")) == "image_only_investigation"
        and _mapping(step.get("metadata")).get("native_interactions")
    ]
    if not investigation_steps:
        _issue(
            report,
            "IMAGE_ONLY_INTERACTIONS_MISSING",
            "image-only trace contains no native investigation interactions",
            location="state.all_steps",
        )
        return

    previous: str | None = None
    segment_count = 1
    for index, step in investigation_steps:
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
            if parent:
                _issue(
                    report,
                    "INTERACTION_CHAIN_ROOT_INVALID",
                    f"image-only segment root must have null parent, got {parent!r}",
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
        if str(step.get("action_type", "")) == "output":
            previous = None
            segment_count += 1

    report.stats["image_only_interaction_steps"] = len(investigation_steps)
    report.stats["image_only_interaction_segments"] = max(1, segment_count - 1)


def _audit_image_only_v2(
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
            elif str(evidence_record.get("task_id", "")).strip() != task_id:
                _issue(
                    report,
                    "FINDING_EVIDENCE_OWNERSHIP_INVALID",
                    "Finding and Evidence must belong to the same ResearchTask",
                    location=location,
                )

    investigation_tool_steps = [
        step
        for step in steps
        if str(step.get("stage", "")) == "image_only_investigation"
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
    for index, step in enumerate(investigation_tool_steps):
        count = _mapping(step.get("metadata")).get("function_call_count")
        if count is not None and int(count or 0) != 1:
            _issue(
                report,
                "IMAGE_ONLY_PARALLEL_TOOL_CALL",
                "each image-only action turn must contain exactly one tool call",
                location=f"image_only_tool_steps[{index}]",
            )

    reflection_counts = [
        int(item.get("action_count", 0) or 0) for item in reflections
    ]
    expected_reflections = list(range(4, action_count + 1, 4))
    if reflection_counts != expected_reflections:
        _issue(
            report,
            "IMAGE_ONLY_REFLECTION_CADENCE_INVALID",
            f"Reflection action counts must be {expected_reflections}, found {reflection_counts}",
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
    if basis_facts != expected_basis_facts:
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
    for fact_id in basis_facts:
        selected_findings = set(findings_by_fact.get(fact_id, [])) & basis_findings
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
            for evidence_id in finding_by_id[finding_id].get("evidence_ids", []) or []
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
                    f"basis Evidence {evidence_id!r} lacks a successful tool call",
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
            "decisive_facts": len(decisive_ids),
            "image_only_actions": action_count,
        }
    )
    _audit_image_only_interaction_chains(steps, report)


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
    is_image_only = str(
        payload.get("input_mode") or state.get("input_mode") or ""
    ) == "image_only"
    verification_case: VerificationCase | None = None
    raw_case = state.get("verification_case")
    if isinstance(raw_case, Mapping):
        try:
            verification_case = VerificationCase.model_validate(dict(raw_case))
        except Exception:
            verification_case = None
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
    if is_image_only:
        _audit_image_only_v2(payload, state, steps, report)
    else:
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
        _audit_external_claims(
            claims,
            sources,
            evidence,
            steps,
            report,
            verification_case,
        )
        _audit_visual_external_evidence(claims, evidence, report)
    _audit_leaks(payload, report)
    if not is_image_only:
        _audit_interaction_chains(steps, report)
        _audit_initial_required_question_service(state, steps, report)
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
