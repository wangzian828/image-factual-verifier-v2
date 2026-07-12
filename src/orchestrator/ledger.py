"""Machine-verifiable ledgers compiled from validated runtime observations."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.orchestrator.source_provenance import classify_source, content_sha256
from src.orchestrator.state import (
    ClaimMode,
    ClaimRecord,
    DiscoveryRecord,
    EvidenceItem,
    EvidenceRecord,
    FailureRecord,
    SourceRecord,
    VerificationCase,
    VerificationLedgers,
    VerificationPlan,
    VerificationResult,
    UnverifiableReason,
)
from src.orchestrator.tool_result import parse_tool_result


def image_sha256(image_path: str) -> str:
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_verification_case(
    image_path: str,
    *,
    case_id: str = "",
    user_claim: Optional[str] = None,
    claim_source_region: Optional[List[float]] = None,
    decision_policy_version: str = "reinspect-v1",
) -> VerificationCase:
    path = str(Path(image_path).resolve())
    return VerificationCase(
        case_id=case_id or Path(path).stem,
        image_path=path,
        image_sha256=image_sha256(path),
        claim_mode=ClaimMode.EXTERNAL if str(user_claim or "").strip() else ClaimMode.EMBEDDED,
        user_claim=str(user_claim).strip() if user_claim is not None else None,
        claim_surface=None,
        claim_source_region=claim_source_region,
        decision_policy_version=decision_policy_version,
    )


def verify_case_image(case: VerificationCase, image_path: str) -> None:
    actual = image_sha256(image_path)
    if actual != case.image_sha256:
        raise ValueError(
            f"VerificationCase image_sha256 mismatch: expected {case.image_sha256}, got {actual}."
        )


class VerificationLedger:
    """The only mutation surface for claim/source/evidence/failure ledgers."""

    def __init__(self, ledgers: Optional[VerificationLedgers] = None) -> None:
        self.data = ledgers.model_copy(deep=True) if ledgers else VerificationLedgers()

    @staticmethod
    def _id(prefix: str, *parts: str) -> str:
        payload = "\x1f".join(str(part) for part in parts)
        return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"

    def add_claim(self, record: ClaimRecord) -> ClaimRecord:
        self._append_unique("claims", record, "claim_id")
        return record

    def add_source(self, record: SourceRecord) -> SourceRecord:
        existing = next(
            (
                item
                for item in self.data.sources
                if item.source_id == record.source_id
            ),
            None,
        )
        if existing is None:
            self.data.sources.append(record)
            return record

        immutable_fields = (
            "canonical_url",
            "hostname",
            "registered_domain",
            "source_family",
            "source_class",
            "artifact_sha256",
        )
        conflicts = [
            field_name
            for field_name in immutable_fields
            if getattr(existing, field_name) != getattr(record, field_name)
        ]
        if conflicts:
            raise ValueError(
                "Duplicate source_id has conflicting immutable content "
                f"({', '.join(conflicts)}): {record.source_id}"
            )
        existing.retrieved_at = min(
            value
            for value in (existing.retrieved_at, record.retrieved_at)
            if value
        ) if existing.retrieved_at or record.retrieved_at else ""
        existing.dependency_source_ids = sorted(
            set(existing.dependency_source_ids) | set(record.dependency_source_ids)
        )
        existing.risk_flags = sorted(
            set(existing.risk_flags) | set(record.risk_flags)
        )
        return existing

    def add_discovery(self, record: DiscoveryRecord) -> DiscoveryRecord:
        self._require_claim(record.claim_id)
        self._append_unique("discoveries", record, "discovery_id")
        return record

    def add_failure(self, record: FailureRecord) -> FailureRecord:
        if record.claim_id:
            self._require_claim(record.claim_id)
        self._append_unique("failures", record, "failure_id")
        return record

    def add_evidence(
        self,
        record: EvidenceRecord,
        *,
        successful_call_ids: Iterable[str],
    ) -> EvidenceRecord:
        self._require_claim(record.claim_id)
        self._require_source(record.source_id)
        if record.function_call_id not in set(successful_call_ids):
            raise ValueError("Evidence must reference one successful immutable function call id.")
        if any(item.function_call_id == record.function_call_id for item in self.data.failures):
            raise ValueError("A failed function call cannot produce evidence.")
        source = next(item for item in self.data.sources if item.source_id == record.source_id)
        if record.artifact_sha256 != source.artifact_sha256:
            raise ValueError("Evidence artifact hash must match its source record.")
        self._append_unique("evidence", record, "evidence_id")
        return record

    def _append_unique(self, collection: str, record: Any, id_field: str) -> None:
        values = getattr(self.data, collection)
        record_id = getattr(record, id_field)
        existing = next((item for item in values if getattr(item, id_field) == record_id), None)
        if existing is None:
            values.append(record)
        elif existing != record:
            raise ValueError(f"Duplicate {id_field} has conflicting content: {record_id}")

    def _require_claim(self, claim_id: str) -> None:
        if claim_id not in {item.claim_id for item in self.data.claims}:
            raise ValueError(f"Unknown claim_id: {claim_id}")

    def _require_source(self, source_id: str) -> None:
        if source_id not in {item.source_id for item in self.data.sources}:
            raise ValueError(f"Unknown source_id: {source_id}")


def compile_runtime_ledgers(
    case: VerificationCase,
    plan: VerificationPlan,
    result: VerificationResult,
    steps: Sequence[Any],
) -> VerificationLedgers:
    ledger = VerificationLedger()
    question_to_claim: Dict[str, str] = {}
    for question in plan.questions:
        claim_id = f"claim-{question.question_id}"
        question_to_claim[question.question_id] = claim_id
        ledger.add_claim(
            ClaimRecord(
                claim_id=claim_id,
                text=question.claim_text,
                question_id=question.question_id,
                criticality="decisive" if question.priority == 1 else (
                    "supporting" if question.priority == 2 else "contextual"
                ),
                unresolved_distinction=question.why,
            )
        )

    step_by_call: Dict[str, Any] = {}
    successful_call_ids: set[str] = set()
    for step in steps:
        if getattr(step, "action_type", "") not in {"tool_call", "format_error"}:
            continue
        tool_name = str(getattr(step, "tool_name", "")).strip()
        if not tool_name:
            continue
        call_id = _step_call_id(step)
        if not call_id:
            continue
        step_by_call[call_id] = step
        try:
            parsed, succeeded = parse_tool_result(getattr(step, "tool_result", ""))
        except Exception as exc:
            parsed, succeeded = {"error": str(exc)}, False
        question_id = str(getattr(step, "tool_args", {}).get("__question_id", ""))
        claim_id = question_to_claim.get(question_id)
        if succeeded:
            successful_call_ids.add(call_id)
            _record_discoveries(ledger, step, parsed, claim_id)
        else:
            ledger.add_failure(
                FailureRecord(
                    failure_id=ledger._id("failure", call_id),
                    function_call_id=call_id,
                    tool_name=tool_name,
                    claim_id=claim_id,
                    code=_failure_code(str(parsed.get("error", ""))),
                    severity="partial",
                    recoverable=True,
                    message=str(parsed.get("error", "tool call failed")),
                )
            )

    for item in result.evidence:
        claim_id = question_to_claim.get(item.related_question)
        step = step_by_call.get(item.function_call_id)
        if not claim_id or step is None or item.function_call_id not in successful_call_ids:
            continue
        data, _ = parse_tool_result(getattr(step, "tool_result", ""))
        if item.tool_used in {"visit", "text_search", "crop_and_search"}:
            record = _find_web_record(data, item.raw_excerpt, item.source)
            if record is None:
                continue
            question = next(
                candidate
                for candidate in plan.questions
                if candidate.question_id == item.related_question
            )
            if str(record.get("goal", "")).strip() != question.claim_text.strip():
                continue
            identity = classify_source(
                item.source,
                injection_flags=record.get("injection_flags", []),
            )
            artifact = str(record.get("artifact_sha256", ""))
            source_id = ledger._id("source", identity.canonical_url, artifact)
            ledger.add_source(
                SourceRecord(
                    source_id=source_id,
                    canonical_url=identity.canonical_url,
                    hostname=identity.hostname,
                    registered_domain=identity.registered_domain,
                    source_family=identity.source_family,
                    source_class=identity.source_class,
                    artifact_sha256=artifact,
                    retrieved_at=str(record.get("retrieved_at", "")),
                    dependency_source_ids=[f"artifact:{artifact}"] if artifact else [],
                    risk_flags=list(identity.risk_flags),
                )
            )
            span = record["evidence_span"]
            exact_text = str(record["evidence"])
            ledger.add_evidence(
                EvidenceRecord(
                    evidence_id=ledger._id(
                        "evidence", item.function_call_id, claim_id, source_id, exact_text
                    ),
                    claim_id=claim_id,
                    source_id=source_id,
                    function_call_id=item.function_call_id,
                    tool_name=item.tool_used,
                    evidence_kind="web_span",
                    exact_text=exact_text,
                    span_start=int(span["start"]),
                    span_end=int(span["end"]),
                    artifact_sha256=artifact,
                    retrieved_at=str(record["retrieved_at"]),
                    stance=_stance(item.direction),
                    quality=item.quality,
                    directness=str(record.get("directness", "direct")),
                ),
                successful_call_ids=successful_call_ids,
            )
        elif item.tool_used == "current_time":
            metadata = getattr(step, "metadata", {}) or {}
            observed_at = str(metadata.get("observed_at") or case.created_at).strip()
            try:
                datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "Runtime observation provenance must contain an ISO-8601 observed_at or case.created_at."
                ) from exc
            artifact = content_sha256(item.raw_excerpt)
            source_id = ledger._id(
                "source", "runtime", item.tool_used, item.function_call_id, artifact
            )
            ledger.add_source(
                SourceRecord(
                    source_id=source_id,
                    canonical_url="",
                    hostname="",
                    registered_domain="",
                    source_family="runtime:current_time",
                    source_class="runtime",
                    artifact_sha256=artifact,
                    retrieved_at=observed_at,
                )
            )
            ledger.add_evidence(
                EvidenceRecord(
                    evidence_id=ledger._id(
                        "evidence", item.function_call_id, claim_id, source_id, item.raw_excerpt
                    ),
                    claim_id=claim_id,
                    source_id=source_id,
                    function_call_id=item.function_call_id,
                    tool_name=item.tool_used,
                    evidence_kind="runtime_anchor",
                    exact_text=item.raw_excerpt,
                    artifact_sha256=artifact,
                    retrieved_at=observed_at,
                    stance=_stance(item.direction),
                    quality=item.quality,
                ),
                successful_call_ids=successful_call_ids,
            )
        else:
            metadata = getattr(step, "metadata", {}) or {}
            observed_at = str(metadata.get("observed_at") or case.created_at).strip()
            for timestamp_name, timestamp in (
                ("case.created_at", case.created_at),
                ("observed_at", observed_at),
            ):
                try:
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(
                        f"Visual observation provenance {timestamp_name} must be ISO-8601."
                    ) from exc
            source_id = ledger._id("source", "image", case.image_sha256)
            ledger.add_source(
                SourceRecord(
                    source_id=source_id,
                    canonical_url="",
                    hostname="",
                    registered_domain="",
                    source_family=f"image:{case.image_sha256}",
                    source_class="visual",
                    artifact_sha256=case.image_sha256,
                    retrieved_at=case.created_at,
                )
            )
            region = getattr(step, "tool_args", {}).get("bbox") or [0.0, 0.0, 1.0, 1.0]
            ledger.add_evidence(
                EvidenceRecord(
                    evidence_id=ledger._id(
                        "evidence", item.function_call_id, claim_id, source_id, item.raw_excerpt
                    ),
                    claim_id=claim_id,
                    source_id=source_id,
                    function_call_id=item.function_call_id,
                    tool_name=item.tool_used,
                    evidence_kind="image_region",
                    exact_text=item.raw_excerpt,
                    image_region=region,
                    artifact_sha256=case.image_sha256,
                    retrieved_at=observed_at,
                    stance=_stance(item.direction),
                    quality=item.quality,
                ),
                successful_call_ids=successful_call_ids,
            )
    _update_claim_statuses(ledger.data)
    return ledger.data


def _update_claim_statuses(ledgers: VerificationLedgers) -> None:
    sources = {item.source_id: item for item in ledgers.sources}
    for claim in ledgers.claims:
        claim_evidence = [item for item in ledgers.evidence if item.claim_id == claim.claim_id]
        supports = [item for item in claim_evidence if item.stance == "support" and item.directness == "direct"]
        refutes = [item for item in claim_evidence if item.stance == "refute" and item.directness == "direct"]
        if supports and refutes:
            claim.status = "conflicted"
            claim.unresolved_distinction = "Direct supporting and refuting evidence remain in conflict."
        elif _direction_is_decisive(refutes, sources):
            claim.status = "refuted"
            claim.unresolved_distinction = ""
        elif _direction_is_decisive(supports, sources):
            claim.status = "supported"
            claim.unresolved_distinction = ""
        elif supports or refutes:
            claim.status = "open"
            claim.unresolved_distinction = "Evidence depends on one non-primary source family."
        else:
            claim.status = "open"


def _direction_is_decisive(
    evidence: Sequence[EvidenceRecord],
    sources: Dict[str, SourceRecord],
) -> bool:
    if not evidence:
        return False
    if any(
        item.evidence_kind == "image_region"
        and item.quality in {"strong", "moderate"}
        for item in evidence
    ):
        return True
    if any(
        sources.get(item.source_id)
        and sources[item.source_id].source_class == "official"
        and item.quality in {"strong", "moderate"}
        for item in evidence
    ):
        return True
    eligible = {
        item.source_id: sources[item.source_id]
        for item in evidence
        if item.source_id in sources
        and sources[item.source_id].source_class != "ugc"
        and not sources[item.source_id].risk_flags
    }
    if len(eligible) < 2:
        return False

    parent = {source_id: source_id for source_id in eligible}

    def find(source_id: str) -> str:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    rows = list(eligible.items())
    for index, (left_id, left) in enumerate(rows):
        left_dependencies = set(left.dependency_source_ids)
        for right_id, right in rows[index + 1 :]:
            if left.source_family == right.source_family:
                union(left_id, right_id)
                continue
            if left.artifact_sha256 and left.artifact_sha256 == right.artifact_sha256:
                union(left_id, right_id)
                continue
            if left_dependencies & set(right.dependency_source_ids):
                union(left_id, right_id)
    return len({find(source_id) for source_id in eligible}) >= 2


def derive_unverifiable_reasons(
    ledgers: VerificationLedgers,
    stop_reason: str = "hard_budget_exhausted",
) -> List[UnverifiableReason]:
    reasons: List[UnverifiableReason] = []
    open_claims = [item for item in ledgers.claims if item.criticality == "decisive" and item.status not in {"supported", "refuted"}]
    if any(item.status == "conflicted" for item in open_claims):
        reasons.append(UnverifiableReason.SOURCES_CONFLICT)
    if any("one non-primary source family" in item.unresolved_distinction for item in open_claims):
        reasons.append(UnverifiableReason.SINGLE_SOURCE_FAMILY)
    if any(item.code == "access_limited" for item in ledgers.failures):
        reasons.append(UnverifiableReason.ACCESS_LIMITED)
    if open_claims and not reasons:
        reasons.append(UnverifiableReason.DECISIVE_EVIDENCE_ABSENT)
    if open_claims:
        reasons.append(
            UnverifiableReason.SEARCH_SATURATED
            if stop_reason == "information_saturated"
            else UnverifiableReason.BUDGET_EXHAUSTED
        )
    return list(dict.fromkeys(reasons))


def _step_call_id(step: Any) -> str:
    metadata = getattr(step, "metadata", {}) or {}
    return str(metadata.get("function_call_id") or f"legacy-{getattr(step, 'stage_name', 'stage')}-{getattr(step, 'round', 0)}")


def _stance(direction: str) -> str:
    return {"supports": "support", "refutes": "refute"}.get(direction, "neutral")


def _failure_code(message: str) -> str:
    lowered = message.lower()
    if "timeout" in lowered:
        return "timeout"
    if "blocked" in lowered or "access" in lowered or "captcha" in lowered:
        return "access_limited"
    if "provider" in lowered or "unavailable" in lowered:
        return "provider_unavailable"
    if "question_id" in lowered or "argument" in lowered or "unknown tool" in lowered:
        return "protocol_error"
    if "json" in lowered or "malformed" in lowered:
        return "malformed_result"
    return "tool_error"


def _record_discoveries(
    ledger: VerificationLedger,
    step: Any,
    data: Dict[str, Any],
    claim_id: Optional[str],
) -> None:
    if not claim_id:
        return
    call_id = _step_call_id(step)
    tool_name = str(getattr(step, "tool_name", ""))
    rows: List[tuple[str, str, str, str]] = []
    if tool_name == "reverse_image_search":
        for item in (data.get("lens_results", []) or []) + (data.get("semantic_results", []) or []):
            if isinstance(item, dict):
                rows.append((str(item.get("url", "")), str(item.get("title", "")), str(item.get("snippet", "")), "reverse_image"))
    elif tool_name == "text_search":
        for query in data.get("queries", []) or []:
            if not isinstance(query, dict):
                continue
            for item in query.get("results", []) or []:
                if isinstance(item, dict):
                    rows.append((str(item.get("url", "")), str(item.get("title", "")), str(item.get("snippet", "")), "serp"))
    elif tool_name == "crop_and_search":
        for region in data.get("regions", []) or []:
            if not isinstance(region, dict):
                continue
            for item in (region.get("lens_results", []) or []) + (region.get("semantic_results", []) or []):
                if isinstance(item, dict):
                    rows.append((str(item.get("url", "")), str(item.get("title", "")), str(item.get("snippet", "")), "visual_reference"))
    for url, title, snippet, candidate_type in _merge_discovery_rows(rows):
        discovery_id = ledger._id("discovery", call_id, url, title)
        ledger.add_discovery(
            DiscoveryRecord(
                discovery_id=discovery_id,
                claim_id=claim_id,
                function_call_id=call_id,
                tool_name=tool_name,
                candidate_url=url,
                title=title,
                snippet=snippet,
                candidate_type=candidate_type,
            )
        )


def _merge_discovery_rows(
    rows: Sequence[tuple[str, str, str, str]],
) -> List[tuple[str, str, str, str]]:
    """Merge duplicate candidates emitted inside one immutable tool response.

    Search providers can return the same URL and title in multiple result groups
    while attaching slightly different snippets. Those entries represent one
    discovery, not conflicting ledger mutations. The merge remains local to one
    function result; ``VerificationLedger`` still rejects conflicting records
    produced elsewhere.
    """

    merged: Dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
    for raw_url, raw_title, raw_snippet, candidate_type in rows:
        url = str(raw_url or "").strip()
        title = " ".join(str(raw_title or "").split())
        snippet = " ".join(str(raw_snippet or "").split())
        if not url:
            continue
        key = (url, title, candidate_type)
        existing = merged.get(key)
        if existing is None or (len(snippet), snippet) > (
            len(existing[2]),
            existing[2],
        ):
            merged[key] = (url, title, snippet, candidate_type)
    return [merged[key] for key in sorted(merged)]


def _find_web_record(data: Any, excerpt: str, source: str) -> Optional[Dict[str, Any]]:
    normalized_excerpt = " ".join(str(excerpt or "").split()).lower()
    candidates: List[Dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            evidence = " ".join(str(value.get("evidence", "")).split()).lower()
            url = str(value.get("selected_url", "") or value.get("url", ""))
            if (
                evidence
                and normalized_excerpt in evidence
                and url
                and bool(value.get("evidence_eligible", False))
                and not value.get("injection_flags")
                and value.get("artifact_sha256")
                and value.get("evidence_span")
                and value.get("retrieved_at")
            ):
                candidates.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    source_identity = classify_source(source)
    matches = [
        item
        for item in candidates
        if classify_source(str(item.get("selected_url", "") or item.get("url", ""))).canonical_url
        == source_identity.canonical_url
    ]
    return min(matches, key=lambda item: len(str(item.get("evidence", "")))) if matches else None
