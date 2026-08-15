"""Private-gold SFT eligibility audit for completed teacher trajectories.

The runtime ImageClaim graph remains the lineage layer.  This module judges the
more general question that matters for SFT: did the completed trajectory reach
the right binary decision about the factual content expressed by the image and
cite decisive, image-relevant Evidence?
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

from pydantic import Field

from src.tools.vision_utils import controlled_image_to_data_url
from src.trajectory.semantic_reward import (
    SemanticRewardJudge,
    _StrictModel,
    _mapping,
    _rows,
    canonical_json,
    sha256_json,
)


SFT_ELIGIBILITY_SCHEMA_VERSION = "ifv-sft-eligibility-v2"
SFT_ELIGIBILITY_INPUT_VERSION = "ifv-sft-eligibility-input-v5"
SFT_ELIGIBILITY_PROMPT_VERSION = "ifv-sft-private-image-fact-gate-v3"
SFT_ELIGIBILITY_GENERATION_VERSION = "minimal-thinking-4096-v5"
SFT_ELIGIBILITY_POSTPROCESS_VERSION = "image-fact-safety-gate-v4"


SFT_ELIGIBILITY_SYSTEM_PROMPT = (
    "You are a frozen post-rollout SFT eligibility auditor. Judge whether the "
    "completed teacher trajectory correctly determined the factual content expressed "
    "by the supplied image and found decisive Evidence for that decision. The "
    "private target describes the image fact and the expected binary verdict. "
    "Runtime ImageClaims are lineage and bookkeeping, not a mandatory semantic "
    "target. Do not require the teacher to reproduce the target wording, a specific "
    "Claim ID, relation slot, URL, original image, source span, or registered "
    "decision path. Accept a different but clearly image-grounded sub-fact when it "
    "decisively establishes the same image-level verdict. Evidence may be a "
    "successful visual observation, OCR/crop result, source passage, same-image "
    "context, or a multi-item chain. Retrieval history describes what the teacher "
    "actually investigated, but is not itself factual Evidence and has no Evidence "
    "IDs. A lack of matching results may only supplement an image-grounded chain "
    "when the history targets a named, plausibly authoritative source or bounded "
    "collection; generic web search failure, topical relatedness, or absence of a "
    "found original never decides the verdict. Select only supplied Evidence IDs. "
    "Assess retrieval_quality as effective when the trajectory's retrieval is "
    "targeted and converted into relevant inspection or Evidence; mixed when its "
    "central route is useful despite some noise or corrected turns; poor when its "
    "central route is generic, repeatedly low-yield, premise-led, ignores useful "
    "candidates, or treats non-results as a conclusion. Mark major overclaiming "
    "when the cited material does not establish the conclusion. Do not search and "
    "do not create human-review work."
)


class SFTEligibilityJudgment(_StrictModel):
    fact_alignment: Literal[
        "same_image_fact",
        "compatible_subfact",
        "different_fact",
        "unclear",
    ]
    decision_support: Literal[
        "supports_real",
        "supports_fake",
        "supporting_only",
        "insufficient",
        "unclear",
    ]
    retrieval_quality: Literal["effective", "mixed", "poor"]
    decisive_evidence_ids: List[str] = Field(
        default_factory=list,
        max_length=40,
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        max_length=40,
    )
    overclaiming: Literal["none", "minor", "major"]
    boundary_assessment: Literal[
        "respected",
        "minor_issue",
        "major_issue",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=1600)


def _text(*values: Any, limit: int = 4000) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()[:limit]
    return ""


def _unique(values: Iterable[Any], *, limit: int = 40) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= limit:
            break
    return result


def _string_values(value: Any, *, limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        return []
    return _unique(
        [
            item
            for item in value
            if isinstance(item, str) and item.strip()
        ],
        limit=limit,
    )


def _target_ids(row: Mapping[str, Any]) -> List[str]:
    return _unique(
        (
            row.get("case_id"),
            row.get("candidate_id"),
            row.get("assignment_id"),
            row.get("source_item_id"),
        ),
        limit=8,
    )


def _expected_verdict(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("expected_verdict")
        or row.get("factual_status")
        or row.get("label")
        or ""
    ).strip().lower()
    if value in {"real", "fake"}:
        return value
    if value == "supported":
        return "real"
    if value == "refuted":
        return "fake"
    raise ValueError("private target must declare supported/refuted or real/fake")


def _chain_rows(binding: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    chain = binding.get("chain")
    return _rows(chain)


def _reference_facts(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    evidence = _mapping(row.get("evidence"))
    binding = _mapping(evidence.get("binding"))

    def add(
        *,
        statement: Any = "",
        evidence_text: Any = "",
        source_url: Any = "",
        role: Any = "",
        verified_value: Any = "",
    ) -> None:
        statement_text = _text(statement, evidence_text)
        evidence_text_value = _text(evidence_text, statement)
        if not statement_text and not evidence_text_value:
            return
        facts.append(
            {
                "statement": statement_text,
                "verified_value": _text(verified_value, limit=1200),
                "evidence_text": evidence_text_value,
                "source_url": _text(source_url, limit=2000),
                "role": _text(role, "source_evidence", limit=120),
            }
        )

    for item in _chain_rows(binding):
        inner = _mapping(item.get("binding")) or item
        add(
            statement=item.get("semantic_audit_reason")
            or inner.get("target_claim"),
            evidence_text=item.get("exact_span")
            or inner.get("source_evidence_span")
            or inner.get("contradiction_relation_identity"),
            source_url=inner.get("source_url") or binding.get("source_url"),
            role=item.get("role") or item.get("type") or "evidence_chain",
            verified_value=inner.get("verified_value"),
        )

    add(
        statement=binding.get("target_claim") or evidence.get("target_claim"),
        evidence_text=(
            binding.get("source_evidence_span")
            or binding.get("candidate_exact_span")
            or binding.get("primary_exact_span")
        ),
        source_url=binding.get("source_url") or row.get("source_url"),
        role=binding.get("match") or "source_evidence",
        verified_value=binding.get("verified_value"),
    )

    for span in _rows(evidence.get("used_exact_spans")):
        add(
            evidence_text=span.get("exact_span") or span.get("text"),
            source_url=row.get("source_url"),
            role=span.get("role") or "source_exact_span",
        )

    for span in _rows(row.get("source_exact_spans")):
        add(
            evidence_text=span.get("text") or span.get("exact_span"),
            source_url=row.get("source_url"),
            role=span.get("role") or "source_exact_span",
        )

    return facts[:40]


def build_sft_target(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the final data-pipeline row into one generic ImageFact target."""

    ids = _target_ids(row)
    if not ids:
        raise ValueError("private target lacks case_id/candidate_id/assignment_id")

    claim_atom = _mapping(row.get("claim_atom"))
    decisive = _text(
        row.get("decisive_visual_atom"),
        _mapping(row.get("automatic_qa")).get("target_visual_atom_observation"),
        limit=4000,
    )
    construction = _mapping(row.get("construction_spec"))
    visible_facts = _string_values(row.get("visible_scene_facts"))
    if not visible_facts:
        visible_facts = _string_values(construction.get("visible_scene_facts"))
    if not visible_facts:
        qa = _mapping(row.get("automatic_qa"))
        visible_facts = _unique(
            [qa.get("target_visual_atom_observation")],
            limit=4,
        )

    visible_anchors = _unique(
        [
            decisive,
            *[_text(item, limit=1200) for item in visible_facts],
        ],
        limit=12,
    )
    statement = _text(
        row.get("target_claim"),
        row.get("primary_claim"),
        _mapping(row.get("decisive_fact")).get("statement"),
        decisive,
        limit=4000,
    )
    image_fact = {
        "statement": statement,
        "visible_anchors": visible_anchors,
        "subject": _text(claim_atom.get("subject"), limit=1200),
        "event_or_context": _text(
            claim_atom.get("event_or_context"),
            row.get("event_identity"),
            limit=1600,
        ),
        "relation": _text(
            claim_atom.get("relation"),
            claim_atom.get("relation_slot"),
            row.get("relation_identity"),
            limit=1200,
        ),
        "depicted_value": _text(
            claim_atom.get("depicted_value"),
            limit=1200,
        ),
    }
    return {
        "schema_version": "ifv-sft-target-v2",
        "case_id": ids[0],
        "case_id_aliases": ids,
        "expected_verdict": _expected_verdict(row),
        "image_fact": image_fact,
        "reference_facts": _reference_facts(row),
    }


def _successful_evidence(item: Mapping[str, Any]) -> bool:
    for key in ("successful_call", "tool_success"):
        if key in item:
            return bool(item.get(key))
    status = str(item.get("status") or item.get("tool_status") or "").lower()
    if status in {"error", "failed", "failure"}:
        return False
    provenance = _mapping(item.get("provenance"))
    if provenance.get("successful_call") is False:
        return False
    return True


def _evidence_text(item: Mapping[str, Any]) -> str:
    return _text(
        item.get("exact_text"),
        item.get("evidence"),
        item.get("observation"),
        item.get("summary"),
        item.get("finding"),
        limit=8000,
    )


def _project_candidate_evidence(
    item: Mapping[str, Any],
    *,
    basis_evidence_ids: set[str],
) -> Dict[str, Any]:
    evidence_id = str(item.get("evidence_id", "")).strip()
    return {
        "evidence_id": evidence_id,
        "task_id": str(item.get("task_id", "")),
        "fact_ids": _unique(item.get("fact_ids", []), limit=12),
        "finding_ids": _unique(item.get("finding_ids", []), limit=12),
        "claim_ids": _unique(item.get("claim_ids", []), limit=12),
        "evidence_kind": str(item.get("evidence_kind", "")),
        "source_url": str(item.get("source_url", "")),
        "source_family": str(item.get("source_family", "")),
        "exact_text": _evidence_text(item),
        "observation": _text(
            item.get("observation"),
            item.get("summary"),
            limit=4000,
        ),
        "successful_call": _successful_evidence(item),
        "basis_selected": evidence_id in basis_evidence_ids,
        "directness": str(item.get("directness", "")),
        "claim_binding": str(item.get("claim_binding", "")),
        "relation_scope": str(item.get("relation_scope", "")),
        "relation_stance": str(item.get("relation_stance", "")),
        "runtime_stance": str(item.get("stance", "")),
        "quality": str(item.get("quality", "")),
        "risk_flags": _unique(item.get("risk_flags", []), limit=12),
    }


def _tool_result_mapping(step: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = step.get("tool_result")
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return _mapping(parsed)


def _string_list(value: Any, *, limit: int) -> List[str]:
    values = value if isinstance(value, list) else [value]
    return _unique(
        [
            str(item).strip()
            for item in values
            if isinstance(item, str) and item.strip()
        ],
        limit=limit,
    )


def _retrieval_history(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compress actual text retrieval into judge-visible process context."""

    history: List[Dict[str, Any]] = []
    for step in _rows(state.get("all_steps")):
        if str(step.get("action_type", "")).strip() != "tool_call":
            continue
        tool_name = str(step.get("tool_name", "")).strip()
        if tool_name not in {"text_search", "visit"}:
            continue
        tool_args = _mapping(step.get("tool_args"))
        result = _tool_result_mapping(step)
        status = str(result.get("status", "")).strip().lower()
        if tool_name == "text_search":
            query_rows = _rows(result.get("queries"))
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(tool_args.get("goal"), limit=800),
                    "queries": _string_list(
                        tool_args.get("queries", tool_args.get("query", [])),
                        limit=3,
                    ),
                    "status": status or "unknown",
                    "result_count": sum(
                        len(_rows(row.get("results")))
                        for row in query_rows
                    ),
                    "search_error": _text(result.get("search_error"), result.get("error"), limit=500),
                }
            )
        else:
            source_urls = _string_list(tool_args.get("url", []), limit=3)
            source_urls = _unique(
                [*source_urls, *_string_list(result.get("url", ""), limit=1)],
                limit=3,
            )
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(
                        tool_args.get("retrieval_goal"),
                        tool_args.get("goal"),
                        limit=800,
                    ),
                    "source_urls": source_urls,
                    "status": status or "unknown",
                    "page_text_extracted": bool(
                        _text(
                            result.get("evidence"),
                            result.get("exact_text"),
                            result.get("content"),
                        )
                    ),
                    "fetch_error": _text(result.get("error"), limit=500),
                }
            )
        if len(history) >= 24:
            break
    return history


def build_sft_eligibility_input(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    image_path: Path | None = None,
) -> Dict[str, Any]:
    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    runtime_case = _mapping(state.get("runtime_case"))
    case_id = str(runtime_case.get("case_id") or trace.get("image_id") or "").strip()
    target = build_sft_target(gold)
    if case_id not in set(target.get("case_id_aliases", [])):
        raise ValueError(
            f"trace case_id {case_id!r} is not present in private target aliases"
        )

    basis = dict(
        _mapping(
            trace.get("verdict_basis")
            or investigation.get("discrepancy_verdict_basis")
        )
    )
    basis_claim_ids = _unique(basis.get("claim_ids", []), limit=12)
    basis_evidence_ids = set(_unique(basis.get("evidence_ids", []), limit=40))
    basis_discrepancy_ids = _unique(basis.get("discrepancy_ids", []), limit=12)

    claims = [
        {
            "claim_id": str(item.get("claim_id", "")),
            "statement": _text(item.get("statement"), limit=2400),
            "salience": str(item.get("salience", "")),
            "status": str(item.get("status", "")),
            "anchor_fact_ids": _unique(item.get("anchor_fact_ids", []), limit=12),
        }
        for item in _rows(investigation.get("image_claims"))
        if str(item.get("claim_id", "")).strip()
    ]
    evidence = [
        _project_candidate_evidence(
            item,
            basis_evidence_ids=basis_evidence_ids,
        )
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", "")).strip()
    ]
    findings = [
        {
            "finding_id": str(item.get("finding_id", "")),
            "task_id": str(item.get("task_id", "")),
            "fact_ids": _unique(item.get("fact_ids", []), limit=12),
            "evidence_ids": _unique(item.get("evidence_ids", []), limit=20),
            "stance": str(item.get("stance", "")),
            "summary": _text(item.get("summary"), limit=2400),
        }
        for item in _rows(investigation.get("findings"))
        if str(item.get("finding_id", "")).strip()
    ]
    discrepancies = [
        {
            "discrepancy_id": str(item.get("discrepancy_id", "")),
            "statement": _text(item.get("statement"), limit=2400),
            "affected_claim_ids": _unique(
                item.get("affected_claim_ids", []),
                limit=12,
            ),
            "visual_anchor_fact_ids": _unique(
                item.get("visual_anchor_fact_ids", []),
                limit=12,
            ),
            "evidence_ids": _unique(item.get("evidence_ids", []), limit=20),
            "materiality": str(item.get("materiality", "")),
            "status": str(item.get("status", "")),
        }
        for item in _rows(investigation.get("material_discrepancies"))
        if str(item.get("discrepancy_id", "")).strip()
    ]
    visual_facts = [
        {
            "fact_id": str(item.get("fact_id", "")),
            "statement": _text(item.get("statement"), item.get("description"), limit=1600),
            "source": str(item.get("source", "")),
        }
        for item in _rows(
            investigation.get("visual_facts") or state.get("visual_facts")
        )
        if str(item.get("fact_id", "")).strip()
    ]

    return {
        "schema_version": SFT_ELIGIBILITY_INPUT_VERSION,
        "case_id": case_id,
        "episode_id": str(trace.get("image_id") or state.get("image_id") or ""),
        "recorded_verdict": str(trace.get("verdict", "")),
        "termination": str(trace.get("termination", "")),
        "image": {
            "image_sha256": str(runtime_case.get("image_sha256", "")),
            "available_to_judge": bool(image_path and image_path.is_file()),
        },
        "candidate": {
            "claims": claims,
            "visual_facts": visual_facts,
            "findings": findings,
            "evidence": evidence,
            "discrepancies": discrepancies,
            "retrieval_history": _retrieval_history(state),
            "basis_claim_ids": basis_claim_ids,
            "basis_discrepancy_ids": basis_discrepancy_ids,
            "verdict_target": _text(basis.get("verdict_target"), limit=4000),
            "unresolved_gaps": _unique(basis.get("unresolved_gaps", []), limit=12),
        },
        "private_target": target,
    }


def _evidence_has_content(row: Mapping[str, Any]) -> bool:
    return bool(
        _text(
            row.get("exact_text"),
            row.get("observation"),
        )
    )


def classify_sft_audit_failures(
    failures: Iterable[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep non-safety audit findings from becoming an all-or-nothing SFT gate."""

    fatal: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    fatal_markers = (
        "ENGINEERING",
        "SOURCE_ACCESS",
        "POLICY_VIOLATION",
        "TRACE_CORRUPT",
        "MALFORMED",
        "TERMINATION_NOT_SUCCESS",
        "POST_VERDICT",
        "PROVIDER_FAILURE",
        "EVIDENCE_CALL_NOT_SUCCESSFUL",
        "INVALID_EVIDENCE",
    )
    fatal_protocol_codes = {
        "PROTOCOL_ERROR",
        "PROTOCOL_REJECTION",
    }
    for raw in failures:
        item = dict(raw)
        code = str(item.get("code", "")).upper()
        if code in fatal_protocol_codes or any(
            marker in code for marker in fatal_markers
        ):
            fatal.append(item)
        else:
            warnings.append(item)
    return fatal, warnings


def sft_eligibility_metrics(
    packet: Mapping[str, Any],
    judgment: SFTEligibilityJudgment | None,
) -> Dict[str, Any]:
    target = _mapping(packet.get("private_target"))
    candidate = _mapping(packet.get("candidate"))
    expected_verdict = str(target.get("expected_verdict", ""))
    recorded_verdict = str(packet.get("recorded_verdict", ""))
    image_available = bool(_mapping(packet.get("image")).get("available_to_judge"))
    expected_support = (
        "supports_real" if expected_verdict == "real" else "supports_fake"
    )
    evidence_by_id = {
        str(item.get("evidence_id", "")): item
        for item in _rows(candidate.get("evidence"))
        if str(item.get("evidence_id", "")).strip()
    }
    if judgment is None:
        judgment_values: Dict[str, Any] = {
            "fact_alignment": "unclear",
            "decision_support": "unclear",
            "retrieval_quality": "poor",
            "decisive_evidence_ids": [],
            "supporting_evidence_ids": [],
            "overclaiming": "major",
            "boundary_assessment": "major_issue",
            "confidence": 0.0,
            "explanation": "judge_not_run",
        }
    else:
        judgment_values = judgment.model_dump(mode="json")

    selected_ids = _unique(
        [
            *judgment_values.get("decisive_evidence_ids", []),
            *judgment_values.get("supporting_evidence_ids", []),
        ],
        limit=40,
    )
    decisive_ids = _unique(
        judgment_values.get("decisive_evidence_ids", []),
        limit=40,
    )
    invalid_ids = [
        evidence_id for evidence_id in selected_ids if evidence_id not in evidence_by_id
    ]
    failed_ids = [
        evidence_id
        for evidence_id in selected_ids
        if evidence_id in evidence_by_id
        and (
            not bool(evidence_by_id[evidence_id].get("successful_call"))
            or not _evidence_has_content(evidence_by_id[evidence_id])
        )
    ]
    valid_decisive_ids = [
        evidence_id
        for evidence_id in decisive_ids
        if evidence_id in evidence_by_id
        and bool(evidence_by_id[evidence_id].get("successful_call"))
        and _evidence_has_content(evidence_by_id[evidence_id])
    ]
    fact_alignment = str(judgment_values.get("fact_alignment", "unclear"))
    decision_support = str(judgment_values.get("decision_support", "unclear"))
    retrieval_quality = str(judgment_values.get("retrieval_quality", "poor"))
    fatal_errors = [
        *(
            ["image_unavailable_to_judge"]
            if not image_available
            else []
        ),
        *(
            ["invalid_judge_evidence_ids"]
            if invalid_ids
            else []
        ),
        *(
            ["selected_evidence_not_successful_or_empty"]
            if failed_ids
            else []
        ),
        *(
            ["different_image_fact"]
            if fact_alignment == "different_fact"
            else []
        ),
        *(
            ["major_overclaiming"]
            if judgment_values.get("overclaiming") == "major"
            else []
        ),
        *(
            ["poor_retrieval_quality"]
            if retrieval_quality == "poor"
            else []
        ),
        *(
            ["no_decisive_evidence"]
            if not valid_decisive_ids
            else []
        ),
    ]
    warnings: List[str] = []
    if fact_alignment == "unclear":
        warnings.append("fact_alignment_unclear")
    if decision_support in {"supporting_only", "insufficient", "unclear"}:
        warnings.append("decision_support_not_decisive")
    if retrieval_quality == "mixed":
        warnings.append("mixed_retrieval_quality")
    if judgment_values.get("overclaiming") == "minor":
        warnings.append("minor_overclaiming")
    if judgment_values.get("boundary_assessment") != "respected":
        warnings.append("boundary_warning")
    basis_ids = set(_unique(candidate.get("basis_claim_ids", []), limit=12))
    if basis_ids and not basis_ids.intersection(
        {
            claim_id
            for evidence_id in valid_decisive_ids
            for claim_id in _unique(
                evidence_by_id[evidence_id].get("claim_ids", []),
                limit=12,
            )
        }
    ):
        warnings.append("decisive_evidence_not_claim_basis_selected")

    return {
        "expected_verdict": expected_verdict,
        "recorded_verdict": recorded_verdict,
        "expected_decision_support": expected_support,
        "verdict_correct": recorded_verdict == expected_verdict,
        "image_available": image_available,
        "fact_alignment": fact_alignment,
        "decision_support": decision_support,
        "retrieval_quality": retrieval_quality,
        "decisive_evidence_ids": valid_decisive_ids,
        "selected_evidence_ids": selected_ids,
        "invalid_judge_evidence_ids": invalid_ids,
        "failed_selected_evidence_ids": failed_ids,
        "fatal_errors": _unique(fatal_errors, limit=20),
        "warnings": _unique(warnings, limit=20),
        "confidence": float(judgment_values.get("confidence", 0.0) or 0.0),
        "explanation": str(judgment_values.get("explanation", "")),
    }


def sft_eligibility_passes(
    metrics: Mapping[str, Any],
    *,
    strict_trace_audit_pass: bool,
    engineering_valid: bool,
    fatal_audit_errors: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Apply only safety gates; non-fatal strict-audit warnings do not veto SFT."""

    expected_support = str(metrics.get("expected_decision_support", ""))
    return bool(
        engineering_valid
        and metrics.get("verdict_correct") is True
        and metrics.get("fact_alignment")
        in {"same_image_fact", "compatible_subfact"}
        and metrics.get("decision_support") == expected_support
        and metrics.get("decisive_evidence_ids")
        and not metrics.get("fatal_errors")
        and not list(fatal_audit_errors)
    )


class SFTEligibilityJudge:
    """Run one standalone structured ImageFact audit after a teacher rollout."""

    def __init__(
        self,
        backend: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._delegate = SemanticRewardJudge(
            backend,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        self.provider = self._delegate.provider
        self.model = self._delegate.model
        self.max_tokens = int(max_tokens)

    @property
    def generation_identity(self) -> str:
        return f"{SFT_ELIGIBILITY_GENERATION_VERSION}:max_tokens={self.max_tokens}"

    async def judge(
        self,
        packet: Mapping[str, Any],
        *,
        image_path: Path | None = None,
    ) -> tuple[SFTEligibilityJudgment, Dict[str, Any]]:
        image_data_url: str | None = None
        image_view: Dict[str, Any] | None = None
        if image_path is not None:
            image_data_url, image_view = controlled_image_to_data_url(
                str(image_path),
                max_long_edge=1280,
                jpeg_quality=88,
            )
        call = await self._delegate._call(
            system_prompt=SFT_ELIGIBILITY_SYSTEM_PROMPT,
            prompt_version=SFT_ELIGIBILITY_PROMPT_VERSION,
            payload=packet,
            response_model=SFTEligibilityJudgment,
            image_data_url=image_data_url,
        )
        return (
            SFTEligibilityJudgment.model_validate(call.parsed),
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_versions": [SFT_ELIGIBILITY_PROMPT_VERSION],
                "generation_version": SFT_ELIGIBILITY_GENERATION_VERSION,
                "max_tokens": self.max_tokens,
                "image_view": image_view,
                "calls": [call.audit],
            },
        )


def build_sft_eligibility_artifact(
    *,
    trace: Mapping[str, Any],
    trace_sha256: str,
    packet: Mapping[str, Any],
    judgment: SFTEligibilityJudgment | None,
    judge_audit: Mapping[str, Any] | None,
    strict_trace_audit_pass: bool,
    strict_trace_audit_failures: Iterable[Mapping[str, Any]] = (),
    strict_trace_audit_warnings: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    engineering_valid = bool(
        str(trace.get("termination", "")) == "success"
        and str(trace.get("verdict", "")) in {"real", "fake"}
    )
    audit_failures = [dict(item) for item in strict_trace_audit_failures]
    explicit_audit_warnings = [
        dict(item) for item in strict_trace_audit_warnings
    ]
    fatal_audit_errors, failure_warnings = classify_sft_audit_failures(
        audit_failures
    )
    audit_warnings = [*failure_warnings, *explicit_audit_warnings]
    metrics = sft_eligibility_metrics(packet, judgment)
    metrics["fatal_audit_errors"] = fatal_audit_errors
    metrics["audit_warnings"] = audit_warnings
    passed = bool(
        judgment is not None
        and sft_eligibility_passes(
            metrics,
            strict_trace_audit_pass=strict_trace_audit_pass,
            engineering_valid=engineering_valid,
            fatal_audit_errors=fatal_audit_errors,
        )
    )
    if not engineering_valid:
        metrics["fatal_errors"] = _unique(
            [*metrics.get("fatal_errors", []), "engineering_invalid"],
            limit=20,
        )
    core = {
        "schema_version": SFT_ELIGIBILITY_SCHEMA_VERSION,
        "postprocess_version": SFT_ELIGIBILITY_POSTPROCESS_VERSION,
        "case_id": str(packet.get("case_id", "")),
        "episode_id": str(packet.get("episode_id", "")),
        "source_trace": {"sha256": trace_sha256},
        "eligibility_input": {
            "schema_version": packet.get("schema_version"),
            "sha256": sha256_json(packet),
            "image": packet.get("image"),
        },
        "judge": dict(judge_audit or {}),
        "structured_judgment": (
            judgment.model_dump(mode="json") if judgment is not None else None
        ),
        "metrics": metrics,
        "gates": {
            "strict_trace_audit_pass": bool(strict_trace_audit_pass),
            "strict_trace_audit_failures": audit_failures,
            "strict_trace_audit_warnings": explicit_audit_warnings,
            "fatal_audit_errors": fatal_audit_errors,
            "audit_warnings": audit_warnings,
            "engineering_valid": engineering_valid,
            "sft_eligibility_pass": passed,
        },
    }
    return {
        **core,
        "artifact_id": f"sha256:{sha256_json(core)}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def sft_eligibility_cache_key(
    *,
    trace_sha256: str,
    packet: Mapping[str, Any],
    provider: str,
    model: str,
    generation_version: str,
) -> str:
    return sha256_json(
        {
            "trace_sha256": trace_sha256,
            "eligibility_input_sha256": sha256_json(packet),
            "provider": provider,
            "model": model,
            "generation_version": generation_version,
            "postprocess_version": SFT_ELIGIBILITY_POSTPROCESS_VERSION,
            "prompt_versions": [SFT_ELIGIBILITY_PROMPT_VERSION],
        }
    )
