"""Private-gold eligibility audit for completed SFT teacher trajectories.

The teacher has already finished when this module runs.  Only the compact
structured target is exposed to the frozen auditor; certifying URLs and source
snapshots are intentionally excluded so eligibility does not mean reproducing a
particular source.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

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


SFT_ELIGIBILITY_SCHEMA_VERSION = "ifv-sft-eligibility-v1"
SFT_ELIGIBILITY_INPUT_VERSION = "ifv-sft-eligibility-input-v2"
SFT_ELIGIBILITY_PROMPT_VERSION = "ifv-sft-private-structured-gate-v3"
SFT_ELIGIBILITY_GENERATION_VERSION = "minimal-thinking-4096-v2"
SFT_ELIGIBILITY_POSTPROCESS_VERSION = "registered-path-gate-v3"
NEW_PATH_REVIEW_CONFIDENCE_THRESHOLD = 0.7

SFT_ELIGIBILITY_SYSTEM_PROMPT = (
    "You are a frozen post-rollout SFT eligibility auditor. Compare the completed "
    "candidate with the private structured target. Judge the semantic identity of "
    "the subject-event-slot relation and its depicted and verified values, whether "
    "the cited exact Evidence directly decides that same relation, and whether the "
    "stated boundary is respected. Every Evidence stance is relative to the candidate "
    "Claim: an exact span stating the verified alternative contradicts a candidate "
    "that depicts another value, even though it supports the private target's correct "
    "value. For a supported target, verified_value_alignment may be equivalent when "
    "the candidate recovers that value; key_error_slot_alignment remains not_applicable. "
    "The private target contains registered decision_paths. Prefer matching the "
    "primary path; if the candidate clearly follows another registered path, set "
    "path_match_status='matched_registered_path' and cite that path_id. If no "
    "registered path matches but the candidate appears to establish a distinct, "
    "image-grounded, directly evidenced reasonable path, set "
    "path_match_status='plausible_new_path_candidate' and fill new_path_candidate; "
    "set matched_path_id to an empty string for a new candidate or rejection. A new "
    "path must identify a visible image anchor, a direct supplied Evidence anchor, "
    "and why it is distinct from every registered path. This does not make the "
    "trajectory eligible automatically and only high-confidence candidates that "
    "survive deterministic screening are sent to human review. Otherwise set "
    "path_match_status='rejected'. Do not search. Do not reward matching wording or "
    "a matching URL. Cite only supplied Claim and Evidence IDs."
)


class SFTTargetEvidenceReview(_StrictModel):
    evidence_id: str = Field(min_length=1, max_length=160)
    relation_match: Literal[
        "same_relation",
        "partial_relation",
        "different_instance",
        "unclear",
    ]
    directness: Literal["direct", "indirect", "unclear"]
    stance: Literal["supports", "contradicts", "background", "unclear"]
    target_value_stated: bool
    explanation: str = Field(min_length=1, max_length=1000)


class SFTNewPathCandidate(_StrictModel):
    proposed_path_type: Literal[
        "identity",
        "time",
        "location",
        "object",
        "action",
        "quantity",
        "scene_constraint",
        "other",
    ]
    visible_image_anchor: str = Field(min_length=1, max_length=1000)
    teacher_evidence_anchor: str = Field(min_length=1, max_length=1000)
    why_not_existing_path: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool


class SFTEligibilityJudgment(_StrictModel):
    path_match_status: Literal[
        "matched_registered_path",
        "plausible_new_path_candidate",
        "rejected",
    ] = "matched_registered_path"
    matched_path_id: str = Field(default="primary", max_length=160)
    new_path_candidate: Optional[SFTNewPathCandidate] = None
    selected_claim_ids: List[str] = Field(default_factory=list, max_length=3)
    claim_relation_match: Literal[
        "same_relation",
        "partial_relation",
        "different_instance",
        "unclear",
    ]
    subject_event_slot_aligned: bool
    depicted_value_alignment: Literal[
        "equivalent",
        "conflicts",
        "missing",
        "unclear",
    ]
    key_error_slot_alignment: Literal[
        "equivalent",
        "wrong",
        "missing",
        "unclear",
        "not_applicable",
    ]
    verified_value_alignment: Literal[
        "equivalent",
        "conflicts",
        "missing",
        "unclear",
        "not_applicable",
    ]
    spurious_error: bool
    evidence_reviews: List[SFTTargetEvidenceReview] = Field(
        default_factory=list,
        max_length=40,
    )
    boundary_respected: bool
    boundary_violations: List[str] = Field(default_factory=list, max_length=8)
    explanation: str = Field(min_length=1, max_length=1600)


def _project_gold(gold: Mapping[str, Any]) -> Dict[str, Any]:
    if str(gold.get("schema_version", "")) != "ifv-scoring-gold-v1":
        raise ValueError("SFT eligibility requires ifv-scoring-gold-v1")
    label = str(gold.get("label", ""))
    if label not in {"supported", "refuted"}:
        raise ValueError("SFT eligibility label must be supported or refuted")
    primary_path = _project_decision_path(
        {**dict(gold), "path_id": "primary"},
    )
    optional_paths = [
        _project_decision_path(path)
        for path in _rows(gold.get("acceptable_decision_paths"))
    ]
    if len(optional_paths) > 3:
        raise ValueError("SFT eligibility accepts at most three optional decision paths")
    path_ids = [str(path.get("path_id", "")) for path in optional_paths]
    if "primary" in path_ids:
        raise ValueError("optional decision path_id must not be primary")
    if len(path_ids) != len(set(path_ids)):
        raise ValueError("optional decision path_id must be unique")
    for path in [primary_path, *optional_paths]:
        _validate_projected_path(path, label=label)
    return {
        "case_id": str(gold.get("case_id", "")),
        "label": label,
        "decisive_fact": dict(_mapping(gold.get("decisive_fact"))),
        "claim_atom": dict(_mapping(gold.get("claim_atom"))),
        "key_error": (
            dict(_mapping(gold.get("key_error")))
            if gold.get("key_error") is not None
            else None
        ),
        "evidence_target": dict(_mapping(gold.get("evidence_target"))),
        "boundary": dict(_mapping(gold.get("boundary"))),
        "decision_paths": [primary_path, *optional_paths],
    }


def _project_decision_path(path: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path_id": str(path.get("path_id", "")),
        "target_claim": str(path.get("target_claim", "")),
        "decisive_fact": dict(_mapping(path.get("decisive_fact"))),
        "claim_atom": dict(_mapping(path.get("claim_atom"))),
        "key_error": (
            dict(_mapping(path.get("key_error")))
            if path.get("key_error") is not None
            else None
        ),
        "evidence_target": dict(_mapping(path.get("evidence_target"))),
        "boundary": dict(_mapping(path.get("boundary"))),
    }


def _validate_projected_path(path: Mapping[str, Any], *, label: str) -> None:
    path_id = str(path.get("path_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}", path_id):
        raise ValueError(f"invalid SFT decision path_id: {path_id!r}")
    if not str(path.get("target_claim", "")).strip():
        raise ValueError(f"SFT decision path {path_id!r} lacks target_claim")
    claim_atom = _mapping(path.get("claim_atom"))
    binding = _mapping(_mapping(path.get("evidence_target")).get("binding"))
    for field_name in ("subject", "event", "slot"):
        expected = str(claim_atom.get(field_name, "")).strip()
        if not expected or str(binding.get(field_name, "")).strip() != expected:
            raise ValueError(
                f"SFT decision path {path_id!r} evidence_target must bind "
                "claim_atom subject/event/slot"
            )
    required_stance = str(
        _mapping(path.get("evidence_target")).get("required_stance", "")
    )
    expected_stance = "supports" if label == "supported" else "contradicts"
    if required_stance != expected_stance:
        raise ValueError(
            f"SFT decision path {path_id!r} stance must agree with label"
        )
    if (
        str(_mapping(path.get("evidence_target")).get("required_directness", ""))
        != "direct"
    ):
        raise ValueError(
            f"SFT decision path {path_id!r} requires direct evidence"
        )
    key_error = path.get("key_error")
    if label == "supported" and key_error is not None:
        raise ValueError("supported SFT decision path must not contain key_error")
    if label == "refuted" and key_error is None:
        raise ValueError("refuted SFT decision path requires key_error")
    if key_error is not None:
        error = _mapping(key_error)
        if str(error.get("slot", "")).strip() != str(
            claim_atom.get("slot", "")
        ).strip():
            raise ValueError(
                f"SFT decision path {path_id!r} key_error slot must match claim_atom"
            )
        if str(error.get("depicted_value", "")).strip() != str(
            claim_atom.get("depicted_value", "")
        ).strip():
            raise ValueError(
                f"SFT decision path {path_id!r} key_error value must match claim_atom"
            )


def _project_candidate_evidence(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": str(item.get("evidence_id", "")),
        "source_url": str(item.get("source_url", "")),
        "source_family": str(item.get("source_family", "")),
        "exact_text": str(item.get("exact_text", ""))[:8000],
        "directness": str(item.get("directness", "")),
        "claim_binding": str(item.get("claim_binding", "")),
        "relation_scope": str(item.get("relation_scope", "")),
        "relation_stance": str(item.get("relation_stance", "")),
        "runtime_stance": str(item.get("stance", "")),
    }


def build_sft_eligibility_input(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    image_path: Path | None = None,
) -> Dict[str, Any]:
    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    runtime_case = _mapping(state.get("runtime_case"))
    case_id = str(runtime_case.get("case_id") or trace.get("image_id") or "")
    target = _project_gold(gold)
    if target["case_id"] != case_id:
        raise ValueError("trace and private gold case_id do not match")

    claims = {
        str(item.get("claim_id", "")): item
        for item in _rows(investigation.get("image_claims"))
        if str(item.get("claim_id", ""))
    }
    evidence = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    discrepancies = {
        str(item.get("discrepancy_id", "")): item
        for item in _rows(investigation.get("material_discrepancies"))
        if str(item.get("discrepancy_id", ""))
    }
    basis = dict(
        _mapping(
            trace.get("verdict_basis")
            or investigation.get("discrepancy_verdict_basis")
        )
    )
    basis_claim_ids = [str(value) for value in basis.get("claim_ids", [])]
    basis_evidence_ids = [str(value) for value in basis.get("evidence_ids", [])]
    basis_discrepancy_ids = [
        str(value) for value in basis.get("discrepancy_ids", [])
    ]
    image_descriptor = {
        "image_sha256": str(runtime_case.get("image_sha256", "")),
        "available_to_judge": bool(image_path and image_path.is_file()),
    }
    return {
        "schema_version": SFT_ELIGIBILITY_INPUT_VERSION,
        "case_id": case_id,
        "episode_id": str(trace.get("image_id") or state.get("image_id") or ""),
        "recorded_verdict": str(trace.get("verdict", "")),
        "termination": str(trace.get("termination", "")),
        "image": image_descriptor,
        "candidate": {
            "claims": [
                {
                    "claim_id": claim_id,
                    "statement": str(claims[claim_id].get("statement", "")),
                    "salience": str(claims[claim_id].get("salience", "")),
                    "anchor_fact_ids": [
                        str(value)
                        for value in claims[claim_id].get("anchor_fact_ids", [])
                    ],
                }
                for claim_id in basis_claim_ids
                if claim_id in claims
            ],
            "discrepancies": [
                {
                    "discrepancy_id": discrepancy_id,
                    "statement": str(
                        discrepancies[discrepancy_id].get("statement", "")
                    ),
                    "affected_claim_ids": [
                        str(value)
                        for value in discrepancies[discrepancy_id].get(
                            "affected_claim_ids", []
                        )
                    ],
                }
                for discrepancy_id in basis_discrepancy_ids
                if discrepancy_id in discrepancies
            ],
            "evidence": [
                _project_candidate_evidence(evidence[evidence_id])
                for evidence_id in basis_evidence_ids
                if evidence_id in evidence
            ],
            "verdict_target": str(basis.get("verdict_target", "")),
            "unresolved_gaps": [
                str(value) for value in basis.get("unresolved_gaps", [])
            ],
        },
        "private_target": target,
    }


def _required_stances(gold: Mapping[str, Any]) -> tuple[str, str]:
    required = str(_mapping(gold.get("evidence_target")).get("required_stance", ""))
    if required == "supports":
        return "support", "supports"
    if required == "contradicts":
        return "refute", "contradicts"
    raise ValueError("evidence_target.required_stance must be supports or contradicts")


def _alignment_flags(
    judgment: SFTEligibilityJudgment,
    *,
    label: str,
    invalid_claim_ids: List[str],
) -> tuple[bool, bool]:
    relation_aligned = bool(
        judgment.selected_claim_ids
        and not invalid_claim_ids
        and judgment.claim_relation_match == "same_relation"
        and judgment.subject_event_slot_aligned
        and judgment.depicted_value_alignment == "equivalent"
    )
    if label == "refuted":
        error_aligned = bool(
            judgment.key_error_slot_alignment == "equivalent"
            and judgment.verified_value_alignment == "equivalent"
            and not judgment.spurious_error
        )
    else:
        error_aligned = bool(
            judgment.key_error_slot_alignment == "not_applicable"
            and judgment.verified_value_alignment
            in {"equivalent", "not_applicable"}
            and not judgment.spurious_error
        )
    return relation_aligned, error_aligned


def _qualifying_evidence_ids(
    judgment: SFTEligibilityJudgment,
    *,
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    runtime_stance: str,
    relation_stance: str,
    required_directness: str,
) -> List[str]:
    result: List[str] = []
    for review in judgment.evidence_reviews:
        row = evidence_by_id.get(review.evidence_id)
        if row is None:
            continue
        if (
            review.relation_match == "same_relation"
            and review.directness == required_directness == "direct"
            and review.stance == relation_stance
            and review.target_value_stated
            and str(row.get("relation_scope", "")) == "same_relation"
            and str(row.get("relation_stance", "")) == relation_stance
            and str(row.get("runtime_stance", "")) == runtime_stance
            and str(row.get("directness", "")) == required_directness
            and bool(str(row.get("exact_text", "")).strip())
        ):
            result.append(review.evidence_id)
    return sorted(set(result))


def sft_eligibility_metrics(
    packet: Mapping[str, Any],
    judgment: SFTEligibilityJudgment,
) -> Dict[str, Any]:
    target = _mapping(packet.get("private_target"))
    candidate = _mapping(packet.get("candidate"))
    label = str(target.get("label", ""))
    expected_verdict = "real" if label == "supported" else "fake"
    decision_paths = _rows(target.get("decision_paths")) or [target]
    path_by_id = {
        str(path.get("path_id", "primary") or "primary"): path
        for path in decision_paths
    }
    path_match_status = str(judgment.path_match_status)
    matched_path_id = str(judgment.matched_path_id or "")
    new_candidate = judgment.new_path_candidate
    judge_path_output_consistent = bool(
        (
            path_match_status == "matched_registered_path"
            and matched_path_id
            and new_candidate is None
        )
        or (
            path_match_status == "plausible_new_path_candidate"
            and not matched_path_id
            and new_candidate is not None
        )
        or (
            path_match_status == "rejected"
            and not matched_path_id
            and new_candidate is None
        )
    )
    selected_path = (
        path_by_id.get(matched_path_id)
        if path_match_status == "matched_registered_path"
        else None
    )
    invalid_path_id = bool(
        path_match_status == "matched_registered_path"
        and selected_path is None
    )
    runtime_stance, relation_stance = (
        _required_stances(selected_path) if selected_path is not None else ("", "")
    )
    required_directness = (
        str(_mapping(selected_path.get("evidence_target")).get("required_directness", ""))
        if selected_path is not None
        else ""
    )
    claim_ids = {
        str(item.get("claim_id", "")) for item in _rows(candidate.get("claims"))
    }
    evidence_by_id = {
        str(item.get("evidence_id", "")): item
        for item in _rows(candidate.get("evidence"))
    }
    invalid_claim_ids = sorted(
        {value for value in judgment.selected_claim_ids if value not in claim_ids}
    )
    invalid_evidence_ids = sorted(
        {
            review.evidence_id
            for review in judgment.evidence_reviews
            if review.evidence_id not in evidence_by_id
        }
    )
    judge_relation_aligned, judge_error_aligned = _alignment_flags(
        judgment,
        label=label,
        invalid_claim_ids=invalid_claim_ids,
    )
    eligible_evidence_ids = (
        _qualifying_evidence_ids(
            judgment,
            evidence_by_id=evidence_by_id,
            runtime_stance=runtime_stance,
            relation_stance=relation_stance,
            required_directness=required_directness,
        )
        if selected_path is not None
        else []
    )
    relation_aligned = bool(
        path_match_status == "matched_registered_path"
        and not invalid_path_id
        and judge_path_output_consistent
        and judge_relation_aligned
    )
    error_aligned = bool(
        selected_path is not None
        and judge_path_output_consistent
        and judge_error_aligned
    )
    verdict_correct = str(packet.get("recorded_verdict", "")) == expected_verdict
    boundary_respected = bool(
        judgment.boundary_respected and not judgment.boundary_violations
    )
    new_candidate_proposed = (
        path_match_status == "plausible_new_path_candidate"
    )
    new_candidate_confidence = (
        float(new_candidate.confidence) if new_candidate is not None else None
    )
    candidate_runtime_stance = "support" if label == "supported" else "refute"
    candidate_relation_stance = (
        "supports" if label == "supported" else "contradicts"
    )
    candidate_evidence_ids = _qualifying_evidence_ids(
        judgment,
        evidence_by_id=evidence_by_id,
        runtime_stance=candidate_runtime_stance,
        relation_stance=candidate_relation_stance,
        required_directness="direct",
    )
    candidate_filter_reasons: List[str] = []
    if new_candidate_proposed:
        if new_candidate is None:
            candidate_filter_reasons.append("candidate_payload_missing")
        if matched_path_id:
            candidate_filter_reasons.append("candidate_has_registered_path_id")
        if not bool(_mapping(packet.get("image")).get("available_to_judge")):
            candidate_filter_reasons.append("image_unavailable_to_judge")
        if new_candidate is not None and not new_candidate.needs_human_review:
            candidate_filter_reasons.append("judge_did_not_request_review")
        if new_candidate is not None and (
            new_candidate_confidence is None
            or new_candidate_confidence < NEW_PATH_REVIEW_CONFIDENCE_THRESHOLD
        ):
            candidate_filter_reasons.append("confidence_below_threshold")
        if not verdict_correct:
            candidate_filter_reasons.append("verdict_incorrect")
        if not judge_relation_aligned:
            candidate_filter_reasons.append("claim_relation_not_aligned")
        if not judge_error_aligned:
            candidate_filter_reasons.append("error_slot_or_value_not_aligned")
        if invalid_evidence_ids:
            candidate_filter_reasons.append("invalid_judge_evidence_ids")
        if not candidate_evidence_ids:
            candidate_filter_reasons.append("no_direct_same_relation_evidence")
        if not boundary_respected:
            candidate_filter_reasons.append("boundary_not_respected")
    new_candidate_screen_pass = bool(
        new_candidate_proposed and not candidate_filter_reasons
    )
    return {
        "expected_verdict": expected_verdict,
        "recorded_verdict": str(packet.get("recorded_verdict", "")),
        "path_match_status": path_match_status,
        "matched_path_id": (
            matched_path_id
            if path_match_status == "matched_registered_path"
            and not invalid_path_id
            else None
        ),
        "invalid_judge_path_id": matched_path_id if invalid_path_id else None,
        "judge_path_output_consistent": judge_path_output_consistent,
        "new_reasonable_path_candidate": new_candidate_proposed,
        "new_path_candidate_confidence": new_candidate_confidence,
        "new_path_candidate_screen_pass": new_candidate_screen_pass,
        "new_path_candidate_filter_reasons": candidate_filter_reasons,
        "candidate_evidence_ids": candidate_evidence_ids,
        "verdict_correct": verdict_correct,
        "claim_relation_aligned": relation_aligned,
        "error_slot_and_value_aligned": error_aligned,
        "eligible_evidence_ids": sorted(set(eligible_evidence_ids)),
        "direct_same_relation_evidence_present": bool(eligible_evidence_ids),
        "boundary_respected": boundary_respected,
        "invalid_judge_claim_ids": invalid_claim_ids,
        "invalid_judge_evidence_ids": invalid_evidence_ids,
    }


def sft_eligibility_passes(
    metrics: Mapping[str, Any],
    *,
    strict_trace_audit_pass: bool,
    engineering_valid: bool,
) -> bool:
    return bool(
        strict_trace_audit_pass
        and engineering_valid
        and metrics.get("path_match_status") == "matched_registered_path"
        and metrics.get("matched_path_id")
        and not metrics.get("invalid_judge_path_id")
        and metrics.get("judge_path_output_consistent") is True
        and metrics.get("verdict_correct") is True
        and metrics.get("claim_relation_aligned") is True
        and metrics.get("error_slot_and_value_aligned") is True
        and metrics.get("direct_same_relation_evidence_present") is True
        and metrics.get("boundary_respected") is True
        and not metrics.get("invalid_judge_claim_ids")
        and not metrics.get("invalid_judge_evidence_ids")
    )


class SFTEligibilityJudge:
    """Run one standalone structured-target audit after a teacher rollout."""

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
) -> Dict[str, Any]:
    engineering_valid = bool(
        str(trace.get("termination", "")) == "success"
        and str(trace.get("verdict", "")) in {"real", "fake"}
    )
    metrics = (
        sft_eligibility_metrics(packet, judgment)
        if judgment is not None
        else {
            "expected_verdict": (
                "real"
                if str(_mapping(packet.get("private_target")).get("label"))
                == "supported"
                else "fake"
            ),
            "recorded_verdict": str(packet.get("recorded_verdict", "")),
            "path_match_status": "rejected",
            "matched_path_id": None,
            "invalid_judge_path_id": None,
            "judge_path_output_consistent": False,
            "new_reasonable_path_candidate": False,
            "new_path_candidate_confidence": None,
            "new_path_candidate_screen_pass": False,
            "new_path_candidate_filter_reasons": ["judge_not_run"],
            "candidate_evidence_ids": [],
            "verdict_correct": False,
            "claim_relation_aligned": False,
            "error_slot_and_value_aligned": False,
            "eligible_evidence_ids": [],
            "direct_same_relation_evidence_present": False,
            "boundary_respected": False,
            "invalid_judge_claim_ids": [],
            "invalid_judge_evidence_ids": [],
        }
    )
    passed = bool(
        judgment is not None
        and sft_eligibility_passes(
            metrics,
            strict_trace_audit_pass=strict_trace_audit_pass,
            engineering_valid=engineering_valid,
        )
    )
    review_blockers = list(metrics.get("new_path_candidate_filter_reasons", []))
    if metrics.get("new_reasonable_path_candidate") is True:
        if not strict_trace_audit_pass:
            review_blockers.append("strict_trace_audit_failed")
        if not engineering_valid:
            review_blockers.append("engineering_invalid")
    human_review_required = bool(
        metrics.get("new_reasonable_path_candidate") is True
        and metrics.get("new_path_candidate_screen_pass") is True
        and not review_blockers
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
            "strict_trace_audit_failures": [
                dict(item) for item in strict_trace_audit_failures
            ],
            "engineering_valid": engineering_valid,
            "sft_eligibility_pass": passed,
            "new_reasonable_path_candidate": bool(
                metrics.get("new_reasonable_path_candidate")
            ),
            "new_path_candidate_screen_pass": bool(
                metrics.get("new_path_candidate_screen_pass")
            ),
            "new_path_candidate_review_blockers": sorted(set(review_blockers)),
            "human_review_required": human_review_required,
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
