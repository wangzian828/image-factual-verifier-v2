# -*- coding: utf-8 -*-
"""Core state models for the image verification pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.redaction import sanitize_for_persistence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ShortListText = Annotated[str, Field(max_length=200)]


class ClaimMode(str, Enum):
    EXTERNAL = "external_claim"
    EMBEDDED = "embedded_claim"


class UnverifiableReason(str, Enum):
    DECISIVE_EVIDENCE_ABSENT = "decisive_evidence_absent"
    SOURCES_CONFLICT = "sources_conflict"
    SINGLE_SOURCE_FAMILY = "single_source_family_dependency"
    UNREADABLE_REGION = "unreadable_region"
    ACCESS_LIMITED = "access_limited"
    BUDGET_EXHAUSTED = "budget_exhausted"


class VerificationCase(StrictModel):
    """Versioned runtime input. Hidden benchmark labels never belong here."""

    case_id: str = Field(min_length=1, max_length=200)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    claim_mode: ClaimMode
    user_claim: Optional[str] = Field(default=None, max_length=2000)
    claim_surface: Optional[str] = Field(default=None, max_length=4000)
    claim_source_region: Optional[List[float]] = None
    decision_policy_version: str = Field(default="reinspect-v1", min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_claim_contract(self) -> "VerificationCase":
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be ISO-8601") from exc
        if self.claim_mode == ClaimMode.EXTERNAL and not str(self.user_claim or "").strip():
            raise ValueError("external_claim requires a non-empty user_claim")
        if self.claim_mode == ClaimMode.EMBEDDED and self.user_claim is not None:
            raise ValueError("embedded_claim must recover the claim from visible pixels")
        if self.claim_source_region is not None:
            region = self.claim_source_region
            if len(region) != 4 or not all(isinstance(value, (int, float)) for value in region):
                raise ValueError("claim_source_region must be [x1,y1,x2,y2]")
            x1, y1, x2, y2 = [float(value) for value in region]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("claim_source_region must be normalized and non-empty")
        return self


class ClaimRecord(StrictModel):
    claim_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)
    question_id: str = Field(default="", max_length=64)
    criticality: Literal["decisive", "supporting", "contextual"] = "decisive"
    status: Literal["open", "supported", "refuted", "conflicted", "unverifiable"] = "open"
    unresolved_distinction: str = Field(default="", max_length=1000)


class SourceRecord(StrictModel):
    source_id: str = Field(min_length=1, max_length=200)
    canonical_url: str = ""
    hostname: str = ""
    registered_domain: str = ""
    source_family: str = Field(min_length=1, max_length=200)
    source_class: Literal["official", "news", "ugc", "visual", "runtime", "unknown"] = "unknown"
    artifact_sha256: str = ""
    retrieved_at: str = ""
    dependency_source_ids: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    claim_id: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=200)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=100)
    evidence_kind: Literal["web_span", "image_region", "runtime_anchor"]
    exact_text: str = Field(min_length=1, max_length=8000)
    span_start: Optional[int] = Field(default=None, ge=0)
    span_end: Optional[int] = Field(default=None, ge=1)
    image_region: Optional[List[float]] = None
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: str = Field(min_length=1)
    stance: Literal["support", "refute", "neutral"]
    quality: Literal["strong", "moderate", "weak"]
    directness: Literal["direct", "indirect"] = "direct"

    @model_validator(mode="after")
    def validate_locator(self) -> "EvidenceRecord":
        if self.evidence_kind == "web_span":
            if self.span_start is None or self.span_end is None:
                raise ValueError("web_span evidence requires exact offsets")
            if self.span_end <= self.span_start or self.span_end - self.span_start != len(self.exact_text):
                raise ValueError("web_span offsets must match exact_text")
        if self.evidence_kind == "image_region" and not self.image_region:
            raise ValueError("image_region evidence requires a region")
        try:
            datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("retrieved_at must be ISO-8601") from exc
        return self


class DiscoveryRecord(StrictModel):
    discovery_id: str = Field(min_length=1, max_length=200)
    claim_id: str = Field(min_length=1, max_length=100)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=100)
    candidate_url: str = ""
    title: str = ""
    snippet: str = ""
    candidate_type: Literal["serp", "reverse_image", "visual_reference"]
    promoted_evidence_id: Optional[str] = None


class FailureRecord(StrictModel):
    failure_id: str = Field(min_length=1, max_length=200)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=100)
    claim_id: Optional[str] = None
    code: Literal[
        "tool_error",
        "provider_unavailable",
        "access_limited",
        "malformed_result",
        "protocol_error",
        "timeout",
        "no_results",
        "budget_exhausted",
    ] = "tool_error"
    severity: Literal["fatal", "partial"] = "partial"
    recoverable: bool = True
    message: str = Field(min_length=1, max_length=4000)
    recovery_call_id: Optional[str] = None


class VerificationLedgers(StrictModel):
    claims: List[ClaimRecord] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    discoveries: List[DiscoveryRecord] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)


class ClaimDecision(StrictModel):
    claim_id: str = Field(min_length=1, max_length=100)
    decision: Literal["support", "refute", "unresolved"]
    evidence_ids: List[str] = Field(default_factory=list)
    reason: Optional[UnverifiableReason] = None


class Entity(StrictModel):
    """A visible entity detected in the image."""

    name: str = ""
    entity_type: str = ""
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attributes: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bbox(self) -> "Entity":
        if not self.bbox:
            return self
        if len(self.bbox) != 4:
            raise ValueError("entity bbox must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("entity bbox must be normalized and non-empty")
        return self


class TextRegion(StrictModel):
    """A visible OCR region."""

    text: str = ""
    bbox_quad: List[List[float]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language: str = "unknown"


class PerceptionReport(StrictModel):
    """Stage 1 output."""

    entities: List[Entity] = Field(default_factory=list)
    text_regions: List[TextRegion] = Field(default_factory=list)
    scene_description: str = ""
    image_type: str = "photo"


class InvestigationQuestion(StrictModel):
    """A question to investigate during verification."""

    question_id: str = Field(default="", max_length=64)
    question: str = Field(default="", max_length=500)
    why: str = Field(default="", max_length=500)
    suggested_tools: List[ShortListText] = Field(default_factory=list, max_length=3)
    suggested_queries: List[ShortListText] = Field(default_factory=list, max_length=3)
    related_entities: List[ShortListText] = Field(default_factory=list, max_length=8)
    priority: int = Field(default=1, ge=1, le=3)


class VerificationPlan(StrictModel):
    """Stage 2 output."""

    questions: List[InvestigationQuestion] = Field(default_factory=list, max_length=4)
    image_intent: str = Field(default="", max_length=500)
    is_trying_to_be_real: bool = True
    risk_assessment: str = Field(default="", max_length=500)
    revision: int = Field(default=0, ge=0)
    revision_reason: str = Field(default="", max_length=500)


class PlanRevision(StrictModel):
    """A bounded delta applied to unresolved questions after coverage audit."""

    question_updates: List[InvestigationQuestion] = Field(
        default_factory=list,
        min_length=1,
        max_length=8,
    )
    revision_reason: str = Field(default="", max_length=500)


class EvidenceItem(StrictModel):
    """A single evidence item collected during verification."""

    function_call_id: str = Field(default="", max_length=200)
    source: str = ""
    summary: str = ""
    raw_excerpt: str = ""
    direction: Literal["supports", "refutes", "neutral"] = "neutral"
    quality: Literal["strong", "moderate", "weak"] = "moderate"
    tool_used: str = ""
    related_question: str = ""


class VisualAnomaly(StrictModel):
    """A concrete anomaly copied from one successful visual tool result."""

    function_call_id: str = Field(default="", max_length=200)
    tool_used: Literal["analyze_visual_anomalies"] = "analyze_visual_anomalies"
    related_question: str = Field(default="", max_length=64)
    name: str = Field(default="", max_length=200)
    region: str = Field(default="", max_length=200)
    phenomenon: str = Field(default="", max_length=1200)
    reasoning: str = Field(default="", max_length=1200)
    severity: int = Field(default=0, ge=0, le=100)
    type: Literal[
        "ai_generation",
        "manipulation",
        "physical_inconsistency",
        "logical_inconsistency",
    ] = "physical_inconsistency"
    entities_involved: List[str] = Field(default_factory=list, max_length=12)


class QuestionResolution(StrictModel):
    """Coverage state for one investigation question."""

    question_id: str = ""
    status: Literal["unanswered", "in_progress", "resolved", "exhausted"] = "unanswered"
    conclusion: str = ""
    remaining_gap: str = ""
    tool_attempts: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)


class CoverageAudit(StrictModel):
    """Deterministic audit of whether the verification plan is covered."""

    iteration: int = Field(default=0, ge=0)
    complete: bool = False
    investigation_complete: bool = False
    question_resolutions: List[QuestionResolution] = Field(default_factory=list)
    unresolved_priority_questions: List[str] = Field(default_factory=list)
    exhausted_priority_questions: List[str] = Field(default_factory=list)
    successful_tool_calls: int = Field(default=0, ge=0)
    distinct_tools: List[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    reason: str = ""
    claim_statuses: Dict[str, str] = Field(default_factory=dict)
    unverifiable_reasons: List[UnverifiableReason] = Field(default_factory=list)


class VerificationResult(StrictModel):
    """Stage 3 output."""

    evidence: List[EvidenceItem] = Field(default_factory=list)
    visual_anomalies: List[VisualAnomaly] = Field(default_factory=list)
    authenticity_assessment: Literal[
        "authentic",
        "likely_ai",
        "likely_manipulated",
        "uncertain",
    ] = "uncertain"
    key_findings: List[str] = Field(default_factory=list)
    source_findings: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    world_model: Dict[str, Any] = Field(default_factory=dict)
    question_resolutions: List[QuestionResolution] = Field(default_factory=list)
    coverage_complete: bool = False
    unresolved_priority_questions: List[str] = Field(default_factory=list)
    exhausted_priority_questions: List[str] = Field(default_factory=list)
    iteration_count: int = Field(default=0, ge=0)


class FinalJudgment(StrictModel):
    """Stage 4 output."""

    verdict: Literal["real", "fake", "unverifiable"] = "unverifiable"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_chain: str = ""
    key_evidence: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    overall_assessment: str = ""


class LedgerJudgment(FinalJudgment):
    """Production Gemini judgment contract bound to immutable ledger ids."""

    claim_decisions: List[ClaimDecision] = Field(default_factory=list)
    selected_evidence_ids: List[str] = Field(default_factory=list)
    policy_rule_id: str = Field(default="reinspect-v1", max_length=100)
    unverifiable_reasons: List[UnverifiableReason] = Field(default_factory=list)


@dataclass
class VerificationState:
    """Aggregate state across the full run."""

    image_path: str = ""
    image_id: str = ""
    verification_case: Optional[VerificationCase] = None
    ledgers: VerificationLedgers = field(default_factory=VerificationLedgers)
    investigation_state: Any = None
    perception: Optional[PerceptionReport] = None
    plan: Optional[VerificationPlan] = None
    verification: Optional[VerificationResult] = None
    judgment: Optional[FinalJudgment] = None
    all_steps: List[Any] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_tool_calls: int = 0
    llm_api_calls: int = 0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    termination: str = ""
    errors: List[str] = field(default_factory=list)
    tool_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    plan_history: List[VerificationPlan] = field(default_factory=list)
    coverage_audits: List[CoverageAudit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full run state."""
        steps_data: List[Dict[str, Any]] = []
        for step in self.all_steps:
            step_dict = {
                "round": getattr(step, "round", 0),
                "stage": getattr(step, "stage_name", "") or getattr(step, "metadata", {}).get("stage", ""),
                "action_type": getattr(step, "action_type", ""),
                "tool_name": getattr(step, "tool_name", ""),
                "tool_args": getattr(step, "tool_args", {}),
                "tool_result": getattr(step, "tool_result", ""),
                "tokens": getattr(step, "tokens", {}),
            }
            thought = getattr(step, "thought", "")
            if thought:
                step_dict["thought"] = thought
            output = getattr(step, "output", None)
            if output is not None:
                step_dict["output"] = output
            metadata = getattr(step, "metadata", None)
            if metadata:
                step_dict["metadata"] = metadata
            steps_data.append(step_dict)

        return sanitize_for_persistence({
            "image_path": self.image_path,
            "image_id": self.image_id,
            "verification_case": self.verification_case.model_dump() if self.verification_case else None,
            "ledgers": self.ledgers.model_dump(),
            "investigation_state": (
                self.investigation_state.model_dump(mode="json")
                if hasattr(self.investigation_state, "model_dump")
                else self.investigation_state
            ),
            "perception": self.perception.model_dump() if self.perception else None,
            "plan": self.plan.model_dump() if self.plan else None,
            "plan_history": [item.model_dump() for item in self.plan_history],
            "verification": self.verification.model_dump() if self.verification else None,
            "coverage_audits": [item.model_dump() for item in self.coverage_audits],
            "judgment": self.judgment.model_dump() if self.judgment else None,
            "all_steps": steps_data,
            "stage_timings": self.stage_timings,
            "total_tool_calls": self.total_tool_calls,
            "llm_api_calls": self.llm_api_calls,
            "token_usage": self.token_usage,
            "termination": self.termination,
            "errors": self.errors,
            "tool_health": self.tool_health,
        })
