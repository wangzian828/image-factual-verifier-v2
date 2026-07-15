"""Strict state models for the image-only VisualFact investigation path."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InvestigationBrief(FrozenStrictModel):
    brief_id: str = Field(min_length=1, max_length=100)
    case_id: str = Field(min_length=1, max_length=200)
    input_mode: Literal["image_only"] = "image_only"
    media_type: Literal[
        "photo",
        "screenshot",
        "document",
        "illustration",
        "meme",
        "unknown",
    ] = "photo"
    objective: str = Field(
        default=(
            "Induce, investigate, and audit image-grounded factual targets using "
            "open-web and visual evidence."
        ),
        min_length=1,
        max_length=500,
    )
    required_output: Tuple[
        Literal[
            "verdict_target",
            "verdict",
            "verdict_basis",
            "evidence",
            "unresolved_gaps",
        ],
        ...,
    ] = Field(
        default_factory=lambda: (
            "verdict_target",
            "verdict",
            "verdict_basis",
            "evidence",
            "unresolved_gaps",
        )
    )
    stop_policy: Literal["coverage_or_bounded_unresolved"] = (
        "coverage_or_bounded_unresolved"
    )


class VisualEntity(StrictModel):
    entity_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=300)
    entity_type: str = Field(min_length=1, max_length=100)
    origin: Literal["input_image", "ocr", "web_discovery"] = "input_image"
    origin_ids: List[str] = Field(default_factory=list, max_length=8)
    region: Optional[List[float]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_region(self) -> "VisualEntity":
        if self.region is None:
            return self
        if len(self.region) != 4:
            raise ValueError("VisualEntity.region must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = [float(value) for value in self.region]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("VisualEntity.region must be normalized and non-empty")
        return self


class RetrievalAnchor(StrictModel):
    anchor_id: str = Field(min_length=1, max_length=100)
    kind: Literal["text", "logo", "entity", "scene_pattern"]
    value: str = Field(min_length=1, max_length=500)
    entity_id: Optional[str] = Field(default=None, max_length=100)
    region: Optional[List[float]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_region(self) -> "RetrievalAnchor":
        if self.region is None:
            return self
        if len(self.region) != 4:
            raise ValueError("RetrievalAnchor.region must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = [float(value) for value in self.region]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(
                "RetrievalAnchor.region must be normalized and non-empty"
            )
        return self


class FactOrigin(StrictModel):
    type: Literal["input_image", "ocr", "web_discovery"]
    origin_ids: List[str] = Field(min_length=1, max_length=8)


class VisualFact(StrictModel):
    fact_id: str = Field(min_length=1, max_length=100)
    kind: Literal[
        "attribute",
        "relation",
        "internal_consistency",
        "text_claim",
    ]
    statement: str = Field(min_length=1, max_length=1200)
    subject_entity_id: str = Field(min_length=1, max_length=100)
    predicate: str = Field(min_length=1, max_length=100)
    object_entity_id: Optional[str] = Field(default=None, max_length=100)
    status: Literal[
        "candidate",
        "active",
        "supported",
        "refuted",
        "conflicted",
        "blocked",
        "exhausted",
        "retired",
    ] = "candidate"
    basis_ids: List[str] = Field(min_length=1, max_length=12)
    decision_relevance: Literal[
        "unknown",
        "supporting",
        "decisive",
    ] = "unknown"
    origin: FactOrigin


class ResearchTask(StrictModel):
    task_id: str = Field(min_length=1, max_length=100)
    fact_ids: List[str] = Field(min_length=1, max_length=6)
    question: str = Field(min_length=1, max_length=800)
    purpose: str = Field(min_length=1, max_length=800)
    priority: int = Field(default=1, ge=1, le=3)
    status: Literal["pending", "active", "resolved", "blocked", "exhausted"] = (
        "active"
    )
    parent_task_id: Optional[str] = Field(default=None, max_length=100)
    origin_ids: List[str] = Field(min_length=1, max_length=12)
    suggested_tools: List[str] = Field(default_factory=list, max_length=4)
    suggested_queries: List[str] = Field(default_factory=list, max_length=3)
    attempt_count: int = Field(default=0, ge=0)
    finding_ids: List[str] = Field(default_factory=list, max_length=20)


class Finding(StrictModel):
    finding_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    fact_ids: List[str] = Field(min_length=1, max_length=6)
    statement: str = Field(min_length=1, max_length=1200)
    stance: Literal["support", "refute", "neutral"]
    evidence_ids: List[str] = Field(min_length=1, max_length=20)
    source_family_ids: List[str] = Field(default_factory=list, max_length=20)
    quality: Literal["decisive", "supporting", "contextual"] = "supporting"


class InvestigationDiscovery(StrictModel):
    discovery_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    fact_ids: List[str] = Field(min_length=1, max_length=6)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: Literal[
        "reverse_image_search",
        "text_search",
        "crop_and_search",
    ]
    candidate_url: str = Field(default="", max_length=4000)
    reference_image_url: str = Field(default="", max_length=4000)
    title: str = Field(default="", max_length=1000)
    snippet: str = Field(default="", max_length=2000)
    candidate_type: Literal["serp", "reverse_image", "visual_reference"]
    promoted_evidence_id: Optional[str] = Field(default=None, max_length=100)


class InvestigationEvidence(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    fact_ids: List[str] = Field(min_length=1, max_length=6)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=100)
    evidence_kind: Literal[
        "web_span",
        "image_region",
        "reference_comparison",
    ]
    source_url: str = Field(default="", max_length=4000)
    source_family: str = Field(min_length=1, max_length=300)
    source_class: Literal[
        "official",
        "news",
        "ugc",
        "visual",
        "unknown",
    ] = "unknown"
    exact_text: str = Field(min_length=1, max_length=8000)
    span_start: Optional[int] = Field(default=None, ge=0)
    span_end: Optional[int] = Field(default=None, ge=1)
    image_region: Optional[List[float]] = None
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: str = Field(min_length=1, max_length=100)
    stance: Literal["support", "refute", "neutral"]
    quality: Literal["strong", "moderate", "weak"]
    directness: Literal["direct", "indirect"] = "direct"
    claim_binding: Literal[
        "none",
        "source_assertion",
        "pixel_observation",
        "same_subject",
        "same_capture",
    ] = "none"
    same_subject_or_scene: Optional[bool] = None
    same_capture_or_near_duplicate: Optional[bool] = None
    likely_different_original_capture: Optional[bool] = None
    edit_evidence_present: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temporal_alignment: str = Field(default="", max_length=100)
    risk_flags: List[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_locator(self) -> "InvestigationEvidence":
        if self.evidence_kind == "web_span":
            if self.span_start is None or self.span_end is None:
                raise ValueError("web_span evidence requires exact offsets")
            if self.span_end <= self.span_start:
                raise ValueError("web_span evidence offsets must be ordered")
        if self.evidence_kind == "image_region":
            if not self.image_region or len(self.image_region) != 4:
                raise ValueError("image_region evidence requires [x1,y1,x2,y2]")
        return self


class InvestigationFailure(StrictModel):
    failure_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    fact_ids: List[str] = Field(min_length=1, max_length=6)
    function_call_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=100)
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
    recoverable: bool = True
    message: str = Field(min_length=1, max_length=4000)


class TaskUpdate(StrictModel):
    task_id: str = Field(min_length=1, max_length=100)
    priority: Optional[int] = Field(default=None, ge=1, le=3)
    reason: str = Field(default="", max_length=800)


class ReflectionOutput(StrictModel):
    task_updates: List[TaskUpdate] = Field(default_factory=list)
    new_tasks: List[ResearchTask] = Field(default_factory=list)
    proposed_decisive_fact_ids: List[str] = Field(default_factory=list)
    recommended_next_task_ids: List[str] = Field(default_factory=list)
    remaining_gaps: List[str] = Field(default_factory=list)
    ready_to_finish: bool = False


class TargetFactProposal(StrictModel):
    statement: str = Field(min_length=1, max_length=1200)
    kind: Literal["attribute", "relation", "internal_consistency"] = "relation"
    predicate: Literal[
        "source_record_matches",
        "visual_integrity",
        "provenance_matches",
        "identified_as",
        "located_at",
        "dated_as",
        "depicts_event",
    ]
    parent_fact_ids: List[str] = Field(min_length=1, max_length=12)
    question: str = Field(min_length=1, max_length=800)
    purpose: str = Field(min_length=1, max_length=800)
    suggested_tools: List[
        Literal[
            "reverse_image_search",
            "text_search",
            "visit",
            "compare_with_reference",
            "check_consistency",
            "analyze_visual_anomalies",
            "crop_and_search",
            "crop_and_inspect",
            "count_objects",
            "ocr_with_position",
        ]
    ] = Field(min_length=1, max_length=4)
    suggested_queries: List[str] = Field(default_factory=list, max_length=3)
    decision_relevance: Literal["supporting", "decisive"] = "decisive"


class TargetPlanningOutput(StrictModel):
    proposals: List[TargetFactProposal] = Field(
        default_factory=list,
        max_length=3,
    )
    remaining_target_gaps: List[str] = Field(
        default_factory=list,
        max_length=4,
    )


class AttributionFactProposal(StrictModel):
    statement: str = Field(min_length=1, max_length=1200)
    kind: Literal["attribute", "relation"] = "relation"
    predicate: Literal[
        "identified_as",
        "attributed_as",
        "created_by",
        "dated_as",
        "located_at",
        "occurred_at",
        "depicts_event",
        "source_record_matches",
    ] = "identified_as"
    parent_fact_ids: List[str] = Field(min_length=1, max_length=3)
    discovery_ids: List[str] = Field(default_factory=list, max_length=8)
    evidence_ids: List[str] = Field(default_factory=list, max_length=8)
    finding_ids: List[str] = Field(default_factory=list, max_length=8)
    suggested_queries: List[str] = Field(default_factory=list, max_length=3)
    decision_relevance: Literal["supporting", "decisive"] = "decisive"


class AttributionOutput(StrictModel):
    proposals: List[AttributionFactProposal] = Field(
        default_factory=list,
        max_length=2,
    )
    remaining_attribution_gaps: List[str] = Field(
        default_factory=list,
        max_length=4,
    )


class InvestigationSegmentOutput(StrictModel):
    segment_summary: str = Field(default="", max_length=1200)
    ready_for_reflection: bool = True


class ReflectionRecord(StrictModel):
    reflection_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1)
    output: ReflectionOutput
    accepted_task_update_ids: List[str] = Field(default_factory=list, max_length=12)
    accepted_new_task_ids: List[str] = Field(default_factory=list, max_length=3)
    accepted_decisive_fact_ids: List[str] = Field(default_factory=list, max_length=2)
    rejected_reasons: List[str] = Field(default_factory=list, max_length=20)
    evidence_gain: bool = False
    decision_gain: bool = False


class FactCoverage(StrictModel):
    fact_id: str = Field(min_length=1, max_length=100)
    status: Literal[
        "supported",
        "refuted",
        "conflicted",
        "blocked",
        "exhausted",
        "unresolved",
    ]
    finding_ids: List[str] = Field(default_factory=list, max_length=20)
    evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    winning_finding_ids: List[str] = Field(default_factory=list, max_length=20)
    winning_evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    support_score: float = Field(default=0.0, ge=0.0)
    refute_score: float = Field(default=0.0, ge=0.0)
    conflict_resolution: Literal[
        "not_applicable",
        "support_wins",
        "refute_wins",
        "needs_discriminating_evidence",
    ] = "not_applicable"
    reason: str = Field(default="", max_length=1200)


class ImageOnlyCoverage(StrictModel):
    audit_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=0)
    decisive_fact_ids: List[str] = Field(default_factory=list, max_length=6)
    facts: List[FactCoverage] = Field(default_factory=list, max_length=6)
    complete: bool = False
    stop_reason: Literal[
        "continue",
        "verdict_determined",
        "coverage_complete",
        "information_saturated",
        "hard_budget_exhausted",
    ] = "continue"
    substantive_gain: bool = False
    low_gain_intervals: int = Field(default=0, ge=0)
    reason: str = Field(default="", max_length=1200)


class VerdictBasis(StrictModel):
    policy_rule_id: Literal["reinspect-v2"] = "reinspect-v2"
    verdict_target: str = Field(min_length=1, max_length=1200)
    fact_ids: List[str] = Field(default_factory=list, max_length=6)
    finding_ids: List[str] = Field(default_factory=list, max_length=20)
    evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    mechanism: Optional[
        Literal[
            "reused_old_image",
            "wrong_event",
            "wrong_place",
            "wrong_identity",
            "landmark_mismatch",
            "impossible_causal_dynamics",
            "inconsistent_lighting_or_reflection",
            "synthetic_presented_as_documentary",
            "decisive_fact_refuted",
        ]
    ] = None
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=12)


class ImageOnlyJudgment(StrictModel):
    verdict: Literal["real", "fake", "unverifiable"]
    confidence: float = Field(ge=0.0, le=1.0)
    policy_rule_id: Literal["reinspect-v2"] = "reinspect-v2"
    selected_fact_ids: List[str] = Field(default_factory=list, max_length=6)
    selected_finding_ids: List[str] = Field(default_factory=list, max_length=20)
    selected_evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=12)


class ImageOnlyInvestigationState(StrictModel):
    brief: InvestigationBrief
    entities: List[VisualEntity] = Field(default_factory=list, max_length=48)
    facts: List[VisualFact] = Field(default_factory=list, max_length=72)
    tasks: List[ResearchTask] = Field(default_factory=list, max_length=12)
    retrieval_anchors: List[RetrievalAnchor] = Field(
        default_factory=list,
        max_length=32,
    )
    discoveries: List[InvestigationDiscovery] = Field(
        default_factory=list,
        max_length=200,
    )
    evidence: List[InvestigationEvidence] = Field(
        default_factory=list,
        max_length=200,
    )
    findings: List[Finding] = Field(default_factory=list, max_length=120)
    failures: List[InvestigationFailure] = Field(default_factory=list, max_length=120)
    reflections: List[ReflectionRecord] = Field(default_factory=list, max_length=6)
    coverage_audits: List[ImageOnlyCoverage] = Field(
        default_factory=list,
        max_length=32,
    )
    decisive_fact_ids: List[str] = Field(default_factory=list, max_length=6)
    recommended_next_task_ids: List[str] = Field(default_factory=list, max_length=4)
    attempted_routes: List[str] = Field(default_factory=list, max_length=120)
    action_count: int = Field(default=0, ge=0, le=24)
    reflection_failure_streak: int = Field(default=0, ge=0, le=2)
    verdict_basis: Optional[VerdictBasis] = None
    judgment: Optional[ImageOnlyJudgment] = None
    stop_reason: Literal[
        "",
        "verdict_determined",
        "coverage_complete",
        "information_saturated",
        "hard_budget_exhausted",
    ] = ""


class BootstrapInvestigation(StrictModel):
    brief: InvestigationBrief
    entities: List[VisualEntity] = Field(default_factory=list, max_length=32)
    facts: List[VisualFact] = Field(default_factory=list, max_length=48)
    tasks: List[ResearchTask] = Field(default_factory=list, max_length=4)
    retrieval_anchors: List[RetrievalAnchor] = Field(
        default_factory=list,
        max_length=24,
    )
    findings: List[Finding] = Field(default_factory=list)
