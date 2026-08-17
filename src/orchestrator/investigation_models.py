"""Strict state models for the image-only VisualFact investigation path."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


RouteFocus = Literal[
    "same_capture_reference",
    "entity_event_identity",
    "relation_value",
    "scene_world_constraints",
    "visual_consistency",
    "media_origin",
]


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
    claim_ids: List[str] = Field(default_factory=list, max_length=3)
    hypothesis_id: Optional[str] = Field(default=None, max_length=100)
    question: str = Field(min_length=1, max_length=800)
    purpose: str = Field(min_length=1, max_length=800)
    priority: int = Field(default=1, ge=1, le=3)
    status: Literal[
        "pending",
        "active",
        "resolved",
        "blocked",
        "exhausted",
        "superseded",
    ] = "active"
    parent_task_id: Optional[str] = Field(default=None, max_length=100)
    origin_ids: List[str] = Field(min_length=1, max_length=12)
    suggested_tools: List[str] = Field(default_factory=list, max_length=5)
    suggested_queries: List[str] = Field(default_factory=list, max_length=3)
    query_replan_count: int = Field(default=0, ge=0, le=1)
    route_replan_count: int = Field(default=0, ge=0, le=1)
    route_replan_focus: str = Field(default="", max_length=800)
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
    candidate_type: Literal[
        "serp",
        "reverse_image",
        "visual_reference",
    ]
    promoted_evidence_id: Optional[str] = Field(default=None, max_length=100)
    abandoned: bool = False
    abandonment_reason: str = Field(default="", max_length=800)


class VisualObservation(StrictModel):
    view_index: int = Field(ge=0, le=4)
    view_kind: Literal[
        "original",
        "anchor_detail",
        "relation_context",
    ]
    region: List[float]
    statement: str = Field(min_length=1, max_length=600)
    property_status: Literal[
        "observed",
        "not_observed",
        "ambiguous",
    ]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_region(self) -> "VisualObservation":
        if len(self.region) != 4:
            raise ValueError("VisualObservation.region must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = [float(value) for value in self.region]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(
                "VisualObservation.region must be normalized and non-empty"
            )
        return self


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
    image_claim: str = Field(default="", max_length=1800)
    retrieval_goal: str = Field(default="", max_length=1800)
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
    relation_scope: Literal[
        "same_relation",
        "partial_relation",
        "different_instance",
        "unclear",
    ] = "unclear"
    relation_stance: Literal[
        "supports",
        "contradicts",
        "background",
        "unclear",
    ] = "unclear"
    same_subject_or_scene: Optional[bool] = None
    same_capture_or_near_duplicate: Optional[bool] = None
    likely_different_original_capture: Optional[bool] = None
    edit_evidence_present: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temporal_alignment: str = Field(default="", max_length=100)
    risk_flags: List[str] = Field(default_factory=list, max_length=12)
    visual_question_id: Optional[str] = Field(default=None, max_length=100)
    visual_scope: Optional[
        Literal["subject", "relation", "scene", "text", "integrity"]
    ] = None
    visual_answer_status: Optional[
        Literal["observed", "not_observed", "ambiguous"]
    ] = None
    visual_observations: List[VisualObservation] = Field(
        default_factory=list,
        max_length=8,
    )

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
        "rate_limited",
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
    recommended_next_task_ids: List[str] = Field(default_factory=list)
    remaining_gaps: List[str] = Field(default_factory=list)
    strategy_decision: Literal[
        "continue",
        "replan",
        "stop_unresolved",
    ] = "continue"
    strategy_task_id: Optional[str] = Field(default=None, max_length=100)
    replacement_query: str = Field(default="", max_length=500)
    expected_information: str = Field(default="", max_length=800)
    strategy_rationale: str = Field(default="", max_length=1200)
    ready_to_finish: bool = False


class QueryConcept(StrictModel):
    concept_id: str = Field(min_length=1, max_length=100)
    evidence_id: str = Field(min_length=1, max_length=100)
    evidence_phrase: str = Field(min_length=1, max_length=800)
    role: Literal[
        "use_or_category",
        "identity",
        "place_or_event",
        "relation_context",
        "presentation_or_mechanics",
        "other",
    ] = "other"


class QueryConceptExtractionOutput(StrictModel):
    task_id: str = Field(min_length=1, max_length=100)
    concepts: List[QueryConcept] = Field(default_factory=list, max_length=8)


class QueryReplanOutput(StrictModel):
    task_id: str = Field(min_length=1, max_length=100)
    selected_concept_id: str = Field(min_length=1, max_length=100)
    concept_term: str = Field(min_length=1, max_length=300)
    replacement_query: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=800)


class TargetFactProposal(StrictModel):
    statement: str = Field(min_length=1, max_length=1200)
    kind: Literal["attribute", "relation", "internal_consistency"] = "relation"
    predicate: Literal[
        "source_record_matches",
        "visual_integrity",
        "provenance_matches",
        "identified_as",
        "depicts_relation",
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
            "crop_and_inspect",
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


class ImageClaimProposal(StrictModel):
    claim_key: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    statement: str = Field(min_length=1, max_length=1200)
    kind: Literal["attribute", "relation", "internal_consistency", "text_claim"]
    predicate: str = Field(min_length=1, max_length=100)
    anchor_fact_ids: List[str] = Field(min_length=1, max_length=12)
    salience: Literal["high", "medium"] = "high"


class SearchHypothesisProposal(StrictModel):
    hypothesis_key: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    route_focus: RouteFocus = Field(
        description=(
            "Primary reason this route exists. Use media_origin only when the "
            "route would primarily identify creator, publisher/platform, "
            "generation method, or publication history of the image itself; "
            "that focus is rejected for factual-image investigation."
        ),
    )
    statement: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "Neutral answer-seeking investigation route about the source, entity, "
            "event, relation, or value; not a proposed verdict or media-origin class."
        ),
    )
    queries: List[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Neutral alternative query formulations built from visible anchors "
            "and relation terms, not a guaranteed execution queue."
        ),
    )
    expected_information: str = Field(
        min_length=1,
        max_length=800,
        description=(
            "The underlying source or relation facts the route should recover, "
            "without presupposing the final verdict."
        ),
    )
    suggested_tools: List[
        Literal[
            "reverse_image_search",
            "text_search",
            "visit",
            "compare_with_reference",
            "check_consistency",
            "analyze_visual_anomalies",
            "crop_and_inspect",
            "ocr_with_position",
        ]
    ] = Field(min_length=1, max_length=4)
    priority: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def validate_first_hop(self) -> "SearchHypothesisProposal":
        if not self.queries and not set(self.suggested_tools) & {
            "reverse_image_search",
            "text_search",
            "check_consistency",
            "analyze_visual_anomalies",
            "crop_and_inspect",
            "ocr_with_position",
        }:
            raise ValueError(
                "search hypothesis requires an executable first-hop tool"
            )
        return self


class ImageAccountPlanningOutput(StrictModel):
    """Image target planning with a legacy ``image_claims`` wire field.

    The wire name remains for replay and training-schema compatibility. These
    rows are semantically image-grounded target-fact bookkeeping, not
    user-supplied Claims or provenance requirements.
    """

    account_summary: str = Field(min_length=1, max_length=1600)
    image_claims: List[ImageClaimProposal] = Field(min_length=1, max_length=3)
    search_hypotheses: List[SearchHypothesisProposal] = Field(
        min_length=1,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_references(self) -> "ImageAccountPlanningOutput":
        claim_keys = [item.claim_key for item in self.image_claims]
        if len(claim_keys) != len(set(claim_keys)):
            raise ValueError("image claim keys must be unique")
        hypothesis_keys = [
            item.hypothesis_key for item in self.search_hypotheses
        ]
        if len(hypothesis_keys) != len(set(hypothesis_keys)):
            raise ValueError("search hypothesis keys must be unique")
        high_claims = [
            item.claim_key
            for item in self.image_claims
            if item.salience == "high"
        ]
        if len(high_claims) != 1:
            raise ValueError(
                "image account requires exactly one high-salience central "
                "target fact"
            )
        return self


class ImageClaim(StrictModel):
    """Legacy target-fact bookkeeping retained for state/replay compatibility."""

    claim_id: str = Field(min_length=1, max_length=100)
    fact_id: str = Field(min_length=1, max_length=100)
    statement: str = Field(min_length=1, max_length=1200)
    anchor_fact_ids: List[str] = Field(min_length=1, max_length=12)
    salience: Literal["high", "medium"] = "high"
    status: Literal[
        "open",
        "supported",
        "refuted",
        "conflicted",
        "unresolved",
    ] = "open"
    task_ids: List[str] = Field(default_factory=list, max_length=8)


class SearchHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=100)
    claim_ids: List[str] = Field(min_length=1, max_length=3)
    route_focus: RouteFocus = "relation_value"
    statement: str = Field(min_length=1, max_length=1200)
    queries: List[str] = Field(default_factory=list, max_length=3)
    expected_information: str = Field(min_length=1, max_length=800)
    suggested_tools: List[str] = Field(min_length=1, max_length=4)
    priority: int = Field(default=1, ge=1, le=3)
    status: Literal[
        "open",
        "active",
        "exhausted",
        "retired",
    ] = "open"
    task_id: Optional[str] = Field(default=None, max_length=100)
    attempt_count: int = Field(default=0, ge=0)


class EvidenceDecisionRefinement(StrictModel):
    slot: Literal[
        "subject_identity",
        "object_category",
        "scene_location",
        "event_identity",
    ]
    statement: str = Field(min_length=1, max_length=1200)
    predicate: Literal[
        "identified_as",
        "depicts_relation",
        "located_at",
        "occurred_at",
        "depicts_event",
        "source_record_matches",
        "provenance_matches",
    ]
    anchor_fact_ids: List[str] = Field(min_length=1, max_length=6)
    grounding_evidence_ids: List[str] = Field(min_length=1, max_length=8)
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
            "crop_and_inspect",
            "ocr_with_position",
        ]
    ] = Field(min_length=1, max_length=4)
    suggested_queries: List[str] = Field(default_factory=list, max_length=3)


class VisualReinspectionRequest(StrictModel):
    reason: Literal[
        "identity",
        "relation",
        "location",
        "event",
        "text",
        "integrity",
    ]
    scope: Literal[
        "subject",
        "relation",
        "scene",
        "text",
        "integrity",
    ]
    question: str = Field(min_length=1, max_length=800)
    expected_property: str = Field(min_length=1, max_length=800)
    anchor_fact_ids: List[str] = Field(min_length=1, max_length=6)
    grounding_evidence_ids: List[str] = Field(min_length=1, max_length=8)


class VisualDiscriminatorCandidate(StrictModel):
    source_phrase: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Concise source-grounded cue that motivates this pixel check. "
            "It may paraphrase or recombine the reviewed Evidence."
        ),
    )
    visible_property: str = Field(
        min_length=1,
        max_length=240,
        description="Directly visible property to inspect in the original pixels.",
    )
    why_discriminative: str = Field(
        min_length=12,
        max_length=800,
        description=(
            "Why this property has information gain beyond generically "
            "confirming the current image-grounded target fact."
        ),
    )
    already_in_claim: bool = Field(
        description=(
            "Whether the current target fact or visual account already asserts "
            "this property."
        )
    )
    expected_if_source_matches: str = Field(
        min_length=1,
        max_length=300,
        description="What should be visible if the source description matches.",
    )


class VisualReinspectionProposal(StrictModel):
    """Model-selected Claim and visual question before runtime-owned evidence binding."""

    claim_id: str = Field(min_length=1, max_length=100)
    reason: Literal[
        "identity",
        "relation",
        "location",
        "event",
        "text",
        "integrity",
    ]
    scope: Literal[
        "subject",
        "relation",
        "scene",
        "text",
        "integrity",
    ]
    question: str = Field(
        min_length=1,
        max_length=800,
        description="One focused question about a directly observable pixel property.",
    )
    expected_property: str = Field(
        min_length=1,
        max_length=240,
        description=(
            "The selected candidate's concise visible property; not an inferred "
            "measurement or broad authenticity classification."
        ),
    )
    candidate_discriminators: List[VisualDiscriminatorCandidate] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Two or three source-grounded visible candidates considered before "
            "choosing the focused reinspection target."
        ),
    )
    selected_discriminator_index: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_selected_discriminator_index(self) -> "VisualReinspectionProposal":
        if (
            self.candidate_discriminators
            and self.selected_discriminator_index >= len(self.candidate_discriminators)
        ):
            raise ValueError(
                "selected_discriminator_index must refer to a supplied candidate"
            )
        return self


class ClaimAssessmentProposal(StrictModel):
    claim_id: str = Field(min_length=1, max_length=100)
    assessment: Literal[
        "supported",
        "refuted",
        "conflicted",
        "insufficient",
    ]
    selected_evidence_ids: List[str] = Field(default_factory=list, max_length=20)
    remaining_gap: str = Field(default="", max_length=800)
    rationale: str = Field(min_length=1, max_length=1600)


class MaterialDiscrepancyProposal(StrictModel):
    statement: str = Field(min_length=1, max_length=1600)
    affected_claim_ids: List[str] = Field(min_length=1, max_length=3)
    visual_anchor_fact_ids: List[str] = Field(min_length=1, max_length=12)
    evidence_ids: List[str] = Field(min_length=1, max_length=20)
    materiality: Literal["decisive", "supporting"] = "decisive"
    status: Literal["established", "conflicted"] = "established"
    rationale: str = Field(min_length=1, max_length=1600)


class MaterialDiscrepancyDraft(StrictModel):
    """Model-authored discrepancy before runtime-owned visual-anchor binding."""

    statement: str = Field(min_length=1, max_length=1600)
    affected_claim_ids: List[str] = Field(min_length=1, max_length=3)
    evidence_ids: List[str] = Field(min_length=1, max_length=20)
    materiality: Literal["decisive", "supporting"] = "decisive"
    status: Literal["established", "conflicted"] = "established"
    rationale: str = Field(min_length=1, max_length=1600)


class NewSearchHypothesis(StrictModel):
    claim_ids: List[str] = Field(min_length=1, max_length=3)
    route_focus: RouteFocus = Field(
        description=(
            "Primary reason this route exists. media_origin is not an allowed "
            "continuation route for factual-image investigation."
        ),
    )
    statement: str = Field(min_length=1, max_length=1200)
    queries: List[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Alternative candidate query formulations, not a guaranteed "
            "execution queue."
        ),
    )
    expected_information: str = Field(min_length=1, max_length=800)
    suggested_tools: List[
        Literal[
            "reverse_image_search",
            "text_search",
            "visit",
            "compare_with_reference",
            "check_consistency",
            "analyze_visual_anomalies",
            "crop_and_inspect",
            "ocr_with_position",
        ]
    ] = Field(min_length=1, max_length=4)
    priority: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def validate_first_hop(self) -> "NewSearchHypothesis":
        if not self.queries and not set(self.suggested_tools) & {
            "reverse_image_search",
            "text_search",
            "check_consistency",
            "analyze_visual_anomalies",
            "crop_and_inspect",
            "ocr_with_position",
        }:
            raise ValueError(
                "new search hypothesis requires an executable first-hop tool"
            )
        return self


class VisualEvidenceDisposition(StrictModel):
    """Explicit audit note for claim-owned pixel Evidence that is not consumed."""

    disposition: Literal["irrelevant_to_current_claim_or_discrepancy"]
    evidence_ids: List[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(min_length=12, max_length=800)


class DiscrepancyDecisionOutput(StrictModel):
    claim_assessments: List[ClaimAssessmentProposal] = Field(
        default_factory=list,
        max_length=3,
    )
    material_discrepancy: Optional[MaterialDiscrepancyProposal] = None
    retire_hypothesis_ids: List[str] = Field(default_factory=list, max_length=6)
    new_hypotheses: List[NewSearchHypothesis] = Field(
        default_factory=list,
        max_length=3,
    )
    visual_reinspection: Optional[VisualReinspectionRequest] = None
    visual_evidence_disposition: Optional[VisualEvidenceDisposition] = None
    verdict_proposal: Literal[
        "continue",
        "fake",
        "real",
    ] = "continue"
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_references(self) -> "DiscrepancyDecisionOutput":
        assessment_claim_ids = [item.claim_id for item in self.claim_assessments]
        if len(assessment_claim_ids) != len(set(assessment_claim_ids)):
            raise ValueError("a decision may assess each image claim at most once")
        if len(self.retire_hypothesis_ids) != len(
            set(self.retire_hypothesis_ids)
        ):
            raise ValueError("retired search hypothesis IDs must be unique")
        return self


class DiscrepancyDecisionProposalOutput(DiscrepancyDecisionOutput):
    """Model-facing v4 Decision output with runtime-bound visual references."""

    material_discrepancy: Optional[MaterialDiscrepancyDraft] = None
    visual_reinspection: Optional[VisualReinspectionProposal] = None

    @model_validator(mode="after")
    def validate_sparse_visual_transition(
        self,
    ) -> "DiscrepancyDecisionProposalOutput":
        if self.visual_reinspection is None:
            return self
        if self.claim_assessments or self.material_discrepancy is not None:
            raise ValueError(
                "visual reinspection proposal must not copy claim or discrepancy IDs"
            )
        if self.retire_hypothesis_ids or self.new_hypotheses:
            raise ValueError(
                "visual reinspection proposal must be the only state transition"
            )
        if self.visual_evidence_disposition is not None:
            raise ValueError(
                "visual reinspection proposal must be the only state transition"
            )
        if self.verdict_proposal != "continue":
            raise ValueError(
                "visual reinspection proposal requires verdict_proposal='continue'"
            )
        return self


class ClaimAssessment(StrictModel):
    assessment_id: str = Field(min_length=1, max_length=100)
    claim_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=0, le=24)
    assessment: Literal[
        "supported",
        "refuted",
        "conflicted",
        "insufficient",
    ]
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)
    finding_ids: List[str] = Field(default_factory=list, max_length=20)
    remaining_gap: str = Field(default="", max_length=800)
    rationale: str = Field(min_length=1, max_length=1600)


class MaterialDiscrepancy(StrictModel):
    discrepancy_id: str = Field(min_length=1, max_length=100)
    statement: str = Field(min_length=1, max_length=1600)
    affected_claim_ids: List[str] = Field(min_length=1, max_length=3)
    visual_anchor_fact_ids: List[str] = Field(min_length=1, max_length=12)
    evidence_ids: List[str] = Field(min_length=1, max_length=20)
    materiality: Literal["decisive", "supporting"] = "decisive"
    status: Literal["established", "conflicted"] = "established"
    rationale: str = Field(min_length=1, max_length=1600)


class DiscrepancyDecisionRecord(StrictModel):
    decision_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=0, le=24)
    trigger: Literal[
        "qualified_evidence",
        "scheduled_boundary",
        "strategy_boundary",
        "before_unresolved",
    ]
    reviewed_evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    output: DiscrepancyDecisionOutput
    accepted_assessment_ids: List[str] = Field(default_factory=list, max_length=3)
    accepted_finding_ids: List[str] = Field(default_factory=list, max_length=20)
    accepted_discrepancy_id: Optional[str] = Field(default=None, max_length=100)
    accepted_hypothesis_ids: List[str] = Field(default_factory=list, max_length=3)
    retired_hypothesis_ids: List[str] = Field(default_factory=list, max_length=6)
    accepted_visual_question_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )
    rejected_reasons: List[str] = Field(default_factory=list, max_length=20)


class ImageClaimCoverage(StrictModel):
    claim_id: str = Field(min_length=1, max_length=100)
    salience: Literal["high", "medium"]
    assessment: Literal[
        "open",
        "supported",
        "refuted",
        "conflicted",
        "insufficient",
    ]
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)
    remaining_gap: str = Field(default="", max_length=800)
    route_status: Literal["open", "closed"] = "open"


class DiscrepancyCoverageAudit(StrictModel):
    audit_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=0, le=24)
    claims: List[ImageClaimCoverage] = Field(default_factory=list, max_length=3)
    decisive_discrepancy_ids: List[str] = Field(
        default_factory=list,
        max_length=12,
    )
    proposed_verdict: Literal[
        "",
        "continue",
        "fake",
        "real",
    ] = ""
    complete: bool = False
    stop_reason: Literal[
        "continue",
        "verdict_determined",
        "meaningful_routes_exhausted",
        "information_saturated",
        "hard_budget_exhausted",
    ] = "continue"
    decision_checkpoint: bool = False
    substantive_gain: bool = False
    reason: str = Field(default="", max_length=1200)


class DiscrepancyVerdictBasis(StrictModel):
    policy_rule_id: Literal["discrepancy-first-v4"] = "discrepancy-first-v4"
    decision_mode: Literal[
        "evidence_determined",
        "bounded_binary_judgment",
    ] = "evidence_determined"
    verdict_target: str = Field(min_length=1, max_length=1600)
    claim_ids: List[str] = Field(default_factory=list, max_length=3)
    discrepancy_ids: List[str] = Field(default_factory=list, max_length=12)
    visual_anchor_fact_ids: List[str] = Field(default_factory=list, max_length=24)
    finding_ids: List[str] = Field(default_factory=list, max_length=20)
    evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    diagnostic_finding_ids: List[str] = Field(default_factory=list, max_length=12)
    diagnostic_evidence_ids: List[str] = Field(default_factory=list, max_length=12)
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=12)


class TerminalVisualRationale(StrictModel):
    """Target-specific pixel rationale for an otherwise unclosed final verdict."""

    target_visible_property: str = Field(min_length=1, max_length=600)
    observed_property: str = Field(min_length=1, max_length=900)
    counterfactual_difference: str = Field(min_length=1, max_length=900)
    relation_to_verdict: Literal["supports_real", "supports_fake"]


class DiscrepancyJudgmentOutput(StrictModel):
    """Model-owned portion of the final binary judgment.

    Basis identifiers are deliberately not generated by the model.  The runtime
    compiles them from the accepted investigation state and injects them into the
    canonical :class:`DiscrepancyJudgment` record.
    """

    verdict: Literal["real", "fake"]
    confidence: float = Field(ge=0.0, le=1.0)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    terminal_visual_rationale: Optional[TerminalVisualRationale] = None


class DiscrepancyJudgment(StrictModel):
    verdict: Literal["real", "fake"]
    confidence: float = Field(ge=0.0, le=1.0)
    policy_rule_id: Literal["discrepancy-first-v4"] = "discrepancy-first-v4"
    selected_claim_ids: List[str] = Field(default_factory=list, max_length=3)
    selected_discrepancy_ids: List[str] = Field(default_factory=list, max_length=12)
    selected_visual_anchor_fact_ids: List[str] = Field(
        default_factory=list,
        max_length=24,
    )
    selected_finding_ids: List[str] = Field(default_factory=list, max_length=20)
    selected_evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    terminal_visual_rationale: Optional[TerminalVisualRationale] = None
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=12)


class ProgressEvent(StrictModel):
    progress_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1, le=24)
    gain: Literal[
        "lead_gain",
        "evidence_gain",
        "decision_gain",
        "visual_understanding_gain",
        "no_gain",
    ]
    source_ids: List[str] = Field(default_factory=list, max_length=40)
    no_substantive_gain_streak: int = Field(ge=0, le=24)
    # Retained in the trace schema for historical replay. The v4 runtime never
    # uses these diagnostic values to schedule a checkpoint or stop a case.
    soft_checkpoint_triggered: bool = False
    grace_remaining: int = Field(default=0, ge=0, le=8)
    rationale: str = Field(min_length=1, max_length=800)


class EvidenceDecisionOutput(StrictModel):
    active_fact_id: str = Field(min_length=1, max_length=100)
    assessment: Literal[
        "supported",
        "refuted",
        "conflicted",
        "insufficient",
    ]
    selected_evidence_ids: List[str] = Field(default_factory=list, max_length=12)
    binding_requirement: Literal[
        "none",
        "text_sufficient",
        "same_capture_helpful",
        "same_capture_required",
    ] = "none"
    remaining_gap: str = Field(default="", max_length=800)
    rationale: str = Field(min_length=1, max_length=1600)
    refinement: Optional[EvidenceDecisionRefinement] = None
    visual_reinspection: Optional[VisualReinspectionRequest] = None


class VisualReinspectionRecord(StrictModel):
    visual_question_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    fact_id: str = Field(min_length=1, max_length=100)
    created_action_count: int = Field(ge=0, le=24)
    request: VisualReinspectionRequest
    anchor_regions: List[List[float]] = Field(default_factory=list, max_length=4)
    status: Literal[
        "pending",
        "running",
        "resolved",
        "failed",
    ] = "pending"
    evidence_ids: List[str] = Field(default_factory=list, max_length=8)
    failure_ids: List[str] = Field(default_factory=list, max_length=4)
    view_artifacts: List[dict[str, Any]] = Field(default_factory=list, max_length=8)


class EvidenceDecisionRecord(StrictModel):
    decision_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1)
    trigger: Literal[
        "decisive_evidence",
        "before_reflection",
        "before_replan",
        "before_unverifiable",
    ]
    reviewed_evidence_ids: List[str] = Field(min_length=1, max_length=40)
    output: EvidenceDecisionOutput
    finding_ids: List[str] = Field(default_factory=list, max_length=12)
    accepted_refinement_fact_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )
    accepted_visual_question_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )
    accepted_visual_task_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )


class InvestigationSegmentOutput(StrictModel):
    segment_summary: str = Field(default="", max_length=1200)
    ready_for_reflection: bool = True


class ReflectionRecord(StrictModel):
    reflection_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1)
    trigger: Literal["interval", "saturation"] = "interval"
    output: ReflectionOutput
    accepted_task_update_ids: List[str] = Field(default_factory=list, max_length=12)
    accepted_new_task_ids: List[str] = Field(default_factory=list, max_length=3)
    rejected_reasons: List[str] = Field(default_factory=list, max_length=20)
    accepted_strategy_decision: Literal[
        "continue",
        "replan",
        "stop_unresolved",
    ] = "continue"
    accepted_replan_query: str = Field(default="", max_length=500)
    strategy_rejected_reason: str = Field(default="", max_length=800)
    evidence_gain: bool = False
    decision_gain: bool = False


class QueryReplanRecord(StrictModel):
    replan_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1)
    trigger: Literal["evidence_boundary"] = "evidence_boundary"
    new_evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    concept_extraction: QueryConceptExtractionOutput
    output: QueryReplanOutput
    accepted_queries: List[str] = Field(default_factory=list, max_length=1)
    rejected_reason: str = Field(default="", max_length=800)


class RouteLocalReplanOutput(StrictModel):
    """One bounded, route-local response to an observed investigation boundary."""

    task_id: str = Field(min_length=1, max_length=100)
    strategy: Literal[
        "replace_query",
        "add_visual_route",
        "continue",
        "stop_route",
    ]
    replacement_query: str = Field(default="", max_length=500)
    visual_focus: str = Field(default="", max_length=800)
    rationale: str = Field(min_length=1, max_length=1200)


class RouteLocalReplanRecord(StrictModel):
    """Audit one free but runtime-bounded route-local replanning decision."""

    replan_id: str = Field(min_length=1, max_length=100)
    action_count: int = Field(ge=1, le=24)
    trigger: Literal[
        "related_unclosed",
        "candidate_exhausted",
        "source_failure",
        "visual_signal",
    ]
    task_id: str = Field(min_length=1, max_length=100)
    output: RouteLocalReplanOutput
    accepted_strategy: Literal[
        "replace_query",
        "add_visual_route",
        "continue",
        "stop_route",
        "rejected",
    ] = "rejected"
    accepted_query: str = Field(default="", max_length=500)
    rejected_reason: str = Field(default="", max_length=800)


class EvidenceGap(StrictModel):
    gap_id: str = Field(min_length=1, max_length=100)
    fact_id: str = Field(min_length=1, max_length=100)
    kind: Literal[
        "image_source_binding",
        "direct_support_or_refute",
        "conflict_resolution",
    ]
    status: Literal[
        "open",
        "resolved",
        "blocked",
        "exhausted",
        "not_required",
    ] = "open"
    evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    reason: str = Field(default="", max_length=800)


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
    decisive_fact_ids: List[str] = Field(default_factory=list, max_length=1)
    facts: List[FactCoverage] = Field(default_factory=list, max_length=1)
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list, max_length=3)
    complete: bool = False
    stop_reason: Literal[
        "continue",
        "verdict_determined",
        "coverage_complete",
        "information_saturated",
        "hard_budget_exhausted",
    ] = "continue"
    reflection_checkpoint: bool = False
    decision_checkpoint: bool = False
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
    reflections: List[ReflectionRecord] = Field(default_factory=list, max_length=7)
    query_replans: List[QueryReplanRecord] = Field(
        default_factory=list,
        max_length=12,
    )
    route_local_replans: List[RouteLocalReplanRecord] = Field(
        default_factory=list,
        max_length=12,
    )
    evidence_decisions: List[EvidenceDecisionRecord] = Field(
        default_factory=list,
        max_length=24,
    )
    visual_reinspections: List[VisualReinspectionRecord] = Field(
        default_factory=list,
        max_length=2,
    )
    coverage_audits: List[ImageOnlyCoverage] = Field(
        default_factory=list,
        max_length=32,
    )
    image_account_summary: str = Field(default="", max_length=1600)
    image_claims: List[ImageClaim] = Field(default_factory=list, max_length=3)
    search_hypotheses: List[SearchHypothesis] = Field(
        default_factory=list,
        max_length=12,
    )
    claim_assessments: List[ClaimAssessment] = Field(
        default_factory=list,
        max_length=72,
    )
    material_discrepancies: List[MaterialDiscrepancy] = Field(
        default_factory=list,
        max_length=12,
    )
    discrepancy_decisions: List[DiscrepancyDecisionRecord] = Field(
        default_factory=list,
        max_length=24,
    )
    discrepancy_coverage_audits: List[DiscrepancyCoverageAudit] = Field(
        default_factory=list,
        max_length=32,
    )
    discrepancy_verdict_basis: Optional[DiscrepancyVerdictBasis] = None
    discrepancy_judgment: Optional[DiscrepancyJudgment] = None
    proposed_verdict: Literal[
        "",
        "continue",
        "fake",
        "real",
    ] = ""
    core_verdict_fact_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )
    core_fact_refinement_count: int = Field(default=0, ge=0, le=1)
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list, max_length=3)
    decisive_fact_ids: List[str] = Field(default_factory=list, max_length=1)
    recommended_next_task_ids: List[str] = Field(default_factory=list, max_length=4)
    attempted_routes: List[str] = Field(default_factory=list, max_length=120)
    recalled_archive_memory_ids: List[str] = Field(
        default_factory=list,
        max_length=120,
    )
    read_archive_memory_ids: List[str] = Field(
        default_factory=list,
        max_length=120,
    )
    pending_archive_read_ids: List[str] = Field(
        default_factory=list,
        max_length=12,
    )
    action_count: int = Field(default=0, ge=0, le=24)
    reflection_failure_streak: int = Field(default=0, ge=0, le=2)
    no_substantive_gain_streak: int = Field(default=0, ge=0, le=24)
    # Legacy trace fields. They remain serializable but have no v4 control-flow
    # effect after no-gain settlement was removed.
    saturation_checkpoint_action: Optional[int] = Field(default=None, ge=0, le=24)
    saturation_grace_remaining: int = Field(default=0, ge=0, le=8)
    # One accepted tool action creates one action-progress event, while a semantic
    # Decision may record an additional decision-progress event without consuming
    # another action.  This audit ledger therefore must not share the 24-action
    # budget's list bound; the action_count fields remain independently capped.
    progress_events: List[ProgressEvent] = Field(default_factory=list)
    verdict_basis: Optional[VerdictBasis] = None
    judgment: Optional[ImageOnlyJudgment] = None
    stop_reason: Literal[
        "",
        "verdict_determined",
        "coverage_complete",
        "meaningful_routes_exhausted",
        "information_saturated",
        "hard_budget_exhausted",
        "engineering_error",
    ] = ""

    @model_validator(mode="after")
    def validate_core_verdict_ownership(
        self,
    ) -> "ImageOnlyInvestigationState":
        if self.core_verdict_fact_id is None and self.decisive_fact_ids:
            self.core_verdict_fact_id = self.decisive_fact_ids[0]
        if self.core_verdict_fact_id is not None:
            if self.core_verdict_fact_id not in {
                fact.fact_id for fact in self.facts
            }:
                raise ValueError(
                    "core_verdict_fact_id must reference an existing fact"
                )
            self.decisive_fact_ids = [self.core_verdict_fact_id]
        elif self.decisive_fact_ids:
            raise ValueError(
                "decisive_fact_ids requires core_verdict_fact_id"
            )
        facts = {fact.fact_id for fact in self.facts}
        if len(facts) != len(self.facts):
            raise ValueError("VisualFact IDs must be unique")
        claims = {claim.claim_id: claim for claim in self.image_claims}
        if len(claims) != len(self.image_claims):
            raise ValueError("ImageClaim IDs must be unique")
        if any(claim.fact_id not in facts for claim in claims.values()):
            raise ValueError("image claims must reference existing VisualFacts")
        fact_by_id = {fact.fact_id: fact for fact in self.facts}
        if any(
            not set(claim.anchor_fact_ids) <= facts
            or any(
                fact_by_id[fact_id].origin.type not in {"input_image", "ocr"}
                for fact_id in claim.anchor_fact_ids
            )
            for claim in claims.values()
        ):
            raise ValueError(
                "image claim anchors must reference pixel/OCR VisualFacts"
            )
        hypotheses = {
            hypothesis.hypothesis_id: hypothesis
            for hypothesis in self.search_hypotheses
        }
        if len(hypotheses) != len(self.search_hypotheses):
            raise ValueError("SearchHypothesis IDs must be unique")
        if any(
            not set(hypothesis.claim_ids) <= set(claims)
            for hypothesis in hypotheses.values()
        ):
            raise ValueError(
                "search hypotheses must reference existing image claims"
            )
        tasks = {task.task_id: task for task in self.tasks}
        if len(tasks) != len(self.tasks):
            raise ValueError("ResearchTask IDs must be unique")
        if any(
            not set(task.claim_ids) <= set(claims)
            or (
                task.hypothesis_id is not None
                and task.hypothesis_id not in hypotheses
            )
            for task in tasks.values()
        ):
            raise ValueError(
                "research tasks must reference existing claims and hypotheses"
            )
        if any(
            not set(claim.task_ids) <= set(tasks)
            for claim in claims.values()
        ):
            raise ValueError("image claim task_ids must reference existing tasks")
        if any(
            hypothesis.task_id is not None
            and (
                hypothesis.task_id not in tasks
                or tasks[hypothesis.task_id].hypothesis_id
                != hypothesis.hypothesis_id
                or set(tasks[hypothesis.task_id].claim_ids)
                != set(hypothesis.claim_ids)
            )
            for hypothesis in hypotheses.values()
        ):
            raise ValueError(
                "search hypothesis tasks must preserve hypothesis and claim ownership"
            )
        assessment_ids = {
            assessment.assessment_id for assessment in self.claim_assessments
        }
        if len(assessment_ids) != len(self.claim_assessments):
            raise ValueError("ClaimAssessment IDs must be unique")
        evidence = {item.evidence_id for item in self.evidence}
        if any(
            assessment.claim_id not in claims
            or not set(assessment.evidence_ids) <= evidence
            for assessment in self.claim_assessments
        ):
            raise ValueError(
                "claim assessments must reference existing claims and Evidence"
            )
        discrepancy_ids = {
            item.discrepancy_id for item in self.material_discrepancies
        }
        if len(discrepancy_ids) != len(self.material_discrepancies):
            raise ValueError("MaterialDiscrepancy IDs must be unique")
        if any(
            not set(item.affected_claim_ids) <= set(claims)
            or not set(item.visual_anchor_fact_ids) <= facts
            or any(
                fact_by_id[fact_id].origin.type not in {"input_image", "ocr"}
                for fact_id in item.visual_anchor_fact_ids
            )
            or not set(item.evidence_ids) <= evidence
            for item in self.material_discrepancies
        ):
            raise ValueError(
                "material discrepancies must reference claims, VisualFacts, and Evidence"
            )
        visual_questions = {
            item.visual_question_id for item in self.visual_reinspections
        }
        if len(visual_questions) != len(self.visual_reinspections):
            raise ValueError("visual reinspection IDs must be unique")
        if any(
            item.task_id not in tasks
            or item.fact_id not in facts
            or not set(item.request.anchor_fact_ids) <= facts
            or not set(item.request.grounding_evidence_ids) <= evidence
            for item in self.visual_reinspections
        ):
            raise ValueError(
                "visual reinspections must reference tasks, facts, and Evidence"
            )
        decision_ids = {
            item.decision_id for item in self.discrepancy_decisions
        }
        if len(decision_ids) != len(self.discrepancy_decisions):
            raise ValueError("DiscrepancyDecision IDs must be unique")
        if any(
            not set(item.reviewed_evidence_ids) <= evidence
            or not set(item.accepted_assessment_ids) <= assessment_ids
            or (
                item.accepted_discrepancy_id is not None
                and item.accepted_discrepancy_id not in discrepancy_ids
            )
            or not set(item.accepted_hypothesis_ids) <= set(hypotheses)
            or not set(item.retired_hypothesis_ids) <= set(hypotheses)
            or (
                item.accepted_visual_question_id is not None
                and item.accepted_visual_question_id not in visual_questions
            )
            for item in self.discrepancy_decisions
        ):
            raise ValueError(
                "discrepancy decisions must reference accepted state records"
            )
        return self


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
