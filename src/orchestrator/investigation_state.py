"""Deterministic per-observation state for evidence-conditioned ReInspect."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import Field, model_validator

from src.orchestrator.source_provenance import classify_source
from src.orchestrator.state import (
    ClaimRecord,
    PerceptionReport,
    StrictModel,
    VerificationLedgers,
    VerificationPlan,
)
from src.orchestrator.tool_result import parse_tool_result


VISUAL_REVISIT_TOOLS = {
    "ocr_with_position",
    "crop_and_inspect",
    "crop_and_search",
    "compare_with_reference",
    "count_objects",
}


class ObservationAssessment(StrictModel):
    assessment_id: str
    function_call_id: str
    claim_id: Optional[str] = None
    question_id: str = ""
    tool_name: str
    status: Literal["success", "error"]
    observation_kind: Literal["discovery", "web_evidence", "visual", "runtime", "failure", "other"]
    evidence_ids: List[str] = Field(default_factory=list)
    discovery_ids: List[str] = Field(default_factory=list)
    source_families: List[str] = Field(default_factory=list)
    stance: Literal["support", "refute", "neutral", "none"] = "none"
    directness: Literal["direct", "indirect", "none"] = "none"
    novelty: Literal["new_claim_signal", "new_source_family", "duplicate", "none"] = "none"
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = ""


class BeliefDelta(StrictModel):
    delta_id: str
    assessment_id: str
    claim_id: Optional[str] = None
    operation: Literal["support", "refute", "split", "create", "unknown", "zero"]
    old_status: str = "open"
    new_status: str = "open"
    explanation: str


class VisualQuestion(StrictModel):
    visual_question_id: str
    claim_id: str
    source_evidence_id: Optional[str] = None
    source_discovery_id: Optional[str] = None
    target_bbox: List[float]
    expected_property: str
    recommended_tools: List[str] = Field(default_factory=list)
    status: Literal["pending", "resolved", "failed", "exhausted"] = "pending"
    resolution_call_id: Optional[str] = None
    failed_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_links(self) -> "VisualQuestion":
        if bool(self.source_evidence_id) == bool(self.source_discovery_id):
            raise ValueError("visual question requires exactly one evidence or discovery source id")
        if len(self.target_bbox) != 4:
            raise ValueError("target_bbox must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = self.target_bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("target_bbox must be normalized and non-empty")
        return self


class RegionObservation(StrictModel):
    region_observation_id: str
    visual_question_id: str
    function_call_id: str
    tool_name: str
    target_bbox: List[float]
    observed_property: str
    status: Literal["success", "error"]


class StoppingAssessment(StrictModel):
    after_function_call_id: str
    can_stop: bool
    unresolved_claim_ids: List[str] = Field(default_factory=list)
    pending_visual_question_ids: List[str] = Field(default_factory=list)
    novel_source_families: int = Field(default=0, ge=0)
    repeated_observations: int = Field(default=0, ge=0)
    remaining_high_value_actions: List[str] = Field(default_factory=list)
    reason: str


class InvestigationState(StrictModel):
    observations: List[ObservationAssessment] = Field(default_factory=list)
    belief_deltas: List[BeliefDelta] = Field(default_factory=list)
    visual_questions: List[VisualQuestion] = Field(default_factory=list)
    region_observations: List[RegionObservation] = Field(default_factory=list)
    stopping_assessments: List[StoppingAssessment] = Field(default_factory=list)


class InvestigationReducer:
    """Reduce one validated tool observation into auditable state changes."""

    @staticmethod
    def reduce(
        state: InvestigationState,
        *,
        step: Any,
        previous_ledgers: VerificationLedgers,
        current_ledgers: VerificationLedgers,
        plan: VerificationPlan,
        perception: PerceptionReport,
    ) -> Dict[str, Any]:
        call_id = _call_id(step)
        question_id = str(getattr(step, "tool_args", {}).get("__question_id", ""))
        claim_id = f"claim-{question_id}" if question_id else None
        tool_name = str(getattr(step, "tool_name", ""))
        try:
            parsed, succeeded = parse_tool_result(getattr(step, "tool_result", ""))
        except Exception as exc:
            parsed, succeeded = {"error": str(exc)}, False

        old_evidence = {item.evidence_id for item in previous_ledgers.evidence}
        old_discoveries = {item.discovery_id for item in previous_ledgers.discoveries}
        new_evidence = [
            item for item in current_ledgers.evidence
            if item.evidence_id not in old_evidence and item.function_call_id == call_id
        ]
        new_discoveries = [
            item for item in current_ledgers.discoveries
            if item.discovery_id not in old_discoveries and item.function_call_id == call_id
        ]
        sources = {item.source_id: item for item in current_ledgers.sources}
        source_families = sorted({sources[item.source_id].source_family for item in new_evidence if item.source_id in sources})
        risk_flags = sorted({flag for item in new_evidence if item.source_id in sources for flag in sources[item.source_id].risk_flags})

        old_claim = _claim(previous_ledgers.claims, claim_id)
        new_claim = _claim(current_ledgers.claims, claim_id)
        stance_values = {item.stance for item in new_evidence if item.stance != "neutral"}
        directness = "direct" if any(item.directness == "direct" for item in new_evidence) else "none"
        if not succeeded:
            kind = "failure"
            operation = "unknown"
        elif new_evidence:
            kind = "web_evidence" if any(item.evidence_kind == "web_span" for item in new_evidence) else "visual"
            operation = _operation(old_claim, new_claim, stance_values)
        elif new_discoveries:
            kind = "discovery"
            operation = "create"
        elif tool_name in VISUAL_REVISIT_TOOLS or tool_name in {"check_consistency", "analyze_visual_anomalies"}:
            kind = "visual"
            operation = "zero"
        elif tool_name == "current_time":
            kind = "runtime"
            operation = "zero"
        else:
            kind = "other"
            operation = "zero"

        prior_families = {item.source_family for item in previous_ledgers.sources}
        if source_families and any(item not in prior_families for item in source_families):
            novelty = "new_source_family"
        elif new_evidence or new_discoveries:
            novelty = "new_claim_signal"
        elif succeeded:
            novelty = "duplicate"
        else:
            novelty = "none"

        assessment_id = _id("obs", call_id)
        assessment = ObservationAssessment(
            assessment_id=assessment_id,
            function_call_id=call_id,
            claim_id=claim_id,
            question_id=question_id,
            tool_name=tool_name,
            status="success" if succeeded else "error",
            observation_kind=kind,
            evidence_ids=[item.evidence_id for item in new_evidence],
            discovery_ids=[item.discovery_id for item in new_discoveries],
            source_families=source_families,
            stance=(next(iter(stance_values)) if len(stance_values) == 1 else ("neutral" if new_evidence else "none")),
            directness=directness,
            novelty=novelty,
            risk_flags=risk_flags,
            summary=_summary(parsed, succeeded),
        )
        delta = BeliefDelta(
            delta_id=_id("delta", assessment_id),
            assessment_id=assessment_id,
            claim_id=claim_id,
            operation=operation,
            old_status=old_claim.status if old_claim else "open",
            new_status=new_claim.status if new_claim else "open",
            explanation=_delta_explanation(operation, new_claim),
        )
        _append_unique(state.observations, assessment, "assessment_id")
        _append_unique(state.belief_deltas, delta, "delta_id")

        created = InvestigationReducer._create_visual_question(
            state,
            step=step,
            new_evidence=new_evidence,
            new_discoveries=new_discoveries,
            plan=plan,
            perception=perception,
            parsed=parsed,
        )
        resolved = InvestigationReducer._resolve_visual_question(state, step, parsed, succeeded)
        stop = InvestigationReducer._stopping_assessment(state, current_ledgers, call_id)
        state.stopping_assessments.append(stop)
        return {
            "observation": assessment.model_dump(mode="json"),
            "belief_delta": delta.model_dump(mode="json"),
            "created_visual_questions": [item.model_dump(mode="json") for item in created],
            "resolved_visual_questions": [item.model_dump(mode="json") for item in resolved],
            "stopping_assessment": stop.model_dump(mode="json"),
        }

    @staticmethod
    def validate_visual_call(state: InvestigationState, tool_name: str, args: Dict[str, Any]) -> str:
        visual_question_id = str(args.get("visual_question_id", "")).strip()
        if not visual_question_id:
            return ""
        question = next((item for item in state.visual_questions if item.visual_question_id == visual_question_id), None)
        if question is None:
            return f"Unknown visual_question_id '{visual_question_id}'."
        if question.status != "pending":
            return f"Visual question '{visual_question_id}' is already {question.status}."
        if tool_name not in VISUAL_REVISIT_TOOLS:
            return f"Tool '{tool_name}' cannot resolve a visual question."
        if question.source_evidence_id and args.get("source_evidence_id") != question.source_evidence_id:
            return "source_evidence_id must match the pending visual question."
        if question.source_discovery_id and args.get("source_discovery_id") != question.source_discovery_id:
            return "source_discovery_id must match the pending visual question."
        if str(args.get("expected_property", "")).strip() != question.expected_property:
            return "expected_property must match the pending visual question."
        if tool_name != "compare_with_reference" and list(args.get("bbox") or []) != question.target_bbox:
            return "bbox must match the pending visual question target_bbox."
        return ""

    @staticmethod
    def _create_visual_question(
        state: InvestigationState,
        *,
        step: Any,
        new_evidence: Sequence[Any],
        new_discoveries: Sequence[Any],
        plan: VerificationPlan,
        perception: PerceptionReport,
        parsed: Dict[str, Any],
    ) -> List[VisualQuestion]:
        question_id = str(getattr(step, "tool_args", {}).get("__question_id", ""))
        if not question_id:
            return []
        claim_id = f"claim-{question_id}"
        plan_question = next((item for item in plan.questions if item.question_id == question_id), None)
        target = _target_bbox(plan_question, perception)
        created: List[VisualQuestion] = []

        discovery = next((item for item in new_discoveries if item.candidate_type == "reverse_image"), None)
        reference_url = str(parsed.get("reference_image_url", ""))
        if discovery is not None and reference_url:
            distinction = _id(
                "reference",
                classify_source(reference_url).canonical_url or reference_url,
            )
            existing = next(
                (
                    item
                    for item in state.visual_questions
                    if item.claim_id == claim_id
                    and item.visual_question_id == _id("vq", claim_id, distinction)
                ),
                None,
            )
            if existing is not None:
                discovery = None
        if discovery is not None and reference_url:
            visual = VisualQuestion(
                visual_question_id=_id("vq", claim_id, distinction),
                claim_id=claim_id,
                source_discovery_id=discovery.discovery_id,
                target_bbox=[0.0, 0.0, 1.0, 1.0],
                expected_property="Whether the discovered reference is the same visual and contains factual edits.",
                recommended_tools=["compare_with_reference"],
            )
            if _append_unique(state.visual_questions, visual, "visual_question_id"):
                created.append(visual)

        visual_tokens = ("image", "photo", "picture", "visual", "logo", "text", "depicted", "图片", "照片", "图像", "文字", "标志", "车型", "人物")
        claim_text = (
            " ".join([plan_question.question, plan_question.claim_text])
            if plan_question
            else ""
        ).lower()
        evidence = next((item for item in new_evidence if item.evidence_kind == "web_span"), None)
        if evidence is not None and any(token in claim_text for token in visual_tokens):
            distinction = _visual_distinction(plan_question, target, evidence)
            existing = next(
                (
                    item
                    for item in state.visual_questions
                    if item.claim_id == claim_id
                    and item.target_bbox == target
                    and item.visual_question_id == _id("vq", claim_id, distinction)
                ),
                None,
            )
            if existing is not None:
                evidence = None
        if evidence is not None and any(token in claim_text for token in visual_tokens):
            visual = VisualQuestion(
                visual_question_id=_id("vq", claim_id, distinction),
                claim_id=claim_id,
                source_evidence_id=evidence.evidence_id,
                target_bbox=target,
                expected_property=f"Whether the target image region is consistent with: {evidence.exact_text[:240]}",
                recommended_tools=["ocr_with_position", "crop_and_inspect"],
            )
            if _append_unique(state.visual_questions, visual, "visual_question_id"):
                created.append(visual)
        return created

    @staticmethod
    def _resolve_visual_question(
        state: InvestigationState,
        step: Any,
        parsed: Dict[str, Any],
        succeeded: bool,
    ) -> List[VisualQuestion]:
        args = getattr(step, "tool_args", {})
        visual_question_id = str(args.get("visual_question_id", "")).strip()
        if not visual_question_id:
            return []
        question = next((item for item in state.visual_questions if item.visual_question_id == visual_question_id), None)
        if question is None or question.status != "pending":
            return []
        if succeeded:
            question.status = "resolved"
        else:
            question.failed_attempts += 1
            question.status = "exhausted" if question.failed_attempts >= 2 else "pending"
        question.resolution_call_id = _call_id(step)
        observed = _summary(parsed, succeeded)
        state.region_observations.append(
            RegionObservation(
                region_observation_id=_id("region", visual_question_id, _call_id(step)),
                visual_question_id=visual_question_id,
                function_call_id=_call_id(step),
                tool_name=str(getattr(step, "tool_name", "")),
                target_bbox=question.target_bbox,
                observed_property=observed,
                status="success" if succeeded else "error",
            )
        )
        return [question] if question.status in {"resolved", "exhausted"} else []

    @staticmethod
    def _stopping_assessment(
        state: InvestigationState,
        ledgers: VerificationLedgers,
        call_id: str,
    ) -> StoppingAssessment:
        unresolved = [
            item.claim_id for item in ledgers.claims
            if item.criticality == "decisive" and item.status not in {"supported", "refuted"}
        ]
        pending = [item.visual_question_id for item in state.visual_questions if item.status == "pending"]
        exhausted_visual = [
            item.visual_question_id
            for item in state.visual_questions
            if item.status == "exhausted"
        ]
        repeated = sum(item.novelty == "duplicate" for item in state.observations)
        actions: List[str] = []
        if pending:
            actions.append("Resolve pending visual questions with their recommended real visual tools.")
        if exhausted_visual:
            actions.append("Record the exhausted visual revisit as typed image-region insufficiency.")
        if unresolved:
            actions.append("Collect direct evidence from a new source family for unresolved claims.")
        can_stop = not unresolved and not pending and not exhausted_visual
        return StoppingAssessment(
            after_function_call_id=call_id,
            can_stop=can_stop,
            unresolved_claim_ids=unresolved,
            pending_visual_question_ids=[*pending, *exhausted_visual],
            novel_source_families=len({item.source_family for item in ledgers.sources}),
            repeated_observations=repeated,
            remaining_high_value_actions=actions,
            reason=("All decisive claims and required visual revisits are resolved." if can_stop else "High-value investigation work remains."),
        )


def pending_visual_question_ids(state: InvestigationState) -> List[str]:
    return [item.visual_question_id for item in state.visual_questions if item.status == "pending"]


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256(chr(31).join(parts).encode('utf-8')).hexdigest()[:20]}"


def _call_id(step: Any) -> str:
    metadata = getattr(step, "metadata", {}) or {}
    return str(metadata.get("function_call_id") or f"legacy-{getattr(step, 'stage_name', 'stage')}-{getattr(step, 'round', 0)}")


def _claim(claims: Sequence[ClaimRecord], claim_id: Optional[str]) -> Optional[ClaimRecord]:
    return next((item for item in claims if item.claim_id == claim_id), None)


def _operation(old: Optional[ClaimRecord], new: Optional[ClaimRecord], stances: set[str]) -> str:
    if new and new.status == "conflicted":
        return "split"
    if new and new.status == "supported" and (old is None or old.status != "supported"):
        return "support"
    if new and new.status == "refuted" and (old is None or old.status != "refuted"):
        return "refute"
    if stances:
        return "unknown"
    return "zero"


def _delta_explanation(operation: str, claim: Optional[ClaimRecord]) -> str:
    if operation == "support":
        return "Validated direct evidence moved the claim to supported."
    if operation == "refute":
        return "Validated direct evidence moved the claim to refuted."
    if operation == "split":
        return "Validated evidence conflicts, so the claim remains split."
    if operation == "create":
        return "The observation created a discovery lead but did not change factual belief."
    if operation == "unknown":
        return "The observation is relevant but insufficient under the source-independence policy."
    return f"No validated belief change; claim remains {(claim.status if claim else 'open')}."


def _summary(parsed: Dict[str, Any], succeeded: bool) -> str:
    if not succeeded:
        return str(parsed.get("error", "Tool call failed."))[:500]
    for key in ("evidence", "answer", "summary", "details", "full_text", "notes"):
        value = str(parsed.get(key, "")).strip()
        if value:
            return value[:500]
    return json.dumps(parsed, ensure_ascii=False, default=str)[:500]


def _target_bbox(question: Any, perception: PerceptionReport) -> List[float]:
    target_tokens = set()
    if question is not None:
        text = " ".join(
            [question.question, question.claim_text, *question.related_entities]
        ).lower()
        target_tokens = set(text.split())
    for region in perception.text_regions:
        if target_tokens and any(token in region.text.lower() for token in target_tokens if len(token) >= 3):
            xs = [point[0] for point in region.bbox_quad if len(point) >= 2]
            ys = [point[1] for point in region.bbox_quad if len(point) >= 2]
            if xs and ys:
                return [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
    for entity in perception.entities:
        if len(entity.bbox) == 4 and (
            not target_tokens or any(token in entity.name.lower() for token in target_tokens if len(token) >= 3)
        ):
            return [float(item) for item in entity.bbox]
    return [0.0, 0.0, 1.0, 1.0]


def _visual_distinction(
    question: Any,
    target_bbox: Sequence[float],
    evidence: Any,
) -> str:
    if question is None:
        target = "visual_claim"
    else:
        target = " ".join(
            [question.claim_text, *question.related_entities]
        ).strip().lower()
    normalized = " ".join(target.split())[:300]
    box = ",".join(f"{float(value):.3f}" for value in target_bbox)
    return _id(
        "distinction",
        normalized or "visual_claim",
        box,
        str(getattr(evidence, "artifact_sha256", "")),
        str(getattr(evidence, "span_start", "")),
        str(getattr(evidence, "span_end", "")),
        " ".join(str(getattr(evidence, "exact_text", "")).split())[:500],
    )


def _append_unique(values: List[Any], item: Any, id_field: str) -> bool:
    item_id = getattr(item, id_field)
    if any(getattr(existing, id_field) == item_id for existing in values):
        return False
    values.append(item)
    return True
