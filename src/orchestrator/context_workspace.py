"""Explicit workspace and stage handoff projections for discrepancy-first v4."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.orchestrator.evidence_semantics import evidence_is_qualified
from src.orchestrator.investigation_models import ImageOnlyInvestigationState
from src.orchestrator.task_store import pending_discrepancy_evidence_ids


HANDOFF_SCHEMA_VERSION = "ifv-stage-handoff-v1"
WORKSPACE_SCHEMA_VERSION = "ifv-explicit-workspace-v1"
MODEL_WORKSPACE_PROJECTION_VERSION = "ifv-model-workspace-projection-v1"


# These stages already compile a complete, stage-owned input projection in
# image_only_prompts.py.  Re-sending the general workspace made the local Qwen
# request carry two copies of the same Claims, routes, Discoveries and Evidence.
# Keep only material that is intentionally absent from the stage projection and
# may be needed to continue a bounded investigation.
_MODEL_WORKSPACE_FIELDS_BY_STAGE: dict[str, tuple[str, ...]] = {
    "verification": (
        "protected_findings",
        "protected_evidence",
        "recalled_materials",
        "visual_reinspections",
        "open_questions",
        "budget",
    ),
    "image_only_discrepancy_decision": (
        "protected_findings",
        "protected_evidence",
        "recalled_materials",
        "visual_reinspections",
        "open_questions",
        "budget",
    ),
    # Judgment receives a runtime-compiled bounded basis with all allowed
    # Claims, anchors, Findings and Evidence.  General workspace history must
    # not override or dilute that basis.
    "image_only_discrepancy_judgment": (),
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProtectedContext(StrictModel):
    claim_ids: list[str] = Field(default_factory=list)
    assessment_ids: list[str] = Field(default_factory=list)
    discrepancy_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    visual_question_ids: list[str] = Field(default_factory=list)

    def flat_ids(self) -> set[str]:
        return {
            item
            for values in self.model_dump().values()
            for item in values
            if str(item).strip()
        }


class ExplicitWorkspace(StrictModel):
    schema_version: str = WORKSPACE_SCHEMA_VERSION
    workspace_version: str
    image_account: dict[str, Any]
    claims: list[dict[str, Any]]
    latest_assessments: list[dict[str, Any]]
    material_discrepancies: list[dict[str, Any]]
    active_hypotheses: list[dict[str, Any]]
    open_tasks: list[dict[str, Any]]
    closed_routes: list[dict[str, Any]]
    protected_findings: list[dict[str, Any]]
    protected_evidence: list[dict[str, Any]]
    recent_evidence: list[dict[str, Any]]
    recent_discoveries: list[dict[str, Any]]
    recent_failures: list[dict[str, Any]]
    visual_reinspections: list[dict[str, Any]]
    recent_actions: list[dict[str, Any]]
    recalled_materials: list[dict[str, Any]]
    attempted_routes: list[Any]
    open_questions: list[str]
    budget: dict[str, Any]
    protected_context: ProtectedContext


class StageHandoffPacket(StrictModel):
    schema_version: str = HANDOFF_SCHEMA_VERSION
    handoff_id: str
    target_stage: str
    task_objective: str
    available_tools: list[str]
    output_contract: str
    workspace: ExplicitWorkspace
    stage_input: Any
    protected_ids: list[str]
    included_protected_ids: list[str]
    missing_protected_ids: list[str]
    protected_coverage: float = Field(ge=0.0, le=1.0)
    estimated_chars: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    compaction: dict[str, Any] = Field(default_factory=dict)


class ContextBudgetResult(StrictModel):
    """Auditable result of fitting one handoff to an explicit token budget."""

    packet: StageHandoffPacket
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)
    target_tokens: int = Field(ge=1)
    removed_item_ids: list[str] = Field(default_factory=list)
    retained_protected_ids: list[str] = Field(default_factory=list)
    protected_context_overflow: bool = False
    all_protected_items_reachable: bool = True


STAGE_OBJECTIVES = {
    "image_account_planning": "建立图像当前表达的主要事实命题和可调查假设。",
    "verification": "选择一个最有价值且未重复的调查动作。",
    "image_only_discrepancy_decision": "结合新增材料更新命题、冲突、图像重检或结论方向。",
    "image_only_reflection": "检查全局调查缺口并决定继续、换路、回读、重检或结算。",
    "image_only_discrepancy_judgment": "依据完整裁决材料给出最终事实结论。",
    "image_only_evidence_decision": "判断新增材料对当前事实状态产生的影响。",
    "image_only_judgment": "依据裁决材料输出最终结论。",
    "image_only_query_concept_extraction": "从新增证据提取可用于下一步调查的概念。",
    "image_only_query_replan": "基于证据生成一个非重复的新调查查询。",
    "image_only_route_local_replan": "在路线边界自由调整查询、视觉核查或停止当前路线。",
    "image_only_planning": "建立初始可核查事实目标。",
}


def build_explicit_workspace(
    state: ImageOnlyInvestigationState,
    *,
    recent_steps: Sequence[Any] = (),
    recalled_materials: Sequence[Mapping[str, Any]] = (),
) -> ExplicitWorkspace:
    latest_assessments: dict[str, Any] = {}
    for assessment in state.claim_assessments:
        latest_assessments[assessment.claim_id] = assessment

    protected = _protected_context(state, latest_assessments)
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    finding_by_id = {item.finding_id: item for item in state.findings}

    protected_evidence = [
        evidence_by_id[evidence_id].model_dump(mode="json")
        for evidence_id in protected.evidence_ids
        if evidence_id in evidence_by_id
    ]
    protected_findings = [
        finding_by_id[finding_id].model_dump(mode="json")
        for finding_id in protected.finding_ids
        if finding_id in finding_by_id
    ]
    recent_evidence = [
        item.model_dump(mode="json")
        for item in state.evidence[-12:]
        if item.evidence_id not in set(protected.evidence_ids)
    ]

    open_tasks = [
        task.model_dump(mode="json")
        for task in state.tasks
        if task.status in {"pending", "active"}
    ]
    closed_routes = [
        {
            "task_id": task.task_id,
            "claim_ids": task.claim_ids,
            "hypothesis_id": task.hypothesis_id,
            "status": task.status,
            "attempt_count": task.attempt_count,
            "finding_ids": task.finding_ids,
        }
        for task in state.tasks
        if task.status not in {"pending", "active"}
    ]

    actions = []
    for step in recent_steps[-8:]:
        actions.append(
            {
                "stage": getattr(step, "stage_name", ""),
                "action_type": getattr(step, "action_type", ""),
                "tool_name": getattr(step, "tool_name", ""),
                "tool_args": getattr(step, "tool_args", {}),
                "context_request_id": getattr(step, "metadata", {}).get(
                    "context_request_id",
                    "",
                ),
                "tool_result_artifact": getattr(step, "metadata", {}).get(
                    "tool_result_artifact"
                ),
            }
        )

    open_questions = []
    for assessment in latest_assessments.values():
        if assessment.assessment in {"insufficient", "conflicted"} and assessment.remaining_gap:
            open_questions.append(assessment.remaining_gap)
    open_questions.extend(
        hypothesis.expected_information
        for hypothesis in state.search_hypotheses
        if hypothesis.status in {"open", "active"}
    )
    open_questions.extend(
        item.request.question
        for item in state.visual_reinspections
        if item.status in {"pending", "running"}
    )

    attempted_routes: list[Any] = []
    for route in state.attempted_routes[-32:]:
        try:
            attempted_routes.append(json.loads(route))
        except (TypeError, ValueError):
            attempted_routes.append(str(route))

    image_account = {
        "summary": state.image_account_summary,
        "visible_facts": [
            item.model_dump(mode="json")
            for item in state.facts
            if item.origin.type in {"input_image", "ocr"}
        ],
        "retrieval_anchors": [
            item.model_dump(mode="json") for item in state.retrieval_anchors
        ],
        "latest_visual_reinspection": (
            state.visual_reinspections[-1].model_dump(mode="json")
            if state.visual_reinspections
            else None
        ),
    }
    fingerprint = _fingerprint(
        {
            "action_count": state.action_count,
            "claims": [item.model_dump(mode="json") for item in state.image_claims],
            "assessments": [
                item.model_dump(mode="json") for item in latest_assessments.values()
            ],
            "evidence_ids": [item.evidence_id for item in state.evidence],
            "task_status": {item.task_id: item.status for item in state.tasks},
            "stop_reason": state.stop_reason,
        }
    )
    return ExplicitWorkspace(
        workspace_version=f"ws-{state.action_count:02d}-{fingerprint[:10]}",
        image_account=image_account,
        claims=[item.model_dump(mode="json") for item in state.image_claims],
        latest_assessments=[
            item.model_dump(mode="json") for item in latest_assessments.values()
        ],
        material_discrepancies=[
            item.model_dump(mode="json") for item in state.material_discrepancies
        ],
        active_hypotheses=[
            item.model_dump(mode="json")
            for item in state.search_hypotheses
            if item.status in {"open", "active"}
        ],
        open_tasks=open_tasks,
        closed_routes=closed_routes,
        protected_findings=protected_findings,
        protected_evidence=protected_evidence,
        recent_evidence=recent_evidence,
        recent_discoveries=[
            item.model_dump(mode="json") for item in state.discoveries[-16:]
        ],
        recent_failures=[
            item.model_dump(mode="json") for item in state.failures[-16:]
        ],
        visual_reinspections=[
            item.model_dump(mode="json") for item in state.visual_reinspections
        ],
        recent_actions=actions,
        recalled_materials=[dict(item) for item in recalled_materials[-4:]],
        attempted_routes=attempted_routes,
        open_questions=list(dict.fromkeys(item for item in open_questions if item)),
        budget={
            "actions_used": state.action_count,
            "actions_remaining": max(0, 24 - state.action_count),
            "no_substantive_gain_streak": state.no_substantive_gain_streak,
            "latest_progress": (
                state.progress_events[-1].model_dump(mode="json")
                if state.progress_events
                else None
            ),
            "stop_reason": state.stop_reason,
            "proposed_verdict": state.proposed_verdict,
            "pending_archive_read_ids": list(state.pending_archive_read_ids),
            "pending_discrepancy_evidence_ids": pending_discrepancy_evidence_ids(
                state
            ),
        },
        protected_context=protected,
    )


def build_stage_handoff(
    state: ImageOnlyInvestigationState,
    *,
    target_stage: str,
    stage_input: Any,
    available_tools: Iterable[str] = (),
    output_contract: str = "",
    recent_steps: Sequence[Any] = (),
    recalled_materials: Sequence[Mapping[str, Any]] = (),
) -> StageHandoffPacket:
    workspace = build_explicit_workspace(
        state,
        recent_steps=recent_steps,
        recalled_materials=recalled_materials,
    )
    protected_ids = sorted(workspace.protected_context.flat_ids())
    payload = {
        "target_stage": target_stage,
        "stage_input": stage_input,
        "workspace": workspace.model_dump(mode="json"),
        "available_tools": sorted(set(str(item) for item in available_tools)),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    included_ids = sorted(
        item for item in protected_ids if item and item in serialized
    )
    missing_ids = sorted(set(protected_ids) - set(included_ids))
    handoff_id = "handoff-" + _fingerprint(payload)[:16]
    return StageHandoffPacket(
        handoff_id=handoff_id,
        target_stage=target_stage,
        task_objective=STAGE_OBJECTIVES.get(
            target_stage,
            "完成当前调查阶段并输出指定结构。",
        ),
        available_tools=sorted(set(str(item) for item in available_tools)),
        output_contract=output_contract,
        workspace=workspace,
        stage_input=stage_input,
        protected_ids=protected_ids,
        included_protected_ids=included_ids,
        missing_protected_ids=missing_ids,
        protected_coverage=(
            len(included_ids) / len(protected_ids) if protected_ids else 1.0
        ),
        estimated_chars=len(serialized),
        estimated_tokens=max(1, (len(serialized) + 3) // 4),
    )


def render_stage_handoff(packet: StageHandoffPacket) -> str:
    return json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2)


def render_stage_request(packet: StageHandoffPacket) -> str:
    """Render the model input without nesting/duplicating stage-specific fields."""

    stage_input: Any = packet.stage_input
    if isinstance(stage_input, str):
        try:
            stage_input = json.loads(stage_input)
        except (TypeError, ValueError):
            stage_input = {"stage_input": stage_input}
    if not isinstance(stage_input, Mapping):
        stage_input = {"stage_input": stage_input}
    runtime_handoff = {
        "schema_version": packet.schema_version,
        "handoff_id": packet.handoff_id,
        "target_stage": packet.target_stage,
        "task_objective": packet.task_objective,
        "available_tools": packet.available_tools,
        "output_contract": packet.output_contract,
        "protected_ids": packet.protected_ids,
        "protected_coverage": packet.protected_coverage,
        "workspace_version": packet.workspace.workspace_version,
        "compaction": packet.compaction,
    }
    if packet.target_stage in _MODEL_WORKSPACE_FIELDS_BY_STAGE:
        fields = _MODEL_WORKSPACE_FIELDS_BY_STAGE[packet.target_stage]
        workspace = packet.workspace.model_dump(mode="json")
        projection = {field: workspace[field] for field in fields}
        runtime_handoff["workspace_projection"] = {
            "schema_version": MODEL_WORKSPACE_PROJECTION_VERSION,
            "mode": "stage_minimal",
            "included_fields": list(fields),
            "full_workspace_archived": True,
        }
        if projection:
            runtime_handoff["workspace"] = projection
    elif packet.target_stage != "image_account_planning":
        runtime_handoff["workspace"] = packet.workspace.model_dump(mode="json")
    payload = {
        **dict(stage_input),
        "runtime_handoff": runtime_handoff,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def fit_stage_handoff_to_budget(
    packet: StageHandoffPacket,
    *,
    target_tokens: int | None = None,
) -> ContextBudgetResult:
    """Remove only archived background rows while retaining protected state.

    This is a structural budgeter, not a factual summarizer.  It never decides
    whether text supports or refutes a claim.  Every removed row remains in the
    immutable runtime archive and the returned report makes the loss explicit.
    """

    resolved_target = max(
        1024,
        int(
            target_tokens
            if target_tokens is not None
            else os.getenv("IFV_CONTEXT_SOFT_TOKEN_BUDGET", "112000")
        ),
    )
    candidate = packet.model_copy(deep=True)
    before_tokens = _estimated_tokens(candidate)
    removed: list[str] = []

    # Ordered from lowest-value/reconstructable history to recent working state.
    # Protected rows live in separate fields and are never candidates here.
    trim_fields = (
        "recent_discoveries",
        "recent_failures",
        "closed_routes",
        "recent_actions",
        "attempted_routes",
        "recent_evidence",
    )
    for field_name in trim_fields:
        values = list(getattr(candidate.workspace, field_name))
        while values and _estimated_tokens(candidate) > resolved_target:
            removed_item = values.pop(0)
            removed.append(_workspace_item_id(field_name, removed_item))
            setattr(candidate.workspace, field_name, list(values))

    after_tokens = _estimated_tokens(candidate)
    protected_ids = sorted(candidate.workspace.protected_context.flat_ids())
    serialized = json.dumps(
        candidate.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    included = sorted(item for item in protected_ids if item in serialized)
    missing = sorted(set(protected_ids) - set(included))
    protected_overflow = after_tokens > resolved_target
    compaction = {
        "schema_version": "ifv-context-compaction-v1",
        "reason": "token_budget" if removed else "within_budget",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "target_tokens": resolved_target,
        "removed_item_ids": removed,
        "protected_source_ids": protected_ids,
        "protected_context_overflow": protected_overflow,
        "validation": {
            "all_protected_items_reachable": not missing,
            "missing_protected_ids": missing,
        },
    }
    candidate.protected_ids = protected_ids
    candidate.included_protected_ids = included
    candidate.missing_protected_ids = missing
    candidate.protected_coverage = (
        len(included) / len(protected_ids) if protected_ids else 1.0
    )
    candidate.estimated_chars = len(serialized)
    candidate.estimated_tokens = after_tokens
    candidate.compaction = compaction
    return ContextBudgetResult(
        packet=candidate,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        target_tokens=resolved_target,
        removed_item_ids=removed,
        retained_protected_ids=included,
        protected_context_overflow=protected_overflow,
        all_protected_items_reachable=not missing,
    )


def _protected_context(
    state: ImageOnlyInvestigationState,
    latest_assessments: Mapping[str, Any],
) -> ProtectedContext:
    claim_ids = [item.claim_id for item in state.image_claims if item.salience == "high"]
    assessment_ids = [item.assessment_id for item in latest_assessments.values()]
    discrepancy_ids = [
        item.discrepancy_id
        for item in state.material_discrepancies
        if item.materiality == "decisive" or item.status == "conflicted"
    ]
    evidence_ids: set[str] = set()
    finding_ids: set[str] = set()
    for assessment in latest_assessments.values():
        evidence_ids.update(assessment.evidence_ids)
        finding_ids.update(getattr(assessment, "finding_ids", []) or [])
    for discrepancy in state.material_discrepancies:
        if discrepancy.discrepancy_id in discrepancy_ids:
            evidence_ids.update(discrepancy.evidence_ids)
    for finding in state.findings:
        if finding.quality == "decisive" or finding.finding_id in finding_ids:
            finding_ids.add(finding.finding_id)
            evidence_ids.update(finding.evidence_ids)
    for evidence in state.evidence:
        if evidence_is_qualified(evidence) and evidence.stance in {"support", "refute"}:
            if evidence.evidence_id in evidence_ids:
                continue
            if any(fact_id in {claim.fact_id for claim in state.image_claims} for fact_id in evidence.fact_ids):
                evidence_ids.add(evidence.evidence_id)
    task_ids = [
        item.task_id for item in state.tasks if item.status in {"pending", "active"}
    ]
    hypothesis_ids = [
        item.hypothesis_id
        for item in state.search_hypotheses
        if item.status in {"open", "active"}
    ]
    visual_question_ids = [
        item.visual_question_id
        for item in state.visual_reinspections
        if item.status in {"pending", "running"}
    ]
    return ProtectedContext(
        claim_ids=sorted(set(claim_ids)),
        assessment_ids=sorted(set(assessment_ids)),
        discrepancy_ids=sorted(set(discrepancy_ids)),
        evidence_ids=sorted(evidence_ids),
        finding_ids=sorted(finding_ids),
        task_ids=sorted(set(task_ids)),
        hypothesis_ids=sorted(set(hypothesis_ids)),
        visual_question_ids=sorted(set(visual_question_ids)),
    )


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _estimated_tokens(value: Any) -> int:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return max(1, (len(serialized) + 3) // 4)


def _workspace_item_id(field_name: str, item: Any) -> str:
    if isinstance(item, Mapping):
        for key in (
            "evidence_id",
            "discovery_id",
            "failure_id",
            "task_id",
            "context_request_id",
            "function_call_id",
        ):
            value = str(item.get(key, "")).strip()
            if value:
                return f"{field_name}:{value}"
    return f"{field_name}:{_fingerprint(item)[:16]}"
