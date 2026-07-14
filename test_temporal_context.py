from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.orchestrator.context import ContextRenderer
from src.orchestrator.ledger import (
    build_verification_case,
    compile_runtime_ledgers,
    evidence_goal_for_case,
)
from src.orchestrator.stage_runner import StageStep
from src.orchestrator.state import (
    ClaimRecord,
    CoverageAudit,
    EvidenceItem,
    EvidenceRecord,
    InvestigationQuestion,
    PerceptionReport,
    SourceRecord,
    VerificationLedgers,
    VerificationPlan,
    VerificationResult,
)
from src.workflow import VerificationWorkflow, WorkflowConfig
from scripts.audit_real_trace import TraceReport, _audit_external_claims


CLAIM = "The event happened."


def _case(
    tmp_path: Path,
    claim_observed_at: str | None = None,
):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"temporal-context-fixture")
    return build_verification_case(
        str(image_path),
        user_claim=CLAIM,
        claim_observed_at=claim_observed_at,
    )


def _plan() -> VerificationPlan:
    return VerificationPlan(
        questions=[
            InvestigationQuestion(
                question_id="q0",
                question="Did the event happen?",
                claim_text=CLAIM,
                why="The event date remains unresolved.",
                priority=1,
            )
        ],
        image_intent="Report an event.",
    )


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-02",
        "2024-01-02T03:04Z",
        "2024-01-02T03:04:05",
        "2024-01-02T03:04:05.123+08:00",
    ],
)
def test_claim_observed_at_accepts_iso_date_or_datetime(
    tmp_path: Path,
    value: str,
) -> None:
    assert _case(tmp_path, value).claim_observed_at == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "2024/01/02",
        "2024-01-02 03:04:05",
        "2024-02-30",
        " 2024-01-02",
    ],
)
def test_claim_observed_at_rejects_non_iso_values(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(ValidationError, match="claim_observed_at"):
        _case(tmp_path, value)


def test_evidence_goal_is_stable_and_preserves_current_time_semantics(
    tmp_path: Path,
) -> None:
    current_case = _case(tmp_path)
    assert evidence_goal_for_case("  current claim  ", current_case) == (
        "  current claim  "
    )

    observed_case = _case(tmp_path, "2024-01-02")
    assert evidence_goal_for_case(CLAIM, observed_case) == (
        "The event happened.\n"
        "As-of constraint: evaluate this claim as of 2024-01-02. "
        "An event first occurring after that time cannot support or refute the claim as "
        "it stood then. A source published later may still provide evidence about the "
        "earlier state when its passage explicitly anchors that earlier time."
    )


def _browse_step(
    goal: str,
    temporal_alignment: str = "before_or_at_cutoff",
) -> StageStep:
    excerpt = "An official record says the event happened."
    return StageStep(
        round=1,
        stage_name="verification",
        action_type="tool_call",
        tool_name="visit",
        tool_args={"__question_id": "q0"},
        tool_result=json.dumps(
            {
                "status": "success",
                "selected_url": "https://source.example/report",
                "goal": goal,
                "evidence": excerpt,
                "summary": excerpt,
                "relevance": "high",
                "stance": "support",
                "artifact_sha256": "a" * 64,
                "evidence_span": {"start": 0, "end": len(excerpt)},
                "retrieved_at": "2024-01-02T12:00:00Z",
                "injection_flags": [],
                "directness": "direct",
                "temporal_alignment": temporal_alignment,
                "evidence_eligible": True,
            }
        ),
        metadata={"tool_success": True, "function_call_id": "call-visit-1"},
    )


def test_runtime_ledger_requires_case_specific_as_of_browse_goal(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "2024-01-02")
    plan = _plan()
    result = VerificationResult(
        evidence=[
            EvidenceItem(
                function_call_id="call-visit-1",
                source="https://source.example/report",
                summary="An official record says the event happened.",
                raw_excerpt="An official record says the event happened.",
                direction="supports",
                quality="strong",
                tool_used="visit",
                related_question="q0",
            )
        ]
    )

    old_goal = compile_runtime_ledgers(case, plan, result, [_browse_step(CLAIM)])
    expected_goal = evidence_goal_for_case(CLAIM, case)
    as_of_goal = compile_runtime_ledgers(
        case,
        plan,
        result,
        [_browse_step(expected_goal)],
    )

    assert old_goal.evidence == []
    assert len(as_of_goal.evidence) == 1

    later_event = compile_runtime_ledgers(
        case,
        plan,
        result,
        [_browse_step(expected_goal, "after_cutoff")],
    )
    unknown_time = compile_runtime_ledgers(
        case,
        plan,
        result,
        [_browse_step(expected_goal, "unknown")],
    )
    assert later_event.evidence == []
    assert unknown_time.evidence == []


def test_strict_auditor_uses_case_as_of_goal_and_temporal_alignment(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "2024-01-02")
    expected_goal = evidence_goal_for_case(CLAIM, case)
    source = SourceRecord(
        source_id="source-temporal",
        canonical_url="https://source.example/report",
        hostname="source.example",
        registered_domain="source.example",
        source_family="domain:source.example",
        artifact_sha256="a" * 64,
        retrieved_at="2024-01-02T12:00:00Z",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-temporal",
        claim_id="claim-q0",
        source_id=source.source_id,
        function_call_id="call-visit-1",
        tool_name="visit",
        evidence_kind="web_span",
        exact_text="An official record says the event happened.",
        span_start=0,
        span_end=43,
        artifact_sha256=source.artifact_sha256,
        retrieved_at=source.retrieved_at,
        stance="support",
        quality="moderate",
    )
    claim = ClaimRecord(
        claim_id="claim-q0",
        text=CLAIM,
        question_id="q0",
        claim_scope="external_fact",
        status="supported",
    )

    def audit(alignment: str) -> TraceReport:
        step = _browse_step(expected_goal, alignment)
        report = TraceReport(path="temporal-fixture.json")
        _audit_external_claims(
            [claim.model_dump(mode="json")],
            [source.model_dump(mode="json")],
            [evidence.model_dump(mode="json")],
            [
                {
                    "action_type": step.action_type,
                    "tool_name": step.tool_name,
                    "tool_result": step.tool_result,
                    "metadata": step.metadata,
                }
            ],
            report,
            case,
        )
        return report

    assert audit("before_or_at_cutoff").issues == []
    assert [item.code for item in audit("after_cutoff").issues] == [
        "EXTERNAL_FACT_MISSING_DIRECT_WEB_EVIDENCE"
    ]


def test_stage_contexts_state_time_semantics_and_replanning_ledger(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "2024-01-02")
    plan = _plan()
    perception = PerceptionReport(scene_description="A public event.")
    source = SourceRecord(
        source_id="source-1",
        canonical_url="https://source.example/private/full/path",
        hostname="source.example",
        registered_domain="source.example",
        source_family="domain:source.example",
        source_class="news",
        artifact_sha256="b" * 64,
        retrieved_at="2024-01-02T12:00:00Z",
    )
    ledgers = VerificationLedgers(
        claims=[
            ClaimRecord(
                claim_id="claim-runtime-1",
                text=CLAIM,
                question_id="q0",
                status="open",
                unresolved_distinction="Whether the report existed by the cutoff.",
            )
        ],
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-1",
                claim_id="claim-runtime-1",
                source_id=source.source_id,
                function_call_id="call-1",
                tool_name="visit",
                evidence_kind="web_span",
                exact_text="proof",
                span_start=0,
                span_end=5,
                artifact_sha256=source.artifact_sha256,
                retrieved_at=source.retrieved_at,
                stance="refute",
                quality="strong",
            )
        ],
    )
    audit = CoverageAudit(unresolved_priority_questions=["q0"])

    planning = ContextRenderer.render_for_planning(perception, case)
    verification = ContextRenderer.render_for_verification(perception, plan, case)
    replanning = ContextRenderer.render_for_replanning(
        perception,
        plan,
        audit,
        VerificationResult(),
        ledgers=ledgers,
        verification_case=case,
    )

    for context in (planning, verification, replanning):
        assert "evaluate as of 2024-01-02" in context
        assert "event first occurring later" in context
        assert "later source may report evidence about the earlier state" in context
    assert "status=open" in replanning
    assert "unresolved_distinction=Whether the report existed by the cutoff." in replanning
    assert "stance=refute; source_class=news; source_family=domain:source.example" in replanning
    assert source.canonical_url not in replanning

    assert "Claim time semantics: current time." in ContextRenderer.render_for_planning(
        perception
    )
    assert "Claim time semantics: current time." in ContextRenderer.render_for_verification(
        perception, plan
    )
    assert "Claim time semantics: current time." in ContextRenderer.render_for_replanning(
        perception,
        plan,
        audit,
        VerificationResult(),
    )


def test_workflow_defaults_to_image_only_and_validates_batch_cases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    image_path = tmp_path / "workflow.jpg"
    image_path.write_bytes(b"workflow-temporal-fixture")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    workflow = VerificationWorkflow(WorkflowConfig(save_traces=False))
    with pytest.raises(
        RuntimeError,
        match=(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini perception"
        ),
    ):
        asyncio.run(
            workflow.run_single(
                str(image_path),
                "single-id",
            )
        )

    batch_cases: list[Any] = []

    async def capture_single(
        _path: str,
        _image_id: str = "",
        *,
        runtime_case: Any = None,
    ) -> dict[str, Any]:
        batch_cases.append(runtime_case)
        return {"verdict": "uncertain"}

    workflow.run_single = capture_single  # type: ignore[method-assign]
    asyncio.run(
        workflow.run_batch(
            ["first.jpg", "second.jpg"],
            image_ids=["first", "second"],
        )
    )
    assert batch_cases == [None, None]
    with pytest.raises(
        ValueError,
        match="runtime_cases must match image_paths length",
    ):
        asyncio.run(
            workflow.run_batch(
                ["first.jpg"],
                runtime_cases=[],
            )
        )
