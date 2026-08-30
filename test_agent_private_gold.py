from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.eval.agent_private_gold import (
    agent_candidate_answer,
    build_agent_private_gold_candidate,
)
from src.eval.private_gold_judge_contract import (
    PRIVATE_GOLD_JUDGE_PROMPT,
    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
)
from src.eval.evaluator_private_gold import private_gold_index
from src.orchestrator.investigation_models import (
    DiscrepancyJudgmentOutput,
    DiscrepancyVerdictBasis,
    FactCheckReport,
)
from src.orchestrator.pipeline import Orchestrator


def _trace() -> dict[str, Any]:
    return {
        "image_id": "case-article",
        "image_path": "/tmp/case-article.jpg",
        "verdict": "fake",
        "termination": "success",
        "verdict_basis": {
            "decision_mode": "evidence_determined",
            "verdict_target": "The image says A won the final.",
            "claim_ids": ["claim-1"],
            "finding_ids": ["finding-1"],
            "evidence_ids": ["evidence-1"],
            "unresolved_gaps": ["The exact venue was not checked."],
        },
        "judgment": {
            "verdict": "fake",
            "overall_assessment": "Official results name B rather than A.",
            "fact_check_report": {
                "headline": "The displayed winner does not match the official result",
                "claim_under_review": "The image says A won the final.",
                "verdict_summary": "Fake: the official result names B.",
                "key_findings": ["The result page identifies B as the winner."],
                "evidence_summary": "The selected official result directly contradicts the displayed winner.",
                "remaining_uncertainties": [
                    "The venue is not material to the winner mismatch."
                ],
            },
            "evidence_citations": [
                {
                    "evidence_id": "evidence-1",
                    "source_url": "https://example.org/final",
                    "source_family": "example.org",
                    "evidence_kind": "web_span",
                    "relation_stance": "contradicts",
                    "excerpt": "B won the final.",
                }
            ],
        },
        "state": {
            "investigation_state": {
                "target_facts": [
                    {
                        "claim_id": "claim-1",
                        "statement": "A won the final.",
                        "status": "refuted",
                        "anchor_fact_ids": ["anchor-1"],
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding-1",
                        "fact_ids": ["fact-1"],
                        "evidence_ids": ["evidence-1"],
                        "stance": "refute",
                        "summary": "The official result names B.",
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "successful_call": True,
                        "tool_name": "visit",
                        "evidence_kind": "web_span",
                        "source_url": "https://example.org/final",
                        "source_family": "example.org",
                        "source_class": "official",
                        "exact_text": "B won the final.",
                        "stance": "refute",
                        "quality": "strong",
                        "directness": "direct",
                        "claim_binding": "source_assertion",
                        "relation_scope": "same_relation",
                        "relation_stance": "contradicts",
                    },
                    {
                        "evidence_id": "evidence-failed",
                        "successful_call": False,
                        "tool_name": "visit",
                        "evidence_kind": "web_span",
                        "source_url": "https://example.org/failed",
                        "source_family": "example.org",
                        "exact_text": "This must not be shown to the judge.",
                    },
                ],
            }
        },
    }


def test_agent_private_gold_projection_uses_actual_successful_trace_evidence() -> None:
    candidate = build_agent_private_gold_candidate(_trace())

    assert candidate["recorded_verdict"] == "fake"
    assert candidate["fact_check_report"]["headline"].startswith("The displayed")
    assert [item["evidence_id"] for item in candidate["selected_evidence"]] == [
        "evidence-1"
    ]
    assert [
        item["evidence_id"] for item in candidate["successful_evidence"]
    ] == ["evidence-1"]
    assert candidate["runtime_evidence_citations"][0]["source_url"].endswith(
        "/final"
    )
    assert agent_candidate_answer(candidate) == {
        "core_fact": "The image says A won the final.",
        "verdict": "fake",
        "reason": "Fake: the official result names B.\n\n"
        "The selected official result directly contradicts the displayed winner.\n\n"
        "The result page identifies B as the winner.\n\n"
        "The official result names B.\n\n"
        "Official results name B rather than A.",
    }


def test_agent_candidate_answer_uses_terminal_assessment_before_internal_target() -> None:
    trace = _trace()
    trace["judgment"].pop("fact_check_report")
    trace["verdict_basis"]["verdict_target"] = "A person gave a speech."
    trace["judgment"]["overall_assessment"] = (
        "The speech is by Vikas Lakhera at LBSNAA, not Vinod Sharma."
    )

    answer = agent_candidate_answer(build_agent_private_gold_candidate(trace))

    assert answer["core_fact"] == (
        "The speech is by Vikas Lakhera at LBSNAA, not Vinod Sharma."
    )


def test_agent_and_direct_audits_share_one_judge_contract() -> None:
    from scripts.audit_direct_qa_baseline import PRIVATE_GOLD_JUDGE_PROMPT as direct_prompt
    from scripts.audit_agent_private_gold import PRIVATE_GOLD_JUDGE_PROMPT as agent_prompt

    assert direct_prompt == PRIVATE_GOLD_JUDGE_PROMPT
    assert agent_prompt == PRIVATE_GOLD_JUDGE_PROMPT
    assert "reason_quality" in PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA["properties"]


def test_private_gold_index_uses_archive_identity_and_drops_ambiguous_aliases() -> None:
    indexed = private_gold_index(
        [
            {
                "archive_source_version_id": "archive-0130",
                "candidate_id": "reused-candidate",
            },
            {
                "archive_source_version_id": "archive-0131",
                "candidate_id": "reused-candidate",
            },
        ]
    )

    assert indexed["archive-0130"]["archive_source_version_id"] == "archive-0130"
    assert indexed["archive-0131"]["archive_source_version_id"] == "archive-0131"
    assert "reused-candidate" not in indexed


def test_agent_audit_selects_sidecar_from_manifest_split() -> None:
    from scripts.audit_agent_private_gold import _default_sidecar_for_manifest

    assert _default_sidecar_for_manifest(Path("test-manifest.jsonl")).name == (
        "test-private-gold.jsonl"
    )
    assert _default_sidecar_for_manifest(Path("train-manifest.jsonl")).name == (
        "train-private-gold.jsonl"
    )


def test_agent_report_sidecar_is_an_in_memory_overlay() -> None:
    from scripts.audit_agent_private_gold import _overlay_report_sidecar

    trace = _trace()
    trace["judgment"].pop("fact_check_report")
    original = trace["judgment"].copy()
    projected, status = _overlay_report_sidecar(
        trace,
        {
            "case_id": "case-article",
            "report": {
                "headline": "Backfilled report",
                "claim_under_review": "The image says A won the final.",
                "verdict_summary": "Fake: official results name B.",
                "key_findings": ["The official result names B."],
                "evidence_summary": "The selected result contradicts A winning.",
                "remaining_uncertainties": [],
            },
            "evidence_citations": [{"evidence_id": "evidence-1"}],
        },
    )

    assert status == "applied"
    assert "fact_check_report" not in trace["judgment"]
    assert trace["judgment"] == original
    assert projected["judgment"]["fact_check_report"]["headline"] == (
        "Backfilled report"
    )
    assert projected["judgment"]["evidence_citations"][0]["evidence_id"] == (
        "evidence-1"
    )


def test_unified_judgment_requires_reader_facing_report() -> None:
    basis = DiscrepancyVerdictBasis(
        decision_mode="evidence_determined",
        verdict_target="The image says A won the final.",
    )
    with pytest.raises(ValueError, match="fact_check_report"):
        DiscrepancyJudgmentOutput(
            verdict="fake",
            confidence=0.8,
            overall_assessment="Official results name B.",
        )

    complete_report = DiscrepancyJudgmentOutput(
        verdict="fake",
        confidence=0.8,
        overall_assessment="Official results name B.",
        fact_check_report=FactCheckReport(
            headline="Official result contradicts displayed winner",
            claim_under_review="The image says A won the final.",
            verdict_summary="Fake: official results name B.",
            key_findings=["The official result names B."],
            evidence_summary="The selected result directly contradicts A winning.",
        ),
    )
    assert Orchestrator._validate_discrepancy_judgment(
        complete_report,
        compiled_verdict="fake",
        basis=basis,
    ) == (True, "")
