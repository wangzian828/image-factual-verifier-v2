from __future__ import annotations

from typing import Any

import pytest

from src.eval.agent_private_gold import (
    build_agent_private_gold_candidate,
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
