from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from src.orchestrator.llm_backend import LLMResponse
from src.trajectory.sft_eligibility import (
    SFT_ELIGIBILITY_SYSTEM_PROMPT,
    SFTEligibilityJudge,
    SFTEligibilityJudgment,
    build_sft_eligibility_artifact,
    build_sft_eligibility_input,
    build_sft_target,
    sft_eligibility_metrics,
    sft_eligibility_passes,
)


def _trace(*, verdict: str = "fake") -> dict[str, Any]:
    return {
        "image_id": "case-1--teacher-r000",
        "verdict": verdict,
        "termination": "success",
        "verdict_basis": {
            "claim_ids": ["claim-1"],
            "discrepancy_ids": ["discrepancy-1"],
            "evidence_ids": ["evidence-1"],
            "verdict_target": "The displayed winner is A, but the official winner is B.",
            "unresolved_gaps": [],
        },
        "state": {
            "image_id": "case-1--teacher-r000",
            "runtime_case": {
                "case_id": "case-1",
                "image_sha256": "a" * 64,
            },
            "investigation_state": {
                "image_claims": [
                    {
                        "claim_id": "claim-1",
                        "statement": "A won the 2026 final.",
                        "salience": "high",
                        "anchor_fact_ids": ["anchor-1"],
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding-1",
                        "task_id": "task-1",
                        "fact_ids": ["fact-1"],
                        "evidence_ids": ["evidence-1"],
                        "stance": "refute",
                        "summary": "The official result names B.",
                    }
                ],
                "material_discrepancies": [
                    {
                        "discrepancy_id": "discrepancy-1",
                        "statement": "The official winner was B, not A.",
                        "affected_claim_ids": ["claim-1"],
                        "evidence_ids": ["evidence-1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "source_url": "https://example.org/final",
                        "source_family": "example.org",
                        "exact_text": "B won the 2026 final.",
                        "directness": "direct",
                        "claim_binding": "source_assertion",
                        "relation_scope": "different_scope",
                        "relation_stance": "contradicts",
                        "stance": "refute",
                        "claim_ids": [],
                        "successful_call": True,
                    },
                    {
                        "evidence_id": "evidence-2",
                        "source_url": "https://example.org/context",
                        "exact_text": "The final took place in Berlin.",
                        "directness": "direct",
                        "claim_binding": "source_assertion",
                        "relation_scope": "location",
                        "relation_stance": "background",
                        "stance": "neutral",
                        "successful_call": True,
                    },
                ],
            },
        },
    }


def _gold() -> dict[str, Any]:
    return {
        "schema_version": "ifv-route-aware-human-review-candidate-v1",
        "candidate_id": "case-1",
        "factual_status": "refuted",
        "target_claim": "The image claims A won the 2026 final.",
        "claim_atom": {
            "subject": "2026 final",
            "event_or_context": "2026 final",
            "relation_slot": "winner",
            "depicted_value": "A",
        },
        "decisive_visual_atom": "The scoreboard displays A as the winner.",
        "visible_scene_facts": ["A is displayed as the winner."],
        "evidence": {
            "binding": {
                "target_claim": "A won the 2026 final.",
                "source_evidence_span": "B won the 2026 final.",
                "source_url": "https://example.org/final",
                "match": "same_event",
            },
            "used_exact_spans": [
                {"exact_span": "B won the 2026 final.", "role": "event_fact"}
            ],
        },
    }


def _judgment(**updates: Any) -> SFTEligibilityJudgment:
    payload: Dict[str, Any] = {
        "fact_alignment": "same_image_fact",
        "decision_support": "supports_fake",
        "decisive_evidence_ids": ["evidence-1"],
        "supporting_evidence_ids": [],
        "overclaiming": "none",
        "boundary_assessment": "respected",
        "confidence": 0.72,
        "explanation": "The evidence directly establishes the incorrect winner.",
    }
    payload.update(updates)
    return SFTEligibilityJudgment.model_validate(payload)


def _packet() -> dict[str, Any]:
    packet = build_sft_eligibility_input(_trace(), _gold())
    packet["image"]["available_to_judge"] = True
    return packet


class _MockBackend:
    provider = "mock"
    model_name = "frozen-mock"

    def __init__(self, judgment: SFTEligibilityJudgment | None = None) -> None:
        self.judgment = judgment or _judgment()
        self.messages: List[List[Dict[str, Any]]] = []

    async def get_response(
        self, messages: List[Dict[str, Any]], **_: Any
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(
            text=self.judgment.model_dump_json(),
            prompt_tokens=120,
            completion_tokens=40,
        )


def test_target_adapter_uses_one_generic_image_fact_shape() -> None:
    target = build_sft_target(_gold())

    assert target["schema_version"] == "ifv-sft-target-v2"
    assert target["expected_verdict"] == "fake"
    assert target["image_fact"]["statement"].startswith("The image claims")
    assert target["image_fact"]["visible_anchors"]
    assert target["reference_facts"][0]["evidence_text"]


def test_target_adapter_accepts_web_chain_without_claim_atom() -> None:
    row = {
        "candidate_id": "web-1",
        "factual_status": "refuted",
        "target_claim": "The image shows a real event.",
        "event_identity": "the claimed event",
        "relation_identity": "the image depicts the event",
        "evidence": {
            "binding": {
                "chain": [
                    {
                        "binding": {
                            "target_claim": "The image shows a real event.",
                            "source_url": "https://example.org/check",
                        },
                        "exact_span": "The image was digitally altered.",
                        "role": "contradiction",
                    }
                ]
            }
        },
    }

    target = build_sft_target(row)

    assert target["expected_verdict"] == "fake"
    assert target["image_fact"]["relation"] == (
        "the image depicts the event"
    )
    assert target["reference_facts"][0]["role"] == "contradiction"
    assert target["reference_facts"][0]["evidence_text"] == (
        "The image was digitally altered."
    )


def test_packet_exposes_all_valid_evidence_and_basis_is_only_a_flag() -> None:
    packet = _packet()
    evidence = {
        row["evidence_id"]: row for row in packet["candidate"]["evidence"]
    }

    assert set(evidence) == {"evidence-1", "evidence-2"}
    assert evidence["evidence-1"]["basis_selected"] is True
    assert evidence["evidence-2"]["basis_selected"] is False
    assert packet["candidate"]["claims"][0]["claim_id"] == "claim-1"


def test_compatible_subfact_can_pass_without_claim_relation_matching() -> None:
    packet = _packet()
    judgment = _judgment(
        fact_alignment="compatible_subfact",
        decision_support="supports_fake",
        decisive_evidence_ids=["evidence-1"],
    )
    metrics = sft_eligibility_metrics(packet, judgment)

    assert metrics["verdict_correct"] is True
    assert metrics["fatal_errors"] == []
    assert sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=False,
        engineering_valid=True,
    )


def test_nonfatal_audit_warning_does_not_veto_sft() -> None:
    packet = _packet()
    artifact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=_judgment(),
        judge_audit={},
        strict_trace_audit_pass=False,
        strict_trace_audit_failures=[
            {
                "code": "BASIS_OMITS_VALID_EVIDENCE",
                "category": "hard",
                "message": "A valid Evidence row was not selected in basis.",
            }
        ],
    )

    assert artifact["gates"]["sft_eligibility_pass"] is True
    assert artifact["gates"]["fatal_audit_errors"] == []
    assert artifact["gates"]["audit_warnings"]


def test_protocol_audit_error_still_blocks_sft() -> None:
    packet = _packet()
    artifact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=_judgment(),
        judge_audit={},
        strict_trace_audit_pass=False,
        strict_trace_audit_failures=[
            {
                "code": "PROTOCOL_ERROR",
                "category": "protocol",
                "message": "Unrecoverable protocol boundary.",
            }
        ],
    )

    assert artifact["gates"]["sft_eligibility_pass"] is False
    assert artifact["gates"]["fatal_audit_errors"]


def test_invalid_selected_evidence_id_blocks_sft() -> None:
    packet = _packet()
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(decisive_evidence_ids=["missing-evidence"]),
    )

    assert metrics["invalid_judge_evidence_ids"] == ["missing-evidence"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_no_decisive_evidence_blocks_sft() -> None:
    packet = _packet()
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(
            decision_support="supporting_only",
            decisive_evidence_ids=[],
            supporting_evidence_ids=["evidence-2"],
        ),
    )

    assert "no_decisive_evidence" in metrics["fatal_errors"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_judge_is_one_post_rollout_call_and_does_not_request_human_review() -> None:
    async def run() -> None:
        packet = _packet()
        backend = _MockBackend()
        judgment, audit = await SFTEligibilityJudge(backend).judge(packet)

        assert judgment.fact_alignment == "same_image_fact"
        assert len(backend.messages) == 1
        request = json.dumps(backend.messages[0], ensure_ascii=False)
        assert "decision_paths" not in request
        assert "human_review" not in request

        artifact = build_sft_eligibility_artifact(
            trace=_trace(),
            trace_sha256="b" * 64,
            packet=packet,
            judgment=judgment,
            judge_audit=audit,
            strict_trace_audit_pass=True,
        )
        assert artifact["gates"]["sft_eligibility_pass"] is True
        assert "human_review_required" not in artifact["gates"]
        assert len(artifact["judge"]["calls"]) == 1

    asyncio.run(run())


def test_prompt_is_image_fact_based_not_claim_path_based() -> None:
    prompt = " ".join(SFT_ELIGIBILITY_SYSTEM_PROMPT.split())
    prompt_lower = prompt.lower()

    assert "factual content expressed by the supplied image" in prompt
    assert "Do not require the teacher to reproduce the target wording" in prompt
    assert "do not create human-review work" in prompt_lower
