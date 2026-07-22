from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from src.orchestrator.llm_backend import LLMResponse
from src.trajectory.sft_eligibility import (
    SFTEligibilityJudge,
    SFTEligibilityJudgment,
    build_sft_eligibility_artifact,
    build_sft_eligibility_input,
    sft_eligibility_metrics,
    sft_eligibility_passes,
)


def _trace() -> dict[str, Any]:
    return {
        "image_id": "case-1--teacher-r000",
        "verdict": "fake",
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
                "material_discrepancies": [
                    {
                        "discrepancy_id": "discrepancy-1",
                        "statement": "The official winner was B, not A.",
                        "affected_claim_ids": ["claim-1"],
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
                        "relation_scope": "same_relation",
                        "relation_stance": "contradicts",
                        "stance": "refute",
                    }
                ],
            },
        },
    }


def _gold() -> dict[str, Any]:
    return {
        "schema_version": "ifv-scoring-gold-v1",
        "case_id": "case-1",
        "label": "refuted",
        "decisive_fact": {
            "statement": "A won the 2026 final.",
            "visual_anchors": ["winner A"],
        },
        "claim_atom": {
            "subject": "2026 final",
            "event": "2026 final",
            "slot": "winner",
            "depicted_value": "A",
        },
        "key_error": {
            "slot": "winner",
            "depicted_value": "A",
            "verified_value": "B",
            "summary": "The displayed winner is wrong.",
        },
        "evidence_target": {
            "binding": {
                "subject": "2026 final",
                "event": "2026 final",
                "slot": "winner",
                "depicted_value": "A",
                "match": "same_relation",
            },
            "required_directness": "direct",
            "required_stance": "contradicts",
            "statement": "B",
        },
        "boundary": {
            "directly_decides": "A won the 2026 final.",
            "does_not_prove": ["A has never won another event."],
        },
    }


def _judgment(**updates: Any) -> SFTEligibilityJudgment:
    payload: Dict[str, Any] = {
        "selected_claim_ids": ["claim-1"],
        "claim_relation_match": "same_relation",
        "subject_event_slot_aligned": True,
        "depicted_value_alignment": "equivalent",
        "key_error_slot_alignment": "equivalent",
        "verified_value_alignment": "equivalent",
        "spurious_error": False,
        "evidence_reviews": [
            {
                "evidence_id": "evidence-1",
                "relation_match": "same_relation",
                "directness": "direct",
                "stance": "contradicts",
                "target_value_stated": True,
                "explanation": "The exact span names B as winner.",
            }
        ],
        "boundary_respected": True,
        "boundary_violations": [],
        "explanation": "The candidate identifies the intended winner substitution.",
    }
    payload.update(updates)
    return SFTEligibilityJudgment.model_validate(payload)


class _MockBackend:
    provider = "mock"
    model_name = "frozen-mock"

    def __init__(self) -> None:
        self.messages: List[List[Dict[str, Any]]] = []

    async def get_response(
        self, messages: List[Dict[str, Any]], **_: Any
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(
            text=_judgment().model_dump_json(),
            prompt_tokens=120,
            completion_tokens=40,
        )


def test_structured_gate_requires_same_relation_value_and_direct_evidence() -> None:
    packet = build_sft_eligibility_input(_trace(), _gold())
    metrics = sft_eligibility_metrics(packet, _judgment())
    assert metrics["verdict_correct"] is True
    assert metrics["claim_relation_aligned"] is True
    assert metrics["error_slot_and_value_aligned"] is True
    assert metrics["eligible_evidence_ids"] == ["evidence-1"]
    assert sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_model_review_cannot_override_runtime_evidence_metadata() -> None:
    trace = _trace()
    trace["state"]["investigation_state"]["evidence"][0][
        "relation_scope"
    ] = "different_instance"
    packet = build_sft_eligibility_input(trace, _gold())
    metrics = sft_eligibility_metrics(packet, _judgment())
    assert metrics["direct_same_relation_evidence_present"] is False
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_private_structured_judge_is_one_post_rollout_call() -> None:
    async def run() -> None:
        packet = build_sft_eligibility_input(_trace(), _gold())
        backend = _MockBackend()
        judgment, audit = await SFTEligibilityJudge(backend).judge(packet)
        assert judgment.claim_relation_match == "same_relation"
        assert len(backend.messages) == 1
        request = json.dumps(backend.messages[0], ensure_ascii=False)
        assert "certifying_evidence" not in request
        assert "snapshot_path" not in request
        artifact = build_sft_eligibility_artifact(
            trace=_trace(),
            trace_sha256="b" * 64,
            packet=packet,
            judgment=judgment,
            judge_audit=audit,
            strict_trace_audit_pass=True,
        )
        assert artifact["gates"]["sft_eligibility_pass"] is True
        assert len(artifact["judge"]["calls"]) == 1

    asyncio.run(run())
