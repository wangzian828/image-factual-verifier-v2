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
        "target_claim": "The image claims A won the 2026 final.",
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


def _optional_path() -> dict[str, Any]:
    return {
        "path_id": "scene_location_mismatch",
        "target_claim": "The image claims the 2026 final happened in Paris.",
        "decisive_fact": {
            "statement": "The depicted venue is Paris.",
            "visual_anchors": ["Paris venue signage"],
        },
        "claim_atom": {
            "subject": "2026 final",
            "event": "2026 final",
            "slot": "location",
            "depicted_value": "Paris",
        },
        "key_error": {
            "type": "CD1",
            "slot": "location",
            "depicted_value": "Paris",
            "verified_value": "Berlin",
            "summary": "The depicted venue is wrong.",
        },
        "evidence_target": {
            "binding": {
                "subject": "2026 final",
                "event": "2026 final",
                "slot": "location",
                "depicted_value": "Paris",
                "match": "same_relation",
            },
            "required_directness": "direct",
            "required_stance": "contradicts",
            "statement": "Berlin",
        },
        "boundary": {
            "directly_decides": "The official venue directly contradicts Paris.",
            "does_not_prove": ["It does not prove the winner."],
        },
        "certifying_evidence": [
            {
                "evidence_id": "gold-evidence-2",
                "source_id": "source-2",
                "url": "https://example.org/location",
                "snapshot_sha256": "c" * 64,
                "exact_span": "The 2026 final was held in Berlin.",
                "stance": "contradicts",
                "directness": "direct",
                "binding": {
                    "subject": "2026 final",
                    "event": "2026 final",
                    "slot": "location",
                    "depicted_value": "Paris",
                    "match": "same_relation",
                },
            }
        ],
        "review_origin": "human_added",
    }


def _location_trace() -> dict[str, Any]:
    trace = _trace()
    trace["verdict_basis"].update(
        {
            "verdict_target": (
                "The image depicts the final in Paris, but the official venue was Berlin."
            ),
        }
    )
    investigation = trace["state"]["investigation_state"]
    investigation["image_claims"][0]["statement"] = (
        "The 2026 final happened in Paris."
    )
    investigation["material_discrepancies"][0]["statement"] = (
        "The official venue was Berlin, not Paris."
    )
    investigation["evidence"][0].update(
        {
            "source_url": "https://example.org/location",
            "exact_text": "The 2026 final was held in Berlin.",
        }
    )
    return trace


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
    assert packet["private_target"]["decision_paths"][0]["path_id"] == "primary"
    metrics = sft_eligibility_metrics(packet, _judgment())
    assert metrics["path_match_status"] == "matched_registered_path"
    assert metrics["matched_path_id"] == "primary"
    assert metrics["verdict_correct"] is True
    assert metrics["claim_relation_aligned"] is True
    assert metrics["error_slot_and_value_aligned"] is True
    assert metrics["eligible_evidence_ids"] == ["evidence-1"]
    assert sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_registered_optional_decision_path_can_pass_sft_gate() -> None:
    gold = _gold()
    gold["acceptable_decision_paths"] = [_optional_path()]
    packet = build_sft_eligibility_input(_location_trace(), gold)
    judgment = _judgment(
        matched_path_id="scene_location_mismatch",
        explanation="The candidate follows the registered venue path.",
    )

    metrics = sft_eligibility_metrics(packet, judgment)

    assert metrics["matched_path_id"] == "scene_location_mismatch"
    assert metrics["claim_relation_aligned"] is True
    assert sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_plausible_unregistered_path_requires_human_review_without_pass() -> None:
    packet = build_sft_eligibility_input(_location_trace(), _gold())
    packet["image"]["available_to_judge"] = True
    judgment = _judgment(
        path_match_status="plausible_new_path_candidate",
        matched_path_id="",
        new_path_candidate={
            "proposed_path_type": "location",
            "visible_image_anchor": "Paris venue signage",
            "teacher_evidence_anchor": "The exact span says Berlin.",
            "why_not_existing_path": "Registered path covers winner, not venue.",
            "confidence": 0.85,
            "needs_human_review": True,
        },
    )

    metrics = sft_eligibility_metrics(packet, judgment)

    assert metrics["new_reasonable_path_candidate"] is True
    assert metrics["new_path_candidate_screen_pass"] is True
    assert metrics["new_path_candidate_filter_reasons"] == []
    assert metrics["claim_relation_aligned"] is False
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )
    artifact = build_sft_eligibility_artifact(
        trace=_location_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=judgment,
        judge_audit={},
        strict_trace_audit_pass=True,
    )
    assert artifact["gates"]["human_review_required"] is True
    assert artifact["gates"]["sft_eligibility_pass"] is False


def test_low_confidence_new_path_candidate_is_not_sent_to_human_review() -> None:
    packet = build_sft_eligibility_input(_location_trace(), _gold())
    packet["image"]["available_to_judge"] = True
    judgment = _judgment(
        path_match_status="plausible_new_path_candidate",
        matched_path_id="",
        new_path_candidate={
            "proposed_path_type": "location",
            "visible_image_anchor": "Paris venue signage",
            "teacher_evidence_anchor": "The exact span says Berlin.",
            "why_not_existing_path": "Registered path covers winner, not venue.",
            "confidence": 0.6,
            "needs_human_review": True,
        },
    )

    artifact = build_sft_eligibility_artifact(
        trace=_location_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=judgment,
        judge_audit={},
        strict_trace_audit_pass=True,
    )

    assert artifact["metrics"]["new_reasonable_path_candidate"] is True
    assert artifact["metrics"]["new_path_candidate_screen_pass"] is False
    assert (
        "confidence_below_threshold"
        in artifact["metrics"]["new_path_candidate_filter_reasons"]
    )
    assert artifact["gates"]["human_review_required"] is False


def test_new_path_candidate_requires_image_and_valid_direct_evidence() -> None:
    trace = _location_trace()
    trace["state"]["investigation_state"]["evidence"][0][
        "relation_scope"
    ] = "different_instance"
    packet = build_sft_eligibility_input(trace, _gold())
    judgment = _judgment(
        path_match_status="plausible_new_path_candidate",
        matched_path_id="",
        new_path_candidate={
            "proposed_path_type": "location",
            "visible_image_anchor": "Paris venue signage",
            "teacher_evidence_anchor": "The exact span says Berlin.",
            "why_not_existing_path": "Registered path covers winner, not venue.",
            "confidence": 0.9,
            "needs_human_review": True,
        },
    )

    metrics = sft_eligibility_metrics(packet, judgment)

    assert metrics["new_path_candidate_screen_pass"] is False
    assert set(metrics["new_path_candidate_filter_reasons"]) == {
        "image_unavailable_to_judge",
        "no_direct_same_relation_evidence",
    }


def test_duplicate_optional_decision_path_ids_fail_closed() -> None:
    gold = _gold()
    gold["acceptable_decision_paths"] = [_optional_path(), _optional_path()]

    try:
        build_sft_eligibility_input(_trace(), gold)
    except ValueError as exc:
        assert "path_id must be unique" in str(exc)
    else:
        raise AssertionError("duplicate optional decision paths must be rejected")


def test_unknown_registered_path_id_fails_closed() -> None:
    packet = build_sft_eligibility_input(_trace(), _gold())
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(matched_path_id="not_registered"),
    )

    assert metrics["invalid_judge_path_id"] == "not_registered"
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_inconsistent_registered_path_output_fails_closed() -> None:
    packet = build_sft_eligibility_input(_trace(), _gold())
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(
            new_path_candidate={
                "proposed_path_type": "location",
                "visible_image_anchor": "Paris venue signage",
                "teacher_evidence_anchor": "The exact span says Berlin.",
                "why_not_existing_path": "It is unrelated to the primary path.",
                "confidence": 0.9,
                "needs_human_review": True,
            }
        ),
    )

    assert metrics["judge_path_output_consistent"] is False
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_new_path_status_without_payload_is_screened_out() -> None:
    packet = build_sft_eligibility_input(_location_trace(), _gold())
    packet["image"]["available_to_judge"] = True
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(
            path_match_status="plausible_new_path_candidate",
            matched_path_id="",
            new_path_candidate=None,
        ),
    )

    assert metrics["new_reasonable_path_candidate"] is True
    assert metrics["new_path_candidate_screen_pass"] is False
    assert "candidate_payload_missing" in metrics[
        "new_path_candidate_filter_reasons"
    ]


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


def test_supported_target_accepts_equivalent_verified_value() -> None:
    trace = _trace()
    trace["verdict"] = "real"
    trace["verdict_basis"]["discrepancy_ids"] = []
    trace["state"]["investigation_state"]["evidence"][0].update(
        {
            "exact_text": "A won the 2026 final.",
            "relation_stance": "supports",
            "stance": "support",
        }
    )
    gold = _gold()
    gold["label"] = "supported"
    gold["key_error"] = None
    gold["evidence_target"]["required_stance"] = "supports"
    judgment = _judgment(
        key_error_slot_alignment="not_applicable",
        verified_value_alignment="equivalent",
        evidence_reviews=[
            {
                "evidence_id": "evidence-1",
                "relation_match": "same_relation",
                "directness": "direct",
                "stance": "supports",
                "target_value_stated": True,
                "explanation": "The exact span states A as the winner.",
            }
        ],
    )

    metrics = sft_eligibility_metrics(
        build_sft_eligibility_input(trace, gold),
        judgment,
    )

    assert metrics["error_slot_and_value_aligned"] is True
    assert sft_eligibility_passes(
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


def test_private_judge_stance_is_relative_to_candidate_claim() -> None:
    prompt = " ".join(SFT_ELIGIBILITY_SYSTEM_PROMPT.split())
    assert "Every Evidence stance is relative to the candidate Claim" in prompt
    assert "verified alternative contradicts a candidate" in prompt
