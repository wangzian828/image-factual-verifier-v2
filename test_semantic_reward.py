from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.orchestrator.llm_backend import LLMResponse
from src.trajectory.semantic_reward import (
    AwareCounterfactualJudgment,
    BlindSemanticJudgment,
    SemanticRewardCache,
    SemanticRewardJudge,
    build_semantic_reward_artifact,
    build_semantic_reward_input,
    semantic_metrics,
    sha256_json,
)


def _trace() -> dict[str, Any]:
    claim_id = "claim-1"
    evidence_id = "evidence-1"
    finding_id = "finding-1"
    basis = {
        "policy_rule_id": "discrepancy-first-v4",
        "decision_mode": "evidence_determined",
        "verdict_target": "The pictured relation is false.",
        "claim_ids": [claim_id],
        "discrepancy_ids": ["discrepancy-1"],
        "visual_anchor_fact_ids": ["anchor-1"],
        "finding_ids": [finding_id],
        "evidence_ids": [evidence_id],
        "unresolved_gaps": [],
    }
    return {
        "image_id": "case-semantic-reward",
        "decision_policy_version": "discrepancy-first-v4",
        "verdict": "fake",
        "termination": "success",
        "verdict_basis": basis,
        "state": {
            "image_id": "case-semantic-reward",
            "runtime_case": {"image_sha256": ""},
            "investigation_state": {
                "image_claims": [
                    {
                        "claim_id": claim_id,
                        "statement": "The person used the pictured object at the event.",
                        "salience": "high",
                        "status": "refuted",
                        "anchor_fact_ids": ["anchor-1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "task_id": "task-1",
                        "fact_ids": ["fact-1"],
                        "evidence_kind": "web_span",
                        "source_url": "https://example.org/source",
                        "source_family": "example.org",
                        "source_class": "official",
                        "exact_text": "The official record identifies a different object.",
                        "artifact_sha256": "a" * 64,
                        "stance": "refute",
                        "quality": "strong",
                        "directness": "direct",
                        "claim_binding": "source_assertion",
                        "risk_flags": [],
                    }
                ],
                "findings": [
                    {
                        "finding_id": finding_id,
                        "task_id": "task-1",
                        "fact_ids": ["fact-1"],
                        "evidence_ids": [evidence_id],
                        "stance": "refute",
                    }
                ],
                "claim_assessments": [
                    {
                        "claim_id": claim_id,
                        "assessment": "refuted",
                        "evidence_ids": [evidence_id],
                        "finding_ids": [finding_id],
                        "remaining_gap": "",
                    }
                ],
                "material_discrepancies": [
                    {
                        "discrepancy_id": "discrepancy-1",
                        "statement": "The central object relation is contradicted.",
                        "affected_claim_ids": [claim_id],
                        "visual_anchor_fact_ids": ["anchor-1"],
                        "evidence_ids": [evidence_id],
                        "materiality": "decisive",
                        "status": "established",
                    }
                ],
            },
        },
    }


def _blind() -> BlindSemanticJudgment:
    return BlindSemanticJudgment(
        claim_reviews=[
            {
                "claim_id": "claim-1",
                "label": "refuted",
                "confidence": 0.95,
                "evidence_ids": ["evidence-1"],
                "entailment_score": 0.92,
                "citation_fidelity": 0.98,
                "explanation": "The exact cited passage contradicts the claim.",
            }
        ],
        predicted_verdict="fake",
        confidence=0.94,
        evidence_sufficient=True,
        explanation="The central claim is refuted.",
    )


def _aware() -> AwareCounterfactualJudgment:
    return AwareCounterfactualJudgment(
        original_verdict_supported=True,
        original_confidence=0.94,
        verdict_sufficiency=0.91,
        swapped_verdict_rejected=True,
        swapped_confidence=0.96,
        dropout_applicable=True,
        dropout_verdict_supported=False,
        dropout_confidence=0.49,
        explanation="The original verdict depends on the cited Evidence.",
    )


class _MockJudgeBackend:
    provider = "mock"
    model_name = "frozen-mock"

    def __init__(self) -> None:
        self.responses = [_blind().model_dump_json(), _aware().model_dump_json()]
        self.messages: List[List[Dict[str, Any]]] = []

    async def get_response(
        self,
        messages: List[Dict[str, Any]],
        **_: Any,
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(
            text=self.responses.pop(0),
            prompt_tokens=100,
            completion_tokens=20,
        )


def test_reward_packet_excludes_reasoning_and_blind_call_hides_policy_labels() -> None:
    trace = _trace()
    trace["state"]["all_steps"] = [
        {
            "stage": "image_account_planning",
            "action_type": "output",
            "metadata": {
                "reasoning": "private policy reasoning",
                "interaction_id": "interaction-plan",
                "policy_input": {"safe": "input"},
                "policy_action": {"safe": "action"},
            },
        }
    ]
    packet = build_semantic_reward_input(trace)
    assert "reasoning" not in json.dumps(packet)
    assert packet["rollout"]["policy_step_ids"] == [
        "case-semantic-reward:image_account_planning:interaction-plan"
    ]


def test_frozen_judge_builds_counterfactual_semantic_artifact() -> None:
    asyncio.run(_run_frozen_judge_builds_counterfactual_semantic_artifact())


async def _run_frozen_judge_builds_counterfactual_semantic_artifact() -> None:
    trace = _trace()
    packet = build_semantic_reward_input(trace)
    backend = _MockJudgeBackend()
    judge = SemanticRewardJudge(backend)
    blind, aware, judge_audit = await judge.judge(packet)

    first_request = json.dumps(backend.messages[0], ensure_ascii=False)
    assert '"recorded_verdict"' not in first_request
    assert '"recorded_status"' not in first_request
    assert '"stance"' not in first_request
    assert "not the probability that the claim is true" in first_request

    artifact = build_semantic_reward_artifact(
        trace=trace,
        trace_sha256="b" * 64,
        packet=packet,
        blind=blind,
        aware=aware,
        judge_audit=judge_audit,
        strict_trace_audit_pass=True,
    )
    assert artifact["gates"]["semantic_audit_pass"] is True
    assert artifact["metrics"]["verdict_swap_rejection"] == 1.0
    assert artifact["metrics"]["evidence_dropout_sensitivity"] == pytest.approx(0.45)
    assert artifact["judge"]["calls"][0]["usage"]["input_tokens"] == 100


def test_unknown_judge_citation_fails_semantic_gate() -> None:
    packet = build_semantic_reward_input(_trace())
    blind = _blind().model_copy(deep=True)
    blind.claim_reviews[0].evidence_ids = ["invented-evidence"]
    metrics = semantic_metrics(packet, blind, _aware())
    assert metrics["evidence_citation_fidelity"] == 0.0
    assert metrics["invalid_judge_evidence_ids"] == ["invented-evidence"]


def test_semantic_reward_cache_is_content_addressed(tmp_path: Path) -> None:
    cache = SemanticRewardCache(tmp_path)
    key = cache.key(
        trace_sha256="a" * 64,
        reward_input_sha256="b" * 64,
        provider="gemini",
        model="judge-model",
    )
    artifact = {"schema_version": "ifv-semantic-reward-v1", "case_id": "case-1"}
    cache.store(key, artifact)
    assert cache.load(key) == artifact
    assert key == sha256_json(
        {
            "trace_sha256": "a" * 64,
            "reward_input_sha256": "b" * 64,
            "provider": "gemini",
            "model": "judge-model",
            "generation_version": "minimal-thinking-4096-v1",
            "prompt_versions": [
                "ifv-semantic-blind-v2",
                "ifv-semantic-aware-counterfactual-v2",
            ],
        }
    )
