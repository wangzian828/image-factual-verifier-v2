from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from src.orchestrator.llm_backend import LLMResponse
from src.trajectory.semantic_reward import (
    SemanticRewardCache,
    SemanticRewardJudge,
    TrajectorySemanticJudgment,
    build_semantic_reward_artifact,
    build_semantic_reward_input,
    semantic_audit_passes,
    semantic_metrics,
    sha256_json,
)


def _trace() -> dict[str, Any]:
    return {
        "image_id": "case-1--group--r000",
        "decision_policy_version": "unified-react-v1",
        "verdict": "fake",
        "termination": "success",
        "verdict_basis": {"evidence_ids": ["evidence-1"]},
        "state": {
            "image_id": "case-1--group--r000",
            "runtime_case": {"case_id": "case-1", "image_sha256": ""},
            "all_steps": [
                {
                    "stage": "image_only_discrepancy_investigation",
                    "action_type": "tool_call",
                    "tool_name": "text_search",
                    "tool_args": {"query": "actual object used at event"},
                    "tool_result": json.dumps(
                        {"status": "success", "results": [{"title": "Official"}]}
                    ),
                    "metadata": {
                        "interaction_id": "turn-1",
                        "reasoning": "hidden policy thinking",
                        "policy_input": {"safe": "input"},
                        "policy_action": {"name": "text_search"},
                    },
                },
                {
                    "stage": "image_only_discrepancy_judgment",
                    "action_type": "output",
                    "metadata": {
                        "interaction_id": "judgment-1",
                        "policy_input": {"safe": "input"},
                        "policy_action": {"verdict": "fake"},
                    },
                },
            ],
            "investigation_state": {
                "progress_events": [
                    {
                        "action_count": 1,
                        "gain": "evidence_gain",
                        "source_ids": ["evidence-1"],
                    }
                ],
                "target_facts": [
                    {
                        "claim_id": "claim-1",
                        "statement": "The pictured relation is factual.",
                        "salience": "high",
                        "status": "refuted",
                        "anchor_fact_ids": ["fact-1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "task_id": "task-1",
                        "fact_ids": ["fact-1"],
                        "evidence_kind": "web_span",
                        "source_url": "https://example.org",
                        "exact_text": "The official record identifies a different object.",
                        "stance": "refute",
                        "quality": "strong",
                        "directness": "direct",
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding-1",
                        "evidence_ids": ["evidence-1"],
                        "stance": "refute",
                        "summary": "Policy-authored summary must stay hidden.",
                    }
                ],
            },
        },
    }


def _judgment(**updates: Any) -> TrajectorySemanticJudgment:
    payload: Dict[str, Any] = {
        "claim_reviews": [
            {
                "claim_id": "claim-1",
                "label": "refuted",
                "confidence": 0.95,
                "evidence_ids": ["evidence-1"],
                "entailment_score": 0.9,
                "citation_fidelity": 0.98,
                "explanation": "The supplied source contradicts the image claim.",
            }
        ],
        "predicted_verdict": "fake",
        "confidence": 0.94,
        "evidence_sufficient": True,
        "evidence_quality": 0.9,
        "investigation_progress": 0.85,
        "search_direction": 0.88,
        "evidence_use": 0.92,
        "belief_revision": 0.8,
        "overall_process_quality": 0.87,
        "evidence_ids": ["evidence-1"],
        "useful_turn_ids": ["turn-001"],
        "problematic_turn_ids": [],
        "explanation": "A productive search yielded decisive official Evidence.",
    }
    payload.update(updates)
    return TrajectorySemanticJudgment.model_validate(payload)


class _MockJudgeBackend:
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
            prompt_tokens=140,
            completion_tokens=40,
        )


def test_packet_is_episode_aware_and_excludes_hidden_answers() -> None:
    packet = build_semantic_reward_input(_trace())
    assert packet["case_id"] == "case-1"
    assert packet["rollout"]["episode_id"] == "case-1--group--r000"
    rendered = json.dumps(packet, ensure_ascii=False)
    assert "hidden policy thinking" not in rendered
    assert packet["investigation_turns"][0]["turn_id"] == "turn-001"
    assert "case-1--group--r000:turn:turn-1" not in rendered
    assert packet["investigation_turns"][0]["state_delta"]["gain"] == "evidence_gain"


def test_frozen_judge_uses_one_blind_trajectory_call() -> None:
    async def run() -> None:
        packet = build_semantic_reward_input(_trace())
        backend = _MockJudgeBackend()
        judgment, audit = await SemanticRewardJudge(backend).judge(packet)
        assert judgment.overall_process_quality == 0.87
        assert len(backend.messages) == 1
        request = json.dumps(backend.messages[0], ensure_ascii=False)
        assert '"recorded_verdict"' not in request
        assert "Policy-authored summary" not in request
        assert '"verdict": "fake"' not in request
        artifact = build_semantic_reward_artifact(
            trace=_trace(),
            trace_sha256="b" * 64,
            packet=packet,
            judgment=judgment,
            judge_audit=audit,
            strict_trace_audit_pass=True,
        )
        assert len(artifact["judge"]["calls"]) == 1
        assert artifact["metrics"]["overall_process_quality"] == 0.87
        assert artifact["gates"]["semantic_audit_pass"] is True

    asyncio.run(run())


def test_unknown_judge_citations_fail_semantic_gate() -> None:
    packet = build_semantic_reward_input(_trace())
    judgment = _judgment(evidence_ids=["invented"])
    metrics = semantic_metrics(packet, judgment)
    assert metrics["invalid_judge_evidence_ids"] == ["invented"]
    assert not semantic_audit_passes(
        metrics, strict_trace_audit_pass=True, engineering_valid=True
    )


def test_semantic_reward_cache_is_versioned_and_content_addressed(
    tmp_path: Path,
) -> None:
    cache = SemanticRewardCache(tmp_path)
    key = cache.key(
        trace_sha256="a" * 64,
        reward_input_sha256="b" * 64,
        provider="gemini",
        model="judge-model",
    )
    artifact = {"schema_version": "ifv-semantic-reward-v2", "case_id": "case-1"}
    cache.store(key, artifact)
    assert cache.load(key) == artifact
    assert key == sha256_json(
        {
            "trace_sha256": "a" * 64,
            "reward_input_sha256": "b" * 64,
            "provider": "gemini",
            "model": "judge-model",
            "generation_version": "minimal-thinking-4096-v5",
            "postprocess_version": "trajectory-semantic-audit-v2",
            "prompt_versions": ["ifv-semantic-trajectory-blind-v2"],
        }
    )
