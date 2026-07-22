from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ifv_training.io import canonical_json, load_json
from ifv_training.rewards import (
    REWARD_LEDGER_SCHEMA_VERSION,
    SEMANTIC_REWARD_SCHEMA_VERSION,
    compose_reward_ledger,
    export_framework_reward,
    load_reward_profile,
    validate_reward_ledger,
    validate_semantic_reward_artifact,
)


def _semantic_artifact() -> dict:
    core = {
        "schema_version": SEMANTIC_REWARD_SCHEMA_VERSION,
        "case_id": "case-reward-1",
        "source_trace": {
            "sha256": "a" * 64,
            "decision_policy_version": "discrepancy-first-v4",
        },
        "reward_input": {
            "schema_version": "ifv-semantic-reward-input-v1",
            "sha256": "b" * 64,
            "image": {"image_sha256": "c" * 64, "available_to_judge": True},
        },
        "rollout": {
            "episode_id": "case-reward-1",
            "policy_step_ids": ["case-reward-1:judgment:1"],
            "terminal_policy_step_id": "case-reward-1:judgment:1",
        },
        "judge": {
            "provider": "gemini",
            "model": "frozen-judge",
            "prompt_versions": ["ifv-semantic-blind-v1", "ifv-semantic-aware-counterfactual-v1"],
            "calls": [
                {
                    "prompt_version": "ifv-semantic-blind-v1",
                    "request_sha256": "d" * 64,
                    "response_sha256": "e" * 64,
                    "interaction_id": "judge-1",
                    "usage": {"input_tokens": 200, "output_tokens": 30, "thought_tokens": 0},
                },
                {
                    "prompt_version": "ifv-semantic-aware-counterfactual-v1",
                    "request_sha256": "f" * 64,
                    "response_sha256": "1" * 64,
                    "interaction_id": "judge-2",
                    "usage": {"input_tokens": 220, "output_tokens": 40, "thought_tokens": 0},
                },
            ],
        },
        "blind_judgment": {
            "predicted_verdict": "fake",
            "claim_reviews": [{"claim_id": "claim-1", "label": "refuted"}],
        },
        "aware_counterfactual_judgment": {
            "original_verdict_supported": True,
            "swapped_verdict_rejected": True,
            "dropout_applicable": True,
            "dropout_verdict_supported": False,
        },
        "metrics": {
            "verdict_blind_agreement": 1.0,
            "claim_label_agreement": 1.0,
            "claim_entailment": 0.9,
            "evidence_citation_fidelity": 0.95,
            "verdict_sufficiency": 0.92,
            "verdict_swap_rejection": 1.0,
            "evidence_dropout_sensitivity": 0.4,
            "dropout_applicable": True,
            "rubber_stamp_risk": 0.0,
            "invalid_judge_evidence_ids": [],
        },
        "gates": {
            "strict_trace_audit_pass": True,
            "strict_trace_audit_failures": [],
            "engineering_valid": True,
            "semantic_audit_pass": True,
        },
    }
    artifact_id = "sha256:" + hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    return {
        **core,
        "artifact_id": artifact_id,
        "created_at": "2026-07-22T00:00:00+00:00",
    }


def test_semantic_artifact_audit_rejects_tampering() -> None:
    artifact = _semantic_artifact()
    assert validate_semantic_reward_artifact(artifact)["passed"] is True
    artifact["metrics"]["claim_entailment"] = 0.1
    audit = validate_semantic_reward_artifact(artifact)
    assert audit["passed"] is False
    assert "artifact_id does not match semantic artifact content" in audit["errors"]


def test_reward_ledger_preserves_dimensions_and_masks_fatal_rollouts() -> None:
    artifact = _semantic_artifact()
    ledger = compose_reward_ledger(
        artifact,
        deterministic={
            "classification_correct": True,
            "episode_id": "rollout-7",
            "step_ids": ["step-1", "step-2"],
        },
    )
    assert ledger["schema_version"] == REWARD_LEDGER_SCHEMA_VERSION
    assert ledger["gates"]["trainable"] is True
    assert ledger["gates"]["eligible_for_positive_buffer"] is True
    assert ledger["scalar_reward"] is not None
    assert ledger["teacher_usage"]["call_count"] == 2
    assert validate_reward_ledger(ledger)["passed"] is True

    masked = compose_reward_ledger(
        artifact,
        deterministic={"fatal_engineering_error": True},
    )
    assert masked["fatal_mask"]["masked"] is True
    assert masked["scalar_reward"] is None
    assert validate_reward_ledger(masked)["passed"] is True


def test_missing_post_rollout_correctness_never_enters_positive_buffer() -> None:
    ledger = compose_reward_ledger(_semantic_artifact())
    assert ledger["gates"]["has_policy_steps"] is True
    assert ledger["gates"]["trainable"] is True
    assert ledger["gates"]["eligible_for_positive_buffer"] is False


def test_rllm_and_verl_exports_use_terminal_reward_and_respect_mask() -> None:
    ledger = compose_reward_ledger(
        _semantic_artifact(),
        deterministic={"step_ids": ["plan", "react", "judgment"]},
    )
    rllm = export_framework_reward(ledger, framework="rllm")
    assert rllm["assignment"] == "terminal_step"
    assert rllm["step_rewards"][0]["reward"] == 0.0
    assert rllm["step_rewards"][-1]["reward"] == ledger["scalar_reward"]

    verl = export_framework_reward(ledger, framework="verl")
    assert verl["reward_assignment"]["mode"] == "terminal_token"
    assert verl["skip_update"] is False


def test_profile_file_is_versioned_and_loadable() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_reward_profile(root / "configs" / "rl" / "semantic-reward-v1.json")
    assert profile["profile_id"] == "ifv-semantic-balanced-v1"
    assert profile["weights"]["classification_correct"] == pytest.approx(0.20)


def test_v2_profile_penalizes_claim_label_disagreement() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_reward_profile(root / "configs" / "rl" / "semantic-reward-v2.json")
    assert profile["profile_id"] == "ifv-semantic-balanced-v2"
    assert profile["weights"]["claim_label_agreement"] == pytest.approx(0.10)

    artifact = _semantic_artifact()
    artifact["metrics"]["claim_label_agreement"] = 0.0
    core = {key: value for key, value in artifact.items() if key not in {"artifact_id", "created_at"}}
    artifact["artifact_id"] = "sha256:" + hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    ledger = compose_reward_ledger(artifact, profile=profile)
    assert ledger["components"]["claim_label_agreement"] == 0.0
    assert ledger["scalar_reward"] < 0.9
