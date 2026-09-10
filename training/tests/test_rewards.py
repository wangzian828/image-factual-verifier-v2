from __future__ import annotations

import hashlib
from pathlib import Path

from ifv_training.io import canonical_json
from ifv_training.rewards import (
    REWARD_LEDGER_SCHEMA_VERSION,
    SEMANTIC_REWARD_SCHEMA_VERSION,
    build_standard_grpo_groups,
    build_ledgers_from_run_artifacts,
    compose_reward_ledger,
    export_framework_reward,
    load_reward_profile,
    validate_reward_ledger,
    validate_semantic_reward_artifact,
)


def _semantic_artifact(episode_id: str = "episode-1") -> dict:
    core = {
        "schema_version": SEMANTIC_REWARD_SCHEMA_VERSION,
        "case_id": "case-reward-1",
        "source_trace": {
            "sha256": "a" * 64,
            "decision_policy_version": "unified-react-v1",
        },
        "reward_input": {
            "schema_version": "ifv-semantic-reward-input-v2",
            "sha256": "b" * 64,
            "image": {"image_sha256": "c" * 64, "available_to_judge": True},
        },
        "rollout": {
            "episode_id": episode_id,
            "policy_step_ids": [f"{episode_id}:judgment:1"],
            "terminal_policy_step_id": f"{episode_id}:judgment:1",
        },
        "judge": {
            "provider": "gemini",
            "model": "frozen-judge",
            "prompt_versions": ["ifv-semantic-trajectory-blind-v1"],
            "calls": [
                {
                    "prompt_version": "ifv-semantic-trajectory-blind-v1",
                    "request_sha256": "d" * 64,
                    "response_sha256": "e" * 64,
                    "interaction_id": "judge-1",
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 30,
                        "thought_tokens": 0,
                    },
                }
            ],
        },
        "trajectory_judgment": {
            "predicted_verdict": "fake",
            "claim_reviews": [{"claim_id": "claim-1", "label": "refuted"}],
        },
        "metrics": {
            "verdict_blind_agreement": 1.0,
            "claim_label_agreement": 1.0,
            "claim_entailment": 0.9,
            "evidence_citation_fidelity": 0.95,
            "evidence_sufficiency": 1.0,
            "evidence_quality": 0.9,
            "investigation_progress": 0.8,
            "search_direction": 0.85,
            "evidence_use": 0.9,
            "belief_revision": 0.75,
            "overall_process_quality": 0.85,
            "invalid_judge_evidence_ids": [],
            "invalid_judge_turn_ids": [],
        },
        "gates": {
            "strict_trace_audit_pass": True,
            "strict_trace_audit_failures": [],
            "engineering_valid": True,
            "semantic_audit_pass": True,
        },
    }
    return {
        **core,
        "artifact_id": "sha256:" + hashlib.sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
        "created_at": "2026-07-22T00:00:00+00:00",
    }


def _ledger(episode_id: str, *, correct: bool, quality: float = 0.85) -> dict:
    return compose_reward_ledger(
        {
            "schema_version": "ifv-post-rollout-deterministic-v1",
            "case_id": "case-reward-1",
            "episode_id": episode_id,
            "classification_correct": correct,
            "strict_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": [f"{episode_id}:judgment:1"],
            "process_components": {
                "evidence_chain_reward": quality,
                "discrepancy_alignment_reward": quality,
                "stop_quality_reward": quality,
            },
        },
    )


def test_one_call_artifact_validates_and_teacher_usage_is_recorded() -> None:
    artifact = _semantic_artifact()
    assert validate_semantic_reward_artifact(artifact)["passed"] is True
    ledger = compose_reward_ledger(
        {
            "case_id": "case-reward-1",
            "episode_id": "episode-1",
            "classification_correct": True,
            "strict_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": ["episode-1:judgment:1"],
            "process_components": {
                "evidence_chain_reward": 0.85,
                "discrepancy_alignment_reward": 0.85,
                "stop_quality_reward": 0.85,
            },
        },
        semantic_artifact=artifact,
    )
    assert ledger["schema_version"] == REWARD_LEDGER_SCHEMA_VERSION
    assert ledger["teacher_usage"]["call_count"] == 1
    assert ledger["diagnostics"]["semantic_reward"]["role"] == "diagnostic_only"
    assert validate_reward_ledger(ledger)["passed"] is True


def test_correctness_dominates_process_quality() -> None:
    incorrect = _ledger("bad", correct=False, quality=1.0)
    weak_correct = _ledger("good", correct=True, quality=0.0)
    assert incorrect["scalar_reward"] == 0.0
    assert weak_correct["scalar_reward"] == 0.35
    assert weak_correct["scalar_reward"] > incorrect["scalar_reward"]


def test_v1_masks_audit_failures_but_v2_keeps_nonfatal_audit_reward() -> None:
    missing = compose_reward_ledger(
        {
            "case_id": "case-reward-1",
            "episode_id": "episode-1",
            "strict_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": ["episode-1:judgment:1"],
        }
    )
    assert missing["scalar_reward"] is None
    assert missing["fatal_mask"]["masked"] is True
    failed = compose_reward_ledger(
        {
            "case_id": "case-reward-1",
            "episode_id": "episode-1",
            "classification_correct": True,
            "strict_trace_audit_pass": False,
            "fatal_engineering_error": False,
            "step_ids": ["episode-1:judgment:1"],
        },
    )
    assert failed["scalar_reward"] is None
    assert failed["fatal_mask"]["masked"] is True

    relaxed_profile = {
        "schema_version": "ifv-rl-reward-profile-v1",
        "profile_id": "ifv-deterministic-process-v2",
        "correct_reward_floor": 0.35,
        "allow_nonfatal_audit_failures": True,
        "weights": {
            "evidence_chain_reward": 0.4,
            "discrepancy_alignment_reward": 0.4,
            "stop_quality_reward": 0.2,
        },
    }
    nonfatal = compose_reward_ledger(
        {
            "case_id": "case-reward-1",
            "episode_id": "episode-nonfatal",
            "classification_correct": True,
            "strict_trace_audit_pass": False,
            "hard_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": ["episode-nonfatal:judgment:1"],
        },
        profile=relaxed_profile,
    )
    assert nonfatal["scalar_reward"] == 0.35
    assert nonfatal["fatal_mask"]["masked"] is False

    hard_failure = compose_reward_ledger(
        {
            "case_id": "case-reward-1",
            "episode_id": "episode-hard",
            "classification_correct": True,
            "strict_trace_audit_pass": False,
            "hard_trace_audit_pass": False,
            "fatal_engineering_error": False,
            "step_ids": ["episode-hard:judgment:1"],
        },
        profile=relaxed_profile,
    )
    assert hard_failure["scalar_reward"] is None
    assert hard_failure["fatal_mask"]["reason"] == "hard_trace_audit_failed"


def test_invalid_semantic_diagnostic_is_rejected_but_never_masks_reward() -> None:
    artifact = _semantic_artifact()
    artifact["metrics"]["invalid_judge_evidence_ids"] = ["invented"]
    core = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "created_at"}
    }
    artifact["artifact_id"] = "sha256:" + hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    deterministic = {
        "case_id": "case-reward-1",
        "episode_id": "episode-1",
        "classification_correct": True,
        "strict_trace_audit_pass": True,
        "fatal_engineering_error": False,
        "step_ids": ["episode-1:judgment:1"],
    }
    with_diagnostic = compose_reward_ledger(
        deterministic,
        semantic_artifact=artifact,
    )
    assert with_diagnostic["scalar_reward"] == 1.0
    assert with_diagnostic["teacher_usage"]["call_count"] == 1
    assert with_diagnostic["diagnostics"]["semantic_reward"]["role"] == (
        "diagnostic_only"
    )
    artifact["artifact_id"] = "sha256:bad"
    try:
        compose_reward_ledger(deterministic, semantic_artifact=artifact)
    except ValueError as exc:
        assert "invalid semantic reward artifact" in str(exc)
    else:
        raise AssertionError("corrupt semantic diagnostic should be rejected")


def test_framework_exports_remain_standard_terminal_episode_rewards() -> None:
    ledger = _ledger("episode-1", correct=True)
    rllm = export_framework_reward(ledger, framework="rllm")
    assert rllm["assignment"] == "terminal_step"
    assert rllm["step_rewards"][-1]["reward"] == ledger["scalar_reward"]
    verl = export_framework_reward(ledger, framework="verl")
    assert verl["reward_assignment"]["mode"] == "terminal_token"


def test_standard_grpo_groups_join_by_episode_and_skip_zero_variance() -> None:
    members = [
        {
            "schema_version": "ifv-rollout-group-member-v1",
            "prompt_group_id": "pg-1",
            "case_id": "case-reward-1",
            "episode_id": f"episode-{index}",
            "rollout_index": index,
            "group_size": 4,
            "sampling_seed": 100 + index,
        }
        for index in range(4)
    ]
    varied = build_standard_grpo_groups(
        [
            _ledger("episode-0", correct=True, quality=0.9),
            _ledger("episode-1", correct=True, quality=0.5),
            _ledger("episode-2", correct=False, quality=1.0),
            _ledger("episode-3", correct=True, quality=0.7),
        ],
        members,
    )[0]
    assert varied["trainable"] is True
    assert varied["valid_member_count"] == 4
    assert varied["reward_std"] > 0

    equal = build_standard_grpo_groups(
        [_ledger(f"episode-{index}", correct=False) for index in range(4)],
        members,
    )[0]
    assert equal["trainable"] is False
    assert equal["skip_reason"] == "zero_reward_variance"


def test_development_group_is_explicitly_prohibited_from_training() -> None:
    members = [
        {
            "schema_version": "ifv-rollout-group-member-v1",
            "prompt_group_id": "pg-frozen",
            "case_id": "case-reward-1",
            "episode_id": f"frozen-{index}",
            "rollout_index": index,
            "group_size": 2,
            "sampling_seed": index,
            "training_prohibited": True,
            "training_eligible": False,
        }
        for index in range(2)
    ]
    group = build_standard_grpo_groups(
        [_ledger("frozen-0", correct=True), _ledger("frozen-1", correct=False)],
        members,
    )[0]
    assert group["trainable"] is False
    assert group["skip_reason"] == "training_prohibited_source"


def test_run_artifact_join_keeps_incorrect_episode_trainable() -> None:
    rows = [
        {
            "episode_id": "episode-0",
            "case_id": "case-reward-1",
            "classification_correct": True,
            "strict_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": ["episode-0:judgment:1"],
            "process_components": {
                "evidence_chain_reward": 0.875,
                "discrepancy_alignment_reward": 0.875,
                "stop_quality_reward": 0.875,
            },
        },
        {
            "episode_id": "episode-1",
            "case_id": "case-reward-1",
            "classification_correct": False,
            "strict_trace_audit_pass": True,
            "fatal_engineering_error": False,
            "step_ids": ["episode-1:judgment:1"],
            "process_components": {
                "evidence_chain_reward": 1.0,
                "discrepancy_alignment_reward": 1.0,
                "stop_quality_reward": 1.0,
            },
        },
    ]
    ledgers = build_ledgers_from_run_artifacts(
        deterministic_rows=rows,
    )
    assert [item["scalar_reward"] for item in ledgers] == [0.91875, 0.0]
    assert all(item["gates"]["trainable"] for item in ledgers)


def test_v2_profile_is_the_default_trajectory_profile() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_reward_profile(root / "configs" / "rl" / "deterministic-process-v2.json")
    assert profile["profile_id"] == "ifv-deterministic-process-v2"
    assert profile["correct_reward_floor"] == 0.35
    assert profile["weights"]["evidence_chain_reward"] == 0.4
    assert profile["weights"]["discrepancy_alignment_reward"] == 0.4
    assert profile["weights"]["stop_quality_reward"] == 0.2
    assert profile["allow_nonfatal_audit_failures"] is True
    assert set(profile["weights"]) == {
        "evidence_chain_reward",
        "discrepancy_alignment_reward",
        "stop_quality_reward",
    }
