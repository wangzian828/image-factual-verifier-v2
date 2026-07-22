"""Validated reward composition for IFV Agent RL.

The module consumes immutable semantic artifacts produced after a rollout.  It
does not call a teacher, execute tools, read gold, or implement an Agent loop.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json, load_json, write_json


SEMANTIC_REWARD_SCHEMA_VERSION = "ifv-semantic-reward-v1"
REWARD_LEDGER_SCHEMA_VERSION = "ifv-rl-reward-ledger-v1"
REWARD_PROFILE_SCHEMA_VERSION = "ifv-rl-reward-profile-v1"

DEFAULT_COMPONENT_WEIGHTS = {
    "classification_correct": 0.20,
    "strict_trace_audit": 0.10,
    "verdict_blind_agreement": 0.15,
    "claim_entailment": 0.10,
    "evidence_citation_fidelity": 0.10,
    "verdict_sufficiency": 0.15,
    "verdict_swap_rejection": 0.10,
    "evidence_dropout_sensitivity": 0.05,
    "rubber_stamp_resistance": 0.05,
}


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _number(value: Any, *, location: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be within [0, 1]")
    return result


def _artifact_core(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in artifact.items()
        if key not in {"artifact_id", "created_at"}
    }


def validate_semantic_reward_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on schema, IDs, metrics, calls, or content hash changes."""

    errors: list[str] = []
    if artifact.get("schema_version") != SEMANTIC_REWARD_SCHEMA_VERSION:
        errors.append("unsupported semantic reward schema_version")
    case_id = str(artifact.get("case_id", "")).strip()
    if not case_id:
        errors.append("case_id is required")
    source_trace = artifact.get("source_trace")
    if not isinstance(source_trace, Mapping):
        errors.append("source_trace is required")
    elif len(str(source_trace.get("sha256", ""))) != 64:
        errors.append("source_trace.sha256 is invalid")
    reward_input = artifact.get("reward_input")
    if not isinstance(reward_input, Mapping):
        errors.append("reward_input is required")
    elif len(str(reward_input.get("sha256", ""))) != 64:
        errors.append("reward_input.sha256 is invalid")
    rollout = artifact.get("rollout")
    if not isinstance(rollout, Mapping):
        errors.append("rollout is required")
    else:
        if not str(rollout.get("episode_id", "")).strip():
            errors.append("rollout.episode_id is required")
        step_ids = rollout.get("policy_step_ids")
        if not isinstance(step_ids, list) or any(
            not str(value).strip() for value in step_ids
        ):
            errors.append("rollout.policy_step_ids must be a list of IDs")
        terminal = str(rollout.get("terminal_policy_step_id", ""))
        if isinstance(step_ids, list) and step_ids:
            if terminal != str(step_ids[-1]):
                errors.append("rollout.terminal_policy_step_id is inconsistent")
        elif terminal:
            errors.append("rollout.terminal_policy_step_id requires a policy step")

    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics is required")
        metrics = {}
    required_metrics = (
        "verdict_blind_agreement",
        "claim_label_agreement",
        "claim_entailment",
        "evidence_citation_fidelity",
        "verdict_sufficiency",
        "verdict_swap_rejection",
        "rubber_stamp_risk",
    )
    for name in required_metrics:
        try:
            _number(metrics.get(name), location=f"metrics.{name}")
        except ValueError as exc:
            errors.append(str(exc))
    try:
        _number(
            metrics.get("evidence_dropout_sensitivity"),
            location="metrics.evidence_dropout_sensitivity",
            nullable=True,
        )
    except ValueError as exc:
        errors.append(str(exc))
    invalid_ids = metrics.get("invalid_judge_evidence_ids", [])
    if not isinstance(invalid_ids, list):
        errors.append("metrics.invalid_judge_evidence_ids must be a list")

    gates = artifact.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates is required")
    else:
        for name in (
            "strict_trace_audit_pass",
            "engineering_valid",
            "semantic_audit_pass",
        ):
            if not isinstance(gates.get(name), bool):
                errors.append(f"gates.{name} must be boolean")

    judge = artifact.get("judge")
    if not isinstance(judge, Mapping):
        errors.append("judge is required")
    else:
        calls = judge.get("calls")
        if not isinstance(calls, list) or len(calls) != 2:
            errors.append("judge.calls must contain blind and aware calls")
        else:
            for index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    errors.append(f"judge.calls[{index}] must be an object")
                    continue
                if len(str(call.get("request_sha256", ""))) != 64:
                    errors.append(f"judge.calls[{index}].request_sha256 is invalid")
                if len(str(call.get("response_sha256", ""))) != 64:
                    errors.append(f"judge.calls[{index}].response_sha256 is invalid")
                usage = call.get("usage")
                if not isinstance(usage, Mapping):
                    errors.append(f"judge.calls[{index}].usage is required")
    blind = artifact.get("blind_judgment")
    if not isinstance(blind, Mapping):
        errors.append("blind_judgment is required")
    else:
        if blind.get("predicted_verdict") not in {"real", "fake", "unclear"}:
            errors.append("blind_judgment.predicted_verdict is invalid")
        claim_reviews = blind.get("claim_reviews")
        if not isinstance(claim_reviews, list) or not claim_reviews:
            errors.append("blind_judgment.claim_reviews must be a non-empty list")
    aware = artifact.get("aware_counterfactual_judgment")
    if not isinstance(aware, Mapping):
        errors.append("aware_counterfactual_judgment is required")
    else:
        for name in (
            "original_verdict_supported",
            "swapped_verdict_rejected",
            "dropout_applicable",
            "dropout_verdict_supported",
        ):
            if not isinstance(aware.get(name), bool):
                errors.append(f"aware_counterfactual_judgment.{name} must be boolean")

    artifact_id = str(artifact.get("artifact_id", ""))
    expected_id = "sha256:" + hashlib.sha256(
        canonical_json(_artifact_core(artifact)).encode("utf-8")
    ).hexdigest()
    if artifact_id != expected_id:
        errors.append("artifact_id does not match semantic artifact content")

    return {
        "schema_version": "ifv-semantic-reward-audit-v1",
        "case_id": case_id,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "artifact_id": artifact_id,
    }


def load_reward_profile(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": REWARD_PROFILE_SCHEMA_VERSION,
            "profile_id": "ifv-semantic-balanced-v1",
            "weights": dict(DEFAULT_COMPONENT_WEIGHTS),
        }
    profile = load_json(path)
    if profile.get("schema_version") != REWARD_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported reward profile schema_version")
    profile_id = str(profile.get("profile_id", "")).strip()
    if not profile_id:
        raise ValueError("reward profile requires profile_id")
    raw_weights = _mapping(profile.get("weights"), location="profile.weights")
    weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        if name not in DEFAULT_COMPONENT_WEIGHTS:
            raise ValueError(f"unknown reward component weight: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weight {name} must be numeric")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"weight {name} cannot be negative")
        weights[str(name)] = numeric
    if not any(weights.values()):
        raise ValueError("reward profile requires at least one positive weight")
    return {**profile, "weights": weights}


def _optional_bool(value: Any, *, location: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean or null")
    return value


def compose_reward_ledger(
    artifact: Mapping[str, Any],
    *,
    deterministic: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = validate_semantic_reward_artifact(artifact)
    if not audit["passed"]:
        raise ValueError("invalid semantic reward artifact: " + "; ".join(audit["errors"]))
    deterministic = deterministic or {}
    profile = dict(profile or load_reward_profile())
    if profile.get("schema_version") != REWARD_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported reward profile")
    weights = _mapping(profile.get("weights"), location="profile.weights")
    metrics = _mapping(artifact.get("metrics"), location="artifact.metrics")
    gates = _mapping(artifact.get("gates"), location="artifact.gates")
    rollout = _mapping(artifact.get("rollout"), location="artifact.rollout")

    classification_correct = _optional_bool(
        deterministic.get("classification_correct"),
        location="deterministic.classification_correct",
    )
    provider_fatal = bool(deterministic.get("fatal_engineering_error", False))
    engineering_valid = bool(gates.get("engineering_valid")) and not provider_fatal
    strict_trace_audit = bool(gates.get("strict_trace_audit_pass")) and bool(
        deterministic.get("strict_trace_audit_pass", True)
    )
    components: dict[str, float | None] = {
        "classification_correct": (
            None if classification_correct is None else float(classification_correct)
        ),
        "strict_trace_audit": float(strict_trace_audit),
        "verdict_blind_agreement": float(metrics["verdict_blind_agreement"]),
        "claim_entailment": float(metrics["claim_entailment"]),
        "evidence_citation_fidelity": float(
            metrics["evidence_citation_fidelity"]
        ),
        "verdict_sufficiency": float(metrics["verdict_sufficiency"]),
        "verdict_swap_rejection": float(metrics["verdict_swap_rejection"]),
        "evidence_dropout_sensitivity": (
            None
            if metrics.get("evidence_dropout_sensitivity") is None
            else float(metrics["evidence_dropout_sensitivity"])
        ),
        "rubber_stamp_resistance": 1.0 - float(metrics["rubber_stamp_risk"]),
    }
    fatal_mask = not engineering_valid or provider_fatal
    mask_reason = ""
    if provider_fatal:
        mask_reason = "fatal_environment_or_provider_error"
    elif not engineering_valid:
        mask_reason = "engineering_invalid_rollout"

    weighted_terms: list[tuple[str, float, float]] = []
    for name, raw_weight in weights.items():
        value = components.get(str(name))
        weight = float(raw_weight)
        if value is None or weight <= 0.0:
            continue
        weighted_terms.append((str(name), float(value), weight))
    weight_sum = sum(weight for _, _, weight in weighted_terms)
    if weight_sum <= 0.0:
        raise ValueError("no active reward components after nullable values")
    scalar_reward = sum(value * weight for _, value, weight in weighted_terms) / weight_sum
    if fatal_mask:
        scalar_reward_value: float | None = None
    else:
        scalar_reward_value = round(max(0.0, min(1.0, scalar_reward)), 8)

    case_id = str(artifact.get("case_id", ""))
    episode_id = str(
        deterministic.get("episode_id") or rollout.get("episode_id") or case_id
    )
    step_ids = deterministic.get("step_ids")
    if step_ids is None:
        step_ids = rollout.get("policy_step_ids", [])
    if not isinstance(step_ids, list):
        raise ValueError("deterministic.step_ids must be a list")
    ledger_core = {
        "schema_version": REWARD_LEDGER_SCHEMA_VERSION,
        "case_id": case_id,
        "episode_id": episode_id,
        "source_semantic_artifact_id": artifact.get("artifact_id"),
        "source_trace_sha256": _mapping(
            artifact.get("source_trace"), location="artifact.source_trace"
        ).get("sha256"),
        "profile": {
            "profile_id": profile.get("profile_id"),
            "weights": dict(weights),
            "active_weight_sum": weight_sum,
        },
        "components": components,
        "semantic_metrics": dict(metrics),
        "gates": {
            "engineering_valid": engineering_valid,
            "strict_trace_audit_pass": strict_trace_audit,
            "semantic_audit_pass": bool(gates.get("semantic_audit_pass")),
            "classification_correct": classification_correct,
            "has_policy_steps": bool(step_ids),
            "trainable": bool(not fatal_mask and step_ids),
            "eligible_for_positive_buffer": bool(
                not fatal_mask
                and step_ids
                and gates.get("semantic_audit_pass")
                and classification_correct is True
            ),
        },
        "fatal_mask": {
            "masked": fatal_mask,
            "reason": mask_reason,
        },
        "scalar_reward": scalar_reward_value,
        "step_ids": [str(value) for value in step_ids],
        "teacher_usage": _teacher_usage(artifact),
    }
    ledger_id = "sha256:" + hashlib.sha256(
        canonical_json(ledger_core).encode("utf-8")
    ).hexdigest()
    return {
        **ledger_core,
        "ledger_id": ledger_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _teacher_usage(artifact: Mapping[str, Any]) -> dict[str, int]:
    judge = _mapping(artifact.get("judge"), location="artifact.judge")
    calls = judge.get("calls", [])
    totals = {"call_count": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    if not isinstance(calls, list):
        return totals
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        totals["call_count"] += 1
        for name in ("input_tokens", "output_tokens", "thought_tokens"):
            totals[name] += int(usage.get(name, 0) or 0)
    return totals


def validate_reward_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if ledger.get("schema_version") != REWARD_LEDGER_SCHEMA_VERSION:
        errors.append("unsupported reward ledger schema_version")
    scalar = ledger.get("scalar_reward")
    fatal_mask = _mapping(ledger.get("fatal_mask"), location="fatal_mask")
    masked = bool(fatal_mask.get("masked"))
    if masked and scalar is not None:
        errors.append("fatal-masked ledger must have scalar_reward=null")
    if not masked:
        try:
            _number(scalar, location="scalar_reward")
        except ValueError as exc:
            errors.append(str(exc))
    core = {
        str(key): value
        for key, value in ledger.items()
        if key not in {"ledger_id", "created_at"}
    }
    expected = "sha256:" + hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    if ledger.get("ledger_id") != expected:
        errors.append("ledger_id does not match reward ledger content")
    return {
        "schema_version": "ifv-rl-reward-ledger-audit-v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "ledger_id": ledger.get("ledger_id"),
    }


def export_framework_reward(
    ledger: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    audit = validate_reward_ledger(ledger)
    if not audit["passed"]:
        raise ValueError("invalid reward ledger: " + "; ".join(audit["errors"]))
    framework = framework.casefold()
    fatal_mask = _mapping(ledger.get("fatal_mask"), location="fatal_mask")
    masked = bool(fatal_mask.get("masked"))
    reward = ledger.get("scalar_reward")
    step_ids = [str(value) for value in ledger.get("step_ids", [])]
    terminal_step_id = step_ids[-1] if step_ids else ""
    skip_update = masked or not step_ids
    mask_reason = str(fatal_mask.get("reason", ""))
    if not mask_reason and not step_ids:
        mask_reason = "no_trainable_policy_steps"
    common = {
        "episode_id": ledger.get("episode_id"),
        "reward": reward,
        "skip_update": skip_update,
        "mask_reason": mask_reason,
        "ledger_id": ledger.get("ledger_id"),
        "components": ledger.get("components"),
    }
    if framework == "rllm":
        return {
            "schema_version": "ifv-rllm-reward-record-v1",
            **common,
            "assignment": "terminal_step",
            "terminal_step_id": terminal_step_id,
            "step_rewards": [
                {
                    "step_id": step_id,
                    "reward": reward if step_id == terminal_step_id else 0.0,
                }
                for step_id in step_ids
            ],
        }
    if framework == "verl":
        return {
            "schema_version": "ifv-verl-reward-record-v1",
            "data_source": "ifv_discrepancy_agent_v4",
            **common,
            "reward_assignment": {
                "mode": "terminal_token",
                "terminal_step_id": terminal_step_id,
            },
            "extra_info": {
                "case_id": ledger.get("case_id"),
                "semantic_audit_pass": _mapping(
                    ledger.get("gates"), location="gates"
                ).get("semantic_audit_pass"),
            },
        }
    raise ValueError("framework must be rllm or verl")


def build_and_write_ledger(
    *,
    semantic_artifact_path: Path,
    output_path: Path,
    deterministic_path: Path | None = None,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    artifact = load_json(semantic_artifact_path)
    deterministic = load_json(deterministic_path) if deterministic_path else {}
    profile = load_reward_profile(profile_path)
    ledger = compose_reward_ledger(
        artifact,
        deterministic=deterministic,
        profile=profile,
    )
    write_json(output_path, ledger)
    return ledger
