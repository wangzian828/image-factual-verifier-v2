"""Strict, gold-gated verification for IFV PSD repair continuations."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import _repo_import  # noqa: F401
from scripts.audit_real_trace import audit_trace
from src.trajectory.scoring import score_process_trace
from src.orchestrator.stage_runner import StageStep

from .psd_repair import VerificationResult, _sha, _text, verify_repair


PSD_LOCAL_VERIFICATION_SCHEMA_VERSION = "ifv-psd-local-verification-v1"


_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "expected_verdict",
        "ground_truth",
        "private_gold",
        "private_target",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _assert_no_private_fields(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            if name.casefold() in _PRIVATE_KEYS:
                raise ValueError(f"private field in student trace: {child_path}")
            _assert_no_private_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_fields(child, path=f"{path}[{index}]")


def _step_row(step: StageStep, *, stage: str, role: str) -> dict[str, Any]:
    metadata = copy.deepcopy(step.metadata or {})
    metadata["psd_role"] = role
    metadata["psd_suffix_step"] = True
    row: dict[str, Any] = {
        "round": int(step.round or 0),
        "stage": stage,
        "action_type": str(step.action_type or ""),
        "tool_name": str(step.tool_name or ""),
        "tool_args": copy.deepcopy(step.tool_args or {}),
        "tool_result": str(step.tool_result or ""),
        "tokens": copy.deepcopy(step.tokens or {}),
        "metadata": metadata,
    }
    if step.thought:
        row["thought"] = step.thought
    if step.output is not None:
        row["output"] = copy.deepcopy(step.output)
    return row


def merge_suffix_into_trace(
    base_trace: Mapping[str, Any],
    suffix_steps: Sequence[StageStep],
    *,
    role: str,
    source_stage: str = "unified_react",
) -> dict[str, Any]:
    """Append a bounded suffix without inventing terminal episode state."""

    if role not in {"teacher", "student"}:
        raise ValueError("PSD repair trace role must be teacher or student")
    trace = copy.deepcopy(dict(base_trace))
    state = trace.get("state")
    if not isinstance(state, dict):
        raise ValueError("PSD repair trace requires a state object")
    all_steps = state.get("all_steps")
    if not isinstance(all_steps, list):
        raise ValueError("PSD repair trace requires state.all_steps")
    if not suffix_steps:
        raise ValueError("PSD repair suffix must contain at least one step")
    _assert_no_private_fields(trace if role == "student" else {})
    all_steps.extend(
        _step_row(step, stage=source_stage, role=role)
        for step in suffix_steps
    )
    tool_count = sum(1 for step in suffix_steps if step.action_type == "tool_call")
    state["total_tool_calls"] = int(state.get("total_tool_calls", 0) or 0) + tool_count
    state["llm_api_calls"] = int(state.get("llm_api_calls", 0) or 0) + sum(
        1 for step in suffix_steps if step.metadata.get("llm_duration_ms") is not None
    )
    trace["psd_repair"] = {
        "role": role,
        "suffix_step_count": len(suffix_steps),
        "terminal_state_replayed": False,
    }
    return trace


def _write_trace(trace: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="ifv-psd-repair-verify-",
        delete=False,
        encoding="utf-8",
    )
    path = Path(handle.name)
    with handle:
        json.dump(trace, handle, ensure_ascii=False)
    return path


def validate_local_verification(
    value: Mapping[str, Any] | None,
    *,
    repair_step_id: str,
) -> dict[str, Any]:
    """Validate a task-verifier artifact; model activity is never a proxy."""

    artifact = _mapping(value)
    errors: list[str] = []
    if artifact.get("schema_version") != PSD_LOCAL_VERIFICATION_SCHEMA_VERSION:
        errors.append("local_verifier_schema_invalid")
    if _text(artifact.get("repair_step_id")) != _text(repair_step_id):
        errors.append("local_verifier_repair_step_mismatch")
    verifier = _mapping(artifact.get("verifier"))
    if _text(verifier.get("kind")) != "task":
        errors.append("local_verifier_kind_not_task")
    if not _text(verifier.get("name")) or not _text(verifier.get("version")):
        errors.append("local_verifier_identity_missing")
    passed = artifact.get("passed")
    if not isinstance(passed, bool):
        errors.append("local_verifier_pass_not_boolean")
    checks = _rows(artifact.get("checks"))
    if not checks:
        errors.append("local_verifier_checks_missing")
    elif any(not isinstance(row.get("passed"), bool) for row in checks):
        errors.append("local_verifier_check_result_invalid")
    elif passed is True and any(row.get("passed") is not True for row in checks):
        errors.append("local_verifier_checks_disagree")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("local_verifier_evidence_missing")
    return {
        "valid": not errors,
        "passed": bool(passed is True and not errors),
        "errors": errors,
        "verifier": {
            "kind": _text(verifier.get("kind")),
            "name": _text(verifier.get("name")),
            "version": _text(verifier.get("version")),
        },
        "repair_step_id": _text(artifact.get("repair_step_id")),
        "artifact_sha256": _sha(artifact) if artifact else "",
    }


def verify_source_rollout_failure(
    trace: Mapping[str, Any],
    *,
    gold: Mapping[str, Any],
    score_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that the original no-hint on-policy rollout actually failed."""

    _assert_no_private_fields(trace)
    reasons: list[str] = []
    result_correct: bool | None = None
    expected_verdict = ""
    try:
        metrics, _score = score_process_trace(
            trace,
            gold,
            score_metadata=score_metadata,
        )
        raw_result = metrics.get("result_correct")
        result_correct = raw_result if isinstance(raw_result, bool) else None
        expected_verdict = _text(metrics.get("expected_verdict"))
        if result_correct is not False:
            reasons.append("source_rollout_not_explicitly_failed")
    except Exception as exc:
        reasons.append(f"source_verification_error:{type(exc).__name__}:{exc}")
    return {
        "passed": result_correct is False and not reasons,
        "result_correct": result_correct,
        "recorded_verdict": _text(trace.get("verdict")),
        "expected_verdict": expected_verdict,
        "reasons": reasons,
    }


def verify_causal_episode(
    hinted_trace: Mapping[str, Any],
    *,
    source_trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    local_verification: Mapping[str, Any] | None,
    repair_step_id: str,
    downstream_patch_count: int = 0,
    score_metadata: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Verify one hinted frozen-policy continuation against its failed source.

    A bounded suffix or unhinted student retry is deliberately insufficient.
    The caller must supply the canonical complete episode generated by the
    frozen round-start policy with the hint, plus a real local task-verifier
    artifact bound to the selected repair step.
    """

    _assert_no_private_fields(hinted_trace)
    source_result = verify_source_rollout_failure(
        source_trace,
        gold=gold,
        score_metadata=score_metadata,
    )
    local_result = validate_local_verification(
        local_verification,
        repair_step_id=repair_step_id,
    )
    path = _write_trace(hinted_trace)
    reasons: list[str] = []
    reasons.extend(source_result["reasons"])
    reasons.extend(f"local:{item}" for item in local_result["errors"])
    if local_result["valid"] and not local_result["passed"]:
        reasons.append("hinted_local_verifier_failed")
    recorded_verdict = _text(hinted_trace.get("verdict"))
    expected_verdict = ""
    strict_pass = False
    full_pass = False
    try:
        report = audit_trace(path)
        failures = report.failures(strict_scheduler=True)
        if failures:
            reasons.extend(f"audit:{item.code}" for item in failures)
        strict_pass = not failures
        metrics, _score = score_process_trace(
            hinted_trace,
            gold,
            score_metadata=score_metadata,
        )
        expected_verdict = _text(metrics.get("expected_verdict"))
        if metrics.get("result_correct") is not True:
            reasons.append("private_gold_verdict_mismatch")
        if metrics.get("engineering_error") is True:
            reasons.append("engineering_error")
        judgment = _mapping(
            hinted_trace.get("judgment")
            or _mapping(hinted_trace.get("state")).get("judgment")
        )
        has_report = isinstance(judgment.get("fact_check_report"), Mapping)
        has_terminal_output = any(
            str(step.get("stage", "")).strip() == "unified_judgment"
            and str(step.get("action_type", "")).strip() == "output"
            for step in _rows(
                _mapping(hinted_trace.get("state")).get("all_steps")
            )
        )
        full_pass = bool(
            strict_pass
            and metrics.get("result_correct") is True
            and has_report
            and has_terminal_output
            and recorded_verdict in {"real", "fake"}
        )
        if not has_terminal_output:
            reasons.append("terminal_judgment_missing")
        if not has_report:
            reasons.append("fact_check_report_missing")
    except Exception as exc:
        reasons.append(f"verification_error:{type(exc).__name__}:{exc}")
    finally:
        path.unlink(missing_ok=True)
    return verify_repair(
        source_rollout_failed=source_result["passed"],
        hinted_local_pass=local_result["passed"],
        hinted_recorded_verdict=recorded_verdict,
        expected_verdict=expected_verdict,
        hinted_strict_trace_audit_pass=strict_pass,
        hinted_episode_pass=full_pass,
        downstream_patch_count=downstream_patch_count,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def verify_continuation_pair(
    *,
    base_trace: Mapping[str, Any],
    teacher_steps: Sequence[StageStep],
    student_steps: Sequence[StageStep],
    hinted_teacher_episode_trace: Mapping[str, Any] | None,
    unhinted_student_episode_trace: Mapping[str, Any] | None,
    gold: Mapping[str, Any],
    local_verification: Mapping[str, Any] | None,
    repair_step_id: str,
) -> tuple[VerificationResult, dict[str, Any]]:
    """Verify the hinted teacher; keep the unhinted student as diagnostics."""

    teacher_trace = merge_suffix_into_trace(
        base_trace,
        teacher_steps,
        role="teacher",
    )
    student_suffix_trace = merge_suffix_into_trace(
        base_trace,
        student_steps,
        role="student",
    )
    if hinted_teacher_episode_trace is None:
        source_result = verify_source_rollout_failure(base_trace, gold=gold)
        local_result = validate_local_verification(
            local_verification,
            repair_step_id=repair_step_id,
        )
        reasons = [*source_result["reasons"]]
        reasons.extend(f"local:{item}" for item in local_result["errors"])
        reasons.append("full_hinted_teacher_episode_trace_required")
        result = verify_repair(
            source_rollout_failed=source_result["passed"],
            hinted_local_pass=local_result["passed"],
            hinted_recorded_verdict="",
            expected_verdict="",
            hinted_strict_trace_audit_pass=False,
            hinted_episode_pass=False,
            reasons=reasons,
        )
    else:
        result = verify_causal_episode(
            hinted_teacher_episode_trace,
            source_trace=base_trace,
            gold=gold,
            local_verification=local_verification,
            repair_step_id=repair_step_id,
        )
    student_diagnostic = None
    if unhinted_student_episode_trace is not None:
        student_diagnostic = verify_source_rollout_failure(
            unhinted_student_episode_trace,
            gold=gold,
        )
    return result, {
        "teacher_suffix_trace": teacher_trace,
        "student_suffix_trace": student_suffix_trace,
        "hinted_teacher_episode_verified": hinted_teacher_episode_trace is not None,
        "unhinted_student_diagnostic": student_diagnostic,
        "unhinted_student_affects_acceptance": False,
    }
