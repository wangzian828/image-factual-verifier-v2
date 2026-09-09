"""Strict, gold-gated verification for IFV OPSD continuations."""

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

from .opsd import VerificationResult, _text, verify_repair


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
    metadata["opsd_role"] = role
    metadata["opsd_suffix_step"] = True
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
        raise ValueError("OPSD trace role must be teacher or student")
    trace = copy.deepcopy(dict(base_trace))
    state = trace.get("state")
    if not isinstance(state, dict):
        raise ValueError("OPSD trace requires a state object")
    all_steps = state.get("all_steps")
    if not isinstance(all_steps, list):
        raise ValueError("OPSD trace requires state.all_steps")
    if not suffix_steps:
        raise ValueError("OPSD suffix must contain at least one step")
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
    trace["opsd"] = {
        "role": role,
        "suffix_step_count": len(suffix_steps),
        "terminal_state_replayed": False,
    }
    return trace


def _write_trace(trace: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="ifv-opsd-verify-",
        delete=False,
        encoding="utf-8",
    )
    path = Path(handle.name)
    with handle:
        json.dump(trace, handle, ensure_ascii=False)
    return path


def verify_causal_episode(
    trace: Mapping[str, Any],
    *,
    gold: Mapping[str, Any],
    local_pass: bool,
    downstream_patch_count: int = 0,
    score_metadata: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Run strict audit and private-gold scoring on one complete episode.

    A bounded suffix without a terminal judgment is deliberately rejected. The
    caller must supply the canonical trace produced by an actual full-episode
    replay, not infer completion from a successful local tool action.
    """

    _assert_no_private_fields(trace)
    path = _write_trace(trace)
    reasons: list[str] = []
    recorded_verdict = _text(trace.get("verdict"))
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
            trace,
            gold,
            score_metadata=score_metadata,
        )
        expected_verdict = _text(metrics.get("expected_verdict"))
        if metrics.get("result_correct") is not True:
            reasons.append("private_gold_verdict_mismatch")
        if metrics.get("engineering_error") is True:
            reasons.append("engineering_error")
        judgment = _mapping(trace.get("judgment") or _mapping(trace.get("state")).get("judgment"))
        has_report = isinstance(judgment.get("fact_check_report"), Mapping)
        has_terminal_output = any(
            str(step.get("stage", "")).strip() == "unified_judgment"
            and str(step.get("action_type", "")).strip() == "output"
            for step in _rows(_mapping(trace.get("state")).get("all_steps"))
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
        local_pass=local_pass,
        recorded_verdict=recorded_verdict,
        expected_verdict=expected_verdict,
        strict_trace_audit_pass=strict_pass,
        full_episode_pass=full_pass,
        downstream_patch_count=downstream_patch_count,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def verify_continuation_pair(
    *,
    base_trace: Mapping[str, Any],
    teacher_steps: Sequence[StageStep],
    student_steps: Sequence[StageStep],
    student_episode_trace: Mapping[str, Any] | None,
    gold: Mapping[str, Any],
    local_pass: bool,
) -> tuple[VerificationResult, dict[str, Any]]:
    """Keep suffix diagnostics and require a separately replayed student episode."""

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
    if student_episode_trace is None:
        result = verify_repair(
            local_pass=local_pass,
            recorded_verdict="",
            expected_verdict="",
            strict_trace_audit_pass=False,
            full_episode_pass=False,
            reasons=("full_student_episode_trace_required",),
        )
    else:
        result = verify_causal_episode(
            student_episode_trace,
            gold=gold,
            local_pass=local_pass,
        )
    return result, {
        "teacher_suffix_trace": teacher_trace,
        "student_suffix_trace": student_suffix_trace,
        "student_episode_verified": student_episode_trace is not None,
    }
