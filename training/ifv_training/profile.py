from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ERROR_PATTERNS = {
    "cuda_oom": re.compile(
        r"(torch\.cuda\.OutOfMemoryError|CUDA out of memory|CUDACachingAllocator)",
        re.IGNORECASE,
    ),
    "cpu_oom_or_sigkill": re.compile(
        r"(exitcode\s*[-=]\s*9|SIGKILL|Killed\b|oom-kill|out of memory killer)",
        re.IGNORECASE,
    ),
    "nccl": re.compile(r"\bNCCL\b.*(error|failed|unhandled|timeout)", re.IGNORECASE),
    "nan_or_inf": re.compile(r"\b(nan|inf)\b", re.IGNORECASE),
    "traceback": re.compile(r"^Traceback \(most recent call last\):"),
}

METRIC_KEYS = {
    "loss",
    "eval_loss",
    "global_step/max_steps",
    "train_speed(s/it)",
    "memory(GiB)",
    "token_acc",
    "grad_norm",
    "learning_rate",
    "epoch",
    "train_runtime",
    "eval_runtime",
}


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", stripped):
            return float(stripped)
    except ValueError:
        return value
    return value


def _parse_metric_dict(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    payload = line[start : end + 1]
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if not METRIC_KEYS.intersection(str(key) for key in parsed):
        return None
    return {str(key): _coerce_number(value) for key, value in parsed.items()}


def _parse_step(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, str) or "/" not in value:
        return None, None
    left, right = value.split("/", 1)
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return None, None


def _metric_value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize_training_log(
    train_log: Path,
    *,
    steady_window: int = 5,
    experiment_id: str = "",
    profile_id: str = "",
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    errors: dict[str, list[int]] = {name: [] for name in ERROR_PATTERNS}
    line_count = 0

    with train_log.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line_count = line_number
            for name, pattern in ERROR_PATTERNS.items():
                if pattern.search(line):
                    errors[name].append(line_number)
            metric = _parse_metric_dict(line)
            if metric is None:
                continue
            current_step, max_steps = _parse_step(metric.get("global_step/max_steps"))
            if current_step is not None:
                metric["global_step"] = current_step
            if max_steps is not None:
                metric["max_steps"] = max_steps
            metrics.append(metric)

    train_step_metrics = [
        row
        for row in metrics
        if "loss" in row and "eval_loss" not in row and "train_loss" not in row
    ]
    eval_metrics = [row for row in metrics if "eval_loss" in row]
    summary_metrics = [row for row in metrics if "train_runtime" in row]
    speed_source = train_step_metrics if train_step_metrics else metrics
    speed_values = [
        value
        for row in speed_source
        if (value := _metric_value(row, "train_speed(s/it)", "train_runtime_seconds_per_step"))
        is not None
    ]
    memory_values = [
        value for row in metrics if (value := _metric_value(row, "memory(GiB)")) is not None
    ]
    loss_values = [value for row in metrics if (value := _metric_value(row, "loss")) is not None]
    eval_loss_values = [
        value for row in metrics if (value := _metric_value(row, "eval_loss")) is not None
    ]
    steps = [int(row["global_step"]) for row in metrics if isinstance(row.get("global_step"), int)]
    steady_values = speed_values[-steady_window:] if steady_window > 0 else []

    detected_errors = {name: lines for name, lines in errors.items() if lines}
    return {
        "schema_version": "ifv-training-profile-v1",
        "experiment_id": experiment_id,
        "profile_id": profile_id,
        "train_log": str(train_log),
        "line_count": line_count,
        "metric_rows": len(metrics),
        "train_step_metric_rows": len(train_step_metrics),
        "eval_metric_rows": len(eval_metrics),
        "summary_metric_rows": len(summary_metrics),
        "detected_errors": detected_errors,
        "passed_basic_log_gate": not detected_errors,
        "steps": {
            "observed": steps,
            "last": max(steps) if steps else None,
        },
        "speed_seconds_per_step": {
            "count": len(speed_values),
            "first": speed_values[0] if speed_values else None,
            "last": speed_values[-1] if speed_values else None,
            "mean": mean(speed_values) if speed_values else None,
            "steady_window": steady_window,
            "steady_mean": mean(steady_values) if steady_values else None,
        },
        "memory_gib": {
            "count": len(memory_values),
            "max": max(memory_values) if memory_values else None,
            "last": memory_values[-1] if memory_values else None,
        },
        "loss": {
            "count": len(loss_values),
            "last": loss_values[-1] if loss_values else None,
        },
        "eval_loss": {
            "count": len(eval_loss_values),
            "last": eval_loss_values[-1] if eval_loss_values else None,
        },
        "runtime_seconds": {
            "train_runtime": (
                _metric_value(summary_metrics[-1], "train_runtime")
                if summary_metrics
                else None
            ),
            "eval_runtime_last": (
                _metric_value(eval_metrics[-1], "eval_runtime") if eval_metrics else None
            ),
        },
    }


def write_training_profile(
    *,
    train_log: Path,
    output: Path,
    steady_window: int = 5,
    experiment_id: str = "",
    profile_id: str = "",
) -> dict[str, Any]:
    result = summarize_training_log(
        train_log,
        steady_window=steady_window,
        experiment_id=experiment_id,
        profile_id=profile_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
