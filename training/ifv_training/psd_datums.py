"""Build framework-neutral sparse top-K PSD cross-entropy datums."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .io import (
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)
from .psd import validate_topk_by_position


PSD_SPARSE_DATUM_SCHEMA_VERSION = "ifv-psd-sparse-topk-datum-v2"
PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION = "ifv-psd-sparse-topk-manifest-v2"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _required_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field}[{index}] must be an integer")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be an integer") from exc
    return result


def _positive_weight(value: Any, *, field: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return weight


def _topk_entry(value: Any) -> tuple[int, float]:
    if isinstance(value, Mapping):
        token_id = value.get("token_id")
        probability = value.get("probability", value.get("prob"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        token_id, probability = value
    else:
        raise ValueError("top-k entry has invalid shape")
    if isinstance(token_id, bool):
        raise ValueError("top-k token ID must be numeric")
    try:
        return int(token_id), float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("top-k entry contains invalid values") from exc


def build_sparse_topk_datum(
    target: Mapping[str, Any],
    *,
    topk: int,
    effective_row_weight: float | None = None,
    max_sequence_length: int = 131_072,
) -> dict[str, Any]:
    """Project one complete PSD target onto causal prediction positions.

    ``input_ids`` excludes the final completion token, matching next-token
    training.  Only positions predicting completion tokens receive non-zero
    weights; prompt and tool-observation tokens are never direct loss targets.
    """

    if _text(target.get("target_status")) != "complete":
        raise ValueError("target_status must be complete")
    target_id = _text(target.get("target_id"))
    if not target_id:
        raise ValueError("target_id_missing")
    kind = _text(target.get("kind"))
    if kind not in {"repair", "preserve"}:
        raise ValueError(f"unsupported target kind: {kind}")

    prompt_ids = _required_int_list(
        target.get("student_prompt_ids"),
        field="student_prompt_ids",
    )
    completion_ids = _required_int_list(
        target.get("completion_ids"),
        field="completion_ids",
    )
    combined = [*prompt_ids, *completion_ids]
    if len(combined) > max_sequence_length:
        raise ValueError(
            f"sequence_length_exceeds_limit:{len(combined)}>{max_sequence_length}"
        )
    if len(combined) < 2:
        raise ValueError("sequence must contain at least two tokens")

    distributions = target.get("teacher_topk_by_position")
    validate_topk_by_position(completion_ids, distributions, topk=topk)
    source_row_weight = _positive_weight(
        target.get("row_weight", 1.0),
        field="row_weight",
    )
    row_weight = _positive_weight(
        source_row_weight
        if effective_row_weight is None
        else effective_row_weight,
        field="effective_row_weight",
    )

    input_ids = combined[:-1]
    target_tokens = [[0] * topk for _ in input_ids]
    weights = [[0.0] * topk for _ in input_ids]
    loss_positions: list[int] = []
    for completion_offset, raw_entries in enumerate(distributions):
        prediction_position = len(prompt_ids) + completion_offset - 1
        if prediction_position < 0 or prediction_position >= len(input_ids):
            raise ValueError(
                f"completion prediction position out of range: {prediction_position}"
            )
        entries = [_topk_entry(entry) for entry in raw_entries]
        for rank, (token_id, probability) in enumerate(entries):
            target_tokens[prediction_position][rank] = token_id
            weights[prediction_position][rank] = probability * row_weight
        loss_positions.append(prediction_position)

    return {
        "schema_version": PSD_SPARSE_DATUM_SCHEMA_VERSION,
        "target_id": target_id,
        "kind": kind,
        "topk": topk,
        "input_ids": input_ids,
        "target_tokens": target_tokens,
        "weights": weights,
        "loss_positions": loss_positions,
        "source_row_weight": source_row_weight,
        "row_weight": row_weight,
        "sequence_tokens": len(combined),
        "input_tokens": len(input_ids),
        "completion_tokens": len(completion_ids),
    }


def _balanced_weights(
    targets: list[Mapping[str, Any]],
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for target in targets:
        kind = _text(target.get("kind"))
        if kind not in {"repair", "preserve"}:
            raise ValueError(f"unsupported target kind: {kind}")
        totals[kind] += _positive_weight(
            target.get("row_weight", 1.0),
            field="row_weight",
        )
    return {
        kind: 1.0 / total
        for kind, total in totals.items()
        if total > 0
    }


def build_sparse_topk_package(
    *,
    targets_path: Path,
    output_dir: Path,
    topk: int = 20,
    max_sequence_length: int = 131_072,
    balance_kinds: bool = True,
    require_both_kinds: bool = True,
) -> dict[str, Any]:
    """Create a fail-closed sparse top-K training interface package."""

    if topk < 1:
        raise ValueError("topk must be positive")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least two")
    require_new_or_empty(output_dir)
    targets = load_jsonl(targets_path)
    if not targets:
        raise ValueError("PSD target input is empty")

    target_ids = [_text(target.get("target_id")) for target in targets]
    if any(not target_id for target_id in target_ids):
        raise ValueError("all targets must have target_id")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("PSD target IDs must be unique")

    input_kinds = {_text(target.get("kind")) for target in targets}
    missing_kinds = sorted({"repair", "preserve"} - input_kinds)
    source_kind_gate = not require_both_kinds or not missing_kinds

    kind_scales = _balanced_weights(targets) if balance_kinds else {}
    datums: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for row_index, target in enumerate(targets):
        kind = _text(target.get("kind"))
        source_weight = target.get("row_weight", 1.0)
        try:
            effective_weight = float(source_weight)
            if balance_kinds:
                effective_weight *= kind_scales[kind]
            datums.append(
                build_sparse_topk_datum(
                    target,
                    topk=topk,
                    effective_row_weight=effective_weight,
                    max_sequence_length=max_sequence_length,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejections.append(
                {
                    "row_index": row_index,
                    "target_id": _text(target.get("target_id")),
                    "kind": kind,
                    "reason": str(exc),
                }
            )

    ready = (
        len(datums) == len(targets)
        and not rejections
        and source_kind_gate
    )
    write_jsonl(output_dir / "candidate_datums.jsonl", datums)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    artifacts = {
        "candidate_datums": "candidate_datums.jsonl",
        "rejections": "rejections.jsonl",
    }
    if ready:
        write_jsonl(output_dir / "datums.jsonl", datums)
        artifacts["datums"] = "datums.jsonl"

    effective_mass = Counter()
    loss_positions = Counter()
    for datum in datums:
        effective_mass[datum["kind"]] += float(datum["row_weight"])
        loss_positions[datum["kind"]] += len(datum["loss_positions"])
    manifest = {
        "schema_version": PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION,
        "source": {
            "targets": str(targets_path),
            "targets_sha256": sha256_file(targets_path),
        },
        "topk": topk,
        "max_sequence_length": max_sequence_length,
        "balance_kinds": balance_kinds,
        "require_both_kinds": require_both_kinds,
        "missing_source_kinds": missing_kinds,
        "counts": {
            "input_targets": len(targets),
            "candidate_datums": len(datums),
            "rejections": len(rejections),
            "loss_positions": sum(loss_positions.values()),
        },
        "targets_by_kind": dict(sorted(Counter(
            datum["kind"] for datum in datums
        ).items())),
        "loss_positions_by_kind": dict(sorted(loss_positions.items())),
        "effective_row_mass_by_kind": dict(sorted(effective_mass.items())),
        "artifacts": artifacts,
        "artifact_sha256": {
            "candidate_datums": sha256_file(output_dir / "candidate_datums.jsonl"),
            "datums": sha256_file(output_dir / "datums.jsonl") if ready else None,
        },
        "status": (
            "ready_for_trainer"
            if ready
            else "blocked_invalid_target"
            if rejections or len(datums) != len(targets)
            else "blocked_missing_source_kind"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
