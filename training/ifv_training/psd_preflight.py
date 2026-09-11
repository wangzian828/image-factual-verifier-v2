"""Fail-closed preflight for materialized sparse top-K PSD datums."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import load_json, load_jsonl, sha256_file
from .psd_datums import (
    PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION,
    PSD_SPARSE_DATUM_SCHEMA_VERSION,
)


PSD_TRAINING_INPUT_GATE_SCHEMA_VERSION = "ifv-psd-training-input-gate-v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _datum_error(row: Mapping[str, Any], *, topk: int, max_context: int) -> str:
    if _text(row.get("schema_version")) != PSD_SPARSE_DATUM_SCHEMA_VERSION:
        return "datum_schema_invalid"
    if _text(row.get("kind")) not in {"repair", "preserve"}:
        return "datum_kind_invalid"
    if _integer(row.get("topk")) != topk:
        return "datum_topk_mismatch"
    input_ids = row.get("input_ids")
    target_tokens = row.get("target_tokens")
    weights = row.get("weights")
    if not isinstance(input_ids, list) or not input_ids:
        return "datum_input_ids_invalid"
    if any(
        isinstance(token, bool) or not isinstance(token, int)
        for token in input_ids
    ):
        return "datum_input_ids_invalid"
    if len(input_ids) > max_context:
        return "datum_context_exceeded"
    if not isinstance(target_tokens, list) or len(target_tokens) != len(input_ids):
        return "datum_target_shape_invalid"
    if not isinstance(weights, list) or len(weights) != len(input_ids):
        return "datum_weight_shape_invalid"
    try:
        row_weight = float(row.get("row_weight"))
    except (TypeError, ValueError):
        return "datum_row_weight_invalid"
    if not math.isfinite(row_weight) or row_weight <= 0:
        return "datum_row_weight_invalid"
    active = False
    for tokens, values in zip(target_tokens, weights, strict=True):
        if not isinstance(tokens, list) or len(tokens) != topk:
            return "datum_target_topk_shape_invalid"
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in tokens
        ):
            return "datum_target_token_invalid"
        if not isinstance(values, list) or len(values) != topk:
            return "datum_weight_topk_shape_invalid"
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            return "datum_weight_non_numeric"
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            return "datum_weight_invalid"
        position_mass = sum(numeric)
        if position_mass > 0 and not math.isclose(
            position_mass,
            row_weight,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return "datum_active_position_mass_mismatch"
        active = active or any(value > 0 for value in numeric)
    if not active:
        return "datum_has_no_active_target"
    return ""


def verify_psd_training_input(
    *,
    datums_path: Path,
    manifest_path: Path,
    expected_topk: int,
    max_context: int,
) -> dict[str, Any]:
    errors: list[str] = []
    datums_path = datums_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if expected_topk < 1:
        raise ValueError("expected_topk must be positive")
    if max_context < 2:
        raise ValueError("max_context must be at least two")

    manifest: Mapping[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if not datums_path.is_file():
        errors.append("datums_file_missing")
    if not manifest_path.is_file():
        errors.append("manifest_file_missing")
    if not errors:
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("manifest_json_invalid")
        try:
            rows = load_jsonl(datums_path)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("datums_jsonl_invalid")

    if manifest:
        if (
            manifest.get("schema_version")
            != PSD_SPARSE_DATUM_MANIFEST_SCHEMA_VERSION
        ):
            errors.append("manifest_schema_invalid")
        if manifest.get("status") != "ready_for_trainer":
            errors.append("manifest_not_ready_for_trainer")
        if _integer(manifest.get("topk")) != expected_topk:
            errors.append("manifest_topk_mismatch")
        if manifest.get("balance_kinds") is not True:
            errors.append("manifest_source_balance_disabled")
        if manifest.get("require_both_kinds") is not True:
            errors.append("manifest_both_source_kinds_not_required")
        if manifest.get("missing_source_kinds") not in ([], None):
            errors.append("manifest_source_kind_missing")
        manifest_max_context = _integer(manifest.get("max_sequence_length"))
        if not 2 <= manifest_max_context <= max_context:
            errors.append("manifest_context_limit_incompatible")
        recorded_datums = _text(_mapping(manifest.get("artifacts")).get("datums"))
        if not recorded_datums:
            errors.append("manifest_datums_artifact_missing")
        elif (manifest_path.parent / recorded_datums).resolve() != datums_path:
            errors.append("manifest_datums_path_mismatch")
        recorded_sha256 = _text(
            _mapping(manifest.get("artifact_sha256")).get("datums")
        )
        if not recorded_sha256:
            errors.append("manifest_datums_sha256_missing")
        elif datums_path.is_file() and recorded_sha256 != sha256_file(datums_path):
            errors.append("manifest_datums_sha256_mismatch")

    identities: set[str] = set()
    kind_counts: Counter[str] = Counter()
    effective_mass: Counter[str] = Counter()
    max_observed_context = 0
    datum_rejections: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        error = _datum_error(row, topk=expected_topk, max_context=max_context)
        target_id = _text(row.get("target_id"))
        if not target_id:
            error = error or "datum_target_id_missing"
        elif target_id in identities:
            error = error or "datum_target_id_duplicate"
        identities.add(target_id)
        if error:
            datum_rejections.append(
                {"row_index": index, "target_id": target_id, "reason": error}
            )
            continue
        kind = _text(row.get("kind"))
        kind_counts[kind] += 1
        effective_mass[kind] += float(row["row_weight"])
        max_observed_context = max(max_observed_context, len(row["input_ids"]))

    if not rows:
        errors.append("datums_empty")
    if datum_rejections:
        errors.append("datum_contract_rejections")
    if set(kind_counts) != {"repair", "preserve"}:
        errors.append("repair_and_preserve_datums_required")
    if not math.isclose(
        effective_mass.get("repair", 0.0),
        effective_mass.get("preserve", 0.0),
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        errors.append("effective_source_mass_not_1_to_1")
    expected_count = _integer(
        _mapping(manifest.get("counts")).get("candidate_datums")
    )
    if manifest and expected_count != len(rows):
        errors.append("manifest_datum_count_mismatch")

    return {
        "schema_version": PSD_TRAINING_INPUT_GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "error_count": len(errors),
        "errors": list(dict.fromkeys(errors)),
        "datums": {
            "path": str(datums_path),
            "sha256": sha256_file(datums_path) if datums_path.is_file() else None,
            "rows": len(rows),
            "by_kind": dict(sorted(kind_counts.items())),
            "effective_row_mass_by_kind": dict(sorted(effective_mass.items())),
            "max_observed_input_tokens": max_observed_context,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
        },
        "expected_topk": expected_topk,
        "max_context": max_context,
        "datum_rejections": datum_rejections,
    }
