"""Offline contracts and target builders for IFV Privileged Self-Distillation.

This module intentionally does not generate hints or call a model.  It creates
the auditable boundary between a privileged repair process and a future PSD
trainer: the exported target contains token IDs and teacher distributions, never
the hint text or private repair reference.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import canonical_json, load_jsonl, require_new_or_empty, sha256_file, write_json, write_jsonl


PSD_HINT_AUDIT_SCHEMA_VERSION = "ifv-psd-hint-audit-v1"
PSD_TARGET_SCHEMA_VERSION = "ifv-psd-target-v1"
PSD_MANIFEST_SCHEMA_VERSION = "ifv-psd-target-manifest-v1"
PSD_TOPK_CACHE_SCHEMA_VERSION = "ifv-psd-teacher-topk-cache-v1"
PSD_TOPK_MATERIALIZATION_SCHEMA_VERSION = "ifv-psd-topk-materialization-v1"

REPAIR_TIERS = frozenset(
    {
        "causal_episode_pass",
        "local_pass_downstream",
        "unrepairable",
        "engineering_error",
    }
)
TRAINABLE_HINT_LEVELS = frozenset({1, 2, 3})

_BINARY_LABEL_RE = re.compile(
    r"(?<![\w-])(real|fake|true|false)(?![\w-])|真实|伪造|真图|假图",
    re.IGNORECASE,
)
_INTERNAL_TERM_RE = re.compile(
    r"\b(?:ground[\s_-]*truth|private[\s_-]*gold|expected[\s_-]*verdict|"
    r"reference[\s_-]*fact|acceptable[\s_-]*evidence)\b",
    re.IGNORECASE,
)
_PRIVATE_CONTAINER_KEYS = frozenset(
    {
        "private",
        "private_gold",
        "private_reference",
        "ground_truth",
        "oracle",
        "repair_oracle",
        "expected",
        "reference_facts",
        "acceptable_evidence",
    }
)
_PRIVATE_VALUE_KEYS = frozenset(
    {
        "expected_verdict",
        "gold_verdict",
        "expected_action",
        "expected_calls",
        "exact_action",
        "exact_query",
        "expected_query",
        "source_url",
        "source_urls",
        "evidence_id",
        "evidence_ids",
        "fact_id",
        "fact_ids",
        "claim_id",
        "claim_ids",
        "image_sha256",
        "exact_text",
        "verified_value",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            yield from _iter_strings(child)


def _private_values(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Collect values that a repair hint must not disclose."""

    values: list[tuple[str, str]] = []

    def walk(value: Any, path: str, inherited_private: bool = False) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key)
            key_lower = key.casefold()
            child_path = f"{path}.{key}" if path else key
            private = inherited_private or key_lower in _PRIVATE_CONTAINER_KEYS
            if private or key_lower in _PRIVATE_VALUE_KEYS:
                values.extend((child_path, item) for item in _iter_strings(child))
            if isinstance(child, Mapping):
                walk(child, child_path, private)

    walk(row, "")
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, value in values:
        normalised = _normalise(value)
        if len(normalised) < 4:
            continue
        identity = (path, normalised)
        if identity not in seen:
            seen.add(identity)
            unique.append((path, value))
    return unique


def _expected_tool_names(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    action = row.get("expected_action") or _mapping(
        row.get("private_reference")
    ).get("expected_action")
    names: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key.casefold() in {"name", "tool", "tool_name", "function"}:
                    if isinstance(child, str) and len(child.strip()) >= 4:
                        names.append((child_path, child.strip()))
                walk(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(action, "expected_action")
    return names


def audit_hint(
    row: Mapping[str, Any],
    *,
    hint: str | None = None,
    hint_level: int | None = None,
) -> dict[str, Any]:
    """Fail closed if a repair hint exposes private answer information.

    L1/L2 cannot name an expected tool. L3 may name a tool class, but still
    cannot repeat a private query, URL, value, Evidence ID, or verdict.
    """

    rendered_hint = _text(hint if hint is not None else row.get("hint"))
    raw_level = hint_level if hint_level is not None else row.get("hint_level", 1)
    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        level = 0

    errors: list[str] = []
    leak_paths: list[str] = []
    if not rendered_hint:
        errors.append("hint_empty")
    if len(rendered_hint) > 1200:
        errors.append("hint_too_long")
    if level not in {1, 2, 3, 4}:
        errors.append("hint_level_invalid")
    if _BINARY_LABEL_RE.search(rendered_hint):
        errors.append("binary_verdict_leak")
    if _INTERNAL_TERM_RE.search(rendered_hint):
        errors.append("internal_term_leak")

    normalised_hint = _normalise(rendered_hint)
    for path, value in _private_values(row):
        if _normalise(value) in normalised_hint:
            leak_paths.append(path)
    if level <= 2:
        for path, name in _expected_tool_names(row):
            if _normalise(name) in normalised_hint:
                leak_paths.append(path)
    if leak_paths:
        errors.append("private_value_leak")

    return {
        "schema_version": PSD_HINT_AUDIT_SCHEMA_VERSION,
        "passed": not errors,
        "hint_level": level,
        "hint_chars": len(rendered_hint),
        "hint_sha256": hashlib.sha256(rendered_hint.encode("utf-8")).hexdigest(),
        "errors": sorted(set(errors)),
        "leak_paths": sorted(set(leak_paths)),
    }


def _int_list(value: Any, *, field: str) -> list[int]:
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


def _prompt_ids(row: Mapping[str, Any], key: str) -> list[int]:
    direct = row.get(key)
    if direct is not None:
        return _int_list(direct, field=key)
    parent = "student" if key == "student_prompt_ids" else "teacher"
    return _int_list(_mapping(row.get(parent)).get("prompt_ids"), field=key)


def _completion_ids(row: Mapping[str, Any]) -> list[int]:
    if row.get("completion_ids") is not None:
        return _int_list(row.get("completion_ids"), field="completion_ids")
    return _int_list(
        _mapping(row.get("completion")).get("token_ids"),
        field="completion_ids",
    )


def _topk_entry(value: Any) -> tuple[int, float]:
    if isinstance(value, Mapping):
        token_id = value.get("token_id")
        probability = value.get("probability", value.get("prob"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        token_id, probability = value
    else:
        raise ValueError("top-k entry must be [token_id, probability] or an object")
    if isinstance(token_id, bool):
        raise ValueError("top-k token ID must be numeric")
    try:
        token = int(token_id)
        probability_value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("top-k entry contains an invalid token/probability") from exc
    if not math.isfinite(probability_value) or probability_value < 0.0:
        raise ValueError("top-k probability must be finite and non-negative")
    return token, probability_value


def validate_topk_by_position(
    completion_ids: Sequence[int],
    topk_by_position: Any,
    *,
    topk: int,
) -> None:
    if topk < 1:
        raise ValueError("topk must be positive")
    if not isinstance(topk_by_position, list):
        raise ValueError("topk_by_position must be a list")
    if len(topk_by_position) != len(completion_ids):
        raise ValueError("topk_by_position length must match completion_ids")
    for index, raw_entries in enumerate(topk_by_position):
        if not isinstance(raw_entries, list) or len(raw_entries) != topk:
            raise ValueError(
                f"topk_by_position[{index}] must contain exactly {topk} entries"
            )
        entries = [_topk_entry(item) for item in raw_entries]
        if len({token for token, _ in entries}) != topk:
            raise ValueError(f"topk_by_position[{index}] has duplicate tokens")
        probability_sum = sum(probability for _, probability in entries)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"topk_by_position[{index}] probabilities must sum to one"
            )


def _assert_trainable_source(row: Mapping[str, Any]) -> None:
    split = _text(
        row.get("split") or _mapping(row.get("source_metadata")).get("split")
    ).casefold()
    if split in {"test", "holdout"}:
        raise ValueError(f"PSD source split is forbidden: {split}")
    if bool(row.get("training_prohibited")):
        raise ValueError("PSD source is training_prohibited")
    if bool(row.get("engineering_error")) or _text(row.get("status")) == "engineering_error":
        raise ValueError("engineering_error cannot become a PSD target")


def _repair_tier(row: Mapping[str, Any]) -> str:
    tier = _text(row.get("repair_tier") or row.get("tier"))
    if tier:
        return tier
    verification = _mapping(row.get("verification"))
    if verification.get("full_episode_pass") is True:
        return "causal_episode_pass"
    if verification.get("local_pass") is True:
        return "local_pass_downstream"
    return ""


def _target_id(
    *,
    kind: str,
    case_id: str,
    episode_id: str,
    step_id: str,
    completion_ids: Sequence[int],
) -> str:
    payload = {
        "kind": kind,
        "case_id": case_id,
        "episode_id": episode_id,
        "step_id": step_id,
        "completion_sha256": hashlib.sha256(
            canonical_json(list(completion_ids)).encode("utf-8")
        ).hexdigest(),
    }
    return "psd:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _topk_payload(
    row: Mapping[str, Any],
    completion_ids: Sequence[int],
    *,
    topk: int,
) -> tuple[str, list[Any]]:
    value = row.get("teacher_topk_by_position")
    if value is None:
        return "pending_topk", []
    validate_topk_by_position(completion_ids, value, topk=topk)
    return "complete", value


def build_repair_target(
    row: Mapping[str, Any],
    *,
    topk: int = 20,
    allow_local_only: bool = False,
) -> dict[str, Any]:
    _assert_trainable_source(row)
    if row.get("accepted") is False:
        raise ValueError("repair row is not accepted")
    tier = _repair_tier(row)
    if tier not in REPAIR_TIERS:
        raise ValueError("repair row needs a valid repair_tier")
    if tier != "causal_episode_pass" and not allow_local_only:
        raise ValueError(f"repair tier is not enabled for primary PSD: {tier}")

    hint_record = _mapping(row.get("hint_record"))
    hint = _text(hint_record.get("text") or row.get("hint"))
    raw_hint_level = hint_record.get("level") or row.get("hint_level", 1)
    hint_audit = audit_hint(row, hint=hint, hint_level=int(raw_hint_level))
    if not hint_audit["passed"]:
        raise ValueError("hint audit failed: " + ",".join(hint_audit["errors"]))
    if int(hint_audit["hint_level"]) not in TRAINABLE_HINT_LEVELS:
        raise ValueError("only L1/L2/L3 hints can create PSD targets")

    student_prompt_ids = _prompt_ids(row, "student_prompt_ids")
    teacher_prompt_ids = _prompt_ids(row, "teacher_prompt_ids")
    completion_ids = _completion_ids(row)
    if student_prompt_ids == teacher_prompt_ids:
        raise ValueError("teacher prompt must differ from student prompt")

    case_id = _text(row.get("case_id"))
    episode_id = _text(row.get("episode_id"))
    repair_site = _mapping(row.get("repair_site"))
    step_id = _text(row.get("repair_step_id") or row.get("step_id") or repair_site.get("step_id"))
    if not case_id or not episode_id or not step_id:
        raise ValueError("repair row requires case_id, episode_id and repair step ID")
    verification = _mapping(row.get("verification"))
    if tier == "causal_episode_pass" and verification.get("full_episode_pass") is not True:
        raise ValueError("causal_episode_pass requires verification.full_episode_pass")

    row_weight = float(row.get("row_weight", 1.0))
    if not math.isfinite(row_weight) or row_weight <= 0:
        raise ValueError("row_weight must be finite and positive")
    target_status, teacher_topk = _topk_payload(row, completion_ids, topk=topk)
    return {
        "schema_version": PSD_TARGET_SCHEMA_VERSION,
        "target_id": _target_id(
            kind="repair",
            case_id=case_id,
            episode_id=episode_id,
            step_id=step_id,
            completion_ids=completion_ids,
        ),
        "kind": "repair",
        "target_status": target_status,
        "case_id": case_id,
        "episode_id": episode_id,
        "repair_site": {
            "step_id": step_id,
            "stage": _text(row.get("stage") or repair_site.get("stage")),
            "example_type": _text(
                row.get("example_type") or repair_site.get("example_type")
            ),
        },
        "repair_tier": tier,
        "hint_level": int(hint_audit["hint_level"]),
        "hint_sha256": hint_audit["hint_sha256"],
        "student_prompt_ids": student_prompt_ids,
        "teacher_prompt_ids": teacher_prompt_ids,
        "completion_ids": completion_ids,
        "teacher_topk_by_position": teacher_topk,
        "row_weight": row_weight,
        "verification": {
            "local_pass": verification.get("local_pass"),
            "full_episode_pass": verification.get("full_episode_pass"),
            "strict_trace_audit_pass": verification.get("strict_trace_audit_pass"),
        },
        "source": {
            "source_run_id": _text(row.get("source_run_id")),
            "runtime_commit": _text(row.get("runtime_commit")),
            "source_trace_sha256": _text(row.get("source_trace_sha256")),
        },
    }


def _preservation_steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_steps = row.get("preservation_steps")
    if raw_steps is None:
        raw_steps = _mapping(row.get("preserve_rollout")).get("steps")
    if raw_steps is None:
        raw_steps = [row]
    if not isinstance(raw_steps, list):
        raise ValueError("preservation_steps must be a list")
    return [step for step in raw_steps if isinstance(step, Mapping)]


def build_preservation_targets(
    row: Mapping[str, Any],
    *,
    topk: int = 20,
) -> list[dict[str, Any]]:
    _assert_trainable_source(row)
    verified_pass = bool(
        row.get("verified_full_task")
        or row.get("base_passed")
        or row.get("base_passed_outright")
        or row.get("class") == "base_pass_preserve"
        or row.get("source") == "base_pass_trace"
    )
    if not verified_pass:
        raise ValueError("preservation row is not a verified policy pass")
    steps = _preservation_steps(row)
    if not steps:
        raise ValueError("preservation row has no usable steps")

    case_id = _text(row.get("case_id"))
    episode_id = _text(row.get("episode_id"))
    if not case_id or not episode_id:
        raise ValueError("preservation row requires case_id and episode_id")
    per_step_weight = float(row.get("row_weight", 1.0)) / len(steps)
    if not math.isfinite(per_step_weight) or per_step_weight <= 0:
        raise ValueError("preservation row_weight must be finite and positive")

    targets: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        merged = {**row, **step}
        student_prompt_ids = _prompt_ids(merged, "student_prompt_ids")
        completion_ids = _completion_ids(merged)
        target_status, teacher_topk = _topk_payload(
            merged, completion_ids, topk=topk
        )
        step_id = _text(
            merged.get("step_id")
            or merged.get("preservation_step_id")
            or f"{episode_id}:preserve:{index}"
        )
        targets.append(
            {
                "schema_version": PSD_TARGET_SCHEMA_VERSION,
                "target_id": _target_id(
                    kind="preserve",
                    case_id=case_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    completion_ids=completion_ids,
                ),
                "kind": "preserve",
                "target_status": target_status,
                "case_id": case_id,
                "episode_id": episode_id,
                "repair_site": {"step_id": step_id, "stage": "", "example_type": ""},
                "repair_tier": "preservation",
                "hint_level": 0,
                "hint_sha256": "",
                "student_prompt_ids": student_prompt_ids,
                "teacher_prompt_ids": student_prompt_ids,
                "completion_ids": completion_ids,
                "teacher_topk_by_position": teacher_topk,
                "row_weight": per_step_weight,
                "verification": {
                    "local_pass": True,
                    "full_episode_pass": True,
                    "strict_trace_audit_pass": row.get("strict_trace_audit_pass"),
                },
                "source": {
                    "source_run_id": _text(row.get("source_run_id")),
                    "runtime_commit": _text(row.get("runtime_commit")),
                    "source_trace_sha256": _text(row.get("source_trace_sha256")),
                },
            }
        )
    return targets


def audit_psd_hints(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(load_jsonl(input_path)):
        rows.append(
            {
                "row_index": index,
                "case_id": _text(row.get("case_id")),
                "episode_id": _text(row.get("episode_id")),
                "repair_step_id": _text(row.get("repair_step_id") or row.get("step_id")),
                **audit_hint(row),
            }
        )
    write_jsonl(output_path, rows)
    return {
        "schema_version": PSD_HINT_AUDIT_SCHEMA_VERSION,
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "passed": sum(item["passed"] for item in rows),
        "rejected": sum(not item["passed"] for item in rows),
    }


def build_psd_target_package(
    *,
    repairs_path: Path,
    preservation_path: Path | None,
    output_dir: Path,
    topk: int = 20,
    allow_local_only: bool = False,
) -> dict[str, Any]:
    require_new_or_empty(output_dir)
    repair_targets: list[dict[str, Any]] = []
    preservation_targets: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, row in enumerate(load_jsonl(repairs_path)):
        try:
            repair_targets.append(
                build_repair_target(
                    row, topk=topk, allow_local_only=allow_local_only
                )
            )
        except ValueError as exc:
            rejected.append(
                {
                    "kind": "repair",
                    "row_index": index,
                    "case_id": _text(row.get("case_id")),
                    "episode_id": _text(row.get("episode_id")),
                    "reason": str(exc),
                }
            )

    if preservation_path is not None:
        for index, row in enumerate(load_jsonl(preservation_path)):
            try:
                preservation_targets.extend(build_preservation_targets(row, topk=topk))
            except ValueError as exc:
                rejected.append(
                    {
                        "kind": "preserve",
                        "row_index": index,
                        "case_id": _text(row.get("case_id")),
                        "episode_id": _text(row.get("episode_id")),
                        "reason": str(exc),
                    }
                )

    targets = repair_targets + preservation_targets
    target_ids = [item["target_id"] for item in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("PSD target IDs must be unique")

    write_jsonl(output_dir / "repair_targets.jsonl", repair_targets)
    write_jsonl(output_dir / "preservation_targets.jsonl", preservation_targets)
    write_jsonl(output_dir / "targets.jsonl", targets)
    write_jsonl(output_dir / "rejections.jsonl", rejected)

    pending_topk = sum(item["target_status"] == "pending_topk" for item in targets)
    manifest = {
        "schema_version": PSD_MANIFEST_SCHEMA_VERSION,
        "topk": topk,
        "allow_local_only": allow_local_only,
        "source": {
            "repairs": str(repairs_path),
            "repairs_sha256": sha256_file(repairs_path),
            "preservation": str(preservation_path) if preservation_path else None,
            "preservation_sha256": (
                sha256_file(preservation_path) if preservation_path else None
            ),
        },
        "counts": {
            "repair_targets": len(repair_targets),
            "preservation_targets": len(preservation_targets),
            "targets": len(targets),
            "pending_topk": pending_topk,
            "complete_topk": len(targets) - pending_topk,
            "rejections": len(rejected),
        },
        "repair_tiers": dict(
            sorted(Counter(item["repair_tier"] for item in repair_targets).items())
        ),
        "artifacts": {
            "repair_targets": "repair_targets.jsonl",
            "preservation_targets": "preservation_targets.jsonl",
            "targets": "targets.jsonl",
            "rejections": "rejections.jsonl",
        },
        "status": (
            "ready_for_topk_cache"
            if targets and pending_topk
            else "ready_for_training"
            if targets
            else "empty"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json(list(token_ids)).encode("utf-8")
    ).hexdigest()


def _cache_topk(row: Mapping[str, Any]) -> Any:
    value = row.get("teacher_topk_by_position")
    return value if value is not None else row.get("topk_by_position")


def materialize_psd_topk_cache(
    *,
    targets_path: Path,
    cache_path: Path,
    output_dir: Path,
    topk: int = 20,
) -> dict[str, Any]:
    """Attach a verified teacher top-k cache to a PSD target package.

    The cache is intentionally keyed by target ID *and* token hashes.  A cache
    row generated for a similar state, a different template, or a different
    completion is unusable.  If any target cannot be verified, no
    ``targets.jsonl`` training artifact is emitted.
    """

    require_new_or_empty(output_dir)
    target_rows = load_jsonl(targets_path)
    if not target_rows:
        raise ValueError("PSD target input is empty")

    cache_by_target: dict[str, Mapping[str, Any]] = {}
    cache_rejections: list[dict[str, Any]] = []
    for row_index, row in enumerate(load_jsonl(cache_path)):
        target_id = _text(row.get("target_id"))
        if not target_id:
            cache_rejections.append(
                {
                    "kind": "cache_row",
                    "row_index": row_index,
                    "reason": "cache_row_missing_target_id",
                }
            )
            continue
        if target_id in cache_by_target:
            cache_rejections.append(
                {
                    "kind": "cache_row",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": "duplicate_cache_target_id",
                }
            )
            continue
        cache_by_target[target_id] = row

    resolved: list[dict[str, Any]] = []
    target_rejections: list[dict[str, Any]] = []
    consumed_target_ids: set[str] = set()
    target_ids: set[str] = set()
    for row_index, target in enumerate(target_rows):
        target_id = _text(target.get("target_id"))
        if not target_id:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "reason": "target_missing_target_id",
                }
            )
            continue
        if target_id in target_ids:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": "duplicate_target_id",
                }
            )
            continue
        target_ids.add(target_id)
        try:
            teacher_prompt_ids = _prompt_ids(target, "teacher_prompt_ids")
            completion_ids = _completion_ids(target)
            cache_row = cache_by_target.get(target_id)
            if cache_row is None:
                raise ValueError("cache_entry_missing")
            teacher_model = _text(
                cache_row.get("teacher_model") or cache_row.get("model")
            )
            if not teacher_model:
                raise ValueError("cache_teacher_model_missing")
            if _text(cache_row.get("teacher_prompt_sha256")) != _token_ids_sha256(
                teacher_prompt_ids
            ):
                raise ValueError("cache_teacher_prompt_sha256_mismatch")
            if _text(cache_row.get("completion_sha256")) != _token_ids_sha256(
                completion_ids
            ):
                raise ValueError("cache_completion_sha256_mismatch")
            cache_topk = _cache_topk(cache_row)
            validate_topk_by_position(completion_ids, cache_topk, topk=topk)
        except ValueError as exc:
            target_rejections.append(
                {
                    "kind": "target",
                    "row_index": row_index,
                    "target_id": target_id,
                    "reason": str(exc),
                }
            )
            continue

        consumed_target_ids.add(target_id)
        resolved_target = dict(target)
        resolved_target["target_status"] = "complete"
        resolved_target["teacher_topk_by_position"] = cache_topk
        resolved_target["teacher"] = {
            "model": teacher_model,
            "teacher_prompt_sha256": _token_ids_sha256(teacher_prompt_ids),
            "completion_sha256": _token_ids_sha256(completion_ids),
        }
        resolved.append(resolved_target)

    cache_orphans = [
        {
            "target_id": target_id,
            "reason": "cache_target_not_present_in_input",
        }
        for target_id in sorted(set(cache_by_target) - consumed_target_ids)
    ]
    rejections = cache_rejections + target_rejections
    all_targets_resolved = len(resolved) == len(target_rows) and not rejections

    write_jsonl(output_dir / "resolved_targets.jsonl", resolved)
    write_jsonl(output_dir / "topk_cache_rejections.jsonl", rejections)
    write_jsonl(output_dir / "topk_cache_orphans.jsonl", cache_orphans)
    artifacts = {
        "resolved_targets": "resolved_targets.jsonl",
        "topk_cache_rejections": "topk_cache_rejections.jsonl",
        "topk_cache_orphans": "topk_cache_orphans.jsonl",
    }
    if all_targets_resolved:
        repair_targets = [
            row for row in resolved if _text(row.get("kind")) == "repair"
        ]
        preservation_targets = [
            row for row in resolved if _text(row.get("kind")) == "preserve"
        ]
        write_jsonl(output_dir / "repair_targets.jsonl", repair_targets)
        write_jsonl(
            output_dir / "preservation_targets.jsonl",
            preservation_targets,
        )
        write_jsonl(output_dir / "targets.jsonl", resolved)
        artifacts.update(
            {
                "repair_targets": "repair_targets.jsonl",
                "preservation_targets": "preservation_targets.jsonl",
                "targets": "targets.jsonl",
            }
        )

    manifest = {
        "schema_version": PSD_TOPK_MATERIALIZATION_SCHEMA_VERSION,
        "topk": topk,
        "source": {
            "targets": str(targets_path),
            "targets_sha256": sha256_file(targets_path),
            "cache": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
        },
        "counts": {
            "input_targets": len(target_rows),
            "resolved_targets": len(resolved),
            "cache_rejections": len(cache_rejections),
            "target_rejections": len(target_rejections),
            "cache_orphans": len(cache_orphans),
        },
        "artifacts": artifacts,
        "status": (
            "ready_for_training"
            if all_targets_resolved
            else "blocked_topk_cache"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
