"""Build public PSD candidate queues from postprocessed Qwen rollouts.

This is deliberately an intermediate layer.  A failed rollout is not itself a
PSD target and this module never claims that its final action is the causal
error.  It records the real model-visible state at a conservative repair
anchor, then leaves causal localization, privileged hinting, teacher decoding,
and verifier-gated repair to later stages.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)


PSD_CANDIDATE_SCHEMA_VERSION = "ifv-psd-candidate-v1"
PSD_CANDIDATE_MANIFEST_SCHEMA_VERSION = "ifv-psd-candidate-manifest-v1"

_FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "acceptable_evidence",
        "expected_status",
        "ground_truth",
        "teacher_score",
        "process_metrics",
        "source_access_policy",
        "excluded_domains",
        "excluded_urls",
    }
)
_REJECTED_ACTION_TYPES = frozenset(
    {"planning_revision", "format_error", "output_rejected", "policy_replan"}
)
_STAGE_EXAMPLE_TYPES = {
    "image_only_investigation": "react",
    "image_only_discrepancy_investigation": "react",
    "image_only_reflection": "reflection",
    "image_only_judgment": "judgment",
    "image_only_evidence_decision": "evidence_decision",
    "image_only_discrepancy_decision": "discrepancy_decision",
    "image_only_query_concept_extraction": "query_concept_extraction",
    "image_only_query_replan": "query_replan",
    "image_only_route_local_replan": "route_local_replan",
    "image_only_planning": "planning",
    "image_only_attribution_planning": "planning",
    "image_account_planning": "image_account_planning",
    "image_only_discrepancy_judgment": "judgment",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assert_public(value: Any, *, path: str = "") -> None:
    """Reject evaluator/private state before it becomes a candidate artifact."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in _FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private/evaluator field in policy snapshot: {child_path}")
            _assert_public(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public(child, path=f"{path}[{index}]")


def _stable_step_id(
    *,
    episode_id: str,
    example_type: str,
    source_step_index: int,
    interaction_id: str,
) -> str:
    return (
        f"{episode_id}:{example_type}:{interaction_id}"
        if interaction_id
        else f"{episode_id}:{example_type}:{source_step_index + 1}"
    )


def _policy_steps(
    trace: Mapping[str, Any],
    *,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Project all usable policy turns, including rejected protocol turns."""

    state = _mapping(trace.get("state"))
    projected: list[dict[str, Any]] = []
    for source_step_index, raw_step in enumerate(_rows(state.get("all_steps"))):
        step = _mapping(raw_step)
        example_type = _STAGE_EXAMPLE_TYPES.get(_text(step.get("stage")))
        metadata = _mapping(step.get("metadata"))
        policy_input = metadata.get("policy_input")
        policy_action = metadata.get("policy_action")
        if (
            example_type is None
            or not isinstance(policy_input, Mapping)
            or not isinstance(policy_action, Mapping)
        ):
            continue
        _assert_public(policy_input, path="policy_input")
        _assert_public(policy_action, path="policy_action")
        action_type = _text(step.get("action_type"))
        protocol_rejected = (
            action_type in _REJECTED_ACTION_TYPES
            or _text(metadata.get("error_class")) == "protocol_error"
            or bool(_text(metadata.get("rejection_reason")))
        )
        interaction_id = _text(metadata.get("interaction_id"))
        projected.append(
            {
                "step_id": _stable_step_id(
                    episode_id=episode_id,
                    example_type=example_type,
                    source_step_index=source_step_index,
                    interaction_id=interaction_id,
                ),
                "source_step_index": source_step_index,
                "stage": _text(step.get("stage")),
                "example_type": example_type,
                "interaction_id": interaction_id,
                "action_type": action_type,
                "protocol_rejected": protocol_rejected,
                "model_visible": {"policy_input": dict(policy_input)},
                "observed_policy_action": dict(policy_action),
            }
        )
    return projected


def _candidate_id(
    *,
    kind: str,
    case_id: str,
    episode_id: str,
    step_id: str,
    source_trace_sha256: str,
) -> str:
    payload = {
        "kind": kind,
        "case_id": case_id,
        "episode_id": episode_id,
        "step_id": step_id,
        "source_trace_sha256": source_trace_sha256,
    }
    return "psd-candidate:" + hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _safe_trace_path(run_dir: Path, recorded: str, episode_id: str) -> Path:
    if recorded:
        candidate = (run_dir / recorded).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"trace_path escapes run directory: {recorded}") from exc
        return candidate
    safe_episode_id = "".join(
        character if character.isalnum() or character in "-_."
        else "_"
        for character in episode_id
    )
    candidates = (
        run_dir / "traces" / f"{episode_id}.json",
        run_dir / "traces" / f"{safe_episode_id}.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


def _trace_case_id(trace: Mapping[str, Any]) -> str:
    state = _mapping(trace.get("state"))
    runtime_case = _mapping(state.get("runtime_case"))
    return _text(
        runtime_case.get("case_id")
        or trace.get("case_id")
        or state.get("case_id")
    )


def _load_train_case_allowlist(path: Path) -> dict[str, str]:
    """Load an explicit train-case manifest, not a generic all-split manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"train case manifest does not exist: {path}")

    if path.suffix.casefold() == ".jsonl":
        raw_rows: Iterable[Any] = load_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, Mapping):
            for key in ("cases", "items", "records"):
                if isinstance(payload.get(key), list):
                    raw_rows = payload[key]
                    break
            else:
                raw_rows = [
                    {"case_id": key, "split": value}
                    for key, value in payload.items()
                    if isinstance(key, str)
                ]
        else:
            raise ValueError("train case manifest must be a JSON object/list or JSONL")

    allowlist: dict[str, str] = {}
    for index, raw_row in enumerate(raw_rows):
        if isinstance(raw_row, str):
            case_id, split = _text(raw_row), "train"
        else:
            row = _mapping(raw_row)
            case_id = _text(
                row.get("case_id")
                or row.get("candidate_id")
                or row.get("assignment_id")
                or row.get("image_id")
                or row.get("id")
            )
            raw_split = row.get("split")
            if isinstance(raw_split, Mapping):
                raw_split = raw_split.get("name")
            split = _text(raw_split) or "train"
        if not case_id:
            raise ValueError(f"train case manifest row {index} lacks a case ID")
        if case_id in allowlist and allowlist[case_id] != split:
            raise ValueError(f"conflicting split for train case {case_id}")
        allowlist[case_id] = split
    if not allowlist:
        raise ValueError("train case manifest is empty")
    return allowlist


def _source_metadata(run_manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_run_id": _text(run_manifest.get("run_id")),
        "runtime_commit": _text(run_manifest.get("git_commit")),
    }


def _rejection(
    *,
    row_index: int,
    reward: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "case_id": _text(reward.get("case_id")),
        "episode_id": _text(reward.get("episode_id")),
        "reason": reason,
    }


def build_psd_candidate_package(
    *,
    run_dir: Path,
    train_cases_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Produce the queues consumed by later PSD localization and repair stages.

    ``train_cases_path`` is intentionally an allowlist.  Merely having a
    non-prohibited rollout row is insufficient to admit a case, so a test case
    can never reach the repair or preservation queue by accident.
    """

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {run_dir}")
    rewards_path = run_dir / "post_rollout_rewards.jsonl"
    if not rewards_path.is_file():
        raise FileNotFoundError(f"missing post-rollout rewards: {rewards_path}")
    require_new_or_empty(output_dir)

    train_cases = _load_train_case_allowlist(train_cases_path)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path) if run_manifest_path.is_file() else {}
    group_paths = {
        _text(item.get("episode_id")): _text(item.get("trace_path"))
        for item in load_jsonl(run_dir / "rollout_groups.jsonl")
        if _text(item.get("episode_id"))
    }

    repair_candidates: list[dict[str, Any]] = []
    preservation_candidates: list[dict[str, Any]] = []
    engineering_requeue: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    source = _source_metadata(run_manifest)

    for row_index, reward in enumerate(load_jsonl(rewards_path)):
        case_id = _text(reward.get("case_id"))
        episode_id = _text(reward.get("episode_id")) or case_id
        split = train_cases.get(case_id)
        if not case_id or not episode_id:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="missing_case_or_episode_id",
                )
            )
            continue
        if split is None:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="case_not_in_explicit_train_allowlist",
                )
            )
            continue
        if split.casefold() != "train":
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason=f"source_split_forbidden:{split}",
                )
            )
            continue
        if bool(reward.get("training_prohibited")):
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason="training_prohibited",
                )
            )
            continue
        if bool(reward.get("fatal_engineering_error")):
            engineering_requeue.append(
                {
                    "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "queue_reason": "fatal_engineering_error",
                    "source": source,
                }
            )
            continue

        try:
            trace_path = _safe_trace_path(
                run_dir,
                group_paths.get(episode_id, ""),
                episode_id,
            )
            if not trace_path.is_file():
                raise FileNotFoundError("canonical_trace_missing")
            trace = load_json(trace_path)
            trace_case_id = _trace_case_id(trace)
            if trace_case_id and trace_case_id != case_id:
                raise ValueError(
                    f"trace_case_id_mismatch:{trace_case_id}!={case_id}"
                )
            trace_sha256 = sha256_file(trace_path)
            steps = _policy_steps(trace, episode_id=episode_id)
            if not steps:
                raise ValueError("trace_has_no_usable_policy_steps")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejections.append(
                _rejection(
                    row_index=row_index,
                    reward=reward,
                    reason=f"trace_rejected:{exc}",
                )
            )
            continue

        strict_audit_pass = bool(reward.get("strict_trace_audit_pass"))
        classification_correct = bool(reward.get("classification_correct"))
        trace_source = {
            **source,
            "source_trace_path": trace_path.relative_to(run_dir).as_posix(),
            "source_trace_sha256": trace_sha256,
        }
        if classification_correct and strict_audit_pass:
            preservation_candidates.append(
                {
                    "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                    "candidate_id": _candidate_id(
                        kind="preserve",
                        case_id=case_id,
                        episode_id=episode_id,
                        step_id="episode",
                        source_trace_sha256=trace_sha256,
                    ),
                    "class": "base_pass_preserve",
                    "candidate_status": "ready_for_teacher_collection",
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "verified_full_task": True,
                    "strict_trace_audit_pass": True,
                    "preservation_steps": steps,
                    "source": trace_source,
                }
            )
            continue

        protocol_steps = [step for step in steps if step["protocol_rejected"]]
        if protocol_steps:
            repair_signal, repair_site = "protocol_rejection", protocol_steps[0]
        elif not strict_audit_pass:
            repair_signal, repair_site = "strict_trace_audit_failure", steps[-1]
        else:
            repair_signal, repair_site = "terminal_outcome_mismatch", steps[-1]

        repair_candidates.append(
            {
                "schema_version": PSD_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": _candidate_id(
                    kind="repair",
                    case_id=case_id,
                    episode_id=episode_id,
                    step_id=repair_site["step_id"],
                    source_trace_sha256=trace_sha256,
                ),
                "class": "repair_seed",
                "candidate_status": "needs_privileged_localization",
                "case_id": case_id,
                "episode_id": episode_id,
                "repair_signal": repair_signal,
                "strict_trace_audit_failure_codes": [
                    _text(value)
                    for value in reward.get("strict_trace_audit_failure_codes", [])
                    if _text(value)
                ],
                "repair_site": repair_site,
                "source": trace_source,
            }
        )

    candidate_ids = [
        row["candidate_id"]
        for row in repair_candidates + preservation_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("PSD candidate IDs must be unique")

    write_jsonl(output_dir / "repair_candidates.jsonl", repair_candidates)
    write_jsonl(
        output_dir / "preservation_candidates.jsonl",
        preservation_candidates,
    )
    write_jsonl(output_dir / "engineering_requeue.jsonl", engineering_requeue)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    manifest = {
        "schema_version": PSD_CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "source": {
            "run_dir": str(run_dir),
            "post_rollout_rewards": str(rewards_path),
            "post_rollout_rewards_sha256": sha256_file(rewards_path),
            "train_cases": str(train_cases_path),
            "train_cases_sha256": sha256_file(train_cases_path),
            **source,
        },
        "counts": {
            "repair_candidates": len(repair_candidates),
            "preservation_candidates": len(preservation_candidates),
            "engineering_requeue": len(engineering_requeue),
            "rejections": len(rejections),
        },
        "repair_signals": dict(
            sorted(
                Counter(
                    _text(item["repair_signal"]) for item in repair_candidates
                ).items()
            )
        ),
        "artifacts": {
            "repair_candidates": "repair_candidates.jsonl",
            "preservation_candidates": "preservation_candidates.jsonl",
            "engineering_requeue": "engineering_requeue.jsonl",
            "rejections": "rejections.jsonl",
        },
        "status": (
            "ready_for_privileged_localization"
            if repair_candidates or preservation_candidates
            else "empty"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
