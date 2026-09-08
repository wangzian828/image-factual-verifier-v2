#!/usr/bin/env python3
"""Verify that a full teacher rollout reached the audited SFT delivery boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "ifv-teacher-sft-final-delivery-v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def verify_teacher_sft_delivery(
    pipeline_dir: Path,
    *,
    expected_case_count: int,
) -> dict[str, Any]:
    pipeline_dir = pipeline_dir.expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def load_required(relative: str) -> dict[str, Any]:
        path = pipeline_dir / relative
        if not path.is_file():
            record(relative, False, "missing")
            return {}
        try:
            value = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            record(relative, False, f"invalid JSON: {exc}")
            return {}
        record(relative, True, "present")
        return value

    preparation = load_required("preparation.json")
    state = load_required("pipeline-state.json")
    classification = load_required("classification/final/classification.json")
    accepted = load_required("accepted-release/accepted_release_manifest.json")
    package = load_required("sft-training-package/MANIFEST.json")
    pipeline_summary = load_required("audits/pipeline-summary.json")

    case_count = int(preparation.get("case_count", 0) or 0)
    record(
        "full_case_count",
        case_count == expected_case_count,
        f"expected={expected_case_count} actual={case_count}",
    )
    record(
        "unbounded_full_run",
        bool(preparation) and preparation.get("limit") is None,
        f"limit={preparation.get('limit')!r}",
    )
    record(
        "pipeline_completed",
        state.get("status") == "completed",
        f"status={state.get('status')!r}",
    )

    classified_count = int(classification.get("case_count", 0) or 0)
    selected_count = int(classification.get("selected_case_count", 0) or 0)
    hard_count = int(classification.get("hard_case_count", 0) or 0)
    record(
        "classification_accounts_for_full_set",
        classified_count == expected_case_count
        and selected_count + hard_count == expected_case_count,
        (
            f"case_count={classified_count} selected={selected_count} "
            f"hard={hard_count}"
        ),
    )

    accepted_count = int(accepted.get("accepted_case_count", 0) or 0)
    record(
        "accepted_release_matches_selection",
        accepted_count > 0 and accepted_count == selected_count,
        f"accepted={accepted_count} selected={selected_count}",
    )

    required_nonempty_files = (
        "accepted-release/selected_episodes.jsonl",
        "accepted-release/trajectory_sft.jsonl",
        "sft-training-package/ms-swift-policy/train.jsonl",
        "sft-training-package/ms-swift-policy/validation.jsonl",
    )
    for relative in required_nonempty_files:
        record(
            relative,
            _nonempty(pipeline_dir / relative),
            "non-empty" if _nonempty(pipeline_dir / relative) else "missing or empty",
        )

    required_files = (
        "accepted-release/perception_trajectories.jsonl",
        "sft-training-package/ms-swift-policy/manifest.json",
        "sft-training-package/ms-swift-perception/manifest.json",
        "sft-training-package/audits/policy.json",
        "sft-training-package/audits/perception.json",
        "sft-training-package/training_plan.json",
    )
    for relative in required_files:
        record(
            relative,
            (pipeline_dir / relative).is_file(),
            "present" if (pipeline_dir / relative).is_file() else "missing",
        )

    counts = package.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    package_selected = int(counts.get("selected_release_cases", 0) or 0)
    policy_rows = int(counts.get("policy_rows", 0) or 0)
    action_only_rows = int(counts.get("action_only_rows", 0) or 0)
    long_holdout_rows = int(counts.get("long_holdout_rows", 0) or 0)
    record(
        "package_counts_match_accepted_release",
        bool(package)
        and accepted_count > 0
        and package_selected == accepted_count
        and policy_rows + action_only_rows + long_holdout_rows == accepted_count,
        (
            f"package_selected={package_selected} policy={policy_rows} "
            f"action_only={action_only_rows} long_holdout={long_holdout_rows} "
            f"accepted={accepted_count}"
        ),
    )

    audits = package.get("audits")
    audits = audits if isinstance(audits, Mapping) else {}
    policy_audit = audits.get("policy")
    perception_audit = audits.get("perception")
    record(
        "policy_sft_audit",
        isinstance(policy_audit, Mapping) and policy_audit.get("passed") is True,
        f"passed={getattr(policy_audit, 'get', lambda *_: None)('passed')!r}",
    )
    record(
        "perception_sft_audit",
        isinstance(perception_audit, Mapping)
        and perception_audit.get("passed") is True,
        f"passed={getattr(perception_audit, 'get', lambda *_: None)('passed')!r}",
    )

    training = package.get("training")
    training = training if isinstance(training, Mapping) else {}
    record(
        "handoff_stops_before_gpu_training",
        training.get("requested") is False and training.get("started") is False,
        (
            f"requested={training.get('requested')!r} "
            f"started={training.get('started')!r}"
        ),
    )
    record(
        "wrapper_trace_audits",
        pipeline_summary.get("all_trace_audits_passed") is True,
        f"passed={pipeline_summary.get('all_trace_audits_passed')!r}",
    )

    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_dir": str(pipeline_dir),
        "expected_case_count": expected_case_count,
        "delivery_scope": "full_train_set",
        "passed": not failed,
        "final_delivery": not failed,
        "failed_check_count": len(failed),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=8490)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_case_count < 1:
        raise SystemExit("--expected-case-count must be positive")

    result = verify_teacher_sft_delivery(
        args.pipeline_dir,
        expected_case_count=args.expected_case_count,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.pipeline_dir.expanduser().resolve()
        / "audits"
        / "final-delivery.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
