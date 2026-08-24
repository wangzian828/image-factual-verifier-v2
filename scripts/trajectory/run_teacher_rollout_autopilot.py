#!/usr/bin/env python3
"""Run the recoverable frozen-teacher trajectory pipeline for one train manifest.

The input manifest may contain evaluator-private facts, labels, and construction
metadata.  This program deliberately creates two separate projections:

* ``runtime-release/`` contains only the exact three-field image runtime contract.
* ``private-gold/`` is used only after a terminal rollout by the frozen SFT judge.

The initial and reroll rollout phases preserve every trace.  Engineering failures
are retried with a fresh seed without re-running a successful case.  SFT rejection
is a quality outcome, not an engineering failure, and is handled by the optional
reroll phase after the initial audit.

Run this on gpu-13 through ``scripts/server/run_gpu13.sh``.  It is intentionally
resumable: the output directory is also the durable state and provenance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.trajectory.stage_accepted_teacher_release import stage_release
from src.eval.release_adapter import (
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    INPUT_MODE,
    RELEASE_SCHEMA_VERSION,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)


SCHEMA_VERSION = "ifv-teacher-rollout-autopilot-v1"
TERMINAL_RUN_STATUSES = frozenset({"completed", "completed_with_errors"})
VALID_VERDICTS = frozenset({"real", "fake"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    pending.replace(path)


def _write_case_list(path: Path, case_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{case_id}\n" for case_id in case_ids),
        encoding="utf-8",
    )


def _relative_file(root: Path, rendered: str, *, field: str) -> Path:
    value = Path(rendered)
    if value.is_absolute():
        raise ValueError(f"{field} must be relative: {rendered}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes dataset root: {rendered}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _case_id(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("unified_case_id")
        or row.get("archive_source_version_id")
        or row.get("case_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("training manifest row lacks unified_case_id")
    if len(value) > 200:
        raise ValueError(f"runtime case_id exceeds 200 characters: {value!r}")
    return value


def _expected_verdict(row: Mapping[str, Any]) -> str:
    supplied = str(row.get("expected_verdict") or row.get("gold_verdict") or "").lower()
    if supplied in VALID_VERDICTS:
        return supplied
    factual_status = str(row.get("factual_status") or "").strip().lower()
    mapping = {"supported": "real", "refuted": "fake"}
    if factual_status not in mapping:
        raise ValueError(
            "private target requires factual_status=supported|refuted or "
            "expected_verdict=real|fake for case "
            f"{_case_id(row)}"
        )
    return mapping[factual_status]


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(source) != _sha256_file(destination):
            raise ValueError(f"existing runtime asset differs from source: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_runtime_release(
    *,
    dataset_root: Path,
    train_manifest: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Project a train manifest into isolated runtime and private-gold artifacts."""

    dataset_root = dataset_root.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not train_manifest.is_file():
        raise FileNotFoundError(f"train manifest does not exist: {train_manifest}")
    try:
        train_manifest.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("train manifest must be under dataset root") from exc

    release_root = output_dir / "runtime-release"
    benchmark_path = release_root / "runtime_input" / "cases.jsonl"
    gold_path = output_dir / "private-gold" / "private_gold.jsonl"
    preparation_path = output_dir / "preparation.json"
    source_sha256 = _sha256_file(train_manifest)

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when supplied")
    if preparation_path.is_file():
        prior = _read_json(preparation_path)
        if (
            prior.get("train_manifest") != str(train_manifest)
            or prior.get("train_manifest_sha256") != source_sha256
            or prior.get("limit") != limit
        ):
            raise ValueError(
                "existing pipeline output was prepared from a different train manifest"
            )
        if not benchmark_path.is_file() or not gold_path.is_file():
            raise FileNotFoundError("existing preparation is missing its projections")
        return prior

    if output_dir.exists() and any(output_dir.iterdir()):
        # Operational launchers commonly create ``logs/`` before handing off
        # the foreground process.  It contains no data artifact and must not
        # make a brand-new pipeline output unusable.  Any other pre-existing
        # entry remains a fail-closed collision.
        unexpected_entries = [
            path.name for path in output_dir.iterdir() if path.name != "logs"
        ]
        if unexpected_entries:
            raise FileExistsError(
                "pipeline output must be new/empty (apart from logs/) or contain "
                f"preparation.json: {output_dir}; unexpected={unexpected_entries}"
            )

    rows = _read_jsonl(train_manifest)
    if not rows:
        raise ValueError(f"train manifest is empty: {train_manifest}")
    seen: set[str] = set()
    prepared: list[tuple[str, dict[str, Any], Path]] = []
    for row in rows:
        case_id = _case_id(row)
        if case_id in seen:
            raise ValueError(f"duplicate case_id in train manifest: {case_id}")
        seen.add(case_id)
        if str(row.get("split") or "train").strip().lower() != "train":
            raise ValueError(f"non-train row found in train manifest: {case_id}")
        image_value = str(
            row.get("unified_image_path") or row.get("local_image_path") or ""
        ).strip()
        if not image_value:
            raise ValueError(f"training manifest row lacks unified_image_path: {case_id}")
        prepared.append(
            (
                case_id,
                dict(row),
                _relative_file(dataset_root, image_value, field="unified_image_path"),
            )
        )
    prepared.sort(key=lambda item: item[0])
    if limit is not None:
        prepared = prepared[:limit]

    runtime_rows: list[dict[str, str]] = []
    gold_rows: list[dict[str, Any]] = []
    materialization: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for index, (case_id, private_row, source_image) in enumerate(prepared, start=1):
        suffix = source_image.suffix.lower() or ".jpg"
        asset_relative = Path("assets") / f"{index:05d}{suffix}"
        destination = release_root / "runtime_input" / asset_relative
        method = _link_or_copy(source_image, destination)
        materialization[method] = materialization.get(method, 0) + 1
        runtime_rows.append(
            {
                "case_id": case_id,
                "image_path": asset_relative.as_posix(),
                "image_sha256": _sha256_file(destination),
            }
        )
        private_projection = dict(private_row)
        private_projection["case_id"] = case_id
        private_projection["expected_verdict"] = _expected_verdict(private_projection)
        gold_rows.append(private_projection)

    _write_jsonl(benchmark_path, runtime_rows)
    _write_jsonl(gold_path, gold_rows)
    _write_json(
        release_root / "manifest.json",
        {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": f"{output_dir.name}-runtime-projection",
            "release_stage": "teacher_rollout_runtime_projection",
            "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "input_mode": INPUT_MODE,
            "decision_policy_version": DATA_PIPELINE_DECISION_POLICY_VERSION,
            "runtime_contract": {
                "allowed_keys": sorted(RUNTIME_CASE_KEYS),
                "private_keys_absent": True,
            },
            "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
            "source_access_policy": {"active": False},
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": _now(),
        "dataset_root": str(dataset_root),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": source_sha256,
        "case_count": len(runtime_rows),
        "limit": limit,
        "runtime_release": str(release_root),
        "benchmark": str(benchmark_path),
        "private_gold": str(gold_path),
        "runtime_cases_sha256": _sha256_file(benchmark_path),
        "private_gold_sha256": _sha256_file(gold_path),
        "runtime_asset_materialization": materialization,
    }
    _write_json(preparation_path, payload)
    return payload


def _trace_case_id(trace: Mapping[str, Any]) -> str:
    state = trace.get("state")
    runtime_case = state.get("runtime_case") if isinstance(state, Mapping) else None
    case_id = runtime_case.get("case_id") if isinstance(runtime_case, Mapping) else ""
    case_id = str(case_id or trace.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("trace lacks state.runtime_case.case_id")
    return case_id


def _trace_episode_id(trace: Mapping[str, Any]) -> str:
    state = trace.get("state")
    episode_id = str(
        trace.get("image_id")
        or (state.get("image_id") if isinstance(state, Mapping) else "")
        or ""
    ).strip()
    if not episode_id:
        raise ValueError("trace lacks image_id")
    return episode_id


def _terminal_success(trace: Mapping[str, Any]) -> bool:
    return (
        str(trace.get("termination") or "") == "success"
        and str(trace.get("verdict") or "").lower() in VALID_VERDICTS
    )


def _trace_entries(run_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    entries: dict[str, tuple[Path, dict[str, Any]]] = {}
    for trace_path in sorted((run_dir / "traces").glob("*.json")):
        trace = _read_json(trace_path)
        case_id = _trace_case_id(trace)
        if case_id in entries:
            raise ValueError(f"duplicate case trace in {run_dir}: {case_id}")
        entries[case_id] = (trace_path, trace)
    return entries


def _attempt_dirs(group_dir: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in group_dir.glob("attempt-*"):
        if not path.is_dir():
            continue
        try:
            number = int(path.name.removeprefix("attempt-"))
        except ValueError:
            continue
        candidates.append((number, path))
    return [path for _, path in sorted(candidates)]


def _successful_trace_sources(
    attempt_dirs: Sequence[Path],
) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for attempt_dir in attempt_dirs:
        for case_id, (trace_path, trace) in _trace_entries(attempt_dir).items():
            if _terminal_success(trace) and case_id not in selected:
                selected[case_id] = (attempt_dir, trace_path, trace)
    return selected


def _copy_or_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _merge_successful_attempts(
    *,
    group_dir: Path,
    group_name: str,
    target_ids: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Materialize one terminal-success trace per case without deleting attempts."""

    merged_dir = group_dir / "merged"
    manifest_path = merged_dir / "run_manifest.json"
    if manifest_path.is_file():
        return merged_dir, _read_json(manifest_path)
    if merged_dir.exists() and any(merged_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial merged run: {merged_dir}")

    attempts = _attempt_dirs(group_dir)
    selected = _successful_trace_sources(attempts)
    target_set = set(target_ids)
    unexpected = sorted(set(selected) - target_set)
    if unexpected:
        raise ValueError(f"{group_name} traces include unexpected case IDs: {unexpected[:3]}")
    unresolved = sorted(target_set - set(selected))
    trace_root = merged_dir / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    materialization: dict[str, int] = {"hardlink": 0, "copy": 0}
    provenance: list[dict[str, Any]] = []
    for case_id in sorted(selected):
        attempt_dir, source_path, trace = selected[case_id]
        destination = trace_root / source_path.name
        if destination.exists():
            raise FileExistsError(f"duplicate merged trace path: {destination}")
        method = _copy_or_link(source_path, destination)
        materialization[method] += 1
        provenance.append(
            {
                "case_id": case_id,
                "episode_id": _trace_episode_id(trace),
                "source_attempt": attempt_dir.name,
                "source_run": str(attempt_dir),
                "source_trace": str(source_path),
                "verdict": trace.get("verdict"),
            }
        )
    manifest = {
        "schema_version": "ifv-merged-teacher-rollout-v1",
        "run_id": merged_dir.name,
        "status": "completed" if not unresolved else "completed_with_errors",
        "started_at": _now(),
        "completed_at": _now(),
        "metadata": {
            "kind": "terminal-success-merge",
            "group": group_name,
            "target_case_count": len(target_ids),
            "merged_success_count": len(selected),
            "attempt_dirs": [str(path) for path in attempts],
            "trace_materialization": materialization,
        },
        "result": {
            "num_cases": len(target_ids),
            "num_episodes": len(target_ids),
            "num_errors": len(unresolved),
            "status_distribution": {
                "success": len(selected),
                "error": len(unresolved),
            },
        },
        "artifacts": {
            "traces": "traces/",
            "trace_provenance": "trace-provenance.jsonl",
            "unresolved_engineering": "unresolved-engineering-case-list.txt",
        },
    }
    _write_jsonl(merged_dir / "trace-provenance.jsonl", provenance)
    _write_case_list(merged_dir / "unresolved-engineering-case-list.txt", unresolved)
    _write_json(manifest_path, manifest)
    return merged_dir, manifest


def _command_log_path(run_dir: Path) -> Path:
    return run_dir / "command.log"


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n[{_now()}] command={json.dumps(list(command))}\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{_now()}] returncode={completed.returncode}\n")
    return int(completed.returncode)


def _run_engineering_retries(
    *,
    group_name: str,
    benchmark: Path,
    pipeline_dir: Path,
    target_ids: Sequence[str],
    profile: str,
    rollout_concurrency: int,
    base_seed: int,
    timeout: float,
    maximum_attempts: int,
) -> tuple[Path, dict[str, Any]]:
    """Run only the unresolved engineering cases in each fresh-seed attempt."""

    if maximum_attempts < 1:
        raise ValueError("maximum engineering attempts must be at least 1")
    group_dir = pipeline_dir / "rollouts" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    target_ids = list(target_ids)
    if len(target_ids) != len(set(target_ids)):
        raise ValueError(f"{group_name} target case IDs must be unique")
    _write_case_list(group_dir / "target-case-list.txt", target_ids)
    state_path = group_dir / "engineering-retry-state.json"

    for attempt_number in range(1, maximum_attempts + 1):
        successful = _successful_trace_sources(_attempt_dirs(group_dir))
        pending = [case_id for case_id in target_ids if case_id not in successful]
        _write_case_list(group_dir / "pending-case-list.txt", pending)
        if not pending:
            break
        attempt_dir = group_dir / f"attempt-{attempt_number:02d}"
        if attempt_dir.exists() and any(attempt_dir.iterdir()):
            # A previous process may have been interrupted.  Its terminal successes
            # already leave the pending set; its unresolved members move to a new
            # output directory rather than overwriting immutable trace artifacts.
            continue
        case_list_path = group_dir / f"attempt-{attempt_number:02d}-case-list.txt"
        _write_case_list(case_list_path, pending)
        command = [
            str(REPO_ROOT / "scripts" / "server" / "run_gpu13.sh"),
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "ifv-agent",
            "python",
            "-m",
            "src.eval.run_cases",
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(attempt_dir),
            "--profile",
            profile,
            "--concurrency",
            str(rollout_concurrency),
            "--rollouts-per-case",
            "1",
            "--base-sampling-seed",
            str(base_seed + attempt_number - 1),
            "--timeout",
            str(timeout),
            "--case-list",
            str(case_list_path),
        ]
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "GEMINI_EVAL_MAX_CONCURRENCY": str(rollout_concurrency),
                "GEMINI_MAX_INFLIGHT_REQUESTS": str(rollout_concurrency),
                "PYTHONUNBUFFERED": "1",
            }
        )
        returncode = _run_command(
            command,
            cwd=REPO_ROOT,
            log_path=_command_log_path(attempt_dir),
            env=env,
        )
        attempt_payload = {
            "schema_version": SCHEMA_VERSION,
            "group": group_name,
            "attempt": attempt_number,
            "started_case_count": len(pending),
            "base_sampling_seed": base_seed + attempt_number - 1,
            "returncode": returncode,
            "completed_at": _now(),
            "summary_exists": (attempt_dir / "summary.json").is_file(),
        }
        _write_json(attempt_dir / "autopilot-attempt.json", attempt_payload)

    selected = _successful_trace_sources(_attempt_dirs(group_dir))
    unresolved = [case_id for case_id in target_ids if case_id not in selected]
    retry_state = {
        "schema_version": SCHEMA_VERSION,
        "group": group_name,
        "target_case_count": len(target_ids),
        "successful_case_count": len(selected),
        "unresolved_engineering_case_count": len(unresolved),
        "maximum_attempts": maximum_attempts,
        "attempt_dirs": [str(path) for path in _attempt_dirs(group_dir)],
        "completed_at": _now(),
    }
    _write_json(state_path, retry_state)
    _write_case_list(group_dir / "unresolved-engineering-case-list.txt", unresolved)
    return _merge_successful_attempts(
        group_dir=group_dir,
        group_name=group_name,
        target_ids=target_ids,
    )


def _sft_artifact_index(eligibility_dir: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in sorted(eligibility_dir.glob("*.sft_eligibility.json")):
        payload = _read_json(path)
        episode_id = str(
            payload.get("episode_id")
            or (payload.get("rollout") or {}).get("episode_id")
            or ""
        ).strip()
        if not episode_id:
            raise ValueError(f"SFT artifact lacks episode_id: {path}")
        if episode_id in indexed:
            raise ValueError(f"duplicate SFT artifact episode_id: {episode_id}")
        indexed[episode_id] = payload
    return indexed


def _sft_audit_complete(eligibility_dir: Path, expected_count: int) -> bool:
    summary_path = eligibility_dir / "sft_eligibility_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    rows = summary.get("rows")
    return (
        isinstance(rows, list)
        and len(rows) == expected_count
        and len(_sft_artifact_index(eligibility_dir)) == expected_count
    )


def _run_sft_audit(
    *,
    run_dir: Path,
    gold: Path,
    pipeline_dir: Path,
    label: str,
    model: str,
    concurrency: int,
    timeout: float,
    maximum_attempts: int,
) -> Path:
    expected_count = len(_trace_entries(run_dir))
    if expected_count < 1:
        raise ValueError(f"cannot audit empty merged run: {run_dir}")
    eligibility_dir = pipeline_dir / "sft-eligibility" / label
    if _sft_audit_complete(eligibility_dir, expected_count):
        return eligibility_dir
    for attempt in range(1, maximum_attempts + 1):
        command = [
            str(REPO_ROOT / "scripts" / "server" / "run_gpu13.sh"),
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "ifv-agent",
            "python",
            "-m",
            "src.eval.score_sft_eligibility",
            "--run-dir",
            str(run_dir),
            "--gold",
            str(gold),
            "--output-dir",
            str(eligibility_dir),
            "--cache-dir",
            str(eligibility_dir / "cache"),
            "--storage-dir",
            str(eligibility_dir / "automatic-storage"),
            "--provider",
            "gemini",
            "--model",
            model,
            "--max-tokens",
            "4096",
            "--timeout",
            str(timeout),
            "--concurrency",
            str(concurrency),
        ]
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        _run_command(
            command,
            cwd=REPO_ROOT,
            log_path=eligibility_dir / "command.log",
            env=env,
        )
        if _sft_audit_complete(eligibility_dir, expected_count):
            return eligibility_dir
    raise RuntimeError(
        f"{label} SFT audit incomplete after {maximum_attempts} attempts: "
        f"{len(_sft_artifact_index(eligibility_dir))}/{expected_count} artifacts"
    )


def _has_early_correct_judgment(trace: Mapping[str, Any], expected: str) -> bool:
    """Require the strict earlier image-only *judgment*, not a proposal."""

    state = trace.get("state")
    steps = state.get("all_steps") if isinstance(state, Mapping) else []
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if step.get("stage") != "image_only_discrepancy_judgment":
            continue
        output = step.get("output")
        verdict = str(output.get("verdict") if isinstance(output, Mapping) else "").lower()
        if verdict == expected:
            return True
    return False


def _classify_initial_outcomes(
    *,
    run_dir: Path,
    eligibility_dir: Path,
    private_gold: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Classify strict early evidence and generate the quality-reroll candidates."""

    result_path = output_dir / "classification.json"
    if result_path.is_file():
        return _read_json(result_path)
    gold_by_case = {
        _case_id(row): _expected_verdict(row) for row in _read_jsonl(private_gold)
    }
    artifacts = _sft_artifact_index(eligibility_dir)
    early: list[dict[str, Any]] = []
    final_only: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reroll_ids: list[str] = []
    seen_reroll: set[str] = set()
    for case_id, (trace_path, trace) in sorted(_trace_entries(run_dir).items()):
        expected = gold_by_case.get(case_id)
        if expected is None:
            raise ValueError(f"private gold lacks successful case: {case_id}")
        episode_id = _trace_episode_id(trace)
        artifact = artifacts.get(episode_id)
        if artifact is None:
            raise ValueError(f"SFT audit artifact missing for episode: {episode_id}")
        passed = (artifact.get("gates") or {}).get("sft_eligibility_pass") is True
        row = {
            "case_id": case_id,
            "episode_id": episode_id,
            "trace_path": str(trace_path),
            "expected_verdict": expected,
            "final_verdict": str(trace.get("verdict") or "").lower(),
            "sft_eligibility_pass": passed,
        }
        if not passed:
            row["bucket"] = "sft_rejected"
            row["rejection_reasons"] = (artifact.get("metrics") or {}).get(
                "fatal_errors"
            ) or []
            rejected.append(row)
        elif _has_early_correct_judgment(trace, expected):
            row["bucket"] = "early_correct_judgment"
            early.append(row)
        else:
            row["bucket"] = "final_only_judgment"
            final_only.append(row)
        if row["bucket"] != "early_correct_judgment" and case_id not in seen_reroll:
            seen_reroll.add(case_id)
            reroll_ids.append(case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "early-correct-judgment.jsonl", early)
    _write_jsonl(output_dir / "final-only-judgment.jsonl", final_only)
    _write_jsonl(output_dir / "sft-rejected.jsonl", rejected)
    _write_case_list(output_dir / "quality-reroll-case-list.txt", reroll_ids)
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_run": str(run_dir),
        "source_eligibility": str(eligibility_dir),
        "successful_case_count": len(early) + len(final_only) + len(rejected),
        "early_correct_judgment_count": len(early),
        "final_only_judgment_count": len(final_only),
        "sft_rejected_count": len(rejected),
        "quality_reroll_count": len(reroll_ids),
        "created_at": _now(),
    }
    _write_json(result_path, result)
    return result


def _load_case_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _final_release(
    *,
    pipeline_dir: Path,
    sources: Sequence[tuple[Path, Path]],
) -> Path:
    output_dir = pipeline_dir / "accepted-release"
    manifest_path = output_dir / "accepted_release_manifest.json"
    if manifest_path.is_file():
        return output_dir
    staged_sources = [(run, audit, None) for run, audit in sources]
    stage_release(
        staged_sources,
        output_dir,
        minimum_accepted_cases=1,
    )
    return output_dir


def _build_package(*, pipeline_dir: Path, accepted_release: Path) -> Path:
    package_dir = pipeline_dir / "sft-training-package"
    manifest_path = package_dir / "MANIFEST.json"
    if manifest_path.is_file():
        return package_dir
    command = [
        str(REPO_ROOT / "scripts" / "server" / "run_gpu13.sh"),
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "ifv-agent",
        "python",
        "scripts/trajectory/build_sft_training_package.py",
        "--accepted-release",
        str(accepted_release),
        "--output-dir",
        str(package_dir),
        "--all-train",
        "--minimum-accepted-cases",
        "1",
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    returncode = _run_command(
        command,
        cwd=REPO_ROOT,
        log_path=package_dir / "command.log",
        env=env,
    )
    if returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"SFT package construction failed: {package_dir}")
    return package_dir


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    train_manifest = (
        args.train_manifest.expanduser().resolve()
        if args.train_manifest is not None
        else dataset_root / "train-manifest.jsonl"
    )
    pipeline_dir = args.output_dir.expanduser().resolve()
    preparation = prepare_runtime_release(
        dataset_root=dataset_root,
        train_manifest=train_manifest,
        output_dir=pipeline_dir,
        limit=args.limit,
    )
    state_path = pipeline_dir / "pipeline-state.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_dir": str(pipeline_dir),
        "prepared": preparation,
        "models": {
            "rollout": args.rollout_model,
            "sft_judge": args.sft_model,
        },
        "concurrency": {
            "rollout": args.rollout_concurrency,
            "sft_judge": args.sft_concurrency,
        },
        "status": "running",
        "updated_at": _now(),
    }
    _write_json(state_path, state)
    if args.prepare_only:
        state["status"] = "prepared"
        state["updated_at"] = _now()
        _write_json(state_path, state)
        return state

    private_gold_path = Path(str(preparation["private_gold"])).expanduser().resolve()
    all_case_ids = [str(row["case_id"]) for row in _read_jsonl(private_gold_path)]
    initial_run, initial_manifest = _run_engineering_retries(
        group_name="initial",
        benchmark=Path(preparation["benchmark"]),
        pipeline_dir=pipeline_dir,
        target_ids=all_case_ids,
        profile=args.rollout_profile,
        rollout_concurrency=args.rollout_concurrency,
        base_seed=args.base_seed,
        timeout=args.rollout_timeout,
        maximum_attempts=args.maximum_engineering_attempts,
    )
    state["initial"] = {
        "merged_run": str(initial_run),
        "unresolved_engineering_case_count": initial_manifest["result"]["num_errors"],
    }
    state["updated_at"] = _now()
    _write_json(state_path, state)

    initial_audit = _run_sft_audit(
        run_dir=initial_run,
        gold=private_gold_path,
        pipeline_dir=pipeline_dir,
        label="initial",
        model=args.sft_model,
        concurrency=args.sft_concurrency,
        timeout=args.sft_timeout,
        maximum_attempts=args.maximum_sft_audit_attempts,
    )
    classification = _classify_initial_outcomes(
        run_dir=initial_run,
        eligibility_dir=initial_audit,
        private_gold=private_gold_path,
        output_dir=pipeline_dir / "classification",
    )
    state["initial"]["sft_eligibility"] = str(initial_audit)
    state["classification"] = classification
    state["updated_at"] = _now()
    _write_json(state_path, state)

    sources: list[tuple[Path, Path]] = [(initial_run, initial_audit)]
    reroll_ids = _load_case_list(
        pipeline_dir / "classification" / "quality-reroll-case-list.txt"
    )
    if reroll_ids:
        reroll_run, reroll_manifest = _run_engineering_retries(
            group_name="quality-reroll",
            benchmark=Path(preparation["benchmark"]),
            pipeline_dir=pipeline_dir,
            target_ids=reroll_ids,
            profile=args.rollout_profile,
            rollout_concurrency=args.rollout_concurrency,
            base_seed=args.base_seed + 1_000_000,
            timeout=args.rollout_timeout,
            maximum_attempts=args.maximum_engineering_attempts,
        )
        reroll_audit = _run_sft_audit(
            run_dir=reroll_run,
            gold=private_gold_path,
            pipeline_dir=pipeline_dir,
            label="quality-reroll",
            model=args.sft_model,
            concurrency=args.sft_concurrency,
            timeout=args.sft_timeout,
            maximum_attempts=args.maximum_sft_audit_attempts,
        )
        state["quality_reroll"] = {
            "merged_run": str(reroll_run),
            "sft_eligibility": str(reroll_audit),
            "unresolved_engineering_case_count": reroll_manifest["result"][
                "num_errors"
            ],
        }
        sources.append((reroll_run, reroll_audit))
        state["updated_at"] = _now()
        _write_json(state_path, state)

    accepted_release = _final_release(pipeline_dir=pipeline_dir, sources=sources)
    package_dir = _build_package(
        pipeline_dir=pipeline_dir,
        accepted_release=accepted_release,
    )
    state["accepted_release"] = str(accepted_release)
    state["sft_training_package"] = str(package_dir)
    state["status"] = "completed"
    state["updated_at"] = _now()
    _write_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        help="defaults to <dataset-root>/train-manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="deterministic case-count cap for an isolated smoke run",
    )
    parser.add_argument("--rollout-profile", default="teacher-gemini")
    parser.add_argument(
        "--rollout-model",
        default="gemini-3.7-flash",
        help=(
            "recorded expected profile model; teacher-gemini is pinned to "
            "gemini-3.7-flash by src.provider_profiles"
        ),
    )
    parser.add_argument("--sft-model", default="gemini-3.7-flash")
    parser.add_argument("--rollout-concurrency", type=int, default=10)
    parser.add_argument("--sft-concurrency", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2026082401)
    parser.add_argument("--rollout-timeout", type=float, default=1800.0)
    parser.add_argument("--sft-timeout", type=float, default=180.0)
    parser.add_argument("--maximum-engineering-attempts", type=int, default=12)
    parser.add_argument("--maximum-sft-audit-attempts", type=int, default=12)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only build and validate the isolated runtime/private projections",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    numeric_values = {
        "--rollout-concurrency": args.rollout_concurrency,
        "--sft-concurrency": args.sft_concurrency,
        "--maximum-engineering-attempts": args.maximum_engineering_attempts,
        "--maximum-sft-audit-attempts": args.maximum_sft_audit_attempts,
    }
    invalid = [name for name, value in numeric_values.items() if value < 1]
    if invalid:
        raise SystemExit(", ".join(invalid) + " must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1 when supplied")
    if (
        args.rollout_profile == "teacher-gemini"
        and args.rollout_model != "gemini-3.7-flash"
    ):
        raise SystemExit(
            "teacher-gemini is pinned to gemini-3.7-flash; do not supply a "
            "different --rollout-model"
        )
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
