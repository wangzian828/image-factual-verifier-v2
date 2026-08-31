#!/usr/bin/env python3
"""Prepare an evaluator-only test subset for image-only Agent rollout.

The Agent receives only ``case_id``, ``image_path`` and ``image_sha256`` from
``runtime-release/runtime_input/cases.jsonl``.  Construction labels and private
gold remain outside that release.  The resulting release is explicitly marked
``development_subset`` and ``training_prohibited`` so it cannot be mistaken for
teacher-rollout training input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.release_adapter import (
    DATA_PIPELINE_DECISION_POLICY_VERSION,
    INPUT_MODE,
    RELEASE_SCHEMA_VERSION,
    RUNTIME_CASE_KEYS,
    RUNTIME_CONTRACT_VERSION,
)


SCHEMA_VERSION = "ifv-agent-test-release-preparation-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(row: Mapping[str, Any]) -> str:
    case_id = str(
        row.get("unified_case_id")
        or row.get("archive_source_version_id")
        or row.get("case_id")
        or ""
    ).strip()
    if not case_id:
        raise ValueError("test manifest row lacks case_id/unified_case_id")
    return case_id


def _source_image(dataset_root: Path, row: Mapping[str, Any]) -> Path:
    rendered = str(
        row.get("unified_image_path") or row.get("local_image_path") or ""
    ).strip()
    if not rendered:
        raise ValueError(f"test row lacks unified_image_path: {_case_id(row)}")
    relative = Path(rendered)
    if relative.is_absolute():
        raise ValueError(
            f"test row image path must be relative to dataset root: {_case_id(row)}"
        )
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"test row image path escapes dataset root: {_case_id(row)}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"test row image is missing: {resolved}")
    return resolved


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


def _load_case_ids(paths: Sequence[Path]) -> set[str]:
    case_ids: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value in case_ids:
                raise ValueError(
                    f"duplicate excluded case_id {value!r} in {path}:{line_number}"
                )
            case_ids.add(value)
    return case_ids


def _stratum_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "unknown").strip() or "unknown" for field in fields)


def _stable_order(case_id: str, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def _select_balanced(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    fields: Sequence[str],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not fields:
        raise ValueError("at least one balanced-by field is required")

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum_key(row, fields)].append(dict(row))
    if not grouped:
        raise ValueError("no available rows after exclusions")
    if limit > len(rows):
        raise ValueError(f"requested {limit} rows but only {len(rows)} are available")

    ordered_keys = sorted(grouped)
    base, remainder = divmod(limit, len(ordered_keys))
    quotas: dict[tuple[str, ...], int] = {
        key: base + (1 if index < remainder else 0)
        for index, key in enumerate(ordered_keys)
    }
    deficits = {
        key: quota - len(grouped[key])
        for key, quota in quotas.items()
        if len(grouped[key]) < quota
    }
    if deficits:
        details = ", ".join(
            f"{'/'.join(key)} needs {quotas[key]}, has {len(grouped[key])}"
            for key in sorted(deficits)
        )
        raise ValueError(f"balanced test selection is undersupplied: {details}")

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key in ordered_keys:
        ranked = sorted(
            grouped[key],
            key=lambda row: _stable_order(_case_id(row), seed=seed),
        )
        selected.extend(ranked[: quotas[key]])
        counts[" / ".join(key)] = quotas[key]
    selected.sort(key=lambda row: _stable_order(_case_id(row), seed=f"{seed}:final"))
    return selected, counts


def _private_gold_case_ids(path: Path) -> set[str]:
    indexed: set[str] = set()
    for row in _read_jsonl(path):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"private-gold row lacks case_id: {path}")
        if case_id in indexed:
            raise ValueError(f"duplicate private-gold case_id: {case_id}")
        indexed.add(case_id)
    return indexed


def prepare_agent_test_release(
    *,
    dataset_root: Path,
    test_manifest: Path,
    private_gold_sidecar: Path,
    output_dir: Path,
    limit: int,
    excluded_case_lists: Sequence[Path] = (),
    balanced_by: Sequence[str] = ("construction_subroute",),
    selection_seed: str = "ifv-agent-test-eval-v1",
    source_access_policy: Path | None = None,
) -> dict[str, Any]:
    """Create a balanced, evaluator-only test release from a unified test manifest."""

    dataset_root = dataset_root.expanduser().resolve()
    test_manifest = test_manifest.expanduser().resolve()
    private_gold_sidecar = private_gold_sidecar.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    policy_path = (
        source_access_policy.expanduser().resolve()
        if source_access_policy is not None
        else None
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    for path, label in (
        (test_manifest, "test manifest"),
        (private_gold_sidecar, "private-gold sidecar"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if policy_path is not None and not policy_path.is_file():
        raise FileNotFoundError(f"source-access policy does not exist: {policy_path}")
    try:
        test_manifest.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("test manifest must be under dataset root") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")

    fields = tuple(str(field).strip() for field in balanced_by if str(field).strip())
    rows = _read_jsonl(test_manifest)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = _case_id(row)
        if case_id in indexed:
            raise ValueError(f"duplicate test case_id: {case_id}")
        if str(row.get("split") or "").strip().lower() != "test":
            raise ValueError(f"non-test row in test manifest: {case_id}")
        indexed[case_id] = row

    excluded = _load_case_ids(
        [Path(path).expanduser().resolve() for path in excluded_case_lists]
    )
    unknown_excluded = sorted(excluded - set(indexed))
    if unknown_excluded:
        raise ValueError(
            "excluded case IDs are absent from test manifest: "
            + ", ".join(unknown_excluded[:3])
        )
    available = [row for case_id, row in indexed.items() if case_id not in excluded]
    selected, stratum_counts = _select_balanced(
        available,
        limit=limit,
        fields=fields,
        seed=selection_seed,
    )
    selected_ids = [_case_id(row) for row in selected]
    gold_ids = _private_gold_case_ids(private_gold_sidecar)
    missing_gold = sorted(set(selected_ids) - gold_ids)
    if missing_gold:
        raise ValueError(
            "private-gold sidecar lacks selected test case IDs: "
            + ", ".join(missing_gold[:3])
        )

    release_root = output_dir / "runtime-release"
    runtime_root = release_root / "runtime_input"
    runtime_rows: list[dict[str, str]] = []
    selection_rows: list[dict[str, Any]] = []
    materialization = {"hardlink": 0, "copy": 0, "existing": 0}
    for index, row in enumerate(selected, start=1):
        case_id = _case_id(row)
        source_image = _source_image(dataset_root, row)
        suffix = source_image.suffix.lower() or ".jpg"
        asset_relative = Path("assets") / f"{index:05d}{suffix}"
        destination = runtime_root / asset_relative
        method = _link_or_copy(source_image, destination)
        materialization[method] += 1
        runtime_rows.append(
            {
                "case_id": case_id,
                "image_path": asset_relative.as_posix(),
                "image_sha256": _sha256_file(destination),
            }
        )
        selection_rows.append(
            {
                "case_id": case_id,
                "stratum": list(_stratum_key(row, fields)),
                "source_record_id": str(row.get("record_id") or ""),
                "source_manifest": str(test_manifest),
            }
        )

    benchmark_path = runtime_root / "cases.jsonl"
    _write_jsonl(benchmark_path, runtime_rows)
    _write_jsonl(output_dir / "selected-cases.jsonl", selection_rows)
    (output_dir / "excluded-case-list.txt").write_text(
        "".join(f"{case_id}\n" for case_id in sorted(excluded)),
        encoding="utf-8",
    )
    (output_dir / "selected-case-list.txt").write_text(
        "".join(f"{case_id}\n" for case_id in selected_ids),
        encoding="utf-8",
    )

    policy_payload: dict[str, Any] = {"active": False}
    if policy_path is not None:
        release_policy = release_root / "evaluator_private" / "source_access_policy.json"
        release_policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(policy_path, release_policy)
        policy_payload = {
            "active": True,
            "path": "evaluator_private/source_access_policy.json",
            "sha256": _sha256_file(policy_path),
        }
    _write_json(
        release_root / "manifest.json",
        {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": f"{output_dir.name}-runtime-projection",
            "release_stage": "development_subset",
            "training_prohibited": True,
            "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "input_mode": INPUT_MODE,
            "decision_policy_version": DATA_PIPELINE_DECISION_POLICY_VERSION,
            "runtime_contract": {
                "allowed_keys": sorted(RUNTIME_CASE_KEYS),
                "private_keys_absent": True,
            },
            "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
            "source_access_policy": policy_payload,
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": _now(),
        "dataset_root": str(dataset_root),
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": _sha256_file(test_manifest),
        "private_gold_sidecar": str(private_gold_sidecar),
        "private_gold_sidecar_sha256": _sha256_file(private_gold_sidecar),
        "training_prohibited": True,
        "case_count": len(runtime_rows),
        "excluded_case_count": len(excluded),
        "balanced_by": list(fields),
        "selection_seed": selection_seed,
        "stratum_counts": stratum_counts,
        "runtime_release": str(release_root),
        "benchmark": str(benchmark_path),
        "runtime_cases_sha256": _sha256_file(benchmark_path),
        "source_access_policy": str(policy_path or ""),
        "source_access_policy_sha256": _sha256_file(policy_path)
        if policy_path is not None
        else "",
        "runtime_asset_materialization": materialization,
    }
    _write_json(output_dir / "preparation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced evaluator-only test release for Agent rollout."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--private-gold-sidecar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--exclude-case-list",
        action="append",
        default=[],
        help="Repeatable text file of test case IDs not to select.",
    )
    parser.add_argument(
        "--balanced-by",
        default="construction_subroute",
        help="Comma-separated manifest fields to balance equally.",
    )
    parser.add_argument("--selection-seed", default="ifv-agent-test-eval-v1")
    parser.add_argument("--source-access-policy")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    result = prepare_agent_test_release(
        dataset_root=Path(args.dataset_root),
        test_manifest=Path(args.test_manifest),
        private_gold_sidecar=Path(args.private_gold_sidecar),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        excluded_case_lists=[Path(value) for value in args.exclude_case_list],
        balanced_by=args.balanced_by.split(","),
        selection_seed=args.selection_seed,
        source_access_policy=Path(args.source_access_policy)
        if args.source_access_policy
        else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
