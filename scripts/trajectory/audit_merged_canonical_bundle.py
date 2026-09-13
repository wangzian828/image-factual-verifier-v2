#!/usr/bin/env python3
"""Independently audit a merged canonical SFT bundle against its sources.

The merge command already fails closed while constructing a release.  This
second implementation reads the finished files again, rebuilds the expected
row/index/media unions from the source bundles, and verifies content and
identity equality without calling the merge implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "training"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ifv_training.audit import audit_derived_dataset


SPLITS = ("train", "validation", "test")
PROVIDER_ARTIFACTS = (
    *SPLITS,
    "index",
    "action_only",
    "media_bindings",
    "media_manifest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class Audit:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.counts: Counter[str] = Counter()

    def require(self, condition: bool, code: str, location: str) -> None:
        if condition:
            return
        self.counts[f"error:{code}"] += 1
        if len(self.errors) < 200:
            self.errors.append({"code": code, "location": location})


def _artifact_rows(
    audit: Audit,
    root: Path,
    manifest: Mapping[str, Any],
    names: Iterable[str],
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    artifacts = _mapping(manifest.get("artifacts"))
    result: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        artifact = _mapping(artifacts.get(name))
        path = (root / str(artifact.get("path", f"{name}.jsonl"))).resolve()
        inside = path == root or root in path.parents
        audit.require(inside, f"{label}_artifact_escape", name)
        audit.require(path.is_file(), f"{label}_artifact_missing", name)
        if not inside or not path.is_file():
            result[name] = []
            continue
        rows = _jsonl(path)
        audit.require(
            _sha256(path) == str(artifact.get("sha256", "")),
            f"{label}_artifact_hash",
            name,
        )
        audit.require(
            len(rows) == int(artifact.get("rows", -1)),
            f"{label}_artifact_rows",
            name,
        )
        result[name] = rows
    return result


def _unique_map(
    audit: Audit,
    rows: Iterable[dict[str, Any]],
    *,
    field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field, "")).strip()
        audit.require(bool(key), f"{label}_missing_key", field)
        audit.require(key not in result, f"{label}_duplicate_key", key)
        if key:
            result[key] = row
    return result


def _resolved_media(image: Any) -> str:
    return str(Path(str(image)).expanduser().resolve())


def audit_merged_bundle(
    merged_root: Path,
    *,
    forbidden_case_files: list[Path] | None = None,
) -> dict[str, Any]:
    audit = Audit()
    root = merged_root.expanduser().resolve()
    bundle_manifest_path = root / "manifest.json"
    bundle = _json(bundle_manifest_path)
    provider_manifest = _mapping(bundle.get("provider_dataset"))
    policy_manifest = _mapping(bundle.get("policy_dataset"))
    provider_root = root / "canonical-dataset"
    policy_root = root / "ms-swift-policy"
    merged = _artifact_rows(
        audit,
        provider_root,
        provider_manifest,
        PROVIDER_ARTIFACTS,
        "merged_provider",
    )
    _artifact_rows(
        audit,
        policy_root,
        policy_manifest,
        (*SPLITS, "index"),
        "merged_policy",
    )

    expected_rows: dict[str, dict[str, dict[str, Any]]] = {
        split: {} for split in SPLITS
    }
    expected_indices: dict[str, dict[str, Any]] = {}
    expected_actions: dict[str, dict[str, Any]] = {}
    expected_bindings: dict[str, dict[str, Any]] = {}
    expected_media: dict[tuple[str, str], dict[str, str]] = {}
    source_bindings: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    seen_trace_hashes: set[str] = set()

    declared_sources = provider_manifest.get("source_bundles")
    declared_sources = declared_sources if isinstance(declared_sources, list) else []
    audit.require(bool(declared_sources), "source_list_missing", "manifest")
    for source_number, declared in enumerate(declared_sources, 1):
        declared = _mapping(declared)
        source_root = Path(str(declared.get("root", ""))).expanduser().resolve()
        source_manifest_path = source_root / "manifest.json"
        audit.require(source_manifest_path.is_file(), "source_manifest_missing", str(source_root))
        if not source_manifest_path.is_file():
            continue
        source_bundle = _json(source_manifest_path)
        audit.require(
            _sha256(source_manifest_path) == str(declared.get("manifest_sha256", "")),
            "source_manifest_hash",
            str(source_root),
        )
        source_provider = _mapping(source_bundle.get("provider_dataset"))
        source_rows = _artifact_rows(
            audit,
            source_root / "canonical-dataset",
            source_provider,
            PROVIDER_ARTIFACTS,
            f"source_{source_number}",
        )
        independent_path = source_root / "independent-audit.json"
        audit.require(independent_path.is_file(), "source_independent_audit_missing", str(source_root))
        if independent_path.is_file():
            independent = _json(independent_path)
            audit.require(
                independent.get("passed") is True and independent.get("error_count") == 0,
                "source_independent_audit_failed",
                str(source_root),
            )
            source_bindings.append(
                {
                    "root": str(source_root),
                    "manifest_sha256": _sha256(source_manifest_path),
                    "independent_audit_sha256": _sha256(independent_path),
                }
            )

        indices = _unique_map(
            audit,
            source_rows.get("index", []),
            field="episode_id",
            label=f"source_{source_number}_index",
        )
        bindings = _unique_map(
            audit,
            source_rows.get("media_bindings", []),
            field="episode_id",
            label=f"source_{source_number}_binding",
        )
        for split in SPLITS:
            for source_row in source_rows.get(split, []):
                row = copy.deepcopy(source_row)
                episode_id = str(row.get("episode_id", "")).strip()
                case_id = str(row.get("case_id", "")).strip()
                index = indices.get(episode_id)
                binding = bindings.get(episode_id)
                audit.require(
                    bool(episode_id and case_id and index is not None and binding is not None),
                    "source_reasoning_provenance",
                    episode_id or str(source_root),
                )
                if not episode_id or index is None or binding is None:
                    continue
                audit.require(
                    episode_id not in expected_rows[split]
                    and all(episode_id not in expected_rows[s] for s in SPLITS if s != split),
                    "cross_source_episode_overlap",
                    episode_id,
                )
                audit.require(case_id not in seen_cases, "cross_source_case_overlap", case_id)
                trace_hash = str(index.get("source_trace_sha256", "")).casefold()
                audit.require(
                    len(trace_hash) == 64 and trace_hash not in seen_trace_hashes,
                    "cross_source_trace_overlap",
                    episode_id,
                )
                seen_cases.add(case_id)
                seen_trace_hashes.add(trace_hash)
                row["split"] = split
                row["images"] = [_resolved_media(image) for image in row.get("images", [])]
                expected_rows[split][episode_id] = row
                expected_index = {
                    **copy.deepcopy(index),
                    "split": split,
                    "source_bundle": str(source_root),
                    "source_bundle_index": source_number,
                }
                expected_indices[episode_id] = expected_index
                expected_binding = copy.deepcopy(binding)
                expected_binding["image_sha256"] = [
                    Path(image).stem.casefold() for image in row["images"]
                ]
                expected_binding["marker_counts"] = [
                    str(message.get("content", "")).count("<image>")
                    for message in row.get("messages", [])
                ]
                expected_bindings[episode_id] = expected_binding
                for image in row["images"]:
                    key = (Path(image).stem.casefold(), image)
                    expected_media[key] = {"sha256": key[0], "path": key[1]}

        for source_row in source_rows.get("action_only", []):
            row = copy.deepcopy(source_row)
            episode_id = str(row.get("episode_id", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            trace_hash = str(row.get("source_trace_sha256", "")).casefold()
            audit.require(episode_id not in expected_actions, "cross_source_action_overlap", episode_id)
            audit.require(case_id not in seen_cases, "cross_source_case_overlap", case_id)
            audit.require(
                len(trace_hash) == 64 and trace_hash not in seen_trace_hashes,
                "cross_source_trace_overlap",
                episode_id,
            )
            seen_cases.add(case_id)
            seen_trace_hashes.add(trace_hash)
            row["source_bundle"] = str(source_root)
            row["source_bundle_index"] = source_number
            row["images"] = [_resolved_media(image) for image in row.get("images", [])]
            expected_actions[episode_id] = row
            for image in row["images"]:
                key = (Path(image).stem.casefold(), image)
                expected_media[key] = {"sha256": key[0], "path": key[1]}

    forbidden: set[str] = set()
    for path in forbidden_case_files or []:
        for row in _jsonl(path.expanduser().resolve()):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                forbidden.add(case_id)
    audit.require(not (seen_cases & forbidden), "forbidden_case_overlap", "dataset")

    for split in SPLITS:
        actual = _unique_map(
            audit,
            merged.get(split, []),
            field="episode_id",
            label=f"merged_{split}",
        )
        audit.require(set(actual) == set(expected_rows[split]), "merged_episode_set", split)
        for episode_id in set(actual) & set(expected_rows[split]):
            audit.require(
                actual[episode_id] == expected_rows[split][episode_id],
                "merged_row_drift",
                episode_id,
            )
            audit.counts["reasoning_rows"] += 1

    actual_indices = _unique_map(audit, merged.get("index", []), field="episode_id", label="merged_index")
    actual_actions = _unique_map(audit, merged.get("action_only", []), field="episode_id", label="merged_action")
    actual_bindings = _unique_map(audit, merged.get("media_bindings", []), field="episode_id", label="merged_binding")
    audit.require(actual_indices == expected_indices, "merged_index_drift", "index")
    audit.require(actual_actions == expected_actions, "merged_action_drift", "action_only")
    audit.require(actual_bindings == expected_bindings, "merged_binding_drift", "media_bindings")
    audit.counts["action_only_rows"] = len(actual_actions)

    actual_media = {
        (str(row.get("sha256", "")).casefold(), str(row.get("path", ""))): row
        for row in merged.get("media_manifest", [])
    }
    audit.require(actual_media == expected_media, "merged_media_manifest_drift", "media_manifest")
    for digest, path_text in actual_media:
        path = Path(path_text)
        audit.require(path.is_file(), "merged_media_missing", path_text)
        if path.is_file():
            audit.require(path.stem.casefold() == digest, "merged_media_name_hash", path_text)
            audit.require(_sha256(path) == digest, "merged_media_content_hash", path_text)
        audit.counts["unique_media_files"] += 1

    policy_audit_path = root / str(_mapping(bundle.get("policy_audit")).get("path", "policy-audit.json"))
    audit.require(policy_audit_path.is_file(), "policy_audit_missing", str(policy_audit_path))
    strict_policy = audit_derived_dataset(policy_root)
    audit.require(strict_policy.get("passed") is True, "strict_policy_audit", "policy")
    if policy_audit_path.is_file():
        audit.require(
            _sha256(policy_audit_path) == str(_mapping(bundle.get("policy_audit")).get("sha256", "")),
            "policy_audit_hash",
            str(policy_audit_path),
        )

    expected_retention = _mapping(provider_manifest.get("row_retention"))
    audit.require(
        expected_retention
        == {
            "raw_rows": len(expected_indices) + len(expected_actions),
            "reasoning_rows": len(expected_indices),
            "action_only_rows": len(expected_actions),
            "dropped_rows": 0,
        },
        "merged_retention_drift",
        "manifest",
    )
    return {
        "schema_version": "ifv-merged-canonical-independent-audit-v1",
        "passed": not audit.errors,
        "error_count": sum(
            count for name, count in audit.counts.items() if name.startswith("error:")
        ),
        "errors": audit.errors,
        "counts": dict(sorted(audit.counts.items())),
        "bindings": {
            "merged_manifest_sha256": _sha256(bundle_manifest_path),
            "sources": source_bindings,
            "forbidden_case_files": [
                {"path": str(path.expanduser().resolve()), "sha256": _sha256(path.expanduser().resolve())}
                for path in forbidden_case_files or []
            ],
        },
        "strict_policy_audit": {
            "passed": strict_policy.get("passed"),
            "production_blockers": _mapping(
                _mapping(strict_policy.get("causal_contract")).get("production_blockers")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--forbidden-case-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_merged_bundle(
        args.merged_root,
        forbidden_case_files=args.forbidden_case_jsonl,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "error_count": report["error_count"],
                "counts": report["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
