#!/usr/bin/env python3
"""Merge independently rebuilt canonical trace bundles behind global gates."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "training"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import load_json, sha256_file, write_json, write_jsonl
from ifv_training.policy import convert_policy_dataset


SPLITS = ("train", "validation", "test")
ARTIFACTS = (*SPLITS, "index", "action_only", "media_bindings", "media_manifest")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index} is not an object")
            yield value


def _bundle(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    manifest = load_json(manifest_path)
    provider = manifest.get("provider_dataset")
    if not isinstance(provider, Mapping):
        raise ValueError(f"bundle has no provider dataset: {resolved}")
    if provider.get("dataset_version") != "ifv-trajectory-sft-dataset-v3":
        raise ValueError(f"unsupported provider dataset: {resolved}")
    if manifest.get("policy_audit", {}).get("passed") is not True:
        raise ValueError(f"source bundle policy audit did not pass: {resolved}")
    provider_root = resolved / "canonical-dataset"
    artifacts = provider.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"provider dataset has no artifacts: {resolved}")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in ARTIFACTS:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"provider artifact is missing: {resolved}:{name}")
        path = (provider_root / str(artifact.get("path", f"{name}.jsonl"))).resolve()
        if provider_root != path and provider_root not in path.parents:
            raise ValueError(f"provider artifact escapes its bundle: {path}")
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"provider artifact hash mismatch: {path}")
        rows[name] = list(_iter_jsonl(path))
        if len(rows[name]) != int(artifact.get("rows", -1)):
            raise ValueError(f"provider artifact row count mismatch: {path}")
    retention = provider.get("row_retention")
    if not isinstance(retention, Mapping):
        raise ValueError(f"provider row-retention record is missing: {resolved}")
    reasoning_count = sum(len(rows[name]) for name in SPLITS)
    if (
        reasoning_count != int(retention.get("reasoning_rows", -1))
        or len(rows["action_only"]) != int(retention.get("action_only_rows", -1))
        or reasoning_count + len(rows["action_only"])
        != int(retention.get("raw_rows", -1))
        or int(retention.get("dropped_rows", -1)) != 0
    ):
        raise ValueError(f"provider row-retention mismatch: {resolved}")
    if len(rows["index"]) != reasoning_count:
        raise ValueError(f"provider reasoning index mismatch: {resolved}")
    return {
        "root": resolved,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "provider": provider,
        "rows": rows,
    }


def _forbidden_cases(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in _iter_jsonl(path.expanduser().resolve()):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                result.add(case_id)
    return result


def _link_media(
    image: str,
    *,
    staging_media: Path,
    final_media: Path,
    digest_cache: dict[Path, str],
) -> tuple[str, str]:
    source = Path(image).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source media is missing: {source}")
    digest = digest_cache.get(source)
    if digest is None:
        digest = sha256_file(source)
        digest_cache[source] = digest
    if source.stem.casefold() != digest:
        raise ValueError(f"source media filename/hash mismatch: {source}")
    suffix = source.suffix.casefold() or ".bin"
    staged = staging_media / f"{digest}{suffix}"
    final = final_media / staged.name
    if staged.exists():
        if sha256_file(staged) != digest:
            raise ValueError(f"merged media collision: {staged}")
    else:
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, staged)
        except OSError:
            shutil.copy2(source, staged)
    return str(final), digest


def merge_bundles(
    source_roots: list[Path],
    output_dir: Path,
    *,
    forbidden_case_files: list[Path] | None = None,
    expected_raw_rows: int | None = None,
    expected_reasoning_rows: int | None = None,
    expected_action_only_rows: int | None = None,
    expected_validation_rows: int | None = None,
) -> dict[str, Any]:
    if len(source_roots) < 2:
        raise ValueError("a merged release requires at least two source bundles")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output must not exist: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".in-progress")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    sources = [_bundle(root) for root in source_roots]
    forbidden = _forbidden_cases(forbidden_case_files or [])

    seen_episodes: set[str] = set()
    seen_cases: set[str] = set()
    seen_trace_hashes: set[str] = set()
    source_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    source_action_only: list[dict[str, Any]] = []
    source_index: list[dict[str, Any]] = []
    source_bindings: dict[str, dict[str, Any]] = {}
    source_manifests: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for source_number, source in enumerate(sources, 1):
        root = source["root"]
        rows = source["rows"]
        index_by_episode = {
            str(item.get("episode_id", "")): item for item in rows["index"]
        }
        binding_by_episode = {
            str(item.get("episode_id", "")): item for item in rows["media_bindings"]
        }
        if len(index_by_episode) != len(rows["index"]):
            raise ValueError(f"duplicate source reasoning index: {root}")
        if len(binding_by_episode) != len(rows["media_bindings"]):
            raise ValueError(f"duplicate source media binding: {root}")
        for split in SPLITS:
            for row in rows[split]:
                episode_id = str(row.get("episode_id", "")).strip()
                case_id = str(row.get("case_id", "")).strip()
                index = index_by_episode.get(episode_id)
                binding = binding_by_episode.get(episode_id)
                trace_hash = str(
                    (index or {}).get("source_trace_sha256", "")
                ).casefold()
                if not episode_id or not case_id or index is None or binding is None:
                    raise ValueError(f"source reasoning row lacks provenance: {root}")
                _claim_identity(
                    episode_id,
                    case_id,
                    trace_hash,
                    forbidden=forbidden,
                    seen_episodes=seen_episodes,
                    seen_cases=seen_cases,
                    seen_trace_hashes=seen_trace_hashes,
                )
                copied = copy.deepcopy(row)
                copied["split"] = split
                source_rows[split].append(copied)
                source_index.append(
                    {
                        **dict(index),
                        "split": split,
                        "source_bundle": str(root),
                        "source_bundle_index": source_number,
                    }
                )
                source_bindings[episode_id] = copy.deepcopy(binding)
                stats["reasoning_rows"] += 1
        for row in rows["action_only"]:
            episode_id = str(row.get("episode_id", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            trace_hash = str(row.get("source_trace_sha256", "")).casefold()
            _claim_identity(
                episode_id,
                case_id,
                trace_hash,
                forbidden=forbidden,
                seen_episodes=seen_episodes,
                seen_cases=seen_cases,
                seen_trace_hashes=seen_trace_hashes,
            )
            copied = copy.deepcopy(row)
            copied["source_bundle"] = str(root)
            copied["source_bundle_index"] = source_number
            source_action_only.append(copied)
            stats["action_only_rows"] += 1
        source_manifests.append(
            {
                "root": str(root),
                "manifest_sha256": sha256_file(source["manifest_path"]),
                "provider_schema_version": source["provider"].get("schema_version"),
                "row_retention": source["provider"].get("row_retention"),
            }
        )

    stats["raw_rows"] = stats["reasoning_rows"] + stats["action_only_rows"]
    expectations = {
        "raw_rows": expected_raw_rows,
        "reasoning_rows": expected_reasoning_rows,
        "action_only_rows": expected_action_only_rows,
        "validation_rows": expected_validation_rows,
    }
    observed = {
        **stats,
        "validation_rows": len(source_rows["validation"]),
    }
    for name, expected in expectations.items():
        if expected is not None and int(observed[name]) != expected:
            raise ValueError(
                f"merged {name} mismatch: expected {expected}, got {observed[name]}"
            )

    staging.mkdir(parents=True)
    provider_root = staging / "canonical-dataset"
    staging_media = staging / "media"
    final_media = output_dir / "media"
    digest_cache: dict[Path, str] = {}
    merged_media: dict[str, dict[str, str]] = {}

    def rebind(row: dict[str, Any]) -> dict[str, Any]:
        copied = copy.deepcopy(row)
        paths: list[str] = []
        for image in copied.get("images", []):
            path, digest = _link_media(
                str(image),
                staging_media=staging_media,
                final_media=final_media,
                digest_cache=digest_cache,
            )
            paths.append(path)
            merged_media.setdefault(digest, {"sha256": digest, "path": path})
        copied["images"] = paths
        marker_count = sum(
            str(message.get("content", "")).count("<image>")
            for message in copied.get("messages", [])
        )
        if marker_count != len(paths):
            raise ValueError(
                f"merged row image-marker mismatch: {copied.get('episode_id')}"
            )
        return copied

    merged_rows = {
        split: sorted(
            (rebind(row) for row in source_rows[split]),
            key=lambda row: row["episode_id"],
        )
        for split in SPLITS
    }
    merged_action_only = sorted(
        (rebind(row) for row in source_action_only), key=lambda row: row["episode_id"]
    )
    rebound_by_episode = {
        row["episode_id"]: row for split in SPLITS for row in merged_rows[split]
    }
    merged_bindings: list[dict[str, Any]] = []
    for episode_id, binding in source_bindings.items():
        row = rebound_by_episode[episode_id]
        digests = [Path(image).stem.casefold() for image in row["images"]]
        copied = copy.deepcopy(binding)
        copied["image_sha256"] = digests
        copied["marker_counts"] = [
            str(message.get("content", "")).count("<image>")
            for message in row["messages"]
        ]
        merged_bindings.append(copied)

    artifacts: dict[str, dict[str, Any]] = {}
    for name in SPLITS:
        path = provider_root / f"{name}.jsonl"
        write_jsonl(path, merged_rows[name])
        artifacts[name] = {
            "path": path.name,
            "rows": len(merged_rows[name]),
            "sha256": sha256_file(path),
        }
    for name, rows in (
        ("index", sorted(source_index, key=lambda row: row["episode_id"])),
        ("action_only", merged_action_only),
        ("media_bindings", sorted(merged_bindings, key=lambda row: row["episode_id"])),
        ("media_manifest", [merged_media[key] for key in sorted(merged_media)]),
    ):
        path = provider_root / f"{name}.jsonl"
        write_jsonl(path, rows)
        artifacts[name] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    provider_manifest = {
        "schema_version": "ifv-merged-canonical-trace-export-v1",
        "dataset_version": "ifv-trajectory-sft-dataset-v3",
        "trajectory_version": "ifv-trajectory-sft-v3",
        "decision_policy_version": "unified-react-v1",
        "source_mode": "globally_gated_canonical_raw_trace_bundles",
        "target_fields_source": "canonical raw traces only",
        "media_source": "content-addressed verified source bundle media",
        "row_retention": {
            "raw_rows": int(stats["raw_rows"]),
            "reasoning_rows": int(stats["reasoning_rows"]),
            "action_only_rows": int(stats["action_only_rows"]),
            "dropped_rows": 0,
        },
        "global_gates": {
            "unique_case_ids": len(seen_cases),
            "unique_episode_ids": len(seen_episodes),
            "unique_trace_hashes": len(seen_trace_hashes),
            "forbidden_case_ids_checked": len(forbidden),
            "forbidden_case_overlap": 0,
            "source_count": len(sources),
        },
        "source_bundles": source_manifests,
        "stats": {
            **dict(sorted(stats.items())),
            "unique_media_files": len(merged_media),
            "image_references": sum(
                len(row.get("images", []))
                for split in SPLITS
                for row in merged_rows[split]
            ),
        },
        "artifacts": artifacts,
    }
    write_json(provider_root / "manifest.json", provider_manifest)

    policy_root = staging / "ms-swift-policy"
    policy_manifest = convert_policy_dataset(provider_root, policy_root)
    policy_audit = audit_derived_dataset(policy_root)
    write_json(staging / "policy-audit.json", policy_audit)
    if not policy_audit.get("passed"):
        raise RuntimeError("merged policy package failed strict audit")
    bundle_manifest = {
        "schema_version": "ifv-merged-canonical-sft-bundle-v1",
        "provider_dataset": provider_manifest,
        "policy_dataset": policy_manifest,
        "policy_audit": {
            "passed": True,
            "path": "policy-audit.json",
            "sha256": sha256_file(staging / "policy-audit.json"),
        },
    }
    write_json(staging / "manifest.json", bundle_manifest)
    os.replace(staging, output_dir)
    return bundle_manifest


def _claim_identity(
    episode_id: str,
    case_id: str,
    trace_hash: str,
    *,
    forbidden: set[str],
    seen_episodes: set[str],
    seen_cases: set[str],
    seen_trace_hashes: set[str],
) -> None:
    if (
        not episode_id
        or not case_id
        or len(trace_hash) != 64
        or case_id in forbidden
        or episode_id in seen_episodes
        or case_id in seen_cases
        or trace_hash in seen_trace_hashes
    ):
        raise ValueError(
            "merged source identity overlap or forbidden evaluation case: "
            f"episode={episode_id!r} case={case_id!r}"
        )
    seen_episodes.add(episode_id)
    seen_cases.add(case_id)
    seen_trace_hashes.add(trace_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--forbidden-case-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument("--expected-raw-rows", type=int)
    parser.add_argument("--expected-reasoning-rows", type=int)
    parser.add_argument("--expected-action-only-rows", type=int)
    parser.add_argument("--expected-validation-rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_bundles(
        args.source,
        args.output,
        forbidden_case_files=args.forbidden_case_jsonl,
        expected_raw_rows=args.expected_raw_rows,
        expected_reasoning_rows=args.expected_reasoning_rows,
        expected_action_only_rows=args.expected_action_only_rows,
        expected_validation_rows=args.expected_validation_rows,
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "row_retention": result["provider_dataset"]["row_retention"],
                "global_gates": result["provider_dataset"]["global_gates"],
                "stats": result["provider_dataset"]["stats"],
                "policy_passed": result["policy_audit"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
