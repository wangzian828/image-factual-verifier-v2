#!/usr/bin/env python3
"""Rebuild training rows from canonical traces plus an attested media map.

The raw-trace archive intentionally omits runtime stores.  A prior audited
delivery remains authoritative only for ordered media files and the message
positions at which those files became visible.  All model targets, tool calls,
tool results, thoughts, and observation locators are rebuilt from canonical
traces by the current exporter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "training"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import (
    canonical_json,
    load_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from ifv_training.policy import convert_policy_dataset
from scripts.trajectory.export_dataset import _cross_case_source_families
from src.trajectory.exporter import (
    export_trajectory_action_only_example,
    export_trajectory_sft_example,
    image_markers,
)
from src.trajectory.schema import DatasetTrajectorySFTExample


SPLITS = ("train", "validation", "test")
EXPECTED_RAW_SCHEMA = "ifv-canonical-trace-v5"
EXPECTED_POLICY_VERSION = "unified-react-v1"


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index + 1} is not an object")
            yield line_index, value


def _safe_indexed_trace(raw_root: Path, row: Mapping[str, Any]) -> Path:
    relative = Path(str(row.get("path", "")))
    path = (raw_root / relative).resolve()
    resolved_root = raw_root.resolve()
    if path == resolved_root or resolved_root not in path.parents:
        raise ValueError("raw trace index path escapes the bundle")
    if not path.is_file():
        raise FileNotFoundError(f"indexed raw trace is missing: {relative}")
    return path


def _source_metadata(source_manifest: Mapping[str, Any]) -> dict[str, str]:
    accepted = source_manifest.get("accepted_dataset")
    accepted = accepted if isinstance(accepted, Mapping) else {}
    source_runs = accepted.get("source_runs")
    source_runs = source_runs if isinstance(source_runs, list) else []
    source = source_runs[0] if source_runs and isinstance(source_runs[0], Mapping) else {}
    return {
        "source_run_id": str(source.get("run_id", "")),
        "runtime_commit": str(source.get("git_commit", "")),
        "release_id": str(source.get("release_id", "")),
        "runtime_contract_version": "",
        "process_reference_protocol_version": "",
    }


def _delivery_checksums(delivery_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = delivery_root / "SHA256SUMS"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid delivery checksum line {line_number}")
        digest, relative = parts[0].casefold(), parts[1].lstrip(" *")
        if (
            len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
            or not relative
            or relative in result
        ):
            raise ValueError(f"invalid delivery checksum entry at line {line_number}")
        result[relative] = digest
    return result


def _verify_delivery_file(
    delivery_root: Path,
    relative: str,
    checksums: Mapping[str, str],
) -> str:
    path = (delivery_root / relative).resolve()
    root = delivery_root.resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise ValueError(f"delivery file escapes or is missing: {relative}")
    expected = str(checksums.get(relative, ""))
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"delivery-level checksum mismatch: {relative}")
    return actual


def _reference_policy_rows(
    delivery_root: Path,
    checksums: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    policy_root = delivery_root / "ms-swift-policy"
    manifest = load_json(policy_root / "manifest.json")
    if manifest.get("dataset_version") != "ifv-ms-swift-qwen-agent-v2":
        raise ValueError("media reference must be the frozen v2 delivery")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("reference policy manifest has no artifacts")
    stale_inner_hashes: list[str] = []
    for name in (*SPLITS, "index"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"reference policy manifest lacks {name}")
        filename = str(artifact.get("path", f"{name}.jsonl"))
        relative = f"ms-swift-policy/{filename}"
        actual = _verify_delivery_file(delivery_root, relative, checksums)
        if actual != str(artifact.get("sha256", "")):
            # The published bundle rewrote image references to absolute paths
            # but retained the pre-relocation inner manifest.  Accept only the
            # file hash bound by the delivery-level checksum list and expose
            # the stale inner entry in the rebuilt manifest.
            stale_inner_hashes.append(name)

    by_location: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in _iter_jsonl(policy_root / "index.jsonl"):
        key = (str(row.get("split", "")), int(row.get("source_index", -1)))
        if key in by_location:
            raise ValueError(f"duplicate reference policy location: {key}")
        by_location[key] = row

    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        source_path = policy_root / str(artifacts[split].get("path", f"{split}.jsonl"))
        observed = 0
        for source_index, row in _iter_jsonl(source_path):
            observed += 1
            index = by_location.get((split, source_index))
            if index is None:
                raise ValueError(
                    f"reference policy row has no index: {split}[{source_index}]"
                )
            episode_id = str(index.get("episode_id", "")).strip()
            if not episode_id or episode_id in result:
                raise ValueError("reference policy episode IDs must be unique")
            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or not isinstance(images, list):
                raise ValueError(f"invalid reference policy row: {episode_id}")
            result[episode_id] = {
                "case_id": str(index.get("case_id", "")),
                "split": split,
                "roles": [str(message.get("role", "")) for message in messages],
                "marker_counts": [
                    str(message.get("content", "")).count("<image>")
                    for message in messages
                ],
                "images": [str(item) for item in images],
                "tool_call_count": int(index.get("tool_call_count", 0) or 0),
            }
        if observed != int(artifacts[split].get("rows", -1)):
            raise ValueError(f"reference policy row count mismatch: {split}")
    if len(result) != int(manifest.get("example_count", -1)):
        raise ValueError("reference policy total row count mismatch")
    return result, stale_inner_hashes


def _case_split(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for _, row in _iter_jsonl(path):
        case_id = str(row.get("case_id", "")).strip()
        split = str(row.get("split", "")).strip()
        group = str(row.get("split_group_id", "")).strip()
        if not case_id or case_id in result or split not in SPLITS or not group:
            raise ValueError("invalid or duplicate fixed case split row")
        result[case_id] = {"split": split, "split_group_id": group}
    return result


def _resolve_media(
    delivery_root: Path,
    reference: str,
    *,
    digest_cache: dict[Path, str],
) -> tuple[str, str]:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("reference media paths must be relative")
    path = (delivery_root / relative).resolve()
    media_root = (delivery_root / "images").resolve()
    if path == media_root or media_root not in path.parents or not path.is_file():
        raise ValueError(f"reference media escapes or is missing: {reference}")
    digest = digest_cache.get(path)
    if digest is None:
        digest = sha256_file(path)
        digest_cache[path] = digest
    if path.stem.casefold() != digest:
        raise ValueError(f"reference media filename/hash mismatch: {reference}")
    return str(path), digest


def bind_reference_media(
    generated_messages: list[dict[str, Any]],
    reference: Mapping[str, Any],
    *,
    delivery_root: Path,
    digest_cache: dict[Path, str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Attach only the reference media layout, never reference target text."""

    messages = copy.deepcopy(generated_messages)
    roles = [str(message.get("role", "")) for message in messages]
    reference_roles = [str(value) for value in reference.get("roles", [])]
    marker_counts = [int(value) for value in reference.get("marker_counts", [])]
    references = [str(value) for value in reference.get("images", [])]
    if roles != reference_roles or len(messages) != len(marker_counts):
        raise ValueError("canonical/reference message role sequence mismatch")
    if sum(marker_counts) != len(references):
        raise ValueError("reference image marker count does not match image list")
    if not references or marker_counts[1] != 1:
        raise ValueError("reference policy must have one initial model-visible image")
    for index, required in enumerate(marker_counts):
        if index != 1 and required and roles[index] != "tool_response":
            raise ValueError("additional images must follow a tool response")
    for index, (message, required) in enumerate(zip(messages, marker_counts)):
        current = str(message.get("content", "")).count("<image>")
        if current > required:
            raise ValueError(
                f"canonical exporter produced excess image markers at message {index}"
            )
        if required - current:
            message["content"] = (
                str(message["content"]).rstrip()
                + "\n\n"
                + image_markers(required - current)
            )
    resolved: list[str] = []
    digests: list[str] = []
    for reference_path in references:
        path, digest = _resolve_media(
            delivery_root,
            reference_path,
            digest_cache=digest_cache,
        )
        resolved.append(path)
        digests.append(digest)
    if sum(
        str(message.get("content", "")).count("<image>") for message in messages
    ) != len(resolved):
        raise ValueError("bound media marker count does not match resolved images")
    return messages, resolved, digests


def _rebind_fallback_image(trace: dict[str, Any], delivery_root: Path) -> tuple[dict[str, Any], str]:
    rebound = copy.deepcopy(trace)
    state = rebound.get("state")
    state = state if isinstance(state, dict) else {}
    runtime_case = state.get("runtime_case")
    runtime_case = runtime_case if isinstance(runtime_case, dict) else {}
    expected = str(runtime_case.get("image_sha256", "")).strip().casefold()
    matches = sorted((delivery_root / "images").glob(expected + ".*"))
    if len(matches) != 1 or sha256_file(matches[0]) != expected:
        raise ValueError("raw runtime image cannot be resolved by SHA-256")
    path = str(matches[0].resolve())
    rebound["image_path"] = path
    state["image_path"] = path
    runtime_case["image_path"] = path
    state["runtime_case"] = runtime_case
    rebound["state"] = state
    return rebound, path


def export_bundle(
    raw_root: Path,
    delivery_root: Path,
    output_dir: Path,
    *,
    raw_archive: Path,
    raw_archive_sha256: str,
    reference_archive: Path,
    reference_archive_sha256: str,
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    delivery_root = delivery_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    raw_archive = raw_archive.expanduser().resolve()
    reference_archive = reference_archive.expanduser().resolve()
    if sha256_file(raw_archive) != raw_archive_sha256.casefold():
        raise ValueError("raw archive SHA-256 mismatch")
    if sha256_file(reference_archive) != reference_archive_sha256.casefold():
        raise ValueError("reference delivery archive SHA-256 mismatch")
    if output_dir.exists():
        raise FileExistsError(f"output must not exist: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".in-progress")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)

    source_manifest_path = delivery_root / "SOURCE_MANIFEST.json"
    case_split_path = delivery_root / "case-split" / "case_split.jsonl"
    delivery_checksums = _delivery_checksums(delivery_root)
    _verify_delivery_file(delivery_root, "SOURCE_MANIFEST.json", delivery_checksums)
    _verify_delivery_file(
        delivery_root,
        "case-split/case_split.jsonl",
        delivery_checksums,
    )
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("schema_version") != "ifv-sft-training-package-v2":
        raise ValueError("unsupported reference delivery manifest")
    counts = source_manifest.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    source = _source_metadata(source_manifest)
    fixed_split = _case_split(case_split_path)
    references, stale_inner_hashes = _reference_policy_rows(
        delivery_root,
        delivery_checksums,
    )

    raw_index_path = raw_root / "index.jsonl"
    raw_rows = [row for _, row in _iter_jsonl(raw_index_path)]
    if len(raw_rows) != int(counts.get("selected_release_cases", -1)):
        raise ValueError("raw index does not match selected release count")
    if len(references) != int(counts.get("policy_rows", -1)):
        raise ValueError("reference policy does not match declared policy count")
    expected_action_only = int(counts.get("action_only_rows", -1))
    if len(raw_rows) - len(references) != expected_action_only:
        raise ValueError("raw/policy difference does not match action-only count")

    output_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    output_index: list[dict[str, Any]] = []
    action_only_rows: list[dict[str, Any]] = []
    media_bindings: list[dict[str, Any]] = []
    media_manifest: dict[str, dict[str, str]] = {}
    digest_cache: dict[Path, str] = {}
    seen_episode: set[str] = set()
    seen_case: set[str] = set()
    stats: Counter[str] = Counter()

    for raw_index, record in enumerate(raw_rows):
        trace_path = _safe_indexed_trace(raw_root, record)
        expected_trace_sha = str(record.get("sha256", "")).strip().casefold()
        if sha256_file(trace_path) != expected_trace_sha:
            raise ValueError(f"raw trace hash mismatch at index {raw_index}")
        trace = load_json(trace_path)
        episode_id = str(record.get("episode_id", "")).strip()
        case_id = str(record.get("case_id", "")).strip()
        state = trace.get("state")
        state = state if isinstance(state, Mapping) else {}
        runtime_case = state.get("runtime_case")
        runtime_case = runtime_case if isinstance(runtime_case, Mapping) else {}
        if (
            not episode_id
            or not case_id
            or episode_id in seen_episode
            or case_id in seen_case
            or str(trace.get("image_id", "")) != episode_id
            or str(runtime_case.get("case_id") or trace.get("case_id", "")) != case_id
        ):
            raise ValueError(f"raw trace identity mismatch at index {raw_index}")
        seen_episode.add(episode_id)
        seen_case.add(case_id)
        if (
            trace.get("schema_version") != EXPECTED_RAW_SCHEMA
            or trace.get("decision_policy_version") != EXPECTED_POLICY_VERSION
            or trace.get("input_mode") != "image_only"
            or trace.get("termination") != "success"
        ):
            raise ValueError(f"raw trace contract mismatch at index {raw_index}")
        split = fixed_split.get(case_id)
        if split is None:
            raise ValueError(f"fixed split lacks selected case: {case_id}")

        rebound, raw_image_path = _rebind_fallback_image(trace, delivery_root)
        reference = references.get(episode_id)
        if reference is None:
            try:
                export_trajectory_sft_example(rebound, source_metadata=source)
            except ValueError as exc:
                if "provider-visible thought" not in str(exc):
                    raise
            else:
                raise ValueError(
                    "reference classifies a reasoning-capable trace as action-only"
                )
            action = export_trajectory_action_only_example(
                rebound,
                source_metadata=source,
            ).model_dump(mode="json")
            action["images"] = [raw_image_path]
            action_only_rows.append(
                {
                    **action,
                    "split": split["split"],
                    "split_group_id": split["split_group_id"],
                    "source_trace_sha256": expected_trace_sha,
                    "media_limit": "initial image only; excluded from reasoning SFT",
                }
            )
            stats["action_only_rows"] += 1
            continue

        if reference["case_id"] != case_id or reference["split"] != split["split"]:
            raise ValueError(f"reference binding mismatch for {episode_id}")
        exported = export_trajectory_sft_example(
            rebound,
            source_metadata=source,
        ).model_dump(mode="json")
        if int(exported["tool_call_count"]) != int(reference["tool_call_count"]):
            raise ValueError(f"tool-call count mismatch for {episode_id}")
        messages, images, image_digests = bind_reference_media(
            exported["messages"],
            reference,
            delivery_root=delivery_root,
            digest_cache=digest_cache,
        )
        exported["messages"] = messages
        exported["images"] = images
        exported["message_count"] = len(messages)
        exported["token_count_estimate"] = len(
            canonical_json(
                {"messages": messages, "tools": exported.get("tools", "")}
            ).encode("utf-8")
        )
        dataset_row = DatasetTrajectorySFTExample(
            **exported,
            split=split["split"],
            split_group_id=split["split_group_id"],
            source_family_keys=_cross_case_source_families(trace),
            teacher_score=0.0,
        ).model_dump(mode="json")
        output_rows[split["split"]].append(dataset_row)
        output_index.append(
            {
                "raw_index": raw_index,
                "episode_id": episode_id,
                "case_id": case_id,
                "split": split["split"],
                "split_group_id": split["split_group_id"],
                "source_trace": str(trace_path),
                "source_trace_sha256": expected_trace_sha,
                "tool_call_count": exported["tool_call_count"],
                "message_count": exported["message_count"],
                "image_count": len(images),
                "token_count_estimate": exported["token_count_estimate"],
            }
        )
        media_bindings.append(
            {
                "episode_id": episode_id,
                "source_trace_sha256": expected_trace_sha,
                "image_sha256": image_digests,
                "marker_counts": reference["marker_counts"],
                "binding_policy": "reference_media_only_canonical_targets",
            }
        )
        for path, digest in zip(images, image_digests):
            media_manifest.setdefault(digest, {"sha256": digest, "path": path})
        stats["reasoning_rows"] += 1
        stats["image_references"] += len(images)

    if stats["reasoning_rows"] != len(references):
        raise ValueError("not every reference reasoning row was rebuilt")
    if stats["action_only_rows"] != expected_action_only:
        raise ValueError("action-only output count mismatch")

    provider_root = staging / "canonical-dataset"
    artifacts: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = sorted(output_rows[split], key=lambda item: item["episode_id"])
        path = provider_root / f"{split}.jsonl"
        write_jsonl(path, rows)
        artifacts[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    for name, rows in (
        ("index", sorted(output_index, key=lambda item: item["episode_id"])),
        ("action_only", sorted(action_only_rows, key=lambda item: item["episode_id"])),
        ("media_bindings", sorted(media_bindings, key=lambda item: item["episode_id"])),
        ("media_manifest", [media_manifest[key] for key in sorted(media_manifest)]),
    ):
        path = provider_root / f"{name}.jsonl"
        write_jsonl(path, rows)
        artifacts[name] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    provider_manifest = {
        "schema_version": "ifv-canonical-trace-media-bound-export-v1",
        "dataset_version": "ifv-trajectory-sft-dataset-v3",
        "trajectory_version": "ifv-trajectory-sft-v3",
        "decision_policy_version": EXPECTED_POLICY_VERSION,
        "source_mode": "canonical_raw_trace_plus_reference_media_projection",
        "target_fields_source": "canonical raw trace only",
        "media_source": "frozen audited v2 delivery only",
        "media_binding": {
            "copies_reference_message_content": False,
            "copies_reference_tool_calls": False,
            "copies_reference_tool_responses": False,
            "copies_reference_final_answers": False,
            "uses_reference_roles_and_marker_counts": True,
            "all_media_content_hash_verified": True,
        },
        "reference_integrity": {
            "archive_sha256_verified": True,
            "delivery_level_checksums_verified": True,
            "stale_inner_policy_manifest_artifacts": stale_inner_hashes,
            "stale_inner_manifest_accepted_only_via_delivery_checksum": True,
        },
        "teacher_score_recovered": False,
        "example_counts": {
            split: int(artifacts[split]["rows"]) for split in SPLITS
        },
        "row_retention": {
            "raw_rows": len(raw_rows),
            "reasoning_rows": int(stats["reasoning_rows"]),
            "action_only_rows": int(stats["action_only_rows"]),
            "dropped_rows": 0,
        },
        "source_artifacts": {
            "raw_archive": {
                "path": str(raw_archive),
                "sha256": raw_archive_sha256.casefold(),
            },
            "reference_archive": {
                "path": str(reference_archive),
                "sha256": reference_archive_sha256.casefold(),
            },
            "raw_index": {
                "path": str(raw_index_path),
                "sha256": sha256_file(raw_index_path),
            },
            "delivery_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
            },
            "case_split": {
                "path": str(case_split_path),
                "sha256": sha256_file(case_split_path),
            },
            "reference_policy_manifest": {
                "path": str(delivery_root / "ms-swift-policy" / "manifest.json"),
                "sha256": sha256_file(
                    delivery_root / "ms-swift-policy" / "manifest.json"
                ),
            },
        },
        "stats": {
            **dict(sorted(stats.items())),
            "unique_media_files": len(media_manifest),
        },
        "artifacts": artifacts,
    }
    write_json(provider_root / "manifest.json", provider_manifest)

    policy_root = staging / "ms-swift-policy"
    policy_manifest = convert_policy_dataset(provider_root, policy_root)
    policy_audit = audit_derived_dataset(policy_root)
    write_json(staging / "policy-audit.json", policy_audit)
    if not policy_audit.get("passed"):
        raise RuntimeError("rebuilt policy package failed strict audit")
    bundle_manifest = {
        "schema_version": "ifv-canonical-sft-rebuild-bundle-v1",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-delivery", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--raw-archive-sha256", required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--reference-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_bundle(
        args.raw_root,
        args.reference_delivery,
        args.output,
        raw_archive=args.raw_archive,
        raw_archive_sha256=args.raw_archive_sha256,
        reference_archive=args.reference_archive,
        reference_archive_sha256=args.reference_archive_sha256,
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "row_retention": result["provider_dataset"]["row_retention"],
                "stats": result["provider_dataset"]["stats"],
                "policy_passed": result["policy_audit"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
