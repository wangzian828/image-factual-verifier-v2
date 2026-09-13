#!/usr/bin/env python3
"""Rebuild canonical SFT rows from an attested request-image sidecar.

The sidecar is authoritative only for immutable raw trace bytes and the exact
ordered images present in each provider request.  Policy targets, thoughts,
tool calls, tool responses, and final answers are rebuilt by the canonical
trajectory exporter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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
    _trajectory_candidate_steps,
    export_trajectory_action_only_example,
    export_trajectory_sft_example,
)
from src.trajectory.media_projection import project_attested_request_media
from src.trajectory.schema import DatasetTrajectorySFTExample


SPLITS = ("train", "validation", "test")
EXPECTED_RAW_SCHEMA = "ifv-canonical-trace-v5"
EXPECTED_POLICY_VERSION = "unified-react-v1"
EXPECTED_DISTRIBUTION_SCHEMA = "ifv-request-image-binding-distribution-v1"


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index + 1} is not an object")
            yield line_index, value


def _safe_file(root: Path, relative: str, *, subtree: str = "") -> Path:
    rel = Path(relative)
    path = (root / rel).resolve()
    allowed = (root / subtree).resolve() if subtree else root.resolve()
    if (
        rel.is_absolute()
        or path == allowed
        or allowed not in path.parents
        or not path.is_file()
    ):
        raise ValueError(f"sidecar file escapes or is missing: {relative}")
    return path


def _verify_extracted_checksums(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("sidecar has no SHA256SUMS")
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid checksum line: {line_number}")
        expected, relative = fields[0].casefold(), fields[1].lstrip(" *")
        if relative in result or len(expected) != 64:
            raise ValueError(f"invalid checksum entry: {line_number}")
        path = _safe_file(root, relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"extracted checksum mismatch: {relative}")
        result[relative] = actual
    if not result:
        raise ValueError("sidecar checksum manifest is empty")
    return result


def _distribution(
    distribution_manifest: Path,
    source_archive: Path,
    source_archive_sha256: str,
) -> dict[str, Any]:
    value = load_json(distribution_manifest)
    audit = value.get("audit")
    audit = audit if isinstance(audit, Mapping) else value
    published_sha = str(
        value.get("archive_sha256") or value.get("sha256") or ""
    ).casefold()
    published_bytes = int(value.get("archive_bytes") or value.get("bytes") or -1)
    expected = source_archive_sha256.casefold()
    if sha256_file(source_archive) != expected:
        raise ValueError("request-image source archive SHA-256 mismatch")
    if published_sha != expected:
        raise ValueError("distribution manifest does not bind the source archive")
    if published_bytes != source_archive.stat().st_size:
        raise ValueError("distribution manifest archive size mismatch")
    if audit.get("passed") is not True or int(audit.get("errors_total", -1)) != 0:
        raise ValueError("request-image distribution audit did not pass")
    if "audit" in value and (
        value.get("independent_extracted_audit_passed") is not True
        or value.get("sha256sum_check_passed") is not True
    ):
        raise ValueError("published extracted sidecar checks did not pass")
    return {
        "schema_version": str(
            value.get("schema_version") or EXPECTED_DISTRIBUTION_SCHEMA
        ),
        "archive_sha256": published_sha,
        "archive_bytes": published_bytes,
        **dict(audit),
    }


def _request_bindings(root: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    rows = 0
    for _, binding in _iter_jsonl(root / "request-image-bindings.jsonl"):
        trace_id = str(binding.get("trace_id", "")).strip()
        request_id = str(binding.get("context_request_id", "")).strip()
        key = (trace_id, request_id)
        if not trace_id or not request_id or key in seen:
            raise ValueError("invalid or duplicate request-image binding")
        seen.add(key)
        grouped[trace_id].append(binding)
        rows += 1
    return grouped, rows


def _rebind_trace_image(
    trace: Mapping[str, Any],
    *,
    image_path: str,
    image_sha256: str,
) -> dict[str, Any]:
    rebound = copy.deepcopy(dict(trace))
    state = rebound.get("state")
    state = state if isinstance(state, dict) else {}
    runtime_case = state.get("runtime_case")
    runtime_case = runtime_case if isinstance(runtime_case, dict) else {}
    rebound["image_path"] = image_path
    state["image_path"] = image_path
    runtime_case["image_path"] = image_path
    runtime_case["image_sha256"] = image_sha256
    state["runtime_case"] = runtime_case
    rebound["state"] = state
    return rebound


def _split_group_id(case_id: str) -> str:
    if len(case_id) <= 100:
        return case_id
    return "case-sha256:" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def export_bundle(
    sidecar_root: Path,
    output_dir: Path,
    *,
    distribution_manifest: Path,
    source_archive: Path,
    source_archive_sha256: str,
    split: str = "train",
) -> dict[str, Any]:
    root = sidecar_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    distribution_manifest = distribution_manifest.expanduser().resolve()
    source_archive = source_archive.expanduser().resolve()
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if output_dir.exists():
        raise FileExistsError(f"output must not exist: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".in-progress")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")

    distribution = _distribution(
        distribution_manifest,
        source_archive,
        source_archive_sha256,
    )
    audit = load_json(root / "audit-report.json")
    if audit.get("passed") is not True or int(audit.get("errors_total", -1)) != 0:
        raise ValueError("extracted request-image audit did not pass")
    checksums = _verify_extracted_checksums(root)
    bindings, binding_count = _request_bindings(root)
    raw_rows = [row for _, row in _iter_jsonl(root / "index.jsonl")]
    expected_rows = int(distribution.get("trace_count", -1))
    if (
        len(raw_rows) != expected_rows
        or int(audit.get("trace_count", -1)) != expected_rows
    ):
        raise ValueError("request-image trace count mismatch")
    if binding_count != int(audit.get("request_count", -1)):
        raise ValueError("request-image request count mismatch")

    staging.mkdir(parents=True)
    output_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    output_index: list[dict[str, Any]] = []
    action_only_rows: list[dict[str, Any]] = []
    media_bindings: list[dict[str, Any]] = []
    media_manifest: dict[str, dict[str, str]] = {}
    digest_cache: dict[Path, str] = {}
    seen_cases: set[str] = set()
    seen_episodes: set[str] = set()
    seen_traces: set[str] = set()
    stats: Counter[str] = Counter()

    for raw_index, record in enumerate(raw_rows):
        trace_path = _safe_file(
            root, str(record.get("trace_path", "")), subtree="traces"
        )
        expected_trace_sha = str(record.get("trace_sha256", "")).casefold()
        if sha256_file(trace_path) != expected_trace_sha:
            raise ValueError(f"raw trace hash mismatch at index {raw_index}")
        trace = load_json(trace_path)
        episode_id = str(record.get("trace_id", "")).strip()
        case_id = str(record.get("case_id", "")).strip()
        state = trace.get("state")
        state = state if isinstance(state, Mapping) else {}
        runtime_case = state.get("runtime_case")
        runtime_case = runtime_case if isinstance(runtime_case, Mapping) else {}
        if (
            not episode_id
            or not case_id
            or episode_id in seen_episodes
            or case_id in seen_cases
            or expected_trace_sha in seen_traces
            or str(trace.get("image_id", "")) != episode_id
            or str(runtime_case.get("case_id") or trace.get("case_id", "")) != case_id
        ):
            raise ValueError(f"raw trace identity mismatch at index {raw_index}")
        if (
            trace.get("schema_version") != EXPECTED_RAW_SCHEMA
            or trace.get("decision_policy_version") != EXPECTED_POLICY_VERSION
            or trace.get("input_mode") != "image_only"
            or trace.get("termination") != "success"
        ):
            raise ValueError(f"raw trace contract mismatch at index {raw_index}")
        seen_episodes.add(episode_id)
        seen_cases.add(case_id)
        seen_traces.add(expected_trace_sha)

        candidates = _trajectory_candidate_steps(trace)
        projection = project_attested_request_media(
            candidate_steps=candidates,
            request_bindings=bindings.get(episode_id, []),
            sidecar_root=root,
            expected_trace_id=episode_id,
            digest_cache=digest_cache,
        )
        first_path = Path(projection.initial_images[0])
        first_digest = digest_cache.get(first_path) or sha256_file(first_path)
        digest_cache[first_path] = first_digest
        rebound = _rebind_trace_image(
            trace,
            image_path=str(first_path),
            image_sha256=first_digest,
        )
        react_steps = [step for _, step, kind in candidates if kind == "react"]
        action_only = any(
            not str(step.get("thought", "") or "").strip() for step in react_steps
        )
        source = {
            "source_run_id": episode_id.split("--", 1)[0],
            "runtime_commit": "",
            "release_id": "",
            "runtime_contract_version": "",
            "process_reference_protocol_version": "",
        }
        if action_only:
            exported_action = export_trajectory_action_only_example(
                rebound,
                source_metadata=source,
                media_projection=projection,
            ).model_dump(mode="json")
            exported_action.update(
                {
                    "split": split,
                    "split_group_id": _split_group_id(case_id),
                    "source_trace_sha256": expected_trace_sha,
                    "media_binding_schema": EXPECTED_DISTRIBUTION_SCHEMA,
                }
            )
            action_only_rows.append(exported_action)
            stats["action_only_rows"] += 1
            continue

        exported = export_trajectory_sft_example(
            rebound,
            source_metadata=source,
            media_projection=projection,
        ).model_dump(mode="json")
        marker_counts = [
            str(message.get("content", "")).count("<image>")
            for message in exported["messages"]
        ]
        if sum(marker_counts) != len(exported["images"]):
            raise ValueError(f"image marker count mismatch for {episode_id}")
        dataset_row = DatasetTrajectorySFTExample(
            **exported,
            split=split,
            split_group_id=_split_group_id(case_id),
            source_family_keys=_cross_case_source_families(trace),
            teacher_score=0.0,
        ).model_dump(mode="json")
        output_rows[split].append(dataset_row)
        image_digests: list[str] = []
        for image in exported["images"]:
            path = Path(image).resolve()
            digest = digest_cache.get(path) or sha256_file(path)
            digest_cache[path] = digest
            image_digests.append(digest)
            media_manifest.setdefault(digest, {"sha256": digest, "path": str(path)})
        output_index.append(
            {
                "raw_index": raw_index,
                "episode_id": episode_id,
                "case_id": case_id,
                "split": split,
                "split_group_id": _split_group_id(case_id),
                "source_trace": str(trace_path),
                "source_trace_sha256": expected_trace_sha,
                "tool_call_count": int(exported["tool_call_count"]),
                "message_count": int(exported["message_count"]),
                "image_count": len(exported["images"]),
                "token_count_estimate": int(exported["token_count_estimate"]),
            }
        )
        media_bindings.append(
            {
                "episode_id": episode_id,
                "source_trace_sha256": expected_trace_sha,
                "image_sha256": image_digests,
                "marker_counts": marker_counts,
                "binding_policy": "attested_request_images_canonical_targets",
            }
        )
        stats["reasoning_rows"] += 1
        stats["image_references"] += len(exported["images"])

    if stats["reasoning_rows"] + stats["action_only_rows"] != len(raw_rows):
        raise ValueError("not every raw trace was retained")
    if set(bindings) != seen_episodes:
        raise ValueError("request-image sidecar contains an unknown or missing trace")

    provider_root = staging / "canonical-dataset"
    artifacts: dict[str, dict[str, Any]] = {}
    for name in SPLITS:
        rows = sorted(output_rows[name], key=lambda item: item["episode_id"])
        path = provider_root / f"{name}.jsonl"
        write_jsonl(path, rows)
        artifacts[name] = {
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
        "schema_version": "ifv-attested-request-image-canonical-export-v1",
        "dataset_version": "ifv-trajectory-sft-dataset-v3",
        "trajectory_version": "ifv-trajectory-sft-v3",
        "decision_policy_version": EXPECTED_POLICY_VERSION,
        "source_mode": "canonical_raw_trace_plus_attested_request_images",
        "target_fields_source": "canonical raw trace only",
        "media_source": "frozen request-image binding sidecar only",
        "media_binding": {
            "copies_provider_input_text": False,
            "copies_target_text": False,
            "uses_ordered_cumulative_request_images": True,
            "preserves_repeated_image_slots": True,
            "all_media_content_hash_verified": True,
        },
        "row_retention": {
            "raw_rows": len(raw_rows),
            "reasoning_rows": int(stats["reasoning_rows"]),
            "action_only_rows": int(stats["action_only_rows"]),
            "dropped_rows": 0,
        },
        "source_artifacts": {
            "archive": {
                "path": str(source_archive),
                "sha256": source_archive_sha256.casefold(),
            },
            "distribution_manifest": {
                "path": str(distribution_manifest),
                "sha256": sha256_file(distribution_manifest),
            },
            "extracted_checksum_manifest": {
                "path": str(root / "SHA256SUMS"),
                "sha256": sha256_file(root / "SHA256SUMS"),
                "verified_files": len(checksums),
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
        raise RuntimeError("attested request-image policy package failed strict audit")
    bundle_manifest = {
        "schema_version": "ifv-attested-request-image-sft-bundle-v1",
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
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--distribution-manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_bundle(
        args.sidecar_root,
        args.output,
        distribution_manifest=args.distribution_manifest,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        split=args.split,
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
