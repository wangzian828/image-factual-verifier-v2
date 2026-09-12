#!/usr/bin/env python3
"""Independently audit a canonical raw-trace SFT rebuild.

This checker deliberately does not call the export functions.  It reconstructs
the raw policy-turn sequence, compares every thought/call/result/final target,
checks the frozen reference media binding, and verifies that the converted
policy keeps every final evidence locator grounded in a prior unmasked success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REJECTED_ACTION_TYPES = {"format_error", "output_rejected", "policy_replan"}
POLICY_STAGES = {"unified_react", "unified_judgment"}
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"^<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _raw_json_prefix(content: str) -> Any:
    text = str(content).strip()
    if text.startswith("<tool_response>"):
        text = text[len("<tool_response>") :].lstrip()
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except Exception:
        return None


def _tool_call(content: str) -> dict[str, Any]:
    call = json.loads(content)
    if not isinstance(call, Mapping):
        raise ValueError("tool call is not an object")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments are not an object")
    return {"name": str(call.get("name", "")), "arguments": dict(arguments)}


def _answer(content: str) -> dict[str, Any]:
    match = ANSWER_RE.search(str(content))
    if match is None:
        raise ValueError("assistant target has no answer block")
    value = json.loads(match.group(1))
    if not isinstance(value, dict):
        raise ValueError("answer block is not an object")
    return value


def _thought(content: str) -> str:
    match = THINK_RE.search(str(content).strip())
    if match is None:
        raise ValueError("assistant action has no think block")
    return match.group(1).strip()


def _canonical_result(raw: Any) -> Any:
    value = raw
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("<tool_response>") and text.endswith("</tool_response>"):
            text = text[len("<tool_response>") : -len("</tool_response>")].strip()
        try:
            value = json.loads(text)
        except Exception:
            return text
    if isinstance(value, Mapping) and "result" in value:
        return value["result"]
    return value


def _candidate_steps(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    state = _mapping(trace.get("state"))
    result: list[Mapping[str, Any]] = []
    for step in state.get("all_steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        if str(step.get("action_type", "")) in REJECTED_ACTION_TYPES:
            continue
        if str(step.get("stage", "")) not in POLICY_STAGES:
            continue
        metadata = _mapping(step.get("metadata"))
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        result.append(step)
    return result


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


def _artifact_checks(audit: Audit, root: Path, manifest: Mapping[str, Any], label: str) -> None:
    artifacts = _mapping(manifest.get("artifacts"))
    for name, artifact in artifacts.items():
        artifact = _mapping(artifact)
        path = root / str(artifact.get("path", ""))
        audit.require(path.is_file(), f"{label}_artifact_missing", str(name))
        if not path.is_file():
            continue
        audit.require(
            _sha256(path) == str(artifact.get("sha256", "")),
            f"{label}_artifact_hash",
            str(name),
        )
        if path.suffix == ".jsonl":
            audit.require(
                len(_jsonl(path)) == int(artifact.get("rows", -1)),
                f"{label}_artifact_rows",
                str(name),
            )


def _rows_by_episode(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split in SPLITS:
        for index, row in enumerate(_jsonl(root / f"{split}.jsonl")):
            rows[(split, index)] = row
    index_rows = _jsonl(root / "index.jsonl")
    by_episode: dict[str, dict[str, Any]] = {}
    index_by_episode: dict[str, dict[str, Any]] = {}
    if index_rows and not all("source_index" in item for item in index_rows):
        for row in rows.values():
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in by_episode:
                raise ValueError("dataset rows lack unique episode IDs")
            by_episode[episode_id] = row
        for item in index_rows:
            episode_id = str(item.get("episode_id", ""))
            if not episode_id or episode_id in index_by_episode:
                raise ValueError("dataset index lacks unique episode IDs")
            index_by_episode[episode_id] = item
        if set(by_episode) != set(index_by_episode):
            raise ValueError("dataset row/index episode sets differ")
        return by_episode, index_by_episode
    for item in index_rows:
        key = (str(item.get("split", "")), int(item.get("source_index", -1)))
        episode_id = str(item.get("episode_id", ""))
        if not episode_id or key not in rows or episode_id in by_episode:
            raise ValueError(f"invalid dataset index binding: {key}")
        by_episode[episode_id] = rows[key]
        index_by_episode[episode_id] = item
    if len(by_episode) != len(rows):
        raise ValueError("dataset index does not cover every row")
    return by_episode, index_by_episode


def _reference_rows(delivery_root: Path) -> dict[str, dict[str, Any]]:
    policy_root = delivery_root / "ms-swift-policy"
    return _rows_by_episode(policy_root)[0]


def _trace_content_audit(
    audit: Audit,
    *,
    trace: Mapping[str, Any],
    row: Mapping[str, Any],
    episode_id: str,
) -> set[str]:
    location = episode_id
    messages = row.get("messages")
    audit.require(isinstance(messages, list), "messages_type", location)
    if not isinstance(messages, list):
        return set()
    candidates = _candidate_steps(trace)
    react_steps = [step for step in candidates if step.get("stage") == "unified_react"]
    judgment_steps = [
        step for step in candidates if step.get("stage") == "unified_judgment"
    ]
    call_indexes = [
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "tool_call"
    ]
    audit.require(len(call_indexes) == len(react_steps), "raw_call_count", location)
    successful_ids: set[str] = set()
    for turn, (message_index, step) in enumerate(zip(call_indexes, react_steps)):
        turn_location = f"{location}:call:{turn}"
        audit.require(message_index > 0, "missing_action_thought", turn_location)
        audit.require(message_index + 1 < len(messages), "missing_tool_response", turn_location)
        if message_index <= 0 or message_index + 1 >= len(messages):
            continue
        assistant = _mapping(messages[message_index - 1])
        response = _mapping(messages[message_index + 1])
        audit.require(assistant.get("role") == "assistant", "action_thought_role", turn_location)
        audit.require(response.get("role") == "tool_response", "tool_response_role", turn_location)
        try:
            observed_thought = _thought(str(assistant.get("content", "")))
        except ValueError:
            observed_thought = "<invalid>"
        audit.require(
            observed_thought == str(step.get("thought", "")).strip(),
            "raw_thought_drift",
            turn_location,
        )
        try:
            observed_call = _tool_call(str(messages[message_index].get("content", "")))
        except Exception:
            observed_call = {}
        audit.require(
            observed_call.get("name") == str(step.get("tool_name", "")),
            "raw_tool_name_drift",
            turn_location,
        )
        audit.require(
            observed_call.get("arguments") == dict(_mapping(step.get("tool_args"))),
            "raw_tool_args_drift",
            turn_location,
        )
        response_payload = _mapping(_raw_json_prefix(str(response.get("content", ""))))
        locator = _mapping(response_payload.get("observation_locator"))
        metadata = _mapping(step.get("metadata"))
        observation_id = str(metadata.get("function_call_id", ""))
        audit.require(
            str(locator.get("observation_id", "")) == observation_id,
            "raw_observation_id_drift",
            turn_location,
        )
        audit.require(
            str(locator.get("tool_name", "")) == str(step.get("tool_name", "")),
            "raw_observation_tool_drift",
            turn_location,
        )
        audit.require(
            response_payload.get("result") == _canonical_result(step.get("tool_result")),
            "raw_tool_result_drift",
            turn_location,
        )
        if locator.get("tool_success") is True and observation_id:
            successful_ids.add(observation_id)
        audit.counts["raw_tool_calls"] += 1
    audit.require(len(judgment_steps) == 1, "raw_judgment_count", location)
    if judgment_steps and messages:
        try:
            observed_answer = _answer(str(messages[-1].get("content", "")))
        except Exception:
            observed_answer = {}
        expected_answer = dict(
            _mapping(_mapping(judgment_steps[-1].get("metadata")).get("policy_action"))
        )
        audit.require(observed_answer == expected_answer, "raw_final_target_drift", location)
        final_ids = [
            str(value).strip()
            for value in observed_answer.get("verdict_observation_ids", []) or []
            if str(value).strip()
        ]
        audit.require(
            set(final_ids).issubset(successful_ids),
            "raw_final_id_not_visible",
            location,
        )
        audit.counts["raw_final_ids"] += len(final_ids)
    return successful_ids


def audit_bundle(
    *,
    raw_root: Path,
    reference_delivery: Path,
    rebuilt_root: Path,
    repo_root: Path,
    verify_source_archives: bool = False,
) -> dict[str, Any]:
    audit = Audit()
    provider_root = rebuilt_root / "canonical-dataset"
    policy_root = rebuilt_root / "ms-swift-policy"
    bundle_manifest = json.loads((rebuilt_root / "manifest.json").read_text(encoding="utf-8"))
    provider_manifest = json.loads((provider_root / "manifest.json").read_text(encoding="utf-8"))
    policy_manifest = json.loads((policy_root / "manifest.json").read_text(encoding="utf-8"))
    _artifact_checks(audit, provider_root, provider_manifest, "provider")
    _artifact_checks(audit, policy_root, policy_manifest, "policy")

    if verify_source_archives:
        for name in ("raw_archive", "reference_archive"):
            artifact = _mapping(_mapping(provider_manifest.get("source_artifacts")).get(name))
            path = Path(str(artifact.get("path", "")))
            audit.require(path.is_file(), "source_archive_missing", name)
            if path.is_file():
                audit.require(
                    _sha256(path) == str(artifact.get("sha256", "")),
                    "source_archive_hash",
                    name,
                )

    raw_index_rows = _jsonl(raw_root / "index.jsonl")
    raw_by_episode: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for item in raw_index_rows:
        episode_id = str(item.get("episode_id", ""))
        path = (raw_root / str(item.get("path", ""))).resolve()
        audit.require(path.is_file(), "raw_trace_missing", episode_id)
        if not path.is_file():
            continue
        digest = _sha256(path)
        audit.require(digest == str(item.get("sha256", "")), "raw_trace_hash", episode_id)
        trace = json.loads(path.read_text(encoding="utf-8"))
        audit.require(str(trace.get("image_id", "")) == episode_id, "raw_identity", episode_id)
        audit.require(episode_id not in raw_by_episode, "raw_duplicate_episode", episode_id)
        raw_by_episode[episode_id] = (item, trace)

    provider_rows, provider_index = _rows_by_episode(provider_root)
    action_rows = {
        str(row.get("episode_id", "")): row
        for row in _jsonl(provider_root / "action_only.jsonl")
    }
    audit.require(
        set(provider_rows).isdisjoint(action_rows),
        "reasoning_action_overlap",
        "dataset",
    )
    audit.require(
        set(provider_rows) | set(action_rows) == set(raw_by_episode),
        "raw_row_partition",
        "dataset",
    )

    references = _reference_rows(reference_delivery)
    audit.require(set(references) == set(provider_rows), "reference_episode_set", "media")
    media_manifest_rows = _jsonl(provider_root / "media_manifest.jsonl")
    media_by_digest = {
        str(item.get("sha256", "")): str(item.get("path", ""))
        for item in media_manifest_rows
    }
    audit.require(
        len(media_by_digest) == len(media_manifest_rows),
        "media_manifest_duplicates",
        "media",
    )
    all_image_digests: list[str] = []

    for episode_id, row in provider_rows.items():
        raw_item, trace = raw_by_episode[episode_id]
        index = provider_index[episode_id]
        location = episode_id
        audit.require(
            str(index.get("source_trace_sha256", "")) == str(raw_item.get("sha256", "")),
            "provider_raw_hash_binding",
            location,
        )
        _trace_content_audit(audit, trace=trace, row=row, episode_id=episode_id)
        reference = references.get(episode_id, {})
        messages = row.get("messages", [])
        reference_messages = reference.get("messages", [])
        audit.require(
            [item.get("role") for item in messages]
            == [item.get("role") for item in reference_messages],
            "reference_role_drift",
            location,
        )
        audit.require(
            [str(item.get("content", "")).count("<image>") for item in messages]
            == [str(item.get("content", "")).count("<image>") for item in reference_messages],
            "reference_marker_drift",
            location,
        )
        images = [str(value) for value in row.get("images", [])]
        reference_images = [str(value) for value in reference.get("images", [])]
        audit.require(
            [Path(value).stem for value in images]
            == [Path(value).stem for value in reference_images],
            "reference_media_order_drift",
            location,
        )
        audit.require(
            sum(str(item.get("content", "")).count("<image>") for item in messages)
            == len(images),
            "provider_marker_count",
            location,
        )
        for image in images:
            path = Path(image)
            digest = path.stem.casefold()
            audit.require(path.is_file(), "media_missing", location)
            if path.is_file():
                audit.require(_sha256(path) == digest, "media_content_hash", location)
            audit.require(media_by_digest.get(digest) == image, "media_manifest_binding", location)
            all_image_digests.append(digest)
        audit.counts["reasoning_rows"] += 1

    for episode_id, row in action_rows.items():
        raw_item, trace = raw_by_episode[episode_id]
        audit.require(
            str(row.get("source_trace_sha256", "")) == str(raw_item.get("sha256", "")),
            "action_raw_hash_binding",
            episode_id,
        )
        _trace_content_audit(audit, trace=trace, row=row, episode_id=episode_id)
        images = [str(value) for value in row.get("images", [])]
        expected = str(
            _mapping(_mapping(trace.get("state")).get("runtime_case")).get("image_sha256", "")
        )
        audit.require(len(images) == 1, "action_image_count", episode_id)
        if images:
            path = Path(images[0])
            audit.require(path.is_file(), "action_image_missing", episode_id)
            if path.is_file():
                audit.require(_sha256(path) == expected, "action_image_hash", episode_id)
        audit.counts["action_only_rows"] += 1

    audit.require(
        set(all_image_digests) == set(media_by_digest),
        "media_manifest_coverage",
        "media",
    )
    audit.counts["image_references"] = len(all_image_digests)
    audit.counts["unique_media_files"] = len(set(all_image_digests))

    policy_rows, _ = _rows_by_episode(policy_root)
    audit.require(set(policy_rows) == set(provider_rows), "policy_episode_set", "policy")
    for episode_id, provider in provider_rows.items():
        policy = policy_rows.get(episode_id)
        if policy is None:
            continue
        source_messages = provider.get("messages", [])
        messages = policy.get("messages", [])
        audit.require(len(messages) == len(source_messages), "policy_message_count", episode_id)
        successful_unmasked: set[str] = set()
        for index, (source, message) in enumerate(zip(source_messages, messages)):
            role = str(source.get("role", ""))
            location = f"{episode_id}:message:{index}"
            audit.require(role == str(message.get("role", "")), "policy_role_drift", location)
            if role == "tool_call":
                audit.counts["policy_tool_calls"] += 1
                if _tool_call(str(source.get("content", ""))) != _tool_call(str(message.get("content", ""))):
                    audit.counts["policy_modified_tool_calls"] += 1
                if message.get("loss") is False:
                    audit.counts["policy_masked_tool_calls"] += 1
                    audit.require(
                        index > 0 and _mapping(messages[index - 1]).get("loss") is False,
                        "policy_mask_pair",
                        location,
                    )
                elif index + 1 < len(messages):
                    payload = _mapping(
                        _raw_json_prefix(str(_mapping(messages[index + 1]).get("content", "")))
                    )
                    locator = _mapping(payload.get("observation_locator"))
                    if locator.get("tool_success") is True:
                        successful_unmasked.add(str(locator.get("observation_id", "")))
                continue
            audit.require(
                str(source.get("content", "")) == str(message.get("content", "")),
                "policy_non_call_content_drift",
                location,
            )
        source_answer = _answer(str(source_messages[-1].get("content", "")))
        policy_answer = _answer(str(messages[-1].get("content", "")))
        audit.require(source_answer == policy_answer, "policy_final_target_drift", episode_id)
        final_ids = [
            str(value).strip()
            for value in policy_answer.get("verdict_observation_ids", []) or []
            if str(value).strip()
        ]
        audit.require(
            set(final_ids).issubset(successful_unmasked),
            "policy_final_id_not_unmasked_success",
            episode_id,
        )
        audit.counts["policy_final_ids"] += len(final_ids)
        audit.counts["policy_rows"] += 1

    # This final schema/runtime audit is intentionally an additional layer;
    # all raw-to-provider comparisons above are implemented independently.
    for import_root in (repo_root, repo_root / "training"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from ifv_training.audit import audit_derived_dataset  # noqa: PLC0415

    strict_policy = audit_derived_dataset(policy_root)
    audit.require(strict_policy.get("passed") is True, "strict_policy_audit", "policy")
    provider_counts = _mapping(provider_manifest.get("row_retention"))
    projection = _mapping(policy_manifest.get("contract_projection"))
    final_projection = _mapping(projection.get("final_observation_ids"))
    audit.require(int(provider_counts.get("raw_rows", -1)) == len(raw_by_episode), "manifest_raw_rows", "manifest")
    audit.require(int(provider_counts.get("dropped_rows", -1)) == 0, "manifest_dropped_rows", "manifest")
    audit.require(
        int(final_projection.get("original", -1))
        == int(final_projection.get("retained", -2))
        == audit.counts["policy_final_ids"],
        "manifest_final_id_retention",
        "manifest",
    )
    audit.require(
        _mapping(bundle_manifest.get("policy_audit")).get("passed") is True,
        "bundle_policy_passed",
        "manifest",
    )
    return {
        "schema_version": "ifv-canonical-rebuild-independent-audit-v1",
        "passed": not audit.errors,
        "error_count": sum(
            count for name, count in audit.counts.items() if name.startswith("error:")
        ),
        "errors": audit.errors,
        "counts": dict(sorted(audit.counts.items())),
        "policy_projection": projection,
        "strict_policy_audit": {
            "passed": strict_policy.get("passed"),
            "production_blockers": _mapping(
                _mapping(strict_policy.get("causal_contract")).get("production_blockers")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-delivery", type=Path, required=True)
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--verify-source-archives", action="store_true")
    args = parser.parse_args()
    report = audit_bundle(
        raw_root=args.raw_root.resolve(),
        reference_delivery=args.reference_delivery.resolve(),
        rebuilt_root=args.rebuilt_root.resolve(),
        repo_root=args.repo_root.resolve(),
        verify_source_archives=args.verify_source_archives,
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
