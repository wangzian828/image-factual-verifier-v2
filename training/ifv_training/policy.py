from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_model_visible
from .io import (
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)


SUPPORTED_DATASET_VERSIONS = frozenset(
    {
        "ifv-trajectory-sft-dataset-v2",
        "ifv-trajectory-sft-dataset-v3",
    }
)
LEGACY_STEP_DATASET_VERSION = "ifv-policy-dataset-v2"
OUTPUT_VERSION = "ifv-ms-swift-trajectory-sft-v1"
SPLITS = ("train", "validation", "test")


def convert_policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("dataset_version") == LEGACY_STEP_DATASET_VERSION:
        raise ValueError(
            "step-level ifv-policy-dataset-v2 rows are no longer accepted "
            "by the default SFT converter; use full trajectory_sft rows."
        )
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("trajectory SFT row requires messages")
    supervised_targets = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = str(message.get("role", ""))
        if role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(f"messages[{index}] has unsupported role {role!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty")
        if role == "assistant":
            if message.get("loss") is not True:
                raise ValueError(f"messages[{index}] must set loss=true")
            supervised_targets += 1
    if supervised_targets < 1:
        raise ValueError("trajectory SFT row has no supervised targets")
    tools = row.get("tools", "")
    if tools is None:
        tools = ""
    if not isinstance(tools, str):
        raise ValueError("trajectory SFT row tools must be a JSON string")
    parsed_tools: Any = []
    if tools.strip():
        try:
            parsed_tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError("trajectory SFT row tools must be valid JSON") from exc
    # Dataset metadata such as teacher_score and split assignment is used for
    # auditing only.  Only the actual model-visible conversation and tool
    # schemas belong under the private-field check.
    assert_model_visible(
        {"messages": messages, "tools": parsed_tools},
        location="trajectory_sft_row",
    )
    output: dict[str, Any] = {
        "messages": [dict(message) for message in messages],
        "channel": "trajectory_sft",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    images = row.get("images", [])
    if images:
        if not isinstance(images, list) or not all(
            isinstance(item, str) and item.strip()
            for item in images
        ):
            raise ValueError("trajectory SFT row images must be non-empty paths")
        first_user = next(
            (
                message
                for message in messages
                if str(message.get("role", "")) == "user"
            ),
            None,
        )
        if first_user is None or "<image>" not in str(
            first_user.get("content", "")
        ):
            raise ValueError(
                "trajectory SFT row images require <image> in the first user "
                "message"
            )
        output["images"] = list(images)
    if isinstance(tools, str) and tools.strip():
        output["tools"] = tools
    return output


def convert_policy_dataset(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = load_json(input_dir / "manifest.json")
    source_version = source_manifest.get("dataset_version")
    if source_version == LEGACY_STEP_DATASET_VERSION:
        raise ValueError(
            "refusing legacy step-level ifv-policy-dataset-v2; export "
            "ifv-trajectory-sft-dataset-v2 instead"
        )
    if source_version not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError(
            "training adapter accepts only "
            f"{sorted(SUPPORTED_DATASET_VERSIONS)}, got "
            f"{source_version!r}"
        )
    require_new_or_empty(output_dir)

    artifacts: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    row_id = 0
    for split in SPLITS:
        converted_rows: list[dict[str, Any]] = []
        for source_index, row in enumerate(load_jsonl(input_dir / f"{split}.jsonl")):
            converted = convert_policy_row(row)
            converted_rows.append(converted)
            index_rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "source_index": source_index,
                    "episode_id": row.get("episode_id"),
                    "case_id": row.get("case_id"),
                    "token_count_estimate": row.get("token_count_estimate"),
                    "tool_call_count": row.get("tool_call_count"),
                    "source_run_id": row.get("source_run_id"),
                    "runtime_commit": row.get("runtime_commit"),
                }
            )
            row_id += 1
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, converted_rows)
        artifacts[split] = {
            "path": output_path.name,
            "rows": len(converted_rows),
            "sha256": sha256_file(output_path),
        }

    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(index_rows),
        "sha256": sha256_file(index_path),
    }
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": {"name": "ms-swift", "version": "4.4.2"},
        "source": {
            "dataset_version": source_manifest["dataset_version"],
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "example_count": row_id,
        "trajectory_format": "one_episode_per_row",
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
