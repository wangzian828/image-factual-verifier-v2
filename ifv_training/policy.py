from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_model_visible
from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)


SUPPORTED_DATASET_VERSION = "ifv-policy-dataset-v2"
OUTPUT_VERSION = "ifv-ms-swift-policy-v1"
SPLITS = ("train", "validation", "test")


def _strip_gemini_wire_instructions(value: str) -> str:
    marker = "Native Gemini Interactions protocol:"
    if marker in value:
        value = value.split(marker, 1)[0]
    return value.rstrip()


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tools[{index}] must be an object")
        if raw.get("type") != "function":
            raise ValueError(f"tools[{index}] must have type=function")
        if isinstance(raw.get("function"), Mapping):
            function = dict(raw["function"])
        else:
            function = {
                "name": raw.get("name"),
                "description": raw.get("description", ""),
                "parameters": raw.get("parameters", {}),
            }
        name = str(function.get("name", "")).strip()
        if not name:
            raise ValueError(f"tools[{index}] has no function name")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(function.get("description", "")),
                    "parameters": function.get("parameters", {}),
                },
            }
        )
    return result


def _tool_action(action: Mapping[str, Any]) -> dict[str, Any]:
    if str(action.get("type", "")) != "tool_call":
        raise ValueError("react policy action must have type=tool_call")
    name = str(action.get("name", "")).strip()
    arguments = action.get("arguments")
    if not name or not isinstance(arguments, Mapping):
        raise ValueError("react policy action requires name and object arguments")
    return {"name": name, "arguments": dict(arguments)}


def convert_policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    example_type = str(row.get("example_type", "")).strip()
    if example_type not in {"planning", "react", "reflection", "judgment"}:
        raise ValueError(f"unsupported policy example_type: {example_type!r}")
    if not bool(row.get("action_valid", False)):
        raise ValueError("invalid policy actions cannot enter SFT")
    if bool(row.get("fatal_boundary", False)):
        raise ValueError("fatal-boundary actions cannot enter SFT")

    policy_input = row.get("policy_input")
    policy_action = row.get("policy_action")
    if not isinstance(policy_input, Mapping):
        raise ValueError("policy_input must be an object")
    if not isinstance(policy_action, Mapping):
        raise ValueError("policy_action must be an object")
    assert_model_visible(policy_input, location="policy_input")
    assert_model_visible(policy_action, location="policy_action")

    system = _strip_gemini_wire_instructions(
        str(policy_input.get("system_instruction", ""))
    )
    system = f"<ifv_stage>{example_type}</ifv_stage>\n{system}"
    user = str(policy_input.get("input_payload", ""))
    if not system or not user:
        raise ValueError("policy input requires system_instruction and input_payload")
    if example_type != "react":
        response_format = policy_input.get("response_format")
        if not isinstance(response_format, Mapping):
            raise ValueError(
                f"{example_type} policy input requires response_format"
            )
        user = (
            f"{user}\n\nRequired structured output contract:\n"
            f"{canonical_json(response_format)}"
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    output: dict[str, Any] = {
        "messages": messages,
        "channel": example_type,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if example_type == "react":
        tools = _normalize_tools(policy_input.get("tools"))
        if not tools:
            raise ValueError("react policy input requires tools")
        messages.append(
            {
                "role": "tool_call",
                "content": canonical_json(_tool_action(policy_action)),
                "loss": True,
            }
        )
        output["tools"] = canonical_json(tools)
    else:
        messages.append(
            {
                "role": "assistant",
                "content": canonical_json(policy_action),
                "loss": True,
            }
        )
    return output


def convert_policy_dataset(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = load_json(input_dir / "manifest.json")
    if source_manifest.get("dataset_version") != SUPPORTED_DATASET_VERSION:
        raise ValueError(
            "training adapter accepts only "
            f"{SUPPORTED_DATASET_VERSION}, got "
            f"{source_manifest.get('dataset_version')!r}"
        )
    require_new_or_empty(output_dir)

    stage_counts: Counter[str] = Counter()
    artifacts: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    stage_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    row_id = 0
    for split in SPLITS:
        converted_rows: list[dict[str, Any]] = []
        for source_index, row in enumerate(load_jsonl(input_dir / f"{split}.jsonl")):
            converted = convert_policy_row(row)
            converted_rows.append(converted)
            example_type = str(row["example_type"])
            stage_rows.setdefault((split, example_type), []).append(converted)
            stage_counts[example_type] += 1
            index_rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "source_index": source_index,
                    "episode_id": row.get("episode_id"),
                    "step_id": row.get("step_id"),
                    "stage": example_type,
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
        for stage in ("planning", "react", "reflection", "judgment"):
            stage_path = output_dir / f"{split}.{stage}.jsonl"
            rows = stage_rows.get((split, stage), [])
            write_jsonl(stage_path, rows)
            artifacts[f"{split}_{stage}"] = {
                "path": stage_path.name,
                "rows": len(rows),
                "sha256": sha256_file(stage_path),
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
        "framework": {"name": "ms-swift", "version": "4.4.1"},
        "source": {
            "dataset_version": source_manifest["dataset_version"],
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "example_count": row_id,
        "stage_counts": dict(sorted(stage_counts.items())),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
