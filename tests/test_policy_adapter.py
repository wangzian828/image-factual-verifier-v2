from __future__ import annotations

import json
from pathlib import Path

import pytest

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import write_json, write_jsonl
from ifv_training.policy import convert_policy_dataset, convert_policy_row


def _policy_row(example_type: str) -> dict:
    action = (
        {
            "type": "tool_call",
            "name": "text_search",
            "arguments": {
                "queries": ["museum object official collection"],
                "question_id": "task-1",
            },
        }
        if example_type == "react"
        else {"ready": True, "reason": "decisive evidence is resolved"}
    )
    return {
        "dataset_version": "ifv-policy-dataset-v2",
        "trajectory_version": "ifv-policy-v1",
        "tokenizer_id": "utf8-byte-v1",
        "episode_id": "case-1",
        "step_id": f"case-1:{example_type}:1",
        "source_run_id": "run-1",
        "runtime_commit": "a" * 40,
        "release_id": "release-1",
        "runtime_contract_version": "ifv-image-only-runtime-v1",
        "process_reference_protocol_version": "protocol-v1",
        "example_type": example_type,
        "runtime_observation_refs": [],
        "policy_input": {
            "system_instruction": (
                "Follow the runtime contract.\n\n"
                "Native Gemini Interactions protocol:\n"
                "Do not leak this provider wire text."
            ),
            "input_payload": "Current public investigation state.",
            "tools": [
                {
                    "type": "function",
                    "name": "text_search",
                    "description": "Search public pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "question_id": {"type": "string"},
                        },
                        "required": ["queries", "question_id"],
                    },
                }
            ],
            "response_format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                },
            },
        },
        "policy_action": action,
        "policy_input_token_ids": [1],
        "policy_action_token_ids": [2],
        "policy_action_loss_mask": [1],
        "action_valid": True,
        "terminated": example_type == "judgment",
        "fatal_boundary": False,
        "split": "train",
        "split_group_id": "group-1",
        "source_family_keys": ["domain:example.org"],
        "teacher_score": 5.0,
    }


def test_react_uses_ms_swift_native_agent_format() -> None:
    converted = convert_policy_row(_policy_row("react"))

    assert [message["role"] for message in converted["messages"]] == [
        "system",
        "user",
        "tool_call",
    ]
    assert converted["messages"][-1]["loss"] is True
    assert json.loads(converted["messages"][-1]["content"]) == {
        "name": "text_search",
        "arguments": {
            "queries": ["museum object official collection"],
            "question_id": "task-1",
        },
    }
    tools = json.loads(converted["tools"])
    assert tools[0]["function"]["name"] == "text_search"
    assert "Gemini Interactions" not in json.dumps(converted)
    assert "policy_action_token_ids" not in converted


def test_structured_stage_retains_response_contract() -> None:
    converted = convert_policy_row(_policy_row("reflection"))

    assert [message["role"] for message in converted["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert "Required structured output contract" in converted["messages"][1][
        "content"
    ]
    assert json.loads(converted["messages"][-1]["content"])["ready"] is True


def test_private_fields_are_rejected() -> None:
    row = _policy_row("react")
    row["policy_input"]["evaluation_gold"] = {"verdict": "fake"}

    with pytest.raises(ValueError, match="evaluation_gold"):
        convert_policy_row(row)


def test_dataset_conversion_is_deterministic_and_auditable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-policy-dataset-v2",
            "schema_version": "ifv-policy-dataset-manifest-v2",
        },
    )
    rows = [_policy_row("react"), _policy_row("reflection")]
    write_jsonl(source / "train.jsonl", rows)
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = convert_policy_dataset(source, first)
    convert_policy_dataset(source, second)
    audit = audit_derived_dataset(first)

    assert manifest["example_count"] == 2
    assert audit["passed"] is True
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_v1_dataset_is_not_silently_accepted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {"dataset_version": "ifv-policy-dataset-v1"},
    )
    with pytest.raises(ValueError, match="ifv-policy-dataset-v2"):
        convert_policy_dataset(source, tmp_path / "output")
