from __future__ import annotations

import json
from pathlib import Path

import pytest

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import write_json, write_jsonl
from ifv_training.policy import convert_policy_dataset, convert_policy_row


def _trajectory_row() -> dict:
    tool_call = {
        "name": "text_search",
        "arguments": {
            "queries": ["museum object official collection"],
            "question_id": "task-1",
        },
    }
    return {
        "dataset_version": "ifv-trajectory-sft-dataset-v1",
        "trajectory_version": "ifv-trajectory-sft-v1",
        "episode_id": "case-1",
        "case_id": "case-1",
        "source_run_id": "run-1",
        "runtime_commit": "a" * 40,
        "release_id": "release-1",
        "runtime_contract_version": "ifv-image-only-runtime-v1",
        "process_reference_protocol_version": "protocol-v1",
        "messages": [
            {
                "role": "system",
                "content": "You are the Image Factual Verifier policy model.",
            },
            {
                "role": "user",
                "content": "Investigate the image and decide the visible fact.",
            },
            {
                "role": "tool_call",
                "content": json.dumps(tool_call),
                "loss": True,
            },
            {
                "role": "tool_response",
                "content": json.dumps(
                    {
                        "name": "text_search",
                        "result": {"status": "ok", "results": []},
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"verdict": "real", "reason": "evidence is sufficient"}
                ),
                "loss": True,
            },
        ],
        "tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
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
                    },
                }
            ]
        ),
        "token_count_estimate": 100,
        "message_count": 5,
        "tool_call_count": 1,
        "split": "train",
        "split_group_id": "group-1",
        "source_family_keys": ["domain:example.org"],
        "teacher_score": 5.0,
    }


def test_react_uses_ms_swift_native_agent_format() -> None:
    converted = convert_policy_row(_trajectory_row())

    assert [message["role"] for message in converted["messages"]] == [
        "system",
        "user",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert converted["messages"][2]["loss"] is True
    assert converted["messages"][3].get("loss") is None
    assert converted["messages"][4]["loss"] is True
    assert json.loads(converted["messages"][2]["content"])["name"] == "text_search"
    tools = json.loads(converted["tools"])
    assert tools[0]["function"]["name"] == "text_search"
    assert converted["channel"] == "trajectory_sft"
    assert "policy_action_token_ids" not in converted


def test_private_fields_are_rejected() -> None:
    row = _trajectory_row()
    row["messages"][1]["evaluation_gold"] = {"verdict": "fake"}

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
            "dataset_version": "ifv-trajectory-sft-dataset-v1",
            "schema_version": "ifv-trajectory-sft-dataset-manifest-v1",
        },
    )
    rows = [
        _trajectory_row(),
        {**_trajectory_row(), "episode_id": "case-2", "case_id": "case-2"},
        {**_trajectory_row(), "episode_id": "case-3", "case_id": "case-3"},
        {**_trajectory_row(), "episode_id": "case-4", "case_id": "case-4"},
    ]
    write_jsonl(source / "train.jsonl", rows)
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = convert_policy_dataset(source, first)
    convert_policy_dataset(source, second)
    audit = audit_derived_dataset(first)

    assert manifest["example_count"] == 4
    assert manifest["artifacts"]["train"]["rows"] == 4
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
    with pytest.raises(ValueError, match="training adapter accepts only"):
        convert_policy_dataset(source, tmp_path / "output")
