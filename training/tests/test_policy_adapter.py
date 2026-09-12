from __future__ import annotations

import json
from pathlib import Path

import pytest

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import sha256_file, write_json, write_jsonl
from ifv_training.policy import (
    convert_policy_dataset,
    convert_policy_row,
    repair_derived_policy_dataset,
)
from src.trajectory.exporter import _render_tool_response_content


def _trajectory_row() -> dict:
    tool_call_arguments = {
        "queries": ["museum object official collection"],
        "question_id": "task-1",
    }
    return {
        "dataset_version": "ifv-trajectory-sft-dataset-v3",
        "trajectory_version": "ifv-trajectory-sft-v3",
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
                "content": (
                    "<image>\nInvestigate the image and decide the visible fact."
                ),
            },
            {
                "role": "assistant",
                "content": "<think>\n先调用搜索工具核对这个事实。\n</think>",
            },
            {
                "role": "tool_call",
                "content": json.dumps(
                    {
                        "name": "text_search",
                        "arguments": json.dumps(
                            tool_call_arguments,
                            ensure_ascii=False,
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "tool_response",
                "content": json.dumps(
                    {
                        "observation_locator": {
                            "observation_id": "obs-1",
                            "tool_name": "text_search",
                            "tool_success": True,
                        },
                        "result": {"status": "ok", "results": []},
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<think>\n证据不足以支持原说法。\n</think>\n\n"
                    "<answer>\n"
                    + json.dumps(
                        {
                            "verdict": "real",
                            "confidence": 0.6,
                            "verdict_observation_ids": ["obs-1"],
                            "overall_assessment": "Evidence remains limited.",
                            "fact_check_report": {
                                "headline": "Limited evidence",
                                "claim_under_review": "The visible museum claim.",
                                "verdict_summary": "The available evidence is compatible.",
                                "key_findings": ["The search did not contradict it."],
                                "evidence_summary": "One retained search observation.",
                                "remaining_uncertainties": ["Primary records were unavailable."],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n</answer>"
                ),
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
        "images": ["data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="],
        "token_count_estimate": 100,
        "message_count": 6,
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
        "assistant",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert "<think>" in converted["messages"][2]["content"]
    call = json.loads(converted["messages"][3]["content"])
    assert call["name"] == "text_search"
    assert json.loads(call["arguments"]) == {
        "queries": "museum object official collection",
    }
    assert converted["messages"][4]["role"] == "tool_response"
    assert "<answer>" in converted["messages"][5]["content"]
    assert "text_search" in {
        item["function"]["name"] for item in json.loads(converted["tools"])
    }
    assert set(converted) == {"messages", "images", "tools"}
    answer = json.loads(
        converted["messages"][-1]["content"].split("<answer>", 1)[1].split(
            "</answer>", 1
        )[0]
    )
    assert answer["verdict_observation_ids"] == ["obs-1"]


def test_policy_adapter_accepts_adjacent_tool_call_batches() -> None:
    row = _trajectory_row()
    row["messages"] = [
        row["messages"][0],
        row["messages"][1],
        row["messages"][2],
        row["messages"][3],
        row["messages"][4],
        {
            "role": "tool_call",
            "content": json.dumps(
                {
                    "name": "visit",
                    "arguments": json.dumps(
                        {"url": "https://example.org"},
                        ensure_ascii=False,
                    ),
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "tool_response",
            "content": json.dumps(
                {"status": "ok", "text": "Example"},
                ensure_ascii=False,
            ),
        },
        row["messages"][-1],
    ]
    row["message_count"] = len(row["messages"])
    row["tool_call_count"] = 2

    converted = convert_policy_row(row)

    assert [message["role"] for message in converted["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool_call",
        "tool_response",
        "tool_call",
        "tool_response",
        "assistant",
    ]


def test_audit_accepts_ms_swift_data_uri_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-trajectory-sft-dataset-v3",
            "schema_version": "ifv-trajectory-sft-dataset-manifest-v1",
        },
    )
    row = _trajectory_row()
    row["images"] = ["data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="]
    row["messages"][1]["content"] = "<image>\nInvestigate the image."
    write_jsonl(source / "train.jsonl", [row])
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])

    output = tmp_path / "output"
    convert_policy_dataset(source, output)

    assert audit_derived_dataset(output)["passed"] is True


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
            "dataset_version": "ifv-trajectory-sft-dataset-v3",
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
    assert manifest["row_retention"] == {
        "source_rows": 4,
        "output_rows": 4,
        "dropped_rows": 0,
    }
    assert audit["passed"] is True
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_dataset_conversion_rejects_source_manifest_count_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-trajectory-sft-dataset-v3",
            "schema_version": "ifv-trajectory-sft-dataset-manifest-v1",
            "example_counts": {"train": 2, "validation": 0, "test": 0},
        },
    )
    write_jsonl(source / "train.jsonl", [_trajectory_row()])
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])

    with pytest.raises(ValueError, match="example counts do not match"):
        convert_policy_dataset(source, tmp_path / "output")


def test_legacy_step_dataset_is_rejected() -> None:
    row = _trajectory_row()
    row["dataset_version"] = "ifv-policy-dataset-v2"

    with pytest.raises(ValueError, match="step-level"):
        convert_policy_row(row)


def test_audit_allows_provider_error_observation_but_rejects_wire_prompt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-trajectory-sft-dataset-v3",
            "schema_version": "ifv-trajectory-sft-dataset-manifest-v1",
        },
    )
    row = _trajectory_row()
    row["messages"][4]["content"] = json.dumps(
        {
            "status": "error",
            "error": (
                "Gemini Interactions request failed with HTTP 500; "
                "retry later"
            ),
        }
    )
    write_jsonl(source / "train.jsonl", [row])
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])
    output = tmp_path / "output"
    convert_policy_dataset(source, output)

    assert audit_derived_dataset(output)["passed"] is True

    converted_rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    converted_rows[0]["messages"][0]["content"] += (
        "\nNative Gemini Interactions protocol:\nprovider-only details"
    )
    write_jsonl(output / "train.jsonl", converted_rows)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["train"]["sha256"] = sha256_file(
        output / "train.jsonl"
    )
    write_json(manifest_path, manifest)

    audit = audit_derived_dataset(output)
    assert audit["passed"] is False
    assert audit["errors"] == [
        "train.jsonl[0] contains provider wire instructions"
    ]


def test_v1_dataset_is_not_silently_accepted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {"dataset_version": "ifv-policy-dataset-v1"},
    )
    with pytest.raises(ValueError, match="training adapter accepts only"):
        convert_policy_dataset(source, tmp_path / "output")


def test_converter_repairs_unknown_arguments_without_dropping_row() -> None:
    row = _trajectory_row()
    call = json.loads(row["messages"][3]["content"])
    arguments = json.loads(call["arguments"])
    arguments["transport_wrapper"] = "legacy-only"
    call["arguments"] = json.dumps(arguments)
    row["messages"][3]["content"] = json.dumps(call)

    converted = convert_policy_row(row)

    repaired = json.loads(converted["messages"][3]["content"])
    assert "transport_wrapper" not in json.loads(repaired["arguments"])
    assert converted["messages"][3].get("loss") is not False


def test_source_exporter_makes_successful_observation_id_causally_visible() -> None:
    rendered = _render_tool_response_content(
        json.dumps({"status": "ok", "result": {"value": 1}}),
        observation_id="call-visible-1",
        tool_name="text_search",
        tool_success=True,
    )

    payload = json.loads(rendered)
    assert payload["observation_locator"] == {
        "observation_id": "call-visible-1",
        "tool_name": "text_search",
        "tool_success": True,
    }
    assert payload["result"] == {"value": 1}


def test_converter_masks_unrepairable_action_but_keeps_complete_trajectory() -> None:
    row = _trajectory_row()
    call = json.loads(row["messages"][3]["content"])
    call["arguments"] = json.dumps({"question_id": "task-1"})
    row["messages"][3]["content"] = json.dumps(call)

    converted = convert_policy_row(row)

    assert len(converted["messages"]) == len(row["messages"])
    assert converted["messages"][2]["loss"] is False
    assert converted["messages"][3]["loss"] is False
    assert converted["messages"][4]["role"] == "tool_response"
    assert converted["messages"][-1].get("loss") is not False


def test_converter_uses_live_schema_and_masks_semantic_duplicate() -> None:
    row = _trajectory_row()
    row["messages"] = [
        *row["messages"][:-1],
        {"role": "assistant", "content": "<think>repeat</think>"},
        row["messages"][3],
        row["messages"][4],
        row["messages"][-1],
    ]

    converted = convert_policy_row(row)
    tools = {
        item["function"]["name"]: item["function"]
        for item in json.loads(converted["tools"])
    }

    assert "finish_investigation" in tools
    assert tools["text_search"]["parameters"]["properties"]["queries"][
        "type"
    ] == "string"
    assert converted["messages"][5]["loss"] is False
    assert converted["messages"][6]["loss"] is False


def test_converter_drops_only_noncausal_final_ids() -> None:
    row = _trajectory_row()
    final = row["messages"][-1]["content"]
    payload = json.loads(final.split("<answer>", 1)[1].split("</answer>", 1)[0])
    payload["verdict_observation_ids"] = ["obs-1", "invisible-id"]
    row["messages"][-1]["content"] = (
        final.split("<answer>", 1)[0]
        + "<answer>"
        + json.dumps(payload)
        + "</answer>"
    )

    converted = convert_policy_row(row)
    converted_payload = json.loads(
        converted["messages"][-1]["content"].split("<answer>", 1)[1].split(
            "</answer>", 1
        )[0]
    )

    assert converted_payload["verdict_observation_ids"] == ["obs-1"]


def test_repair_dataset_retains_every_row_and_records_masks(tmp_path: Path) -> None:
    source = tmp_path / "derived-v2"
    source.mkdir()
    good = convert_policy_row(_trajectory_row())
    bad = json.loads(json.dumps(good))
    bad_call = json.loads(bad["messages"][3]["content"])
    bad_call["arguments"] = json.dumps({"question_id": "task-1"})
    bad["messages"][3]["content"] = json.dumps(bad_call)
    write_jsonl(source / "train.jsonl", [good, bad])
    write_jsonl(source / "validation.jsonl", [])
    write_jsonl(source / "test.jsonl", [])
    write_jsonl(
        source / "index.jsonl",
        [
            {"row_id": 0, "split": "train", "source_index": 0},
            {"row_id": 1, "split": "train", "source_index": 1},
        ],
    )
    artifacts = {}
    for name in ("train", "validation", "test", "index"):
        path = source / f"{name}.jsonl"
        artifacts[name] = {
            "path": path.name,
            "rows": sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ),
            "sha256": sha256_file(path),
        }
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-ms-swift-qwen-agent-v2",
            "artifacts": artifacts,
        },
    )

    output = tmp_path / "repaired-v3"
    manifest = repair_derived_policy_dataset(source, output)

    assert manifest["row_retention"] == {
        "source_rows": 2,
        "output_rows": 2,
        "dropped_rows": 0,
    }
    assert manifest["contract_projection"]["masked_unrepairable_tool_calls"] == {
        "missing_required": 1
    }
    repaired_rows = [
        json.loads(line)
        for line in (output / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(repaired_rows) == 2
    assert repaired_rows[1]["messages"][3]["loss"] is False
    assert audit_derived_dataset(output)["passed"] is True
