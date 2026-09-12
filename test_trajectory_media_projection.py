from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.trajectory.exporter import export_trajectory_sft_example
from src.trajectory.media_projection import project_trajectory_media


def _artifact(runtime_root: Path, content: bytes) -> dict:
    digest = hashlib.sha256(content).hexdigest()
    relative = f"{digest[:2]}/{digest}.jpg"
    path = runtime_root / "artifacts" / "sha256" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "sha256": digest,
        "media_type": "image/jpeg",
        "byte_count": len(content),
        "artifact_path": relative,
    }


def _context(runtime_root: Path, request_id: str, media: list[dict]) -> None:
    path = runtime_root / "context" / f"{request_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "context_items": [
                    {
                        "kind": "input_payload",
                        "media": media,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _step(
    *,
    stage: str,
    action_type: str,
    request_id: str,
    policy_action: dict,
    thought: str,
    tool_result: str = "",
) -> dict:
    tool_name = str(policy_action.get("name", ""))
    tool_args = policy_action.get("arguments", {})
    return {
        "stage": stage,
        "action_type": action_type,
        "tool_name": tool_name,
        "tool_args": tool_args if isinstance(tool_args, dict) else {},
        "thought": thought,
        "tool_result": tool_result,
        "metadata": {
            "context_request_id": request_id,
            "function_call_id": request_id if action_type == "tool_call" else "",
            "policy_input": {
                "system_instruction": f"{stage} instruction",
                "input_payload": {"request_id": request_id},
                "tools": (
                    [
                        {
                            "name": "example_tool",
                            "description": "Inspect evidence.",
                            "parameters": {"type": "object"},
                        }
                    ]
                    if stage == "unified_react"
                    else []
                ),
            },
            "policy_action": policy_action,
        },
    }


def _trace(tmp_path: Path) -> dict:
    runtime_root = tmp_path / "runtime" / "case-1" / "attempt-1"
    original = _artifact(runtime_root, b"original-image")
    candidate = _artifact(runtime_root, b"candidate-image")
    crop = _artifact(runtime_root, b"crop-image")
    _context(runtime_root, "req-000001", [original])
    _context(runtime_root, "req-000002", [original, candidate])
    _context(runtime_root, "req-000003", [original, candidate, crop])

    image_path = tmp_path / "original.jpg"
    image_path.write_bytes(b"fallback-that-should-not-be-used")
    steps = [
        _step(
            stage="unified_react",
            action_type="tool_call",
            request_id="req-000001",
            policy_action={
                "type": "tool_call",
                "name": "example_tool",
                "arguments": {"query": "first"},
            },
            thought="Inspect the first route.",
            tool_result=json.dumps({"status": "success", "result": "first"}),
        ),
        _step(
            stage="unified_react",
            action_type="tool_call",
            request_id="req-000002",
            policy_action={
                "type": "tool_call",
                "name": "example_tool",
                "arguments": {"query": "second"},
            },
            thought="Inspect the candidate image.",
            tool_result=json.dumps({"status": "success", "result": "second"}),
        ),
        _step(
            stage="unified_judgment",
            action_type="output",
            request_id="req-000003",
            policy_action={
                "type": "output",
                "verdict": "real",
                "reason": "The evidence is consistent.",
            },
            thought="",
        ),
    ]
    return {
        "image_id": "case-1",
        "input_mode": "image_only",
        "termination": "success",
        "decision_policy_version": "unified-react-v1",
        "state": {
            "input_mode": "image_only",
            "runtime_store": {"runtime_path": str(runtime_root)},
            "runtime_case": {
                "case_id": "case-1",
                "image_path": str(image_path),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            },
            "investigation_state": {
                "schema_version": "ifv-unified-react-raw-history-v1",
                "case_id": "case-1",
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "objective": "Verify the factual content expressed by the image.",
                "action_count": 2,
                "stop_reason": "model_finished",
                "finish_rationale": "The retained observations are sufficient.",
            },
            "all_steps": steps,
        },
    }


def test_projection_preserves_new_runtime_images_once(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    steps = trace["state"]["all_steps"]

    projection = project_trajectory_media(
        trace,
        candidate_steps=steps,
        fallback_image_path=trace["state"]["runtime_case"]["image_path"],
    )

    assert len(projection.initial_images) == 1
    assert {index: len(images) for index, images in projection.images_after_step.items()} == {
        0: 1,
        1: 1,
    }
    assert len(projection.images) == 3
    assert len(set(projection.images)) == 3
    assert projection.missing == []


def test_trajectory_export_matches_image_markers_to_portable_media(
    tmp_path: Path,
) -> None:
    exported = export_trajectory_sft_example(_trace(tmp_path))

    assert len(exported.images) == 3
    assert all(image.startswith("data:image/jpeg;base64,") for image in exported.images)
    assert sum(
        message["content"].count("<image>") for message in exported.messages
    ) == 3
    tool_responses = [
        message["content"]
        for message in exported.messages
        if message["role"] == "tool_response"
    ]
    assert len(tool_responses) == 2
    assert all(content.count("<image>") == 1 for content in tool_responses)


def test_trajectory_export_binds_call_to_executed_args_and_observation_id(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    first = trace["state"]["all_steps"][0]
    first["tool_args"] = {"query": "runtime-normalized"}

    exported = export_trajectory_sft_example(trace)
    first_call = next(
        message for message in exported.messages if message["role"] == "tool_call"
    )
    first_response = next(
        message for message in exported.messages if message["role"] == "tool_response"
    )
    call = json.loads(first_call["content"])
    response = json.loads(first_response["content"].split("\n\n<image>", 1)[0])

    assert json.loads(call["arguments"]) == {"query": "runtime-normalized"}
    assert response["observation_locator"] == {
        "observation_id": "req-000001",
        "tool_name": "example_tool",
        "tool_success": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda step: step.pop("tool_args"), "executed tool arguments"),
        (
            lambda step: step["metadata"].pop("function_call_id"),
            "model-visible observation ID",
        ),
        (lambda step: step.__setitem__("tool_result", ""), "recorded tool result"),
    ],
)
def test_trajectory_export_fails_closed_when_execution_binding_is_missing(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    trace = _trace(tmp_path)
    mutation(trace["state"]["all_steps"][0])

    with pytest.raises(ValueError, match=message):
        export_trajectory_sft_example(trace)
