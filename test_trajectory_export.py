from __future__ import annotations

import json
from pathlib import Path

from src.trajectory.exporter import export_policy_examples
from test_image_only_v2_trajectory import (
    test_scripted_image_only_v2_complete_trajectory,
)


def _trace(tmp_path: Path) -> dict:
    test_scripted_image_only_v2_complete_trajectory(tmp_path)
    return json.loads(
        (tmp_path / "traces" / "case_scripted_v2.json").read_text(
            encoding="utf-8"
        )
    )


def test_exporter_uses_actual_policy_boundaries_and_aligned_masks(
    tmp_path: Path,
) -> None:
    examples = export_policy_examples(_trace(tmp_path))

    assert [item.example_type for item in examples] == [
        "react",
        "react",
        "react",
        "react",
        "reflection",
        "judgment",
    ]
    assert not any(item.example_type == "planning" for item in examples)
    assert all(item.tokenizer_id == "utf8-byte-v1" for item in examples)
    assert all(
        len(item.policy_action_token_ids)
        == len(item.policy_action_loss_mask)
        for item in examples
    )
    assert all(set(item.policy_action_loss_mask) == {1} for item in examples)
    first_action = examples[0].policy_action
    assert first_action["type"] == "tool_call"
    assert set(first_action["arguments"]) == {
        "question_id",
        "url",
        "goal",
    }
    assert "__claim_text" not in json.dumps(first_action)
    assert examples[-1].terminated is True


def test_fatal_boundary_is_zero_masked(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    trace["termination"] = "error"
    trace["state"]["termination"] = "error"

    examples = export_policy_examples(trace)

    assert examples[-1].fatal_boundary is True
    assert set(examples[-1].policy_action_loss_mask) == {0}
    assert all(
        set(item.policy_action_loss_mask) == {1}
        for item in examples[:-1]
    )


def test_exporter_rejects_evaluator_private_leak(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    step = next(
        item
        for item in trace["state"]["all_steps"]
        if item.get("metadata", {}).get("policy_input")
    )
    step["metadata"]["policy_input"]["evaluation_gold"] = {"verdict": "real"}

    try:
        export_policy_examples(trace)
    except ValueError as exc:
        assert "evaluation_gold" in str(exc)
    else:
        raise AssertionError("private evaluator data must be rejected")
