from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from ifv_training.psd import (
    PSD_TARGET_SCHEMA_VERSION,
    audit_hint,
    build_preservation_targets,
    build_psd_target_package,
    build_repair_target,
    materialize_psd_topk_cache,
    validate_topk_by_position,
)


def _repair_row(**overrides: object) -> dict:
    row = {
        "schema_version": "ifv-psd-repair-v1",
        "case_id": "case-1",
        "episode_id": "episode-1",
        "repair_step_id": "episode-1:react:17",
        "stage": "react",
        "accepted": True,
        "repair_tier": "causal_episode_pass",
        "hint": "先检查当前尚未验证的事实关系，再选择一个能产生新证据的动作。",
        "hint_level": 1,
        "student_prompt_ids": [1, 2, 3],
        "teacher_prompt_ids": [1, 2, 3, 4],
        "completion_ids": [5, 6],
        "expected_action": {
            "name": "text_search",
            "arguments": {"query": "private exact query"},
        },
        "private_reference": {
            "expected_verdict": "fake",
            "source_url": "https://private.example/item",
        },
        "verification": {
            "source_rollout_failed": True,
            "hinted_local_pass": True,
            "hinted_episode_pass": True,
            "hinted_strict_trace_audit_pass": True,
        },
        "model_roles": {
            "hint_constructor": {
                "provider": "frontier",
                "model": "strong-hint-constructor",
                "supplies_training_distribution": False,
            },
            "frozen_self_teacher": {
                "provider": "qwen_local",
                "model": "qwen-round-start",
                "round_start_checkpoint": "checkpoint-round-0",
                "supplies_training_distribution": True,
            },
            "trainable_student": {
                "provider": "qwen_local",
                "model": "qwen-round-start",
                "initial_checkpoint": "checkpoint-round-0",
            },
        },
    }
    row.update(overrides)
    return row


def test_hint_audit_accepts_local_procedural_hint() -> None:
    result = audit_hint(_repair_row())
    assert result["passed"] is True
    assert result["errors"] == []


def test_hint_audit_rejects_verdict_and_exact_action_values() -> None:
    verdict = audit_hint(_repair_row(hint="这张图是 fake。"))
    assert verdict["passed"] is False
    assert "binary_verdict_leak" in verdict["errors"]

    exact = audit_hint(_repair_row(hint="请使用 private exact query"))
    assert exact["passed"] is False
    assert "private_value_leak" in exact["errors"]


def test_repair_target_is_pending_until_teacher_topk_exists() -> None:
    target = build_repair_target(_repair_row())
    assert target["schema_version"] == PSD_TARGET_SCHEMA_VERSION
    assert target["target_status"] == "pending_topk"
    assert target["teacher_topk_by_position"] == []
    assert target["model_roles"]["hint_constructor"]["model"] == (
        "strong-hint-constructor"
    )
    assert target["model_roles"]["frozen_self_teacher"]["model"] == (
        "qwen-round-start"
    )


def test_topk_validation_requires_exact_length_and_normalized_mass() -> None:
    validate_topk_by_position(
        [5, 6],
        [
            [[5, 0.7], [8, 0.3]],
            [[6, 0.6], [9, 0.4]],
        ],
        topk=2,
    )
    with pytest.raises(ValueError, match="sum to one"):
        validate_topk_by_position(
            [5],
            [[[5, 0.8], [8, 0.3]]],
            topk=2,
        )


def test_preservation_row_weight_is_split_across_steps() -> None:
    row = {
        "case_id": "case-1",
        "episode_id": "episode-pass",
        "class": "base_pass_preserve",
        "verified_full_task": True,
        "preservation_steps": [
            {"step_id": "s1", "student_prompt_ids": [1], "completion_ids": [2]},
            {"step_id": "s2", "student_prompt_ids": [1, 2], "completion_ids": [3]},
        ],
    }
    targets = build_preservation_targets(row)
    assert len(targets) == 2
    assert [item["row_weight"] for item in targets] == [0.5, 0.5]
    assert all(item["target_status"] == "pending_topk" for item in targets)


def test_target_package_rejects_test_rows_and_writes_manifest(tmp_path: Path) -> None:
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text(
        json.dumps(_repair_row(), ensure_ascii=False)
        + "\n"
        + json.dumps(
            _repair_row(case_id="test-case", split="test"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "package"
    manifest = build_psd_target_package(
        repairs_path=repairs,
        preservation_path=None,
        output_dir=out_dir,
    )
    assert manifest["counts"]["repair_targets"] == 1
    assert manifest["counts"]["rejections"] == 1
    assert manifest["status"] == "ready_for_topk_cache"
    assert (out_dir / "targets.jsonl").is_file()
    stored = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["schema_version"].startswith("ifv-psd-target-manifest")


def _token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(
            token_ids,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_materialize_topk_cache_requires_exact_token_hashes(tmp_path: Path) -> None:
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text(
        json.dumps(_repair_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    package = tmp_path / "package"
    build_psd_target_package(
        repairs_path=repairs,
        preservation_path=None,
        output_dir=package,
        topk=2,
    )
    target = json.loads(
        (package / "targets.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    cache = tmp_path / "topk.jsonl"
    cache.write_text(
        json.dumps(
            {
                "schema_version": "ifv-psd-teacher-topk-cache-v1",
                "target_id": target["target_id"],
                "teacher_model": "qwen-test",
                "teacher_prompt_sha256": _token_hash(
                    target["teacher_prompt_ids"]
                ),
                "completion_sha256": _token_hash(target["completion_ids"]),
                "teacher_topk_by_position": [
                    [[5, 0.7], [8, 0.3]],
                    [[6, 0.6], [9, 0.4]],
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "materialized"
    result = materialize_psd_topk_cache(
        targets_path=package / "targets.jsonl",
        cache_path=cache,
        output_dir=output,
        topk=2,
    )
    assert result["status"] == "ready_for_training"
    completed = json.loads(
        (output / "targets.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert completed["target_status"] == "complete"
    assert completed["teacher"]["model"] == "qwen-test"


def test_materialize_topk_cache_fails_closed_when_target_is_missing(
    tmp_path: Path,
) -> None:
    target = build_repair_target(_repair_row(), topk=2)
    targets = tmp_path / "targets.jsonl"
    targets.write_text(json.dumps(target) + "\n", encoding="utf-8")
    cache = tmp_path / "topk.jsonl"
    cache.write_text("", encoding="utf-8")
    output = tmp_path / "blocked"
    result = materialize_psd_topk_cache(
        targets_path=targets,
        cache_path=cache,
        output_dir=output,
        topk=2,
    )
    assert result["status"] == "blocked_topk_cache"
    assert not (output / "targets.jsonl").exists()
    reasons = (output / "topk_cache_rejections.jsonl").read_text(
        encoding="utf-8"
    )
    assert "cache_entry_missing" in reasons
