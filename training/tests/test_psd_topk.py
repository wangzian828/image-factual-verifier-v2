from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ifv_training.io import sha256_file
from ifv_training.psd import build_repair_target
from ifv_training.psd_topk import collect_psd_topk_cache, normalized_topk


def _repair_row() -> dict[str, Any]:
    checkpoint = "/checkpoints/round-0"
    model = "ifv-qwen-round-0"
    return {
        "schema_version": "ifv-psd-repair-v1",
        "case_id": "case-1",
        "episode_id": "episode-1",
        "repair_step_id": "episode-1:react:2",
        "stage": "react",
        "accepted": True,
        "repair_tier": "causal_episode_pass",
        "hint": "检查尚未验证的关系，再选择能产生新证据的动作。",
        "hint_level": 1,
        "student_prompt_ids": [1, 2, 3],
        "teacher_prompt_ids": [1, 2, 3, 4],
        "completion_ids": [5, 6],
        "verification": {
            "source_rollout_failed": True,
            "hinted_local_pass": True,
            "hinted_episode_pass": True,
            "hinted_strict_trace_audit_pass": True,
        },
        "model_roles": {
            "hint_constructor": {
                "provider": "gemini",
                "model": "gemini-hint-model",
                "supplies_training_distribution": False,
            },
            "frozen_self_teacher": {
                "provider": "qwen_local",
                "model": model,
                "round_start_checkpoint": checkpoint,
                "checkpoint_manifest_sha256": "placeholder",
                "sees_hint": True,
                "supplies_training_distribution": True,
            },
            "trainable_student": {
                "provider": "qwen_local",
                "model": model,
                "initial_checkpoint": checkpoint,
                "checkpoint_manifest_sha256": "placeholder",
                "sees_hint": False,
            },
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint-manifest.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "ifv-qwen-checkpoint-manifest-v1",
                "checkpoint": {"path": "/checkpoints/round-0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_sha = sha256_file(checkpoint)
    repair = _repair_row()
    repair["model_roles"]["frozen_self_teacher"][
        "checkpoint_manifest_sha256"
    ] = checkpoint_sha
    repair["model_roles"]["trainable_student"][
        "checkpoint_manifest_sha256"
    ] = checkpoint_sha
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps(build_repair_target(repair), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    profile = tmp_path / "serving-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "ifv-qwen-serving-profile-v1",
                "profile_id": "ifv-qwen-round-0",
                "model_path": "/checkpoints/round-0",
                "engine": "vllm",
                "wire_api": "chat_completions",
                "base_url": "http://127.0.0.1:8901/v1",
                "context_length": 131072,
                "checkpoint_manifest_sha256": checkpoint_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return targets, profile, checkpoint


def _position(*, actual: int, start: int) -> dict[str, Mapping[str, Any]]:
    # vLLM may include the forced token in addition to the requested top-k.
    return {
        str(start): {"logprob": -0.1, "rank": 1},
        str(start + 1): {"logprob": -0.3, "rank": 2},
        str(actual): {"logprob": -9.0, "rank": 100},
    }


def test_normalized_topk_drops_forced_token_outside_ranked_topk() -> None:
    result = normalized_topk(_position(actual=99, start=10), topk=2)
    assert [item[0] for item in result] == [10, 11]
    assert sum(float(item[1]) for item in result) == pytest.approx(1.0)


def test_visual_scoring_is_rejected_before_network(tmp_path: Path) -> None:
    targets, profile, checkpoint = _write_inputs(tmp_path)
    row = json.loads(targets.read_text(encoding="utf-8"))
    row["teacher_prompt_ids"].append(248056)
    targets.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def requester(*args):
        pytest.fail("visual token-only targets must never reach the server")

    with pytest.raises(ValueError, match="psd_multimodal_not_supported"):
        collect_psd_topk_cache(
            targets_path=targets, serving_profile_path=profile,
            checkpoint_manifest_path=checkpoint, output_dir=tmp_path / "cache",
            topk=2, requester=requester,
        )


def test_collect_topk_uses_forced_ids_and_resumes_without_server(
    tmp_path: Path,
) -> None:
    targets, profile, checkpoint = _write_inputs(tmp_path)
    requests: list[tuple[str, Mapping[str, Any] | None]] = []

    def requester(
        url: str,
        payload: Mapping[str, Any] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        assert timeout == 12.0
        requests.append((url, payload))
        if url.endswith("/models"):
            return {"data": [{"id": "ifv-qwen-round-0"}]}
        assert payload is not None
        assert payload["prompt"] == [1, 2, 3, 4, 5, 6]
        assert payload["prompt_logprobs"] == 2
        return {
            "choices": [
                {
                    "prompt_token_ids": payload["prompt"],
                    "prompt_logprobs": [
                        None,
                        None,
                        None,
                        None,
                        _position(actual=5, start=20),
                        _position(actual=6, start=30),
                    ],
                }
            ]
        }

    output = tmp_path / "cache"
    result = collect_psd_topk_cache(
        targets_path=targets,
        serving_profile_path=profile,
        checkpoint_manifest_path=checkpoint,
        output_dir=output,
        topk=2,
        retries=1,
        timeout=12.0,
        requester=requester,
    )
    assert result["status"] == "ready_for_materialization"
    assert result["counts"]["collected_this_run"] == 1
    assert len(requests) == 2
    cache_row = json.loads(
        (output / "teacher_topk_cache.jsonl").read_text(encoding="utf-8")
    )
    assert cache_row["collection"]["method"] == (
        "forced_token_ids_prompt_logprobs"
    )
    assert cache_row["teacher_topk_by_position"][0][0][0] == 20

    def offline(*args: object, **kwargs: object) -> Mapping[str, Any]:
        raise AssertionError("completed collection must resume without vLLM")

    resumed = collect_psd_topk_cache(
        targets_path=targets,
        serving_profile_path=profile,
        checkpoint_manifest_path=checkpoint,
        output_dir=output,
        topk=2,
        retries=1,
        requester=offline,
    )
    assert resumed["status"] == "ready_for_materialization"
    assert resumed["counts"]["cached_before_run"] == 1
    assert resumed["counts"]["collected_this_run"] == 0


def test_collect_topk_persists_failure_and_can_retry_same_directory(
    tmp_path: Path,
) -> None:
    targets, profile, checkpoint = _write_inputs(tmp_path)

    def broken(
        url: str,
        payload: Mapping[str, Any] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        if url.endswith("/models"):
            return {"data": [{"id": "ifv-qwen-round-0"}]}
        return {"choices": [{"prompt_token_ids": [999]}]}

    output = tmp_path / "cache"
    result = collect_psd_topk_cache(
        targets_path=targets,
        serving_profile_path=profile,
        checkpoint_manifest_path=checkpoint,
        output_dir=output,
        topk=2,
        retries=1,
        requester=broken,
    )
    assert result["status"] == "blocked_collection_errors"
    assert result["counts"]["remaining"] == 1
    assert not (output / "teacher_topk_cache.jsonl").exists()
    assert "prompt_token_ids" in (output / "failures.jsonl").read_text(
        encoding="utf-8"
    )


def test_collect_topk_rejects_profile_checkpoint_mismatch(tmp_path: Path) -> None:
    targets, profile, checkpoint = _write_inputs(tmp_path)
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["model_path"] = "/checkpoints/wrong"
    profile.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="round-start checkpoint"):
        collect_psd_topk_cache(
            targets_path=targets,
            serving_profile_path=profile,
            checkpoint_manifest_path=checkpoint,
            output_dir=tmp_path / "cache",
            topk=2,
            retries=1,
            requester=lambda *_: {},
        )
