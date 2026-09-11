from __future__ import annotations

import json
from pathlib import Path

import pytest

from ifv_training.io import sha256_file
from ifv_training.psd_round import (
    complete_psd_round,
    validate_rollout_gate_for_candidates,
    verify_psd_round_rollout,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rollout(
    root: Path,
    *,
    run_id: str,
    checkpoint_manifest: Path,
    checkpoint_path: Path,
) -> tuple[Path, Path, Path]:
    run_dir = root / run_id
    train_cases = root / "train-cases.jsonl"
    train_cases.write_text('{"case_id":"a","split":"train"}\n', encoding="utf-8")
    serving = root / f"{run_id}-serving.json"
    _write(
        serving,
        {
            "schema_version": "ifv-qwen-serving-profile-v1",
            "profile_id": "qwen-round-policy",
            "model_path": str(checkpoint_path),
            "base_url": "http://127.0.0.1:8000/v1",
            "wire_api": "chat_completions",
            "multimodal": True,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        },
    )
    _write(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "status": "completed",
            "completed_at": "2026-09-11T00:00:00Z",
            "benchmark": {"training_prohibited": False},
            "agent": {
                "model": "qwen-round-policy",
                "base_url": "http://127.0.0.1:8000/v1",
                "llm_wire_api": "chat_completions",
            },
        },
    )
    (run_dir / "post_rollout_rewards.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "rollout_groups.jsonl").write_text("{}\n", encoding="utf-8")
    return run_dir, train_cases, serving


def _checkpoint(path: Path, *, step: int, dataset_sha: str = "dataset-sha") -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(f"test weights {step}".encode())
    (path / "optimizer.pt").write_bytes(b"test optimizer state")
    manifest = path.parent / f"checkpoint-{step}-manifest.json"
    _write(
        manifest,
        {
            "schema_version": "ifv-qwen-checkpoint-manifest-v1",
            "checkpoint": {"path": str(path), "global_step": step,
                "optimizer_state_available": True, "scheduler_state_available": True,
                "rng_state_available": True},
            "training_dataset": {"manifest_sha256": dataset_sha},
            "artifacts": [{"scope": scope, "path": name, "sha256": sha256_file(path / name)}
                          for scope, name in (("model", "model.safetensors"), ("training_state", "optimizer.pt"))],
        },
    )
    return manifest


def test_psd_round_chain_requires_fresh_rollout_from_previous_output(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base"
    base_manifest = _checkpoint(base_path, step=0)
    run1, cases, serving1 = _rollout(
        tmp_path,
        run_id="round-1-rollout",
        checkpoint_manifest=base_manifest,
        checkpoint_path=base_path,
    )
    rollout_gate1 = tmp_path / "round-1-rollout-gate.json"
    first = verify_psd_round_rollout(
        round_index=1,
        run_dir=run1,
        train_cases_path=cases,
        serving_profile_path=serving1,
        round_start_checkpoint_manifest_path=base_manifest,
        output=rollout_gate1,
    )
    assert first["passed"] is True

    trained_path = tmp_path / "checkpoint-1"
    trained_manifest = _checkpoint(trained_path, step=1)
    training_profile = tmp_path / "round-1-training-profile.json"
    _write(
        training_profile,
        {
            "passed_production_gate": True,
            "checkpoint_save": {"states": [{"path": str(trained_path)}]},
            "raw_dataset_gate": {
                "verification": {"manifest": {"sha256": "dataset-sha"}}
            },
        },
    )
    completion1 = tmp_path / "round-1-completion.json"
    completed = complete_psd_round(
        rollout_gate_path=rollout_gate1,
        training_profile_path=training_profile,
        output_checkpoint_manifest_path=trained_manifest,
        output=completion1,
    )
    assert completed["passed"] is True
    for check in ("optimizer_state_available", "scheduler_state_available", "rng_state_available"):
        changed = json.loads(trained_manifest.read_text())
        changed["checkpoint"][check] = False
        tampered = tmp_path / "tampered-manifest.json"
        _write(tampered, changed)
        rejected = complete_psd_round(rollout_gate_path=rollout_gate1,
            training_profile_path=training_profile, output_checkpoint_manifest_path=tampered,
            output=tmp_path / "rejected.json")
        assert not rejected["passed"]
    original_weights = (trained_path / "model.safetensors").read_bytes()
    (trained_path / "model.safetensors").write_bytes(b"changed after manifest")
    assert not complete_psd_round(rollout_gate_path=rollout_gate1,
        training_profile_path=training_profile, output_checkpoint_manifest_path=trained_manifest,
        output=tmp_path / "rejected.json")["passed"]
    (trained_path / "model.safetensors").write_bytes(original_weights)

    run2, cases2, serving2 = _rollout(
        tmp_path,
        run_id="round-2-rollout",
        checkpoint_manifest=trained_manifest,
        checkpoint_path=trained_path,
    )
    second = verify_psd_round_rollout(
        round_index=2,
        run_dir=run2,
        train_cases_path=cases2,
        serving_profile_path=serving2,
        round_start_checkpoint_manifest_path=trained_manifest,
        previous_round_completion_path=completion1,
        output=tmp_path / "round-2-rollout-gate.json",
    )
    assert second["passed"] is True
    assert second["checks"]["fresh_rollout_run_id"]["passed"] is True


def test_candidate_gate_rejects_rollout_mutation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "base"
    checkpoint_manifest = _checkpoint(checkpoint_path, step=0)
    run_dir, cases, serving = _rollout(
        tmp_path,
        run_id="round-1-rollout",
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_path=checkpoint_path,
    )
    gate = tmp_path / "gate.json"
    verify_psd_round_rollout(
        round_index=1,
        run_dir=run_dir,
        train_cases_path=cases,
        serving_profile_path=serving,
        round_start_checkpoint_manifest_path=checkpoint_manifest,
        output=gate,
    )
    (run_dir / "post_rollout_rewards.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rewards"):
        validate_rollout_gate_for_candidates(
            rollout_gate_path=gate,
            run_dir=run_dir,
            train_cases_path=cases,
        )
