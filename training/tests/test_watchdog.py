from __future__ import annotations

import json
from pathlib import Path

from ifv_training.watchdog import (
    build_sft_watchdog_snapshot,
    concise_watchdog_status,
)


def _write_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
    return checkpoint


def test_watchdog_reports_running_training_and_latest_checkpoint(
    tmp_path: Path,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --max_steps 10 --eval_strategy steps",
                "{'loss': '1.2', 'grad_norm': '3.0', 'token_acc': '0.5', "
                "'global_step/max_steps': '1/10', 'remaining_time': '9m'}",
                "{'loss': '0.9', 'grad_norm': '2.0', 'token_acc': '0.6', "
                "'global_step/max_steps': '2/10', 'remaining_time': '8m'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resources = tmp_path / "resource-samples.jsonl"
    resources.write_text(
        json.dumps(
            {
                "timestamp_epoch": 100.0,
                "root_alive": True,
                "gpu": {
                    "utilization_percent_by_physical_gpu": {"0": 95},
                    "whole_gpu_memory_mib_by_physical_gpu": {"0": 30000},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root, 2)

    result = build_sft_watchdog_snapshot(
        train_log=train_log,
        checkpoint_root=checkpoint_root,
        resource_samples=resources,
        now_epoch=train_log.stat().st_mtime + 5,
    )

    assert result["status"] == "running"
    assert result["progress"]["percent"] == 20.0
    assert result["optimization"]["latest_loss"] == 0.9
    assert result["checkpoints"]["latest"]["global_step"] == 2
    assert "step=2/10" in concise_watchdog_status(result)


def test_watchdog_detects_stalled_process_and_numerical_regression(
    tmp_path: Path,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --max_steps 20 --eval_strategy steps",
                "{'loss': '1.0', 'grad_norm': '2.0', "
                "'global_step/max_steps': '1/20'}",
                "{'loss': '0.8', 'grad_norm': '2.2', "
                "'global_step/max_steps': '2/20'}",
                "{'loss': '0.7', 'grad_norm': '2.1', "
                "'global_step/max_steps': '3/20'}",
                "{'loss': '2.0', 'grad_norm': '20.0', "
                "'global_step/max_steps': '4/20'}",
                "{'eval_loss': '0.5', 'global_step/max_steps': '2/20'}",
                "{'eval_loss': '0.7', 'global_step/max_steps': '4/20'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resources = tmp_path / "resource-samples.jsonl"
    resources.write_text(
        json.dumps({"timestamp_epoch": 100.0, "root_alive": True}) + "\n",
        encoding="utf-8",
    )

    result = build_sft_watchdog_snapshot(
        train_log=train_log,
        resource_samples=resources,
        stale_seconds=60,
        now_epoch=train_log.stat().st_mtime + 120,
    )

    codes = {alert["code"] for alert in result["alerts"]}
    assert result["status"] == "critical"
    assert "stalled_progress" in codes
    assert "training_loss_spike" in codes
    assert "gradient_norm_spike" in codes
    assert "eval_loss_regression" in codes


def test_watchdog_detects_semantic_collapse(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "Executing: swift sft --max_steps 10\n"
        "{'loss': '0.8', 'global_step/max_steps': '5/10'}\n",
        encoding="utf-8",
    )
    behavior = tmp_path / "behavior.jsonl"
    behavior.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "global_step": 0,
                        "metrics": {
                            "verdict_accuracy": 0.80,
                            "strict_evidence_sufficiency_rate": 0.50,
                            "format_valid_rate": 1.0,
                            "think_present_rate": 1.0,
                            "engineering_failure_rate": 0.0,
                            "majority_verdict_rate": 0.60,
                        },
                    }
                ),
                json.dumps(
                    {
                        "global_step": 5,
                        "metrics": {
                            "verdict_accuracy": 0.60,
                            "strict_evidence_sufficiency_rate": 0.25,
                            "format_valid_rate": 0.80,
                            "think_present_rate": 0.70,
                            "engineering_failure_rate": 0.10,
                            "majority_verdict_rate": 0.95,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_sft_watchdog_snapshot(
        train_log=train_log,
        behavior_metrics=behavior,
        now_epoch=train_log.stat().st_mtime,
    )

    codes = {alert["code"] for alert in result["alerts"]}
    assert result["health"] == "critical"
    assert "semantic_accuracy_regression" in codes
    assert "semantic_evidence_regression" in codes
    assert "format_valid_rate_collapse" in codes
    assert "think_present_rate_collapse" in codes
    assert "engineering_failure_regression" in codes
    assert "verdict_mode_collapse" in codes


def test_watchdog_marks_completed_run(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --max_steps 2 --eval_strategy steps",
                "{'loss': '1.0', 'global_step/max_steps': '1/2'}",
                "{'loss': '0.8', 'global_step/max_steps': '2/2'}",
                "{'eval_loss': '0.7', 'global_step/max_steps': '2/2'}",
                "[INFO:swift] End time of running main: 2026-09-09 12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_sft_watchdog_snapshot(
        train_log=train_log,
        now_epoch=train_log.stat().st_mtime,
    )

    assert result["lifecycle"] == "completed"
    assert result["status"] == "completed"
    assert result["recommendation"] == "select_checkpoint"
