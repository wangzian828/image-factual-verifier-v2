from __future__ import annotations

import json
import subprocess
import sys
import pytest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "verify_psd_training_profile.py"
)


def _command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--mode",
        "production",
        "--world-size",
        "8",
        "--sequence-parallel-size",
        "8",
        "--padding-free",
        "true",
        "--use-logits-to-keep",
        "false",
        "--max-context",
        "131072",
        "--topk",
        "20",
        "--loss-chunk-tokens",
        "256",
        "--tuner-type",
        "lora",
        "--lora-rank",
        "32",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0",
        "--target-modules",
        "all-linear",
        "--train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "32",
        "--learning-rate",
        "4e-5",
        "--num-train-epochs",
        "5",
        "--output",
        str(output),
    ]


def test_production_psd_profile_gate_accepts_upstream_recipe(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    result = subprocess.run(_command(output), check=False, capture_output=True)

    assert result.returncode == 0, result.stderr.decode()
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["parallelism"]["data_parallel_size"] == 1
    assert gate["optimization"]["unique_targets_per_step"] == 32


def test_production_psd_profile_gate_accepts_dp8_with_sparse_logits(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    command[command.index("--sequence-parallel-size") + 1] = "1"
    command[command.index("--gradient-accumulation-steps") + 1] = "4"
    command[command.index("--use-logits-to-keep") + 1] = "true"
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 0, result.stderr.decode()
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["parallelism"]["data_parallel_size"] == 8
    assert gate["optimization"]["unique_targets_per_step"] == 32


def test_production_psd_profile_gate_rejects_dp_without_sparse_logits(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    command[command.index("--sequence-parallel-size") + 1] = "1"
    command[command.index("--gradient-accumulation-steps") + 1] = "4"
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 1
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert "data_parallel_training_requires_logits_to_keep" in gate["errors"]


def test_h20_batched_dp4_preserves_optimizer_batch(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    command[command.index("--world-size") + 1] = "4"
    command[command.index("--sequence-parallel-size") + 1] = "1"
    command[command.index("--padding-free") + 1] = "false"
    command[command.index("--use-logits-to-keep") + 1] = "true"
    command[command.index("--train-batch-size") + 1] = "4"
    command[command.index("--gradient-accumulation-steps") + 1] = "2"
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 0, result.stderr.decode()
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["parallelism"]["data_parallel_size"] == 4
    assert gate["optimization"]["unique_targets_per_step"] == 32


def test_sequence_parallel_profile_still_requires_padding_free(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    command[command.index("--padding-free") + 1] = "false"
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 1
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert "sequence_parallel_requires_padding_free" in gate["errors"]


def test_h20_sp4_preserves_optimizer_batch(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    for option in ("--world-size", "--sequence-parallel-size"):
        command[command.index(option) + 1] = "4"
    result = subprocess.run(command, check=False, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["optimization"]["unique_targets_per_step"] == 32
    assert gate["parallelism"]["data_parallel_size"] == 1


def test_h20_rejects_halved_accumulation(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    for option in ("--world-size", "--sequence-parallel-size"):
        command[command.index(option) + 1] = "4"
    command[command.index("--gradient-accumulation-steps") + 1] = "16"
    result = subprocess.run(command, check=False, capture_output=True)
    assert result.returncode == 1
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert "production_unique_targets_per_step_must_be_32" in gate["errors"]


@pytest.mark.parametrize("option,value", [("lr-scheduler-type", "cosine"),
    ("loss-reduction", "mean"), ("adam-beta2", "0.999"),
    ("adam-epsilon", "1e-8"), ("weight-decay", "0.1")])
def test_backend_defaults_cannot_silently_change_recipe(tmp_path, option, value):
    output = tmp_path / "gate.json"
    command = _command(output) + ["--" + option, value]
    result = subprocess.run(command, check=False, capture_output=True)
    assert result.returncode == 1
    assert json.loads(output.read_text())["passed"] is False
