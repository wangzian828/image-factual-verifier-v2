from __future__ import annotations

import json
import subprocess
import sys
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


def test_production_psd_profile_gate_rejects_sp1(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    command = _command(output)
    index = command.index("--sequence-parallel-size") + 1
    command[index] = "1"
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 1
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert "sequence_parallel_size_must_be_8" in gate["errors"]
