from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "verify_qwen35_rl_environment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen35_rl_environment", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rl_preflight_requires_real_online_rl_arguments() -> None:
    required = MODULE.RLHF_ARGUMENT_FIELDS
    assert {
        "rlhf_type",
        "use_vllm",
        "vllm_mode",
        "use_gym_env",
        "multi_turn_scheduler",
        "max_turns",
        "num_generations",
    }.issubset(required)
    assert MODULE.missing_rlhf_fields(sorted(required)) == []
    assert MODULE.missing_rlhf_fields([]) == sorted(required)


def test_rl_preflight_records_command_timeout(monkeypatch) -> None:
    def expire(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(MODULE.subprocess, "run", expire)
    result = MODULE.run_command(["slow-command"], 7)

    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["returncode"] is None


def test_rl_preflight_uses_same_native_core_as_long_sft() -> None:
    assert {
        "torch",
        "transformers",
        "ms-swift",
        "deepspeed",
        "flash-attn",
        "flash-linear-attention",
        "causal-conv1d",
        "liger-kernel",
    } == MODULE.CORE_REQUIRED_PACKAGES
