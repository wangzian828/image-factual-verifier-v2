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


def test_rl_preflight_uses_ms_swift_rlhf_arguments_class() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from swift.arguments import RLHFArguments" in source
    assert "RlhfArguments" not in source


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


def test_rl_preflight_runs_core_verifier_with_same_timeout(monkeypatch) -> None:
    captured = {}

    def fake_run(command, timeout):
        captured["command"] = command
        captured["timeout"] = timeout
        return {
            "passed": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": 3.0,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
        }

    monkeypatch.setattr(MODULE, "run_command", fake_run)
    result = MODULE.verify_core_environment(
        model=Path("/model"),
        max_context=131072,
        expected_versions={"torch": "2.11.0"},
        expected_python="3.12",
        expected_torch_cuda="13.0",
        expected_gpu_count=1,
        expected_gpu_name="NVIDIA A100-SXM4-40GB",
        expected_gpu_memory_mib=40960,
        gpu_memory_tolerance_mib=128,
        timeout=17,
    )

    assert result["passed"] is False
    assert captured["timeout"] == 17
    assert str(MODULE.BASE_SCRIPT) in captured["command"]
