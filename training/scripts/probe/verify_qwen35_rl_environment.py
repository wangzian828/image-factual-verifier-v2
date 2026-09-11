#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("verify_qwen35_sft_environment.py")
RLHF_ARGUMENT_FIELDS = {
    "rlhf_type",
    "use_vllm",
    "vllm_mode",
    "vllm_gpu_memory_utilization",
    "vllm_tensor_parallel_size",
    "use_gym_env",
    "multi_turn_scheduler",
    "max_turns",
    "max_completion_length",
    "num_generations",
    "steps_per_generation",
}
CORE_REQUIRED_PACKAGES = {
    "torch",
    "transformers",
    "ms-swift",
    "deepspeed",
    "flash-attn",
    "flash-linear-attention",
    "causal-conv1d",
    "liger-kernel",
}


def _load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "verify_qwen35_sft_environment", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base environment verifier: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def missing_rlhf_fields(actual_fields: list[str]) -> list[str]:
    return sorted(RLHF_ARGUMENT_FIELDS.difference(actual_fields))


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "passed": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout_excerpt": stdout[-2000:],
            "stderr_excerpt": stderr[-2000:],
        }
    return {
        "command": command,
        "passed": process.returncode == 0,
        "timed_out": False,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_excerpt": process.stdout[-2000:],
        "stderr_excerpt": process.stderr[-2000:],
    }


def verify_rl_contract(timeout: int) -> dict[str, Any]:
    code = f"""
import dataclasses
import json
import jiter
import openai
from swift.arguments import RlhfArguments

required = set({sorted(RLHF_ARGUMENT_FIELDS)!r})
actual = {{field.name for field in dataclasses.fields(RlhfArguments)}}
print(json.dumps({{
    "field_count": len(actual),
    "required_fields_present": sorted(required.intersection(actual)),
    "missing_fields": sorted(required.difference(actual)),
    "openai_module": openai.__file__,
    "jiter_module": jiter.__file__,
}}))
"""
    command_result = run_command([sys.executable, "-c", code], timeout)
    actual_fields: list[str] = []
    parse_error = ""
    if command_result["passed"]:
        try:
            payload = json.loads(command_result["stdout_excerpt"])
            actual_fields = payload["required_fields_present"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            parse_error = f"could not parse RL argument probe: {exc}"
    missing = missing_rlhf_fields(actual_fields)
    return {
        "passed": command_result["passed"] and not parse_error and not missing,
        "required_fields": sorted(RLHF_ARGUMENT_FIELDS),
        "actual_fields": actual_fields,
        "missing_fields": missing,
        "parse_error": parse_error,
        "process": command_result,
    }


def verify(
    *,
    model: Path,
    max_context: int,
    expected_versions: dict[str, str],
    expected_python: str,
    expected_torch_cuda: str,
    expected_gpu_count: int,
    expected_gpu_name: str,
    expected_gpu_memory_mib: int,
    gpu_memory_tolerance_mib: int,
    command_timeout: int,
) -> dict[str, Any]:
    base = BASE.verify(
        model=model,
        max_context=max_context,
        expected_versions=expected_versions,
        expected_python=expected_python,
        expected_torch_cuda=expected_torch_cuda,
        expected_gpu_count=expected_gpu_count,
        expected_gpu_name=expected_gpu_name,
        expected_gpu_memory_mib=expected_gpu_memory_mib,
        gpu_memory_tolerance_mib=gpu_memory_tolerance_mib,
        required_packages=CORE_REQUIRED_PACKAGES,
    )
    environment_bin = Path(sys.executable).resolve().parent
    commands = {
        "pip_check": run_command(
            [sys.executable, "-m", "pip", "check"], command_timeout
        ),
        "swift_rlhf_help": run_command(
            [str(environment_bin / "swift"), "rlhf", "--help"],
            command_timeout,
        ),
        "vllm_serve_help": run_command(
            [str(environment_bin / "vllm"), "serve", "--help"],
            command_timeout,
        ),
    }
    rlhf_contract = verify_rl_contract(command_timeout)
    errors = list(base["errors"])
    for name, result in commands.items():
        if not result["passed"]:
            errors.append(f"RL environment command failed: {name}")
    if not rlhf_contract["passed"]:
        errors.append("ms-swift RL argument/import contract failed")
    return {
        "schema_version": "ifv-qwen35-rl-environment-preflight-v1",
        "passed": not errors,
        "errors": errors,
        "core_environment": base,
        "rlhf_argument_contract": rlhf_contract,
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--expected-package-version", action="append", default=[])
    parser.add_argument("--expected-python", default="3.12")
    parser.add_argument("--expected-torch-cuda", default="13.0")
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument(
        "--expected-gpu-name", default="NVIDIA A100-SXM4-40GB"
    )
    parser.add_argument("--expected-gpu-memory-mib", type=int, default=40960)
    parser.add_argument("--gpu-memory-tolerance-mib", type=int, default=128)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.expected_gpu_count < 1:
        parser.error("--expected-gpu-count must be positive")
    if args.command_timeout < 1:
        parser.error("--command-timeout must be positive")
    try:
        expected_versions = BASE.parse_expected_versions(
            args.expected_package_version
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = verify(
        model=args.model,
        max_context=args.max_context,
        expected_versions=expected_versions,
        expected_python=args.expected_python,
        expected_torch_cuda=args.expected_torch_cuda,
        expected_gpu_count=args.expected_gpu_count,
        expected_gpu_name=args.expected_gpu_name,
        expected_gpu_memory_mib=args.expected_gpu_memory_mib,
        gpu_memory_tolerance_mib=args.gpu_memory_tolerance_mib,
        command_timeout=args.command_timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
