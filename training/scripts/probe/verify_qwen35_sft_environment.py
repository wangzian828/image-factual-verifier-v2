#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "ms-swift": "swift",
    "deepspeed": "deepspeed",
    "flash-attn": "flash_attn",
    "flash-linear-attention": "fla",
    "causal-conv1d": "causal_conv1d",
    "liger-kernel": "liger_kernel",
}
NATIVE_EXTENSION_MODULES = {
    "flash-attn": "flash_attn_2_cuda",
    "causal-conv1d": "causal_conv1d_cuda",
}
TEMPLATE_PARAMETERS = {
    "max_length",
    "truncation_strategy",
    "max_pixels",
    "padding_free",
    "loss_scale",
    "sequence_parallel_size",
    "enable_thinking",
    "add_non_thinking_prefix",
}
SFT_ARGUMENT_FIELDS = {
    "fsdp",
    "max_length",
    "attn_impl",
    "padding_free",
    "sequence_parallel_size",
    "use_logits_to_keep",
    "gradient_checkpointing",
    "vit_gradient_checkpointing",
    "save_strategy",
    "resume_from_checkpoint",
    "add_non_thinking_prefix",
    "max_pixels",
    "loss_scale",
}


def parse_expected_versions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        package, separator, version = value.partition("=")
        package = package.strip()
        version = version.strip()
        if not separator or not package or not version:
            raise ValueError(
                "expected package versions must use PACKAGE=VERSION: "
                f"{value!r}"
            )
        if package in result:
            raise ValueError(f"duplicate expected package: {package}")
        result[package] = version
    return result


def parse_visible_gpu_ids(value: str) -> list[int]:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        raise ValueError("CUDA_VISIBLE_DEVICES is empty")
    try:
        result = [int(entry) for entry in entries]
    except ValueError as exc:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain physical integer GPU ids"
        ) from exc
    if len(result) != len(set(result)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU ids")
    return result


def parse_gpu_inventory(output: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi inventory row: {line!r}")
        index = int(parts[0])
        result[index] = {
            "physical_index": index,
            "name": parts[1],
            "memory_total_mib": int(parts[2]),
        }
    return result


def parse_glibc_versions(output: str) -> list[str]:
    versions = {
        match.group(1)
        for match in re.finditer(r"\bGLIBC_(\d+(?:\.\d+)+)\b", output)
    }
    return sorted(versions, key=_version_tuple)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def inspect_native_extension(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"native extension module is missing: {module_name}")
    path = Path(spec.origin).resolve()
    if path.suffix != ".so":
        raise RuntimeError(
            f"native extension does not resolve to a shared object: {path}"
        )
    process = subprocess.run(
        ["readelf", "--version-info", "--wide", str(path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"readelf failed for {path}: {detail}")
    required_versions = parse_glibc_versions(process.stdout)
    libc_name, host_version = platform.libc_ver()
    if libc_name != "glibc" or not host_version:
        raise RuntimeError(
            f"could not determine host glibc version: {libc_name!r} {host_version!r}"
        )
    maximum_required = required_versions[-1] if required_versions else ""
    compatible = not maximum_required or (
        _version_tuple(maximum_required) <= _version_tuple(host_version)
    )
    return {
        "module": module_name,
        "path": str(path),
        "host_glibc": host_version,
        "required_glibc_versions": required_versions,
        "maximum_required_glibc": maximum_required,
        "compatible": compatible,
    }


def read_model_contract(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen3.5 config is missing text_config")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or not all(
        isinstance(value, str) for value in layer_types
    ):
        raise ValueError("Qwen3.5 text_config is missing layer_types")
    return {
        "path": str(model.resolve()),
        "config_path": str(config_path.resolve()),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text_config.get("model_type"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "layer_type_counts": dict(sorted(Counter(layer_types).items())),
    }


def _gpu_inventory() -> dict[int, dict[str, Any]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"nvidia-smi inventory failed: {detail}")
    return parse_gpu_inventory(process.stdout)


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
    required_packages: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    packages: dict[str, dict[str, Any]] = {}
    imported_modules: dict[str, dict[str, str]] = {}
    native_extensions: dict[str, dict[str, Any]] = {}

    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        errors.append(
            f"Python version mismatch: expected {expected_python}, got {actual_python}"
        )

    for package, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = ""
            errors.append(f"missing package: {package}")
        packages[package] = {"expected": expected, "actual": actual}
        if actual and actual != expected:
            errors.append(
                f"package version mismatch for {package}: "
                f"expected {expected}, got {actual}"
            )

    for package in sorted(required_packages):
        module_name = MODULES.get(package)
        if module_name is None:
            errors.append(f"no import mapping is defined for package: {package}")
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on target CUDA env
            errors.append(
                f"failed to import {package} ({module_name}): "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        imported_modules[package] = {
            "module": module_name,
            "path": str(getattr(module, "__file__", "") or ""),
        }

    for package, module_name in NATIVE_EXTENSION_MODULES.items():
        if package not in required_packages:
            continue
        try:
            native_contract = inspect_native_extension(module_name)
        except Exception as exc:  # pragma: no cover - depends on target env
            errors.append(
                f"failed to inspect native extension for {package}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        native_extensions[package] = native_contract
        if not native_contract["compatible"]:
            errors.append(
                f"native extension for {package} requires GLIBC_"
                f"{native_contract['maximum_required_glibc']}, but the host has "
                f"glibc {native_contract['host_glibc']}"
            )

    torch_cuda = ""
    torch_cuda_available = False
    logical_gpu_count = 0
    try:
        import torch

        torch_cuda = str(torch.version.cuda or "")
        torch_cuda_available = bool(torch.cuda.is_available())
        logical_gpu_count = int(torch.cuda.device_count())
        if torch_cuda != expected_torch_cuda:
            errors.append(
                f"PyTorch CUDA mismatch: expected {expected_torch_cuda}, "
                f"got {torch_cuda or 'none'}"
            )
        if not torch_cuda_available:
            errors.append("CUDA is unavailable to PyTorch")
        if logical_gpu_count != expected_gpu_count:
            errors.append(
                f"PyTorch logical GPU count mismatch: expected "
                f"{expected_gpu_count}, got {logical_gpu_count}"
            )
    except Exception as exc:  # pragma: no cover - depends on target CUDA env
        errors.append(f"failed to inspect PyTorch CUDA: {type(exc).__name__}: {exc}")

    template_parameters: list[str] = []
    try:
        from swift import get_template

        template_parameters = list(inspect.signature(get_template).parameters)
        missing = sorted(TEMPLATE_PARAMETERS.difference(template_parameters))
        if missing:
            errors.append(
                "ms-swift get_template is missing required parameters: "
                + ", ".join(missing)
            )
    except Exception as exc:  # pragma: no cover - depends on target env
        errors.append(
            f"failed to inspect ms-swift template contract: "
            f"{type(exc).__name__}: {exc}"
        )

    training_argument_fields: list[str] = []
    try:
        from swift.arguments import SftArguments

        training_argument_fields = [field.name for field in fields(SftArguments)]
        missing = sorted(SFT_ARGUMENT_FIELDS.difference(training_argument_fields))
        if missing:
            errors.append(
                "ms-swift SftArguments is missing required fields: "
                + ", ".join(missing)
            )
    except Exception as exc:  # pragma: no cover - depends on target env
        errors.append(
            f"failed to inspect ms-swift SFT argument contract: "
            f"{type(exc).__name__}: {exc}"
        )

    model_contract: dict[str, Any] = {}
    try:
        model_contract = read_model_contract(model)
        if model_contract.get("model_type") != "qwen3_5":
            errors.append(
                f"unexpected model type: {model_contract.get('model_type')!r}"
            )
        architectures = model_contract.get("architectures")
        if not isinstance(architectures, list) or (
            "Qwen3_5ForConditionalGeneration" not in architectures
        ):
            errors.append(f"unexpected model architectures: {architectures!r}")
        model_context = model_contract.get("max_position_embeddings")
        if not isinstance(model_context, int) or model_context < max_context:
            errors.append(
                f"model context is below requested length: "
                f"model={model_context!r}, requested={max_context}"
            )
        layer_counts = model_contract.get("layer_type_counts") or {}
        if (
            layer_counts.get("linear_attention", 0) < 1
            or layer_counts.get("full_attention", 0) < 1
        ):
            errors.append(
                f"Qwen3.5 hybrid layer contract is missing: {layer_counts!r}"
            )
    except Exception as exc:
        errors.append(f"failed to inspect model config: {type(exc).__name__}: {exc}")

    visible_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpu_ids: list[int] = []
    selected_gpus: list[dict[str, Any]] = []
    try:
        visible_gpu_ids = parse_visible_gpu_ids(visible_value)
        if len(visible_gpu_ids) != expected_gpu_count:
            errors.append(
                f"visible GPU count mismatch: expected {expected_gpu_count}, "
                f"got {len(visible_gpu_ids)}"
            )
        inventory = _gpu_inventory()
        for index in visible_gpu_ids:
            gpu = inventory.get(index)
            if gpu is None:
                errors.append(f"physical GPU {index} is missing from nvidia-smi")
                continue
            selected_gpus.append(gpu)
            if gpu["name"] != expected_gpu_name:
                errors.append(
                    f"GPU {index} name mismatch: expected {expected_gpu_name!r}, "
                    f"got {gpu['name']!r}"
                )
            memory_delta = abs(
                int(gpu["memory_total_mib"]) - expected_gpu_memory_mib
            )
            if memory_delta > gpu_memory_tolerance_mib:
                errors.append(
                    f"GPU {index} memory mismatch: expected "
                    f"{expected_gpu_memory_mib} MiB +/- "
                    f"{gpu_memory_tolerance_mib} MiB, got "
                    f"{gpu['memory_total_mib']} MiB"
                )
    except Exception as exc:
        errors.append(f"failed to inspect GPU inventory: {type(exc).__name__}: {exc}")

    return {
        "schema_version": "ifv-qwen35-sft-environment-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "python": {
            "expected_major_minor": expected_python,
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "required_package_imports": imported_modules,
        "native_extensions": native_extensions,
        "torch_cuda": {
            "expected": expected_torch_cuda,
            "actual": torch_cuda,
            "available": torch_cuda_available,
            "logical_gpu_count": logical_gpu_count,
        },
        "template_contract": {
            "required_parameters": sorted(TEMPLATE_PARAMETERS),
            "actual_parameters": template_parameters,
        },
        "sft_argument_contract": {
            "required_fields": sorted(SFT_ARGUMENT_FIELDS),
            "actual_fields": training_argument_fields,
        },
        "model": model_contract,
        "requested_max_context": max_context,
        "gpu_contract": {
            "cuda_visible_devices": visible_value,
            "visible_physical_ids": visible_gpu_ids,
            "expected_count": expected_gpu_count,
            "expected_name": expected_gpu_name,
            "expected_memory_mib": expected_gpu_memory_mib,
            "memory_tolerance_mib": gpu_memory_tolerance_mib,
            "selected": selected_gpus,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--expected-package-version", action="append", default=[])
    parser.add_argument("--required-package", action="append", default=[])
    parser.add_argument("--expected-python", default="3.12")
    parser.add_argument("--expected-torch-cuda", default="12.8")
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument(
        "--expected-gpu-name", default="NVIDIA A100-SXM4-40GB"
    )
    parser.add_argument("--expected-gpu-memory-mib", type=int, default=40960)
    parser.add_argument("--gpu-memory-tolerance-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.expected_gpu_count < 1:
        parser.error("--expected-gpu-count must be positive")
    if args.expected_gpu_memory_mib < 1:
        parser.error("--expected-gpu-memory-mib must be positive")
    if args.gpu_memory_tolerance_mib < 0:
        parser.error("--gpu-memory-tolerance-mib must not be negative")
    try:
        expected_versions = parse_expected_versions(
            args.expected_package_version
        )
    except ValueError as exc:
        parser.error(str(exc))
    required_packages = set(args.required_package)
    missing_version_contracts = sorted(
        required_packages.difference(expected_versions)
    )
    if missing_version_contracts:
        parser.error(
            "required packages must also have exact expected versions: "
            + ", ".join(missing_version_contracts)
        )
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
        required_packages=required_packages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
