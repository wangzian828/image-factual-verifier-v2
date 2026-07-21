from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import sha256_file, write_json


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "ms-swift",
    "deepspeed",
    "vllm",
    "lmdeploy",
    "xformers",
    "flashinfer-python",
    "xgrammar",
    "llguidance",
    "compressed-tensors",
    "ray",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "qwen-vl-utils",
    "rllm",
    "verl",
    "sglang",
)

MS_SWIFT_RELEASE = {
    "name": "ms-swift",
    "version": "4.4.1",
    "git_tag": "v4.4.1",
    "git_commit": "98a09c18cdf95ff07051324b9b8cc90f5184b24b",
}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def environment_manifest(repo_root: Path) -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    installed = sorted(
        {
            distribution.metadata["Name"].casefold(): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }.items()
    )
    lock_payload = json.dumps(
        installed,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "ifv-training-environment-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "package_lock": {
            "algorithm": "sha256",
            "normalized_distribution_count": len(installed),
            "sha256": hashlib.sha256(lock_payload).hexdigest(),
        },
        "framework_release": MS_SWIFT_RELEASE,
        "torch": torch_info,
        "repository": {
            "path": str(repo_root.resolve()),
            "commit": _git_commit(repo_root),
        },
    }


def artifact_manifest(
    *,
    kind: str,
    root: Path,
    files: Iterable[Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifacts = []
    for raw_path in files:
        path = raw_path.resolve()
        artifacts.append(
            {
                "path": str(path.relative_to(root.resolve())),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "ifv-training-artifact-manifest-v1",
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "artifacts": artifacts,
    }


def write_environment_manifest(output: Path, repo_root: Path) -> dict[str, Any]:
    value = environment_manifest(repo_root)
    write_json(output, value)
    return value
