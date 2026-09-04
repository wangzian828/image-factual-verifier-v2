#!/usr/bin/env python3
"""Preflight a host before IFV rollout, export, audit, or training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _command(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _path_check(raw: str, *, kind: str) -> dict[str, Any]:
    if not raw.strip():
        return {"configured": False, "exists": False, "path": ""}
    path = Path(raw).expanduser().resolve()
    exists = path.is_dir() if kind == "directory" else path.is_file()
    return {"configured": True, "exists": exists, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-provider",
        choices=("none", "gemini", "qwen", "local"),
        default="none",
    )
    parser.add_argument("--require-training", action="store_true")
    args = parser.parse_args()

    data_root = os.getenv("IFV_DATA_ROOT", "").strip()
    env_file = os.getenv("IFV_ENV_FILE", "").strip()
    model = (
        os.getenv("IFV_MODEL_ID", "").strip()
        or os.getenv("QWEN_LOCAL_MODEL_PATH", "").strip()
    )
    checks: dict[str, Any] = {
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "repository": {
            "root": str(REPO_ROOT),
            "git": _command("git", "status", "--short", "--branch"),
        },
        "paths": {
            "data_root": _path_check(data_root, kind="directory"),
            "env_file": _path_check(env_file, kind="file"),
            "model": _path_check(model, kind="directory"),
        },
        "runtime": {
            "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
            "proxy_configured": bool(
                os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            ),
            "no_proxy": os.getenv("NO_PROXY", os.getenv("no_proxy", "")),
        },
        "modules": {
            name: _module(name)
            for name in (
                "httpx",
                "requests",
                "pydantic",
                "PIL",
                "dotenv",
            )
        },
        "commands": {
            name: shutil.which(name) or ""
            for name in ("git", "curl", "nvidia-smi")
        },
    }

    failures: list[str] = []
    if checks["runtime"]["omp_num_threads"] != "1":
        failures.append("OMP_NUM_THREADS must equal 1")
    if data_root and not checks["paths"]["data_root"]["exists"]:
        failures.append("IFV_DATA_ROOT does not exist")
    if env_file and not checks["paths"]["env_file"]["exists"]:
        failures.append("IFV_ENV_FILE does not exist")
    if any(not present for present in checks["modules"].values()):
        failures.append("one or more runtime Python modules are missing")
    if args.require_provider == "gemini" and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        failures.append("Gemini credentials are not configured")
    if args.require_provider == "qwen" and not os.getenv("DASHSCOPE_API_KEY"):
        failures.append("DASHSCOPE_API_KEY is not configured")
    if args.require_provider == "local" and not (
        os.getenv("QWEN_LOCAL_BASE_URL") or os.getenv("LMDEPLOY_BASE_URL")
    ):
        failures.append("local model endpoint is not configured")
    if args.require_training:
        checks["modules"]["swift"] = _module("swift")
        checks["modules"]["torch"] = _module("torch")
        if not checks["modules"]["swift"] or not checks["modules"]["torch"]:
            failures.append("training requires both swift and torch")
        if not model or not checks["paths"]["model"]["exists"]:
            failures.append("training requires an existing IFV_MODEL_ID")

    report = {
        "schema_version": "ifv-server-doctor-v1",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"IFV doctor: {'PASS' if report['passed'] else 'FAIL'}")
        for failure in failures:
            print(f"- {failure}")
        print(f"- repo: {REPO_ROOT}")
        print(f"- python: {sys.executable}")
        print(f"- data root: {data_root or '(default/unspecified)'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
