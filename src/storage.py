"""Shared storage paths for local development and server runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def data_root() -> Optional[Path]:
    raw = os.getenv("IFV_DATA_ROOT", "").strip()
    return Path(raw).expanduser() if raw else None


def data_path(relative: PathLike, local_default: PathLike) -> Path:
    root = data_root()
    return root / Path(relative) if root is not None else Path(local_default)


def default_trace_dir() -> str:
    return str(data_path("runs/traces", "outputs/traces"))


def default_eval_root() -> Path:
    return data_path("runs/eval", "outputs/eval_runs")


def default_tool_cache_dir() -> str:
    return str(data_path("cache/tools", ".cache/tool_results"))
