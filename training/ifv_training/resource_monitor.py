from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _read_children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, PermissionError, ValueError):
        return []


def process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen or not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        pending.extend(_read_children(pid))
    return seen


def _rss_kib(pid: int) -> int:
    path = Path(f"/proc/{pid}/status")
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return 0
    return 0


def _process_cpu_seconds(pid: int) -> float:
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text().split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks / float(os.sysconf("SC_CLK_TCK"))
    except (FileNotFoundError, PermissionError, ValueError, OSError, IndexError):
        return 0.0


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_nvidia_smi(query: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _gpu_snapshot(selected_gpu_ids: list[int], pids: set[int]) -> dict[str, Any]:
    gpu_rows = _run_nvidia_smi(
        "gpu=index,uuid,memory.used,utilization.gpu"
    )
    uuid_to_index: dict[str, int] = {}
    whole_gpu_memory: dict[str, int] = {}
    utilization: dict[str, int] = {}
    for row in gpu_rows:
        if len(row) != 4:
            continue
        try:
            index = int(row[0])
            memory_used = int(row[2])
            gpu_utilization = int(row[3])
        except ValueError:
            continue
        uuid_to_index[row[1]] = index
        if index in selected_gpu_ids:
            whole_gpu_memory[str(index)] = memory_used
            utilization[str(index)] = gpu_utilization

    process_memory = {str(index): 0 for index in selected_gpu_ids}
    for row in _run_nvidia_smi("compute-apps=gpu_uuid,pid,used_memory"):
        if len(row) != 3:
            continue
        try:
            pid = int(row[1])
            memory_used = int(row[2])
        except ValueError:
            continue
        index = uuid_to_index.get(row[0])
        if pid in pids and index in selected_gpu_ids:
            process_memory[str(index)] += memory_used
    return {
        "process_memory_mib_by_physical_gpu": process_memory,
        "whole_gpu_memory_mib_by_physical_gpu": whole_gpu_memory,
        "utilization_percent_by_physical_gpu": utilization,
    }


def sample_process_tree(root_pid: int, selected_gpu_ids: list[int]) -> dict[str, Any]:
    pids = process_tree_pids(root_pid)
    rss_kib = sum(_rss_kib(pid) for pid in pids)
    return {
        "timestamp_epoch": time.time(),
        "root_pid": root_pid,
        "root_alive": Path(f"/proc/{root_pid}").exists(),
        "process_count": len(pids),
        "process_tree_rss_mib": round(rss_kib / 1024.0, 3),
        "process_tree_cpu_seconds": round(
            sum(_process_cpu_seconds(pid) for pid in pids),
            6,
        ),
        "pids": sorted(pids),
        "gpu": _gpu_snapshot(selected_gpu_ids, pids),
    }


def summarize_resource_samples(
    samples: Iterable[dict[str, Any]],
    *,
    command: list[str],
    exit_code: int,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    selected_gpu_ids: list[int],
) -> dict[str, Any]:
    rows = list(samples)
    peak_process_memory = {str(index): 0 for index in selected_gpu_ids}
    peak_whole_memory = {str(index): 0 for index in selected_gpu_ids}
    utilization_values: dict[str, list[int]] = {
        str(index): [] for index in selected_gpu_ids
    }
    for row in rows:
        gpu = row.get("gpu")
        gpu = gpu if isinstance(gpu, dict) else {}
        for key, value in (
            gpu.get("process_memory_mib_by_physical_gpu") or {}
        ).items():
            peak_process_memory[str(key)] = max(
                peak_process_memory.get(str(key), 0),
                int(value),
            )
        for key, value in (
            gpu.get("whole_gpu_memory_mib_by_physical_gpu") or {}
        ).items():
            peak_whole_memory[str(key)] = max(
                peak_whole_memory.get(str(key), 0),
                int(value),
            )
        for key, value in (
            gpu.get("utilization_percent_by_physical_gpu") or {}
        ).items():
            utilization_values.setdefault(str(key), []).append(int(value))
    process_cpu_core_equivalents: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        elapsed = float(current.get("timestamp_epoch") or 0.0) - float(
            previous.get("timestamp_epoch") or 0.0
        )
        cpu_delta = float(
            current.get("process_tree_cpu_seconds") or 0.0
        ) - float(previous.get("process_tree_cpu_seconds") or 0.0)
        if elapsed > 0 and cpu_delta >= 0:
            process_cpu_core_equivalents.append(cpu_delta / elapsed)
    gpu_distributions = {
        key: {
            "count": len(values),
            "mean": round(mean(values), 3) if values else None,
            "median": round(median(values), 3) if values else None,
            "p90": (
                round(float(_quantile([float(item) for item in values], 0.90)), 3)
                if values
                else None
            ),
            "max": max(values) if values else None,
            "busy_fraction_ge_80": (
                round(sum(value >= 80 for value in values) / len(values), 6)
                if values
                else None
            ),
        }
        for key, values in utilization_values.items()
    }
    return {
        "schema_version": "ifv-training-resource-summary-v1",
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "exit_code": exit_code,
        "sample_count": len(rows),
        "selected_physical_gpu_ids": selected_gpu_ids,
        "process_tree_peak_rss_mib": (
            max(
                (float(row.get("process_tree_rss_mib") or 0.0) for row in rows),
                default=0.0,
            )
        ),
        "process_tree_peak_process_count": max(
            (int(row.get("process_count") or 0) for row in rows),
            default=0,
        ),
        "process_tree_cpu_core_equivalents": {
            "count": len(process_cpu_core_equivalents),
            "mean": (
                round(mean(process_cpu_core_equivalents), 6)
                if process_cpu_core_equivalents
                else None
            ),
            "median": (
                round(median(process_cpu_core_equivalents), 6)
                if process_cpu_core_equivalents
                else None
            ),
            "p90": (
                round(
                    float(_quantile(process_cpu_core_equivalents, 0.90)),
                    6,
                )
                if process_cpu_core_equivalents
                else None
            ),
            "max": (
                round(max(process_cpu_core_equivalents), 6)
                if process_cpu_core_equivalents
                else None
            ),
        },
        "gpu_peak_process_memory_mib_by_physical_gpu": peak_process_memory,
        "gpu_peak_whole_memory_mib_by_physical_gpu": peak_whole_memory,
        "gpu_mean_utilization_percent_by_physical_gpu": {
            key: value["mean"] for key, value in gpu_distributions.items()
        },
        "gpu_utilization_percent_by_physical_gpu": gpu_distributions,
    }


def run_with_resource_monitor(
    *,
    command: list[str],
    summary_output: Path,
    samples_output: Path,
    selected_gpu_ids: list[int],
    sample_interval: float,
) -> int:
    if not command:
        raise ValueError("resource monitor command is empty")
    if sample_interval <= 0:
        raise ValueError("sample interval must be positive")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(command)
    samples: list[dict[str, Any]] = []
    with samples_output.open("w", encoding="utf-8") as handle:
        while True:
            sample = sample_process_tree(process.pid, selected_gpu_ids)
            samples.append(sample)
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()
            exit_code = process.poll()
            if exit_code is not None:
                break
            time.sleep(sample_interval)
    finished_at = datetime.now(timezone.utc).isoformat()
    summary = summarize_resource_samples(
        samples,
        command=command,
        exit_code=int(exit_code),
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=time.monotonic() - started,
        selected_gpu_ids=selected_gpu_ids,
    )
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return int(exit_code)
