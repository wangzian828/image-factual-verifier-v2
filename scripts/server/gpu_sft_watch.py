#!/usr/bin/env python3
"""Watch contiguous GPU windows and notify Feishu when SFT capacity is free."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


GPU_COUNT = 8
WINDOW_SIZE = 4
MEMORY_LIMIT_MIB = 1024
UTILIZATION_LIMIT = 5
POLL_SECONDS = 30
CONSECUTIVE_CHECKS = 2


def _run_nvidia_smi(query: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[list[str]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append([item.strip() for item in line.split(",")])
    return rows


def gpu_snapshot() -> dict[int, dict[str, Any]]:
    gpu_rows = _run_nvidia_smi("gpu=index,uuid,memory.used,utilization.gpu")
    process_rows = _run_nvidia_smi(
        "compute-apps=gpu_uuid,pid,used_memory"
    )
    processes_by_uuid: dict[str, list[dict[str, str]]] = {}
    for row in process_rows:
        if len(row) < 3:
            continue
        processes_by_uuid.setdefault(row[0], []).append(
            {"pid": row[1], "memory_mib": row[2]}
        )

    snapshot: dict[int, dict[str, Any]] = {}
    for row in gpu_rows:
        if len(row) < 4:
            continue
        index = int(row[0])
        uuid = row[1]
        memory_mib = int(float(row[2]))
        utilization = int(float(row[3]))
        processes = processes_by_uuid.get(uuid, [])
        snapshot[index] = {
            "memory_mib": memory_mib,
            "utilization": utilization,
            "processes": processes,
            "idle": (
                memory_mib < MEMORY_LIMIT_MIB
                and utilization < UTILIZATION_LIMIT
                and not processes
            ),
        }
    return snapshot


def contiguous_idle_windows(snapshot: dict[int, dict[str, Any]]) -> list[list[int]]:
    windows: list[list[int]] = []
    for start in range(GPU_COUNT - WINDOW_SIZE + 1):
        window = list(range(start, start + WINDOW_SIZE))
        if all(snapshot.get(index, {}).get("idle") is True for index in window):
            windows.append(window)
    return windows


def _read_webhook(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"webhook file is empty: {path}")


def notify(webhook: str, windows: list[list[int]], snapshot: dict[int, dict[str, Any]]) -> None:
    window_text = ", ".join(
        f"{window[0]}-{window[-1]}" for window in windows
    )
    payload = {
        "msg_type": "text",
        "content": {
            "text": (
                "GPU-SFT available: contiguous GPUs "
                f"{window_text} are idle for 1 minute. "
                "Please ask me before starting training."
            )
        },
    }
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"Feishu webhook HTTP {response.status}")
    print(
        json.dumps(
            {
                "event": "notified",
                "windows": windows,
                "snapshot": {
                    str(index): snapshot[index] for index in sorted(snapshot)
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"streak": 0, "notified_windows": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"streak": 0, "notified_windows": []}
    return value if isinstance(value, dict) else {"streak": 0, "notified_windows": []}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def watch(webhook_file: Path, state_file: Path, poll_seconds: int) -> int:
    webhook = _read_webhook(webhook_file)
    state = _load_state(state_file)
    while True:
        try:
            snapshot = gpu_snapshot()
            windows = contiguous_idle_windows(snapshot)
            if windows:
                state["streak"] = int(state.get("streak", 0)) + 1
            else:
                state["streak"] = 0
                state["notified_windows"] = []

            window_keys = [",".join(str(index) for index in window) for window in windows]
            notified = set(str(item) for item in state.get("notified_windows", []))
            if state["streak"] >= CONSECUTIVE_CHECKS:
                new_windows = [window for window, key in zip(windows, window_keys) if key not in notified]
                if new_windows:
                    notify(webhook, new_windows, snapshot)
                    notified.update(
                        ",".join(str(index) for index in window)
                        for window in new_windows
                    )
                    state["notified_windows"] = sorted(notified)

            state["last_snapshot"] = snapshot
            state["last_windows"] = windows
            state["last_checked_at"] = int(time.time())
            _write_state(state_file, state)
        except Exception as exc:
            print(f"gpu_sft_watch error: {exc}", file=sys.stderr, flush=True)
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webhook-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    return watch(args.webhook_file, args.state_file, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
