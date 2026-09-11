#!/usr/bin/env python3
"""Protect rented H20 GPUs from prolonged low-utilization reclamation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_GPU_IDS = (0, 1, 2, 3)
DEFAULT_VLLM_PORTS = (8902, 8903, 8904, 8905)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--utilization-threshold-percent", type=float, default=10.0)
    parser.add_argument("--low-window-seconds", type=int, default=90 * 60)
    parser.add_argument("--free-memory-limit-mib", type=int, default=1024)
    parser.add_argument("--model", default="ifv-qwen3.5-9b")
    parser.add_argument("--pulse-tokens", type=int, default=1024)
    parser.add_argument("--pulse-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--keeper-script",
        type=Path,
        default=Path(__file__).with_name("gpu_memory_keeper.sh"),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if not 0 < args.utilization_threshold_percent <= 100:
        parser.error("--utilization-threshold-percent must be in (0, 100]")
    if args.low_window_seconds < args.poll_seconds:
        parser.error("--low-window-seconds must be at least --poll-seconds")
    if args.free_memory_limit_mib < 0:
        parser.error("--free-memory-limit-mib must be non-negative")
    if args.pulse_tokens < 1:
        parser.error("--pulse-tokens must be positive")
    if args.pulse_timeout_seconds <= 0:
        parser.error("--pulse-timeout-seconds must be positive")
    return args


def gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(float(fields[1])),
                "memory_total_mib": int(float(fields[2])),
                "utilization_percent": float(fields[3]),
            }
        )
    return rows


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "ifv-gpu-utilization-guard-v1", "gpus": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "ifv-gpu-utilization-guard-v1", "gpus": {}}
    if not isinstance(value, dict):
        return {"schema_version": "ifv-gpu-utilization-guard-v1", "gpus": {}}
    value.setdefault("schema_version", "ifv-gpu-utilization-guard-v1")
    value.setdefault("gpus", {})
    return value


def update_state(
    previous: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    now: float,
    threshold_percent: float,
) -> dict[str, Any]:
    previous_gpus = previous.get("gpus")
    if not isinstance(previous_gpus, Mapping):
        previous_gpus = {}
    current: dict[str, Any] = {}
    for sample in samples:
        index = int(sample["index"])
        key = str(index)
        old = previous_gpus.get(key)
        if not isinstance(old, Mapping):
            old = {}
        utilization = float(sample["utilization_percent"])
        below_since: float | None
        if utilization >= threshold_percent:
            below_since = None
        else:
            prior = old.get("below_since")
            below_since = float(prior) if isinstance(prior, (int, float)) else now
        current[key] = {
            **dict(sample),
            "below_since": below_since,
            "low_seconds": round(max(0.0, now - below_since), 3)
            if below_since is not None
            else 0.0,
        }
    return {
        "schema_version": "ifv-gpu-utilization-guard-v1",
        "updated_at": now,
        "gpus": current,
        "last_action": previous.get("last_action"),
    }


def at_risk_gpu_ids(state: Mapping[str, Any], low_window_seconds: float) -> list[int]:
    gpus = state.get("gpus")
    if not isinstance(gpus, Mapping):
        return []
    return sorted(
        int(index)
        for index, row in gpus.items()
        if isinstance(row, Mapping)
        and float(row.get("low_seconds") or 0.0) >= low_window_seconds
    )


def all_gpus_free(
    samples: Sequence[Mapping[str, Any]],
    *,
    memory_limit_mib: int,
    utilization_threshold_percent: float,
) -> bool:
    return bool(samples) and all(
        int(sample["memory_used_mib"]) < memory_limit_mib
        and float(sample["utilization_percent"]) < utilization_threshold_percent
        for sample in samples
    )


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def healthy_vllm_ports(ports: Sequence[int]) -> set[int]:
    return {
        port
        for port in ports
        if _http_ok(f"http://127.0.0.1:{port}/health")
    }


def pulse_vllm(
    *,
    port: int,
    model: str,
    tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "GPU keepalive diagnostic. Produce a long stream of arbitrary "
                    "Chinese characters without tools or early stopping. Nonce: "
                    + uuid.uuid4().hex
                ),
            }
        ],
        "temperature": 1.0,
        "max_tokens": tokens,
        "min_tokens": tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"vLLM keepalive returned HTTP {response.status}")
    return {
        "port": port,
        "status": "success",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def protect(
    args: argparse.Namespace,
    state: dict[str, Any],
    samples: Sequence[Mapping[str, Any]],
    risk_ids: Sequence[int],
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "at": time.time(),
        "risk_gpu_ids": list(risk_ids),
        "status": "dry_run" if args.dry_run else "pending",
    }
    if args.dry_run:
        action["mode"] = "none"
        return action

    port_by_gpu = dict(zip(DEFAULT_GPU_IDS, DEFAULT_VLLM_PORTS))
    healthy_ports = healthy_vllm_ports(DEFAULT_VLLM_PORTS)
    pulse_targets = [
        (gpu, port_by_gpu[gpu])
        for gpu in risk_ids
        if gpu in port_by_gpu and port_by_gpu[gpu] in healthy_ports
    ]
    if pulse_targets:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(pulse_targets)) as executor:
            futures = {
                executor.submit(
                    pulse_vllm,
                    port=port,
                    model=args.model,
                    tokens=args.pulse_tokens,
                    timeout_seconds=args.pulse_timeout_seconds,
                ): gpu
                for gpu, port in pulse_targets
            }
            for future in as_completed(futures):
                gpu = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append({"gpu": gpu, **result})
        action.update({"mode": "vllm_pulse", "status": "completed", "results": results})
        successful = {
            str(result["gpu"])
            for result in results
            if result.get("status") == "success"
        }
        for gpu in successful:
            row = state["gpus"].get(gpu)
            if isinstance(row, dict):
                row["below_since"] = action["at"]
                row["low_seconds"] = 0.0
        return action

    if all_gpus_free(
        samples,
        memory_limit_mib=args.free_memory_limit_mib,
        utilization_threshold_percent=args.utilization_threshold_percent,
    ):
        completed = subprocess.run(
            [str(args.keeper_script), "start"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        action.update(
            {
                "mode": "start_keeper",
                "status": "completed" if completed.returncode == 0 else "error",
                "exit_code": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        return action

    action.update(
        {
            "mode": "deferred",
            "status": "active_process_or_partial_service",
            "healthy_vllm_ports": sorted(healthy_ports),
        }
    )
    return action


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_log(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    while True:
        now = time.time()
        try:
            samples = gpu_snapshot()
            state = update_state(
                load_state(args.state_file),
                samples,
                now=now,
                threshold_percent=args.utilization_threshold_percent,
            )
            risk_ids = at_risk_gpu_ids(state, args.low_window_seconds)
            event: dict[str, Any] = {
                "timestamp": now,
                "event": "sample",
                "risk_gpu_ids": risk_ids,
                "gpus": state["gpus"],
            }
            if risk_ids:
                action = protect(args, state, samples, risk_ids)
                state["last_action"] = action
                event["action"] = action
            write_json_atomic(args.state_file, state)
            append_log(args.log_file, event)
        except Exception as exc:
            append_log(
                args.log_file,
                {
                    "timestamp": now,
                    "event": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
