#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ifv_training.watchdog import (
    build_sft_watchdog_snapshot,
    concise_watchdog_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resource-samples", type=Path)
    parser.add_argument("--behavior-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=900.0)
    parser.add_argument("--gpu-memory-target-min-mib", type=int)
    parser.add_argument("--gpu-memory-target-max-mib", type=int)
    parser.add_argument("--gpu-memory-max-imbalance-mib", type=int)
    parser.add_argument("--gpu-utilization-target-min-percent", type=float)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-on-terminal", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive")
    for name in (
        "gpu_memory_target_min_mib",
        "gpu_memory_target_max_mib",
        "gpu_memory_max_imbalance_mib",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.gpu_memory_target_min_mib is not None
        and args.gpu_memory_target_max_mib is not None
        and args.gpu_memory_target_min_mib
        >= args.gpu_memory_target_max_mib
    ):
        parser.error(
            "--gpu-memory-target-min-mib must be below "
            "--gpu-memory-target-max-mib"
        )
    if (
        args.gpu_utilization_target_min_percent is not None
        and not 0 <= args.gpu_utilization_target_min_percent <= 100
    ):
        parser.error("--gpu-utilization-target-min-percent must be in [0, 100]")

    while True:
        snapshot = build_sft_watchdog_snapshot(
            train_log=args.train_log,
            checkpoint_root=args.checkpoint_root,
            resource_samples=args.resource_samples,
            behavior_metrics=args.behavior_metrics,
            output=args.output,
            stale_seconds=args.stale_seconds,
            gpu_memory_target_min_mib=args.gpu_memory_target_min_mib,
            gpu_memory_target_max_mib=args.gpu_memory_target_max_mib,
            gpu_memory_max_imbalance_mib=args.gpu_memory_max_imbalance_mib,
            gpu_utilization_target_min_percent=(
                args.gpu_utilization_target_min_percent
            ),
        )
        print(concise_watchdog_status(snapshot), flush=True)
        if args.once:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            return 1 if snapshot["health"] == "critical" else 0
        if args.stop_on_terminal and snapshot["lifecycle"] in {
            "completed",
            "failed",
        }:
            return 1 if snapshot["health"] == "critical" else 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
