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
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-on-terminal", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive")

    while True:
        snapshot = build_sft_watchdog_snapshot(
            train_log=args.train_log,
            checkpoint_root=args.checkpoint_root,
            resource_samples=args.resource_samples,
            behavior_metrics=args.behavior_metrics,
            output=args.output,
            stale_seconds=args.stale_seconds,
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
