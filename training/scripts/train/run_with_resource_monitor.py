#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ifv_training.resource_monitor import run_with_resource_monitor


def _gpu_ids(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    try:
        result = [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be a comma-separated integer list"
        ) from exc
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU ids must not contain duplicates")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument(
        "--gpu-ids",
        type=_gpu_ids,
        default=_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(
        run_with_resource_monitor(
            command=command,
            summary_output=args.summary_output,
            samples_output=args.samples_output,
            selected_gpu_ids=args.gpu_ids,
            sample_interval=args.sample_interval,
        )
    )


if __name__ == "__main__":
    main()
