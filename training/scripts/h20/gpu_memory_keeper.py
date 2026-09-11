#!/usr/bin/env python3
"""Keep one idle CUDA device above provider memory and utilization floors."""

from __future__ import annotations

import argparse
import signal
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--gib", type=float, default=12.0)
    parser.add_argument("--duty-cycle", type=float, default=0.25)
    parser.add_argument("--matrix-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gib <= 0:
        raise SystemExit("--gib must be positive")
    if not 0 < args.duty_cycle <= 1:
        raise SystemExit("--duty-cycle must be in (0, 1]")
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be positive")

    torch.cuda.set_device(args.device)
    elements = int(args.gib * 1024**3 / torch.tensor([], dtype=torch.uint8).element_size())
    allocation = torch.empty(elements, dtype=torch.uint8, device=f"cuda:{args.device}")
    allocation.fill_(1)
    matrix_a = torch.randn(
        (args.matrix_size, args.matrix_size),
        dtype=torch.bfloat16,
        device=f"cuda:{args.device}",
    )
    matrix_b = torch.randn_like(matrix_a)
    matrix_out = torch.empty_like(matrix_a)
    torch.cuda.synchronize(args.device)

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"holding {args.gib:.1f} GiB and {args.duty_cycle:.0%} compute duty "
        f"on CUDA device {args.device}",
        flush=True,
    )
    while not stopping:
        cycle_started = time.monotonic()
        compute_until = cycle_started + args.duty_cycle
        while not stopping and time.monotonic() < compute_until:
            torch.mm(matrix_a, matrix_b, out=matrix_out)
            torch.cuda.synchronize(args.device)
        remaining = 1.0 - (time.monotonic() - cycle_started)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
