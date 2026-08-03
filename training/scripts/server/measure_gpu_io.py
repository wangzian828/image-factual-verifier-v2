#!/usr/bin/env python3
"""Measure basic GPU H2D, D2H, and local HBM bandwidth.

This is a lightweight training-node diagnostic for CPU-offload locality issues. It is
not a substitute for full training profiling.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch


def _measure_gbps(operation, bytes_per_iter: int, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        operation()
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    return bytes_per_iter * max(1, iters) / seconds / 1e9


def _parse_gpus(value: str) -> list[int]:
    if value == "visible":
        return list(range(torch.cuda.device_count()))
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


def measure_gpu(gpu: int, *, size_mib: int, warmup: int, iters: int) -> dict[str, Any]:
    torch.cuda.set_device(gpu)
    num_bytes = size_mib * 1024**2
    device = torch.device(f"cuda:{gpu}")

    host_src = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
    host_dst = torch.empty_like(host_src, pin_memory=True)
    gpu_buffer = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    host_src.zero_()
    gpu_buffer.zero_()

    h2d_gbps = _measure_gbps(
        lambda: gpu_buffer.copy_(host_src, non_blocking=True),
        num_bytes,
        warmup,
        iters,
    )
    d2h_gbps = _measure_gbps(
        lambda: host_dst.copy_(gpu_buffer, non_blocking=True),
        num_bytes,
        warmup,
        iters,
    )

    elements = max(1, num_bytes // torch.empty((), dtype=torch.float32).element_size())
    x = torch.empty(elements, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    x.fill_(1.0)
    y.fill_(2.0)
    z.zero_()
    tensor_bytes = x.numel() * x.element_size()

    local_copy_payload_gbps = _measure_gbps(lambda: z.copy_(x), tensor_bytes, warmup, iters)
    stream_add_hbm_gbps = _measure_gbps(
        lambda: torch.add(x, y, out=z),
        3 * tensor_bytes,
        warmup,
        iters,
    )

    return {
        "logical_gpu": gpu,
        "name": torch.cuda.get_device_name(gpu),
        "size_mib": size_mib,
        "h2d_payload_gbps": round(h2d_gbps, 3),
        "d2h_payload_gbps": round(d2h_gbps, 3),
        "local_copy_payload_gbps": round(local_copy_payload_gbps, 3),
        "local_copy_hbm_read_write_gbps": round(2 * local_copy_payload_gbps, 3),
        "stream_add_hbm_gbps": round(stream_add_hbm_gbps, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="visible", help="'visible' or comma-separated logical GPU ids")
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    result = {
        "schema_version": "ifv-gpu-io-diagnostic-v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "measurements": [
            measure_gpu(gpu, size_mib=args.size_mib, warmup=args.warmup, iters=args.iters)
            for gpu in _parse_gpus(args.gpus)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
