from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"expected world_size={args.expected_world_size}, got {world_size}"
            )

        value = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2
        actual = value.item()
        if actual != expected:
            raise RuntimeError(f"NCCL all_reduce expected {expected}, got {actual}")

        devices = [None] * world_size
        dist.all_gather_object(
            devices,
            {
                "rank": rank,
                "local_rank": local_rank,
                "name": torch.cuda.get_device_name(),
                "uuid": str(torch.cuda.get_device_properties(local_rank).uuid),
            },
        )
        if len({item["uuid"] for item in devices}) != world_size:
            raise RuntimeError(f"tensor-parallel ranks did not use unique GPUs: {devices}")

        if rank == 0:
            result = {
                "schema_version": "ifv-nccl-tensor-parallel-verification-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "passed": True,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "all_reduce_value": actual,
                "nccl_cumem_host_enable": os.environ.get(
                    "NCCL_CUMEM_HOST_ENABLE", ""
                ),
                "devices": devices,
            }
            rendered = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
