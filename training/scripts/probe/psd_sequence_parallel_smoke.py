#!/usr/bin/env python3
"""Distributed CPU smoke for IFV PSD target splitting and global SP loss."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh

from ifv_training.psd_ms_swift import (
    sequence_parallel_sparse_topk_cross_entropy,
    sparse_topk_cross_entropy,
    split_psd_targets_for_sequence_parallel,
)
from swift.sequence_parallel import GatherLoss, sequence_parallel


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("PSD sequence-parallel smoke requires at least two ranks")

    sequence_parallel.world_size = world_size
    sequence_parallel.sp_world_size = world_size
    sequence_parallel.rp_world_size = 1
    sequence_parallel.dp_world_size = 1
    sequence_parallel.device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, world_size),
        mesh_dim_names=("data", "sequence"),
    )
    sequence_parallel.tokenizer = SimpleNamespace(pad_token_id=0)

    sequence_length = world_size * 4
    vocab_size = 17
    topk = 2
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    sequence_parallel.extra_kwargs["text_position_ids"] = position_ids.clone()
    full_targets = torch.zeros((1, sequence_length, topk), dtype=torch.long)
    full_weights = torch.zeros((1, sequence_length, topk), dtype=torch.float32)
    # Keep the target short and on the tail shards. Most SP ranks therefore
    # exercise the valid zero-local-target path.
    for offset, position in enumerate((sequence_length - 2, sequence_length - 1)):
        full_targets[0, position] = torch.tensor([3 + offset, 8 + offset])
        full_weights[0, position] = torch.tensor([0.75, 0.25])

    base = torch.linspace(
        -1.5,
        1.5,
        steps=sequence_length * vocab_size,
        dtype=torch.float32,
    ).reshape(1, sequence_length, vocab_size)
    reference_logits = base.clone().requires_grad_(True)
    reference_loss = sparse_topk_cross_entropy(
        SimpleNamespace(logits=reference_logits),
        psd_target_tokens=full_targets,
        psd_weights=full_weights,
        chunk_size=1,
    )
    reference_loss.backward()
    reference_gradient = reference_logits.grad.detach().clone()

    local_targets, local_weights = split_psd_targets_for_sequence_parallel(
        full_targets,
        full_weights,
        sequence_parallel_instance=sequence_parallel,
    )
    local_logits = sequence_parallel.split(
        base,
        dim=1,
        position_ids=position_ids,
    ).detach().requires_grad_(True)
    distributed_loss = sequence_parallel_sparse_topk_cross_entropy(
        SimpleNamespace(logits=local_logits),
        psd_target_tokens=local_targets,
        psd_weights=local_weights,
        sequence_parallel_instance=sequence_parallel,
        gather_loss=GatherLoss,
    )
    distributed_loss.backward()
    expected_local_gradient = sequence_parallel.split(
        reference_gradient,
        dim=1,
        position_ids=position_ids,
    ) * world_size

    local_passed = bool(
        torch.allclose(distributed_loss.detach(), reference_loss.detach(), atol=1e-6)
        and torch.allclose(
            local_logits.grad,
            expected_local_gradient,
            atol=1e-6,
            rtol=1e-5,
        )
    )
    passed = torch.tensor(int(local_passed), dtype=torch.int64)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        result = {
            "schema_version": "ifv-psd-sequence-parallel-smoke-v1",
            "passed": bool(passed.item()),
            "world_size": world_size,
            "sequence_length": sequence_length,
            "topk": topk,
            "reference_loss": float(reference_loss.detach()),
            "distributed_loss": float(distributed_loss.detach()),
            "gathers_full_vocab_logits": False,
        }
        output = os.getenv("IFV_PSD_SP_SMOKE_OUTPUT", "").strip()
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    dist.barrier()
    dist.destroy_process_group()
    if not passed.item():
        raise SystemExit("PSD sequence-parallel smoke failed")


if __name__ == "__main__":
    main()
