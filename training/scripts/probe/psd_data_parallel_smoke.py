#!/usr/bin/env python3
"""Distributed CPU smoke for PSD's global summed objective under DP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ifv_training.psd_ms_swift import (
    install_ms_swift_psd_plugin,
    sparse_topk_cross_entropy,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.model_info = SimpleNamespace(is_moe_model=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        features = torch.tensor(
            [-1.5, -0.5, 0.25, 1.0, 2.0],
            dtype=self.scale.dtype,
            device=self.scale.device,
        )
        logits = self.scale * features.view(1, 1, -1)
        return SimpleNamespace(
            logits=logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        )


class Template:
    sequence_parallel_size = 1
    padding_free = False

    @staticmethod
    def compute_sft_loss(model, inputs, **_):
        return model(**inputs)


class Accelerator:
    gradient_accumulation_steps = 1

    def __init__(self, world_size: int) -> None:
        self.num_processes = world_size

    @staticmethod
    def unwrap_model(model):
        return model.module if isinstance(model, DistributedDataParallel) else model


def loss_function(outputs, labels, **kwargs):
    return sparse_topk_cross_entropy(
        outputs,
        psd_target_tokens=kwargs["psd_target_tokens"],
        psd_weights=kwargs["psd_weights"],
    )


def trainer_facade(world_size: int, model: TinyPolicy) -> SimpleNamespace:
    args = SimpleNamespace(
        average_tokens_across_devices=False,
        enable_channel_loss=False,
        enable_dft_loss=False,
        past_index=-1,
        router_aux_loss_coef=None,
        tuner_backend="unsloth",
        use_liger_kernel=False,
    )
    return SimpleNamespace(
        accelerator=Accelerator(world_size),
        args=args,
        custom_metrics={"train": {}, "eval": {}},
        label_smoother=None,
        model=model,
        model_accepts_loss_kwargs=True,
        template=Template(),
    )


def rank_inputs(rank: int) -> dict[str, torch.Tensor | object]:
    weights = torch.zeros((1, 3, 2), dtype=torch.float32)
    weights[0, 1] = torch.tensor([0.75, 0.25]) * (rank + 1)
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.full((1, 3), -100, dtype=torch.long),
        "psd_target_tokens": torch.tensor([[[0, 0], [3, 4], [0, 0]]]),
        "psd_weights": weights,
        "compute_loss_func": loss_function,
    }


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("PSD data-parallel smoke requires at least two ranks")

    install_ms_swift_psd_plugin()
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    reference_model = TinyPolicy()
    reference_inputs = rank_inputs(rank)
    reference_outputs = reference_model(reference_inputs["input_ids"])
    reference_loss = loss_function(
        reference_outputs,
        reference_inputs["labels"],
        psd_target_tokens=reference_inputs["psd_target_tokens"],
        psd_weights=reference_inputs["psd_weights"],
    )
    expected_gradient = torch.autograd.grad(reference_loss, reference_model.scale)[0]
    dist.all_reduce(expected_gradient, op=dist.ReduceOp.SUM)

    model = TinyPolicy()
    distributed_model = DistributedDataParallel(model)
    facade = trainer_facade(world_size, model)
    loss = Seq2SeqTrainer.compute_loss(
        facade,
        distributed_model,
        rank_inputs(rank),
    )
    loss.backward()

    local_passed = bool(
        torch.allclose(model.scale.grad, expected_gradient, atol=1e-6, rtol=1e-5)
    )
    passed = torch.tensor(int(local_passed), dtype=torch.int64)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    gradients = [torch.zeros_like(model.scale.grad) for _ in range(world_size)]
    dist.all_gather(gradients, model.scale.grad)

    if rank == 0:
        result = {
            "schema_version": "ifv-psd-data-parallel-smoke-v1",
            "passed": bool(passed.item()),
            "world_size": world_size,
            "backward_objective": "global_datum_sum",
            "expected_gradient": float(expected_gradient),
            "rank_gradients": [float(value) for value in gradients],
            "uses_actual_swift_compute_loss_bridge": True,
            "uses_actual_ddp_gradient_reduction": True,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        output = os.getenv("IFV_PSD_DP_SMOKE_OUTPUT", "").strip()
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")

    dist.barrier()
    dist.destroy_process_group()
    if not passed.item():
        raise SystemExit("PSD data-parallel summed objective smoke failed")


if __name__ == "__main__":
    main()
