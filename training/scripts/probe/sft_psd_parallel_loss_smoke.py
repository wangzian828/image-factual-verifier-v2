"""Read-only CPU numerical audit of installed Swift SFT and PSD SP scaling.

Run with torchrun --nproc_per_node 4, CUDA_VISIBLE_DEVICES='', PYTHONPATH=training.
Uses native GatherLoss and actual Swift compute_loss; tiny linear model, no API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from swift.sequence_parallel import GatherLoss, sequence_parallel as sp
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from ifv_training.psd_ms_swift import (
    install_ms_swift_psd_plugin, sequence_parallel_sparse_topk_cross_entropy,
)


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    sp.world_size = sp.sp_world_size = world
    sp.rp_world_size = sp.dp_world_size = 1
    sp.device_mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("data", "sequence"))
    sp.tokenizer = SimpleNamespace(pad_token_id=0)
    length = world * 4
    positions = torch.arange(length).unsqueeze(0)
    sp.extra_kwargs["text_position_ids"] = positions
    features = torch.linspace(-1, 1, length * 5).reshape(1, length, 5)
    initial = torch.linspace(-0.4, 0.3, 5 * 17).reshape(5, 17)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(initial.clone())
            self.model_info = SimpleNamespace(is_moe_model=False)

        def forward(self, features):
            return SimpleNamespace(logits=features @ self.weight)

    class Template:
        sequence_parallel_size = world
        padding_free = True

        @staticmethod
        def compute_sft_loss(model, inputs, **kwargs):
            return model(**inputs)

    accelerator = SimpleNamespace(num_processes=world, gradient_accumulation_steps=1,
                                  unwrap_model=lambda model: model)
    options = SimpleNamespace(use_liger_kernel=False, enable_dft_loss=False,
        enable_channel_loss=False, router_aux_loss_coef=None, tuner_backend="unsloth",
        average_tokens_across_devices=True)
    local_features = sp.split(features, dim=1, position_ids=positions)
    labels = torch.full((1, length), -100)
    labels[0, -3:] = torch.tensor([3, 7, 9])  # other ranks have no supervised token
    # prepare_inputs rolls labels BEFORE SP splitting; mirror that boundary.
    shifted = labels.roll(-1, -1)
    local_labels = sp.split(shifted, dim=1, position_ids=positions)
    checks = {}
    for mode in ("sft", "psd"):
        model, reference = Model(), Model()
        trainer = SimpleNamespace(template=Template(), model=model, accelerator=accelerator,
            args=options, label_smoother=None, model_accepts_loss_kwargs=True,
            custom_metrics={"train": {}, "eval": {}})
        if mode == "psd":
            install_ms_swift_psd_plugin()
        total_loss = torch.tensor(0.)
        expected_loss = torch.tensor(0.)
        for micro in range(3):
            batch = {"features": local_features, "labels": local_labels.clone()}
            if mode == "sft":
                batch["compute_loss_func"] = None
                denominator = int((labels != -100).sum()) * world * 3
                expected = torch.nn.functional.cross_entropy(reference(features).logits.flatten(0, 1),
                    shifted.flatten(), reduction="sum") / int((labels != -100).sum()) / 3
            else:
                targets = torch.zeros(1, length, 2, dtype=torch.long)
                weights = torch.zeros(1, length, 2)
                targets[0, -3:-1] = torch.tensor([[3, 7], [9, 11]])
                weights[0, -3:-1] = torch.tensor([0.75, 0.25]) * (micro + 1)
                batch.update(psd_target_tokens=targets, psd_weights=weights,
                    labels=torch.full_like(local_labels, -100))
                batch["compute_loss_func"] = lambda outputs, labels, **kwargs: (
                    sequence_parallel_sparse_topk_cross_entropy(outputs,
                        psd_target_tokens=kwargs["psd_target_tokens"], psd_weights=kwargs["psd_weights"],
                        sequence_parallel_instance=sp, gather_loss=GatherLoss))
                # The old bridge incorrectly multiplied this zero count by world.
                denominator = 0
                selected = reference(features).logits.log_softmax(-1).gather(-1, targets)
                expected = -(selected * weights).sum()
            loss = Seq2SeqTrainer.compute_loss(trainer, model, batch, num_items_in_batch=denominator)
            loss.backward()
            expected.backward()
            total_loss += loss.detach()
            expected_loss += expected.detach()
        dist.all_reduce(model.weight.grad)
        model.weight.grad /= world  # actual DDP/FSDP parameter gradient reduction
        loss_error = float((total_loss - expected_loss).abs())
        grad_error = float((model.weight.grad - reference.weight.grad).abs().max())
        checks[mode] = {"loss": float(total_loss), "reference_loss": float(expected_loss),
            "absolute_loss_error": loss_error, "max_gradient_error": grad_error,
            "passed": loss_error < 1e-5 and grad_error < 1e-5}
    passed = torch.tensor(int(all(row["passed"] for row in checks.values())))
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if dist.get_rank() == 0:
        report = {"passed": bool(passed), "world_size": world, "microbatches": 3,
            "checks": checks, "native_swift_gather_and_compute_loss": True,
            "gpu_or_real_model_acceptance": False}
        output = Path(os.environ["IFV_PARALLEL_LOSS_AUDIT_OUTPUT"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()
    if not passed:
        raise SystemExit("parallel loss/gradient mismatch")


if __name__ == "__main__":
    main()
