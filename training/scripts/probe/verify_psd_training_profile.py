#!/usr/bin/env python3
"""Fail-closed profile gate for the four/eight-GPU IFV PSD recipe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("memory_probe", "production"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--sequence-parallel-size", type=int, required=True)
    parser.add_argument("--padding-free", type=_bool, required=True)
    parser.add_argument("--use-logits-to-keep", type=_bool, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--loss-chunk-tokens", type=int, required=True)
    parser.add_argument("--tuner-type", required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--lora-dropout", type=float, required=True)
    parser.add_argument("--target-modules", required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-12)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-reduction", default="sum")
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    if args.lr_scheduler_type != "constant":
        errors.append("published_scheduler_must_be_constant")
    if args.loss_reduction != "sum":
        errors.append("published_backward_reduction_must_be_sum")
    for name, expected in (("adam_beta1", 0.9), ("adam_beta2", 0.95),
                           ("adam_epsilon", 1e-12), ("weight_decay", 0.0)):
        if not math.isclose(getattr(args, name), expected, rel_tol=1e-12, abs_tol=0.0):
            errors.append("published_" + name + "_mismatch")
    if args.world_size not in {4, 8}:
        errors.append("world_size_must_be_4_or_8")
    if (
        args.sequence_parallel_size < 1
        or args.sequence_parallel_size > args.world_size
        or args.world_size % args.sequence_parallel_size != 0
    ):
        errors.append("sequence_parallel_size_must_divide_world_size")
    if not args.padding_free:
        errors.append("padding_free_required")
    if args.max_context != 131_072:
        errors.append("max_context_must_be_131072")
    if args.topk != 20:
        errors.append("topk_must_be_20")
    if args.loss_chunk_tokens < 1:
        errors.append("loss_chunk_tokens_invalid")
    if args.tuner_type != "lora":
        errors.append("tuner_type_must_be_lora")
    if args.lora_rank != 32:
        errors.append("lora_rank_must_be_32")
    if args.lora_alpha != 32:
        errors.append("lora_alpha_must_be_32")
    if not math.isclose(args.lora_dropout, 0.0, abs_tol=1e-12):
        errors.append("lora_dropout_must_be_0")
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if target_modules != ["all-linear"]:
        errors.append("target_modules_must_be_all_linear")
    if args.train_batch_size != 1:
        errors.append("one_datum_per_device_required")
    if args.gradient_accumulation_steps < 1:
        errors.append("gradient_accumulation_steps_invalid")

    data_parallel_size = (
        args.world_size // args.sequence_parallel_size
        if args.sequence_parallel_size > 0
        and args.world_size % args.sequence_parallel_size == 0
        else 0
    )
    if args.sequence_parallel_size == 1 and not args.use_logits_to_keep:
        errors.append("data_parallel_training_requires_logits_to_keep")
    if args.sequence_parallel_size > 1 and args.use_logits_to_keep:
        errors.append("sequence_parallel_training_forbids_logits_to_keep")
    unique_targets_per_step = (
        args.train_batch_size
        * data_parallel_size
        * args.gradient_accumulation_steps
    )
    if args.mode == "production":
        if args.max_steps is not None:
            errors.append("production_max_steps_must_be_unset")
        if args.num_train_epochs is None or not math.isclose(
            args.num_train_epochs,
            5.0,
            abs_tol=1e-12,
        ):
            errors.append("production_epochs_must_be_5")
        if not math.isclose(args.learning_rate, 4e-5, rel_tol=1e-12):
            errors.append("production_learning_rate_must_be_4e-5")
        if unique_targets_per_step != 32:
            errors.append("production_unique_targets_per_step_must_be_32")
    else:
        if args.max_steps != 1:
            errors.append("memory_probe_max_steps_must_be_1")
        if args.num_train_epochs is not None:
            errors.append("memory_probe_epochs_must_be_unset")

    result = {
        "schema_version": "ifv-psd-training-profile-gate-v2",
        "passed": not errors,
        "mode": args.mode,
        "errors": errors,
        "parallelism": {
            "world_size": args.world_size,
            "sequence_parallel_size": args.sequence_parallel_size,
            "data_parallel_size": data_parallel_size,
            "use_logits_to_keep": args.use_logits_to_keep,
        },
        "optimization": {
            "unique_targets_per_step": unique_targets_per_step,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "loss_reduction": args.loss_reduction,
            "lr_scheduler_type": args.lr_scheduler_type,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
        },
        "context": {
            "max_tokens": args.max_context,
            "topk": args.topk,
            "loss_chunk_tokens": args.loss_chunk_tokens,
            "padding_free": args.padding_free,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
