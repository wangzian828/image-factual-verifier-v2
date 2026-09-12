"""Isolated CPU smoke through the real Swift SFT entry point and Qwen3.5.

Uses random tiny weights, real tokenizer, compact PSD datums, LoRA and full
optimizer checkpoints. No network, provider calls, production weights or GPU.
This is integration acceptance, not 9B/multimodal quality acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(root / "hf-cache")
    os.environ["MODELSCOPE_CACHE"] = str(root / "modelscope-cache")
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from ifv_training.psd_hf_teacher import enable_cpu_reference_kernels
    from ifv_training.psd_datums import build_sparse_topk_datum, compact_datum
    enable_cpu_reference_kernels()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    text = config.text_config
    for key, value in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           layer_types=["full_attention"], use_cache=False).items():
        setattr(text, key, value)
    text.rope_parameters = dict(rope_type="default", rope_theta=10000.0,
        partial_rotary_factor=1.0, mrope_interleaved=True, mrope_section=[2, 3, 3])
    for key, value in dict(depth=1, hidden_size=32, intermediate_size=64,
                           num_heads=2, out_hidden_size=32).items():
        setattr(config.vision_config, key, value)
    model_dir = root / "tiny-model"
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa")
    parameter_count = sum(p.numel() for p in model.parameters())
    model.save_pretrained(model_dir)
    AutoProcessor.from_pretrained(args.base_model, local_files_only=True).save_pretrained(model_dir)
    del model
    rows = []
    for i in range(3):
        target = dict(target_id=f"tiny-{i}", kind="repair" if i == 0 else "preserve",
            target_status="complete", student_prompt_ids=[5, 6, 7, 8],
            completion_ids=[100 + i, 101 + i], row_weight=1.0,
            teacher_topk_by_position=[[[token, 0.05] for token in range(100, 120)]] * 2)
        rows.append(compact_datum(build_sparse_topk_datum(target, topk=20)))
    dataset = root / "datums.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    from swift.pipelines import sft_main
    output = root / "train"
    arguments = dict(model=str(model_dir), model_type="qwen3_5", dataset=str(dataset),
        template="ifv_psd_topk", loss_type="ifv_psd_topk",
        external_plugins=str(Path(__file__).resolve().parents[2] / "plugins/ifv_psd_topk_plugin.py"),
        split_dataset_ratio=0, strict=True, remove_unused_columns=False,
        padding_free=False, sequence_parallel_size=1, max_length=131072,
        use_cpu=True, torch_dtype="float32", bf16=False, fp16=False,
        attn_impl="sdpa", use_liger_kernel=False, use_logits_to_keep=False,
        tuner_type="lora", lora_rank=32, lora_alpha=32, target_modules="all-linear",
        freeze_llm=False, freeze_vit=False, freeze_aligner=False,
        gradient_checkpointing=False, num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=4e-5, lr_scheduler_type="constant", warmup_ratio=0,
        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.95,
        adam_epsilon=1e-12, weight_decay=0, max_grad_norm=1, seed=0,
        save_strategy="steps", save_steps=1, save_total_limit=2, save_only_model=False,
        add_version=False, output_dir=str(output), report_to="none",
        dataset_num_proc=1, dataloader_num_workers=0, load_from_cache_file=False,
        logging_steps=1)
    argv = [part for key, value in arguments.items() for part in
            ("--" + key, str(value).lower() if isinstance(value, bool) else str(value))]
    (root / "command.json").write_text(json.dumps(argv, indent=2), encoding="utf-8")
    sft_main(argv)
    checkpoint = output / "checkpoint-2"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    losses = [row["loss"] for row in state["log_history"] if "loss" in row]
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    checks = dict(two_optimizer_steps=state["global_step"] == 2,
        one_epoch=state["epoch"] == 1.0, finite_positive_losses=bool(losses) and
        all(math.isfinite(loss) and loss > 0 for loss in losses),
        full_state=all((checkpoint / name).is_file() for name in required))
    report = dict(schema_version="ifv-psd-training-entrypoint-smoke-v1",
        passed=all(checks.values()), checks=checks, loss=losses,
        random_model_parameters=parameter_count, compact_datums=3,
        actual_entrypoint="swift.pipelines.sft_main", gpu_or_quality_acceptance=False)
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PSD real training entrypoint smoke failed")


if __name__ == "__main__":
    main()
