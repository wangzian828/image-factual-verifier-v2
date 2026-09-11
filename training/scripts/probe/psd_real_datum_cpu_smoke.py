"""Two real admitted PSD datums: 9B CPU update, full-state resume and next-round initialization.

Diagnostic only: not a production optimizer batch, full PSD round or GPU capacity test.
No target truncation, generated test image or replacement teacher distribution.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("datums", "datum-manifest", "serving-profile", "checkpoint-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import get_peft_model, LoraConfig, PeftModel
    from ifv_training.io import load_json, load_jsonl, write_json, sha256_file
    from ifv_training.psd_preflight import verify_psd_training_input
    from ifv_training.psd_initialization import verify_initialization
    from ifv_training.psd_hf_teacher import enable_cpu_reference_kernels
    from ifv_training.psd_ms_swift import install_ms_swift_psd_plugin, sparse_topk_cross_entropy
    from swift.template.register import TEMPLATE_MAPPING
    started = time.monotonic()
    profile = load_json(args.serving_profile)
    base = profile.get("engine_model_path") or profile["model_path"]
    adapter = profile.get("adapter_path")
    initialization = verify_initialization(datum_manifest_path=args.datum_manifest,
        serving_profile_path=args.serving_profile, checkpoint_manifest_path=args.checkpoint_manifest,
        model_path=base, adapter_path=adapter)
    gate = verify_psd_training_input(datums_path=args.datums, manifest_path=args.datum_manifest,
        expected_topk=20, max_context=131072)
    if not gate["passed"]:
        raise ValueError("PSD real-datum input gate failed")
    selected = []
    rows = load_jsonl(args.datums)
    for kind in ("repair", "preserve"):
        choices = sorted((row for row in rows if row["kind"] == kind), key=lambda row: (len(row["input_ids"]), row["target_id"]))
        if not choices:
            raise ValueError("real canary requires both admitted source kinds")
        selected.append(choices[0])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "selection.json", {"datums_sha256": sha256_file(args.datums),
        "selection_rule": "shortest complete datum of each source kind; no truncation",
        "targets": [{"target_id": row["target_id"], "kind": row["kind"], "tokens": len(row["input_ids"])} for row in selected]})
    torch.set_num_threads(16)
    torch.manual_seed(0)
    enable_cpu_reference_kernels()
    install_ms_swift_psd_plugin()
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    def load_base():
        return AutoModelForImageTextToText.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa")
    model = load_base()
    model = (PeftModel.from_pretrained(model, adapter, is_trainable=True) if adapter else
        get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0,
                                      target_modules="all-linear", task_type="CAUSAL_LM")))
    def configure(model):
        model.train()
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        return optimizer, scheduler
    def fingerprint(model):
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            if value.requires_grad:
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    def step(model, optimizer, scheduler):
        losses, visual_grad = [], 0.0
        optimizer.zero_grad(set_to_none=True)
        for datum in selected:
            write_json(args.output_dir / "progress.json", {"phase": "optimizer_diagnostic",
                "target_id": datum["target_id"], "kind": datum["kind"],
                "sequence_tokens": len(datum["input_ids"]), "seconds": time.monotonic() - started})
            template = TEMPLATE_MAPPING["ifv_psd_topk"].template_cls.__new__(TEMPLATE_MAPPING["ifv_psd_topk"].template_cls)
            template.processor, template.padding_free, template.sequence_parallel_size = processor, False, 1
            template._get_get_rope_index = lambda: model.get_base_model().model.get_rope_index
            batch = template.data_collator([template.encode(datum)])
            tokens, weights = batch.pop("psd_target_tokens"), batch.pop("psd_weights")
            batch.pop("labels")
            positions = torch.where(weights.sum(-1)[0] > 0)[0]
            output = model(**batch, use_cache=False, logits_to_keep=positions)
            loss = sparse_topk_cross_entropy(output, psd_target_tokens=tokens[:, positions],
                                            psd_weights=weights[:, positions])
            if not torch.isfinite(loss):
                raise ValueError("non-finite real PSD loss")
            (loss / len(selected)).backward()
            losses.append(float(loss.detach()))
            del output, loss, batch, tokens, weights
        visual_grad = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                          if "visual" in name and p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return {"losses": losses, "visual_gradient_l1": visual_grad, "gradient_norm": float(norm)}
    optimizer, scheduler = configure(model)
    before = fingerprint(model)
    first = step(model, optimizer, scheduler)
    write_json(args.output_dir / "first-step.json", first)
    updated = fingerprint(model)
    if before == updated or first["visual_gradient_l1"] <= 0:
        raise ValueError("real PSD step did not update image-conditioned policy")
    checkpoint = args.output_dir / "checkpoint-1"
    model.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(torch.get_rng_state(), checkpoint / "rng_state.pth")
    write_json(checkpoint / "trainer_state.json", {"global_step": 1, "scope": "diagnostic_two_datum_batch"})
    second = step(model, optimizer, scheduler)
    expected = fingerprint(model)
    del model, optimizer, scheduler
    gc.collect()
    restored = PeftModel.from_pretrained(load_base(), checkpoint, is_trainable=True)
    optimizer, scheduler = configure(restored)
    # This is next-round initialization: previous weights, fresh optimizer.
    next_round_initialization_passed = fingerprint(restored) == updated and not optimizer.state
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", weights_only=True))
    torch.set_rng_state(torch.load(checkpoint / "rng_state.pth", weights_only=True))
    resumed = step(restored, optimizer, scheduler)
    resume_passed = fingerprint(restored) == expected and resumed["losses"] == second["losses"]
    result = {"passed": resume_passed and next_round_initialization_passed,
        "scope": "real_admitted_repair_and_preservation_9b_cpu_optimizer_diagnostic",
        "production_round_completed": False, "gpu_capacity_tested": False,
        "initialization": initialization, "first_step": first, "second_step": second, "resumed_step": resumed,
        "full_state_resume_exact": resume_passed, "next_round_weights_and_fresh_optimizer": next_round_initialization_passed,
        "seconds": time.monotonic() - started}
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("PSD real optimizer/resume acceptance failed")


if __name__ == "__main__":
    main()
