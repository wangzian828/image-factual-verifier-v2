"""Real Qwen3.5 processor/model image-conditioning and PSD gradient smoke.

Default uses small random weights with the deployed architecture and tokenizer.
It is an implementation test, not a trained-9B quality or H20 capacity claim.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint", action="store_true")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from ifv_training.psd_media import bind_media, load_media
    from ifv_training.psd_hf_teacher import enable_cpu_reference_kernels, FrozenTeacher
    from ifv_training.psd_topk import normalized_topk
    from ifv_training.psd_datums import build_sparse_topk_datum
    from ifv_training.psd_ms_swift import sparse_topk_cross_entropy, install_ms_swift_psd_plugin
    from swift.template.register import TEMPLATE_MAPPING

    torch.set_num_threads(16 if args.full_checkpoint else 4)
    enable_cpu_reference_kernels()
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 4096}
    image = Image.new("RGB", (64, 64), "red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}
    grid = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
    count = int(grid.prod()) // processor.image_processor.merge_size ** 2
    visual = [248053] + [248056] * count + [248054]
    prompt = [1] + visual + [2] + visual + [3]
    media = bind_media({"content": [block, block]}, processor=processor,
                       output_dir=args.output_dir / "media", prompt_ids=prompt, processor_id=args.model)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text = config.text_config
    for key, value in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=["linear_attention", "full_attention"]).items():
        setattr(text, key, value)
    text.rope_parameters.update(partial_rotary_factor=1.0, mrope_section=[2, 3, 3])
    for key, value in dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64).items():
        setattr(config.vision_config, key, value)
    if args.full_checkpoint:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        model = AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa").float()
    model.eval()
    completion = [20, 21]
    teacher = FrozenTeacher.__new__(FrozenTeacher)
    teacher.model = model
    teacher.device = "cpu"
    teacher_prompt = prompt + [19]
    response = teacher.score({"teacher_prompt_ids": teacher_prompt, "completion_ids": completion,
                              "target": {"psd_media": media}}, 20)
    distributions = [normalized_topk(entries, topk=20) for entries in
                     response["choices"][0]["prompt_logprobs"][len(teacher_prompt):]]
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0,
                                            target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()
    datum = build_sparse_topk_datum({"target_status": "complete", "target_id": "smoke", "kind": "repair",
        "student_prompt_ids": prompt, "completion_ids": completion,
        "teacher_topk_by_position": distributions, "psd_media": media}, topk=20)
    install_ms_swift_psd_plugin()
    cls = TEMPLATE_MAPPING["ifv_psd_topk"].template_cls
    template = cls.__new__(cls)
    template.processor = processor
    template.padding_free = False
    template.sequence_parallel_size = 1
    # Exercise the actual inherited Qwen multimodal position implementation.
    template._get_get_rope_index = lambda: model.get_base_model().model.get_rope_index
    encoded = template.encode(datum)
    batch = template.data_collator([encoded])
    tokens, weights = batch.pop("psd_target_tokens"), batch.pop("psd_weights")
    batch.pop("labels")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-5)
    output = model(**batch, use_cache=False)
    loss = sparse_topk_cross_entropy(output, psd_target_tokens=tokens, psd_weights=weights)
    loss.backward()
    visual_grad = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                      if "visual" in name and p.grad is not None)
    if not torch.isfinite(loss) or visual_grad <= 0:
        raise RuntimeError("multimodal PSD produced invalid loss or no visual gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    checkpoint = args.output_dir / "checkpoint"
    model.save_pretrained(checkpoint)
    torch.save({"optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, checkpoint / "training_state.pt")
    restored = torch.load(checkpoint / "training_state.pt", weights_only=True)
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"])
    model.eval()
    with torch.no_grad():
        original = model(**batch, use_cache=False).logits
        altered = dict(batch, pixel_values=torch.zeros_like(batch["pixel_values"]))
        difference = float((original - model(**altered, use_cache=False).logits).abs().max())
        model.load_adapter(str(checkpoint), adapter_name="restored", is_trainable=True)
        model.set_adapter("restored")
        reloaded = model(**batch, use_cache=False).logits
        reload_delta = float((original - reloaded).abs().max())
    if reload_delta != 0:
        raise RuntimeError(f"adapter reload changed model outputs: {reload_delta}")
    if difference <= 0:
        raise RuntimeError("model output is independent of images")
    result = {"passed": True, "scope": "pretrained_9b_cpu_lora" if args.full_checkpoint else "tiny_random_qwen35_real_processor_and_plugin",
              "images": 2, "loss": float(loss.detach()), "visual_gradient_l1": visual_grad,
              "pixel_change_logit_delta": difference, "position_shape": list(batch["position_ids"].shape),
              "teacher_topk": 20, "optimizer_state_reloaded": bool(optimizer.state),
              "adapter_reload_logit_delta": reload_delta}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
