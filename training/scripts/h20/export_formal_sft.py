"""Export inference BF16 weights from a completed FSDP2 full-state checkpoint.

Source optimizer/scheduler/RNG and fp32 master weights remain untouched. Only
one BF16 export is written; no intermediate 35-GiB fp32 merged file is created.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ifv_training.io import sha256_file, write_json
from ifv_training.checkpoints import SERVING_ASSET_FILES, audit_full_parameter_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    checkpoint, base, output = (p.resolve() for p in (args.checkpoint, args.base_model, args.output))
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != state["max_steps"] or state["epoch"] < 1:
        raise ValueError("formal training has not completed")
    for name in ("pytorch_model_fsdp_0/.metadata", "optimizer_0/.metadata", "scheduler.pt"):
        if not (checkpoint / name).is_file():
            raise ValueError("missing full-state checkpoint component: " + name)
    if len(list(checkpoint.glob("rng_state_*.pth"))) != 4:
        raise ValueError("four-rank RNG state is incomplete")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "model"
    model_dir.mkdir()
    started = time.monotonic()
    def progress(stage):
        write_json(output / "progress.json", {"stage": stage,
            "elapsed_seconds": round(time.monotonic() - started, 2)})
        print(stage, flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import _load_state_dict, _EmptyStateDictLoadPlanner
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    progress("loading_fsdp_master_weights_on_cpu")
    tensors = {}
    _load_state_dict(tensors, storage_reader=dcp.FileSystemReader(checkpoint / "pytorch_model_fsdp_0"),
        planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    if len(tensors) == 1 and isinstance(next(iter(tensors.values())), dict):
        tensors = next(iter(tensors.values()))
    progress("converting_inference_copy_to_bf16")
    for name in tensors:
        tensor = tensors[name]
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite checkpoint tensor: " + name)
        tensors[name] = tensor.to(torch.bfloat16).contiguous() if tensor.is_floating_point() else tensor.contiguous()
    count = sum(t.numel() for t in tensors.values())
    if count < 9_000_000_000:
        raise ValueError("full 9B model weights are missing")
    progress("saving_sharded_bf16_export")
    shards = split_torch_state_dict_into_shards(tensors, filename_pattern="model{suffix}.safetensors", max_shard_size="5GB")
    for filename, names in shards.filename_to_tensors.items():
        save_file({name: tensors.pop(name) for name in names}, model_dir / filename, metadata={"format": "pt"})
    if shards.is_sharded:
        write_json(model_dir / "model.safetensors.index.json",
            {"metadata": shards.metadata, "weight_map": shards.tensor_to_filename})
    for name in SERVING_ASSET_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, model_dir / name)
    shutil.copy2(checkpoint.parent / "args.json", model_dir / "args.json")
    progress("auditing_full_parameter_update_and_resume_state")
    audit = audit_full_parameter_checkpoint(checkpoint_dir=model_dir, base_model_dir=base,
        state_checkpoint_dir=checkpoint, output_path=output / "full-parameter-audit.json")
    if not audit["passed"]:
        raise ValueError("full-parameter export audit failed")
    record = {"schema_version": "ifv-h20-formal-sft-export-v1", "passed": True,
        "source_checkpoint": str(checkpoint), "base_model": str(base), "model_path": str(model_dir),
        "global_step": state["global_step"], "epoch": state["epoch"], "dtype": "bfloat16",
        "parameter_elements": count, "source_full_state_preserved": True,
        "source_metadata_sha256": {name: sha256_file(checkpoint / name) for name in
            ("trainer_state.json", "pytorch_model_fsdp_0/.metadata", "optimizer_0/.metadata", "scheduler.pt")},
        "model_artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in model_dir.iterdir() if p.is_file()}}
    write_json(output / "export.json", record)
    progress("ready_for_inference")


if __name__ == "__main__":
    main()
