"""Merge a validated PEFT adapter into a standalone Qwen checkpoint.

This is the serving fallback for multimodal adapters that the installed vLLM
cannot apply dynamically.  The source adapter and base model stay untouched;
the output is written atomically and carries a content-addressed export record.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(base_model: Path, adapter: Path) -> dict[str, Any]:
    base_config = base_model / "config.json"
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    for path in (base_config, adapter_config, adapter_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "base_model": str(base_model),
        "base_config_sha256": sha256_file(base_config),
        "adapter": str(adapter),
        "adapter_config_sha256": sha256_file(adapter_config),
        "adapter_weights_sha256": sha256_file(adapter_weights),
    }


def _existing_export(output: Path, source: dict[str, Any]) -> dict[str, Any] | None:
    record_path = output / "merge-export.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("passed") is not True or record.get("source") != source:
            return None
        for name, artifact in record["model_artifacts"].items():
            path = output / name
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                return None
        return record
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def merge_for_serving(*, base_model: Path, adapter: Path, output: Path) -> dict[str, Any]:
    base_model = base_model.expanduser().resolve()
    adapter = adapter.expanduser().resolve()
    output = output.expanduser().resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(base_model)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    source = _source_identity(base_model, adapter)
    if output.exists():
        existing = _existing_export(output, source)
        if existing is not None:
            return existing
        raise FileExistsError(f"existing merged export is not valid for this source: {output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(16)
    started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    adapter_parameters = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    if adapter_parameters <= 0:
        raise ValueError("PEFT did not load any LoRA parameters")
    model = model.merge_and_unload(safe_merge=True, progressbar=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("merged model still exposes LoRA parameters")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count < 9_000_000_000:
        raise ValueError("merged Qwen3.5 model is incomplete")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        AutoProcessor.from_pretrained(
            base_model,
            local_files_only=True,
        ).save_pretrained(temporary)
        del model
        gc.collect()

        weight_files = sorted(temporary.glob("model*.safetensors"))
        if not weight_files:
            raise ValueError("merged export has no safetensors weights")
        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(p for p in temporary.iterdir() if p.is_file()):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        record = {
            "schema_version": "ifv-merged-lora-serving-export-v1",
            "passed": True,
            "source": source,
            "model_path": str(output),
            "dtype": "bfloat16",
            "parameter_count": parameter_count,
            "adapter_parameter_count": adapter_parameters,
            "weight_file_count": len(weight_files),
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "model_artifacts": artifacts,
        }
        (temporary / "merge-export.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = merge_for_serving(
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
