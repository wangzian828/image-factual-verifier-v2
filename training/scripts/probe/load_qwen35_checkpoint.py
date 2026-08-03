#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a merged Qwen3.5 checkpoint on CPU as a reload smoke test."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    import torch
    import transformers

    auto_model = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_model is None:
        auto_model = transformers.AutoModelForVision2Seq
    config = transformers.AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = auto_model.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    result = {
        "schema_version": "ifv-qwen35-checkpoint-load-smoke-v1",
        "model_dir": str(model_dir),
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "parameter_dtypes": dtypes,
        "passed": parameter_count > 9_000_000_000
        and "Qwen3_5" in type(model).__name__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
