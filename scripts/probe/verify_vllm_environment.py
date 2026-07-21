from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED = {
    "vllm": "0.11.2",
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "torchaudio": "2.9.0",
    "transformers": "4.57.6",
    "tokenizers": "0.22.1",
    "huggingface-hub": "0.36.0",
    "xformers": "0.0.33.post1",
    "flashinfer-python": "0.5.2",
    "xgrammar": "0.1.25",
    "llguidance": "1.3.0",
    "compressed-tensors": "0.12.2",
}


def _installed_versions() -> dict[str, str]:
    actual: dict[str, str] = {}
    errors: list[str] = []
    for package, expected in EXPECTED.items():
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}")
            continue
        actual[package] = value
        if value != expected:
            errors.append(f"{package}: expected {expected}, got {value}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return actual


def verify(model: Path) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, got {sys.version}")
    if not model.is_dir():
        raise RuntimeError(f"model directory does not exist: {model}")

    versions = _installed_versions()

    import torch
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.reasoning import ReasoningParserManager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the locked serving environment")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected PyTorch CUDA 12.8 build, got {torch.version.cuda}")

    config = AutoConfig.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    architectures = list(getattr(config, "architectures", None) or [])
    if "Qwen3VLForConditionalGeneration" not in architectures:
        raise RuntimeError(f"unexpected model architectures: {architectures}")
    if "Qwen3VLForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        raise RuntimeError("vLLM does not register Qwen3-VL")
    if "qwen3" not in ReasoningParserManager.list_registered():
        raise RuntimeError("vLLM does not register the qwen3 reasoning parser")
    if "qwen3_xml" not in ToolParserManager.list_registered():
        raise RuntimeError("vLLM does not register the qwen3_xml tool parser")

    devices = [
        {
            "logical_index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]
    return {
        "schema_version": "ifv-vllm-qwen3vl-environment-verification-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "devices": devices,
        "model": {
            "path": str(model.resolve()),
            "architectures": architectures,
            "model_type": getattr(config, "model_type", ""),
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "model_max_length": getattr(tokenizer, "model_max_length", None),
        },
        "protocol": {
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_xml",
            "structured_output_backend": "xgrammar",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.model)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
