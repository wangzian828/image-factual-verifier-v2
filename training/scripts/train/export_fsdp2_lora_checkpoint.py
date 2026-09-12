"""Export an ms-swift FSDP2 LoRA checkpoint for PEFT/vLLM serving.

FSDP2 writes the trainable adapter tensors as a distributed-checkpoint (DCP)
state dictionary.  PEFT and vLLM instead expect ``adapter_config.json`` and
``adapter_model.safetensors``.  This conversion preserves the source training
checkpoint and records enough hashes to make a repeated invocation idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lora_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the FSDP model wrapper and reject non-LoRA tensors."""

    wrapped = [name.startswith("model.") for name in state]
    if any(wrapped) and not all(wrapped):
        raise ValueError("mixed wrapped and unwrapped FSDP LoRA key prefixes")
    normalized: dict[str, Any] = {}
    for source_name, tensor in state.items():
        name = source_name.removeprefix("model.") if all(wrapped) else source_name
        if not name.startswith("base_model."):
            raise ValueError(f"unexpected FSDP LoRA key prefix: {source_name}")
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"non-LoRA tensor in FSDP adapter state: {source_name}")
        if name in normalized:
            raise ValueError(f"duplicate normalized LoRA key: {name}")
        normalized[name] = tensor
    if not normalized:
        raise ValueError("FSDP adapter state is empty")
    return normalized


def infer_target_modules(state: Mapping[str, Any]) -> list[str]:
    suffixes = (".lora_A.weight", ".lora_B.weight")
    targets: set[str] = set()
    variants: dict[str, set[str]] = {}
    for name in state:
        suffix = next((value for value in suffixes if name.endswith(value)), None)
        if suffix is None:
            raise ValueError(f"not a normalized LoRA tensor key: {name}")
        module_path = name[: -len(suffix)]
        terminal = module_path.rsplit(".", 1)[-1]
        targets.add(terminal)
        variants.setdefault(module_path, set()).add(suffix)
    incomplete = sorted(path for path, kinds in variants.items() if len(kinds) != 2)
    if incomplete:
        raise ValueError(f"LoRA A/B tensor pairs are incomplete: {incomplete[:8]}")
    return sorted(targets)


def _validate_rank(state: Mapping[str, Any], expected_rank: int) -> None:
    for name, tensor in state.items():
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != 2:
            raise ValueError(f"LoRA tensor is not a matrix: {name} {shape}")
        rank_axis = 0 if name.endswith(".lora_A.weight") else 1
        if shape[rank_axis] != expected_rank:
            raise ValueError(
                f"LoRA rank mismatch for {name}: expected {expected_rank}, got {shape}"
            )


def _existing_export_is_valid(output: Path, source_metadata_sha256: str) -> bool:
    report_path = output / "export.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("source_metadata_sha256") != source_metadata_sha256:
            return False
        for name, record in report["adapter_artifacts"].items():
            path = output / name
            if not path.is_file() or _sha256_file(path) != record.get("sha256"):
                return False
        return report.get("passed") is True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def export_adapter(
    *,
    checkpoint: Path,
    base_model: Path,
    output: Path,
    rank: int,
    alpha: int,
    dropout: float,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    base_model = base_model.expanduser().resolve()
    output = output.expanduser().resolve()
    dcp_dir = checkpoint / "pytorch_model_fsdp_0"
    metadata_path = dcp_dir / ".metadata"
    if not metadata_path.is_file():
        raise ValueError(f"missing FSDP2 LoRA metadata: {metadata_path}")
    if not base_model.is_dir():
        raise FileNotFoundError(base_model)
    source_sha = _sha256_file(metadata_path)
    if output.exists():
        if _existing_export_is_valid(output, source_sha):
            return json.loads((output / "export.json").read_text(encoding="utf-8"))
        raise FileExistsError(f"existing adapter export is not valid for this source: {output}")

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import (
        _EmptyStateDictLoadPlanner,
        _load_state_dict,
    )
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.monotonic()
    raw_state: dict[str, Any] = {}
    _load_state_dict(
        raw_state,
        storage_reader=dcp.FileSystemReader(dcp_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if len(raw_state) == 1 and isinstance(next(iter(raw_state.values())), dict):
        raw_state = next(iter(raw_state.values()))
    state = normalize_lora_state_dict(raw_state)
    _validate_rank(state, rank)
    targets = infer_target_modules(state)
    for name, tensor in state.items():
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid LoRA tensor: {name}")
        state[name] = tensor.contiguous()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        weights_path = temporary / "adapter_model.safetensors"
        save_file(state, weights_path, metadata={"format": "pt"})
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
            inference_mode=True,
            base_model_name_or_path=str(base_model),
        )
        config.save_pretrained(temporary)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            exported_keys = list(handle.keys())
            if set(exported_keys) != set(state):
                raise ValueError("saved adapter tensor keys differ from the DCP state")
            for name in exported_keys:
                if tuple(handle.get_slice(name).get_shape()) != tuple(state[name].shape):
                    raise ValueError(f"saved adapter tensor shape mismatch: {name}")
        config_payload = json.loads(
            (temporary / "adapter_config.json").read_text(encoding="utf-8")
        )
        if int(config_payload.get("r", -1)) != rank:
            raise ValueError("saved adapter config rank mismatch")
        artifacts = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = temporary / name
            artifacts[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        report = {
            "schema_version": "ifv-fsdp2-lora-export-v1",
            "passed": True,
            "source_checkpoint": str(checkpoint),
            "source_metadata_sha256": source_sha,
            "base_model": str(base_model),
            "adapter_path": str(output),
            "tensor_count": len(state),
            "parameter_elements": sum(int(tensor.numel()) for tensor in state.values()),
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": targets,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "adapter_artifacts": artifacts,
        }
        (temporary / "export.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    args = parser.parse_args()
    report = export_adapter(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        output=args.output,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
