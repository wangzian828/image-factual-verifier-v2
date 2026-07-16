from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "args.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)


def build_checkpoint_manifest(
    *,
    checkpoint_dir: Path,
    dataset_manifest_path: Path,
    output_path: Path,
    base_model_id: str,
    model_revision: str,
    processor_revision: str,
    method: str,
    framework_version: str = "4.4.1",
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    dataset_manifest_path = dataset_manifest_path.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    dataset_manifest = load_json(dataset_manifest_path)
    artifacts: list[dict[str, Any]] = []
    for name in CHECKPOINT_FILES:
        path = checkpoint_dir / name
        if path.is_file():
            artifacts.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not artifacts:
        raise ValueError(f"no checkpoint artifacts found in {checkpoint_dir}")
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = (
        load_json(trainer_state_path) if trainer_state_path.is_file() else {}
    )
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_config = (
        load_json(adapter_config_path) if adapter_config_path.is_file() else {}
    )
    manifest = {
        "schema_version": "ifv-qwen-checkpoint-manifest-v1",
        "base_model": {
            "id_or_path": base_model_id,
            "revision": model_revision,
        },
        "processor_revision": processor_revision,
        "framework": {"name": "ms-swift", "version": framework_version},
        "training_method": method,
        "training_dataset": {
            "dataset_version": dataset_manifest.get("dataset_version"),
            "manifest_sha256": sha256_file(dataset_manifest_path),
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "global_step": trainer_state.get("global_step"),
            "optimizer_state_available": (
                checkpoint_dir / "optimizer.pt"
            ).is_file(),
            "scheduler_state_available": (
                checkpoint_dir / "scheduler.pt"
            ).is_file(),
        },
        "adapter_config": adapter_config,
        "artifacts": artifacts,
    }
    write_json(output_path, manifest)
    return manifest


def build_serving_profile(
    *,
    output_path: Path,
    profile_id: str,
    model_path: str,
    engine: str,
    port: int,
    tensor_parallel_size: int,
    dtype: str,
    context_length: int,
    checkpoint_manifest_path: Path | None,
) -> dict[str, Any]:
    profile = {
        "schema_version": "ifv-qwen-serving-profile-v1",
        "profile_id": profile_id,
        "model_path": model_path,
        "engine": engine,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "wire_api": "chat_completions",
        "tool_call_parser": "ms-swift-model-template",
        "multimodal": True,
        "context_length": context_length,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "quantization": None,
        "health_probe": {
            "method": "GET",
            "path": "/v1/models",
        },
        "checkpoint_manifest_sha256": (
            sha256_file(checkpoint_manifest_path)
            if checkpoint_manifest_path
            else None
        ),
    }
    write_json(output_path, profile)
    return profile
