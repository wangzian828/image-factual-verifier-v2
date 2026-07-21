from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


BASE_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "args.json",
    "config.json",
    "generation_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
FULL_WEIGHT_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
)
OPTIMIZER_PATTERNS = (
    "*optim_states.pt",
    "optimizer.pt",
)
SCHEDULER_PATTERNS = (
    "scheduler.pt",
    "*model_states.pt",
)
RNG_PATTERNS = (
    "rng_state*.pth",
    "random_states*.pkl",
)


def _bool_arg(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _full_weight_files(checkpoint_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in FULL_WEIGHT_PATTERNS:
        files.extend(path for path in checkpoint_dir.glob(pattern) if path.is_file())
    return sorted(set(files))


def _state_files(checkpoint_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(
            path for path in checkpoint_dir.rglob(pattern) if path.is_file()
        )
    return sorted(set(files))


def _optimizer_state_files(checkpoint_dir: Path) -> list[Path]:
    return _state_files(checkpoint_dir, OPTIMIZER_PATTERNS)


def _scheduler_state_files(checkpoint_dir: Path) -> list[Path]:
    return _state_files(checkpoint_dir, SCHEDULER_PATTERNS)


def _rng_state_files(checkpoint_dir: Path) -> list[Path]:
    return _state_files(checkpoint_dir, RNG_PATTERNS)


def _weight_index(root: Path) -> dict[str, Path]:
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if index_paths:
        payload = json.loads(index_paths[0].read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors weight map: {index_paths[0]}")
        return {
            str(name): root / str(relative)
            for name, relative in weight_map.items()
        }

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for full-parameter checkpoint audit"
        ) from exc

    result: dict[str, Path] = {}
    for path in sorted(root.glob("model*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                result[str(name)] = path
    return result


def _tensor_shape(path: Path, name: str) -> tuple[int, ...]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return tuple(int(value) for value in handle.get_slice(name).get_shape())


def _load_tensor(path: Path, name: str) -> Any:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _component_candidates(
    names: set[str],
    component: str,
) -> list[str]:
    def matches(name: str) -> bool:
        lowered = name.casefold()
        if not lowered.endswith(".weight"):
            return False
        if component == "language":
            return (
                "language_model.layers." in lowered
                or (
                    ".model.layers." in lowered
                    and ".visual." not in lowered
                )
            )
        if component == "vision":
            return (
                ".visual." in lowered
                and "merger" not in lowered
                and "projector" not in lowered
                and "aligner" not in lowered
            )
        if component == "aligner":
            return any(
                marker in lowered
                for marker in ("merger", "projector", "aligner")
            )
        raise ValueError(component)

    return sorted(name for name in names if matches(name))


def _component_weight_deltas(
    base_model_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, dict[str, Any]]:
    import torch

    base_index = _weight_index(base_model_dir)
    checkpoint_index = _weight_index(checkpoint_dir)
    shared_names = set(base_index).intersection(checkpoint_index)
    result: dict[str, dict[str, Any]] = {}
    for component in ("language", "vision", "aligner"):
        candidates = _component_candidates(shared_names, component)
        inspected = 0
        best_delta = 0.0
        best_name = ""
        errors: list[str] = []
        ranked: list[tuple[int, str]] = []
        for name in candidates:
            try:
                shape = _tensor_shape(checkpoint_index[name], name)
                numel = 1
                for value in shape:
                    numel *= value
                if 0 < numel <= 10_000_000:
                    ranked.append((numel, name))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        for _, name in sorted(ranked)[:8]:
            try:
                base = _load_tensor(base_index[name], name).float()
                trained = _load_tensor(checkpoint_index[name], name).float()
                if base.shape != trained.shape:
                    errors.append(f"{name}: shape mismatch")
                    continue
                delta = float(torch.max(torch.abs(trained - base)).item())
                inspected += 1
                if delta > best_delta:
                    best_delta = delta
                    best_name = name
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        result[component] = {
            "updated": best_delta > 0.0,
            "sample_weight": best_name,
            "max_abs_delta": best_delta,
            "inspected_weight_count": inspected,
            "errors": errors[:8],
        }
    return result


def audit_full_parameter_checkpoint(
    *,
    checkpoint_dir: Path,
    base_model_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    base_model_dir = base_model_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    if not base_model_dir.is_dir():
        raise FileNotFoundError(base_model_dir)

    args_path = checkpoint_dir / "args.json"
    args = load_json(args_path) if args_path.is_file() else {}
    tuner_type = str(args.get("tuner_type", "")).casefold()
    freeze_flags = {
        name: _bool_arg(args.get(name))
        for name in ("freeze_llm", "freeze_vit", "freeze_aligner")
    }
    weight_files = _full_weight_files(checkpoint_dir)
    optimizer_files = _optimizer_state_files(checkpoint_dir)
    scheduler_files = _scheduler_state_files(checkpoint_dir)
    rng_files = _rng_state_files(checkpoint_dir)
    adapter_files = [
        path
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if (path := checkpoint_dir / name).is_file()
    ]
    component_updates: dict[str, dict[str, Any]] = {}
    comparison_error = ""
    if weight_files:
        try:
            component_updates = _component_weight_deltas(
                base_model_dir,
                checkpoint_dir,
            )
        except Exception as exc:
            comparison_error = f"{type(exc).__name__}: {exc}"

    checks = {
        "tuner_type_full": tuner_type == "full",
        "llm_unfrozen": freeze_flags["freeze_llm"] is False,
        "vit_unfrozen": freeze_flags["freeze_vit"] is False,
        "aligner_unfrozen": freeze_flags["freeze_aligner"] is False,
        "full_weights_present": bool(weight_files),
        "optimizer_state_present": bool(optimizer_files),
        "scheduler_state_present": bool(scheduler_files),
        "rng_state_present": bool(rng_files),
        "adapter_weights_absent": not adapter_files,
        "language_weights_updated": bool(
            component_updates.get("language", {}).get("updated")
        ),
        "vision_weights_updated": bool(
            component_updates.get("vision", {}).get("updated")
        ),
        "aligner_weights_updated": bool(
            component_updates.get("aligner", {}).get("updated")
        ),
    }
    result = {
        "schema_version": "ifv-full-parameter-checkpoint-audit-v1",
        "checkpoint_dir": str(checkpoint_dir),
        "base_model_dir": str(base_model_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "configured_freeze_flags": freeze_flags,
        "full_weight_files": [
            str(path.relative_to(checkpoint_dir)) for path in weight_files
        ],
        "optimizer_state_files": [
            str(path.relative_to(checkpoint_dir)) for path in optimizer_files
        ],
        "scheduler_state_files": [
            str(path.relative_to(checkpoint_dir)) for path in scheduler_files
        ],
        "rng_state_files": [
            str(path.relative_to(checkpoint_dir)) for path in rng_files
        ],
        "adapter_files": [
            str(path.relative_to(checkpoint_dir)) for path in adapter_files
        ],
        "component_updates": component_updates,
        "comparison_error": comparison_error,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


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
    candidate_paths: list[Path] = []
    for name in BASE_CHECKPOINT_FILES:
        path = checkpoint_dir / name
        if path.is_file():
            candidate_paths.append(path)
    candidate_paths.extend(_full_weight_files(checkpoint_dir))
    candidate_paths.extend(_optimizer_state_files(checkpoint_dir))
    candidate_paths.extend(_scheduler_state_files(checkpoint_dir))
    candidate_paths.extend(_rng_state_files(checkpoint_dir))
    for path in sorted(set(candidate_paths)):
        artifacts.append(
            {
                "path": str(path.relative_to(checkpoint_dir)),
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
            "optimizer_state_available": bool(
                _optimizer_state_files(checkpoint_dir)
            ),
            "scheduler_state_available": bool(
                _scheduler_state_files(checkpoint_dir)
            ),
            "rng_state_available": bool(_rng_state_files(checkpoint_dir)),
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
    tool_call_parser: str,
    reasoning_parser: str,
    thinking_enabled: bool,
    checkpoint_manifest_path: Path | None,
) -> dict[str, Any]:
    profile = {
        "schema_version": "ifv-qwen-serving-profile-v1",
        "profile_id": profile_id,
        "model_path": model_path,
        "engine": engine,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "wire_api": "chat_completions",
        "tool_call_parser": tool_call_parser,
        "reasoning_parser": reasoning_parser,
        "thinking_enabled": thinking_enabled,
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
