"""Verify sampled merged weights equal the base plus their LoRA deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _weight_index(root: Path) -> dict[str, Path]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return {str(name): root / str(path) for name, path in mapping.items()}
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise ValueError(f"no model safetensors in {root}")
    from safetensors import safe_open

    result = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            result.update({str(name): path for name in handle.keys()})
    return result


def _tensor(path: Path, name: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _sample_modules(adapter: Path) -> list[str]:
    from safetensors import safe_open

    candidates: dict[str, list[tuple[int, str]]] = {"language": [], "vision": []}
    with safe_open(
        adapter / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        for name in handle.keys():
            if not name.endswith(".lora_A.weight"):
                continue
            module = name.removesuffix(".lora_A.weight")
            b_name = module + ".lora_B.weight"
            if b_name not in handle.keys():
                raise ValueError(f"missing LoRA B tensor for {module}")
            a_shape = handle.get_slice(name).get_shape()
            b_shape = handle.get_slice(b_name).get_shape()
            full_elements = int(a_shape[1]) * int(b_shape[0])
            category = "vision" if ".visual." in module else "language"
            candidates[category].append((full_elements, module))
    selected = []
    for category in ("language", "vision"):
        if not candidates[category]:
            raise ValueError(f"adapter has no {category} LoRA module")
        selected.append(sorted(candidates[category])[0][1])
    return selected


def audit(*, base_model: Path, adapter: Path, merged_model: Path, output: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    base_model = base_model.expanduser().resolve()
    adapter = adapter.expanduser().resolve()
    merged_model = merged_model.expanduser().resolve()
    base_index = _weight_index(base_model)
    merged_index = _weight_index(merged_model)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(config["lora_alpha"]) / float(config["r"])
    samples = []
    with safe_open(
        adapter / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        for module in _sample_modules(adapter):
            model_name = module.removeprefix("base_model.model.") + ".weight"
            if model_name not in base_index or model_name not in merged_index:
                raise ValueError(f"merged/base model lacks adapter target: {model_name}")
            base = _tensor(base_index[model_name], model_name)
            merged = _tensor(merged_index[model_name], model_name)
            a = handle.get_tensor(module + ".lora_A.weight")
            b = handle.get_tensor(module + ".lora_B.weight")
            expected = base + (b @ a * scale).to(dtype=base.dtype)
            error = float((merged.float() - expected.float()).abs().max().item())
            changed = int(torch.count_nonzero(merged != base).item())
            samples.append({
                "module": module,
                "model_weight": model_name,
                "shape": list(base.shape),
                "changed_elements": changed,
                "max_abs_expected_error": error,
                "passed": changed > 0 and error == 0.0,
            })
    result = {
        "schema_version": "ifv-merged-lora-audit-v1",
        "base_model": str(base_model),
        "adapter": str(adapter),
        "merged_model": str(merged_model),
        "scale": scale,
        "samples": samples,
        "passed": all(sample["passed"] for sample in samples),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "adapter", "merged-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
