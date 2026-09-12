from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "train"
sys.path.insert(0, str(SCRIPT_DIR))

from export_fsdp2_lora_checkpoint import (  # noqa: E402
    infer_target_modules,
    normalize_lora_state_dict,
)


def test_normalizes_fsdp_wrapper_and_infers_paired_targets() -> None:
    state = {
        "model.base_model.model.layers.0.q_proj.lora_A.weight": object(),
        "model.base_model.model.layers.0.q_proj.lora_B.weight": object(),
        "model.base_model.model.visual.blocks.0.attn.proj.lora_A.weight": object(),
        "model.base_model.model.visual.blocks.0.attn.proj.lora_B.weight": object(),
    }
    normalized = normalize_lora_state_dict(state)
    assert sorted(normalized) == [
        "base_model.model.layers.0.q_proj.lora_A.weight",
        "base_model.model.layers.0.q_proj.lora_B.weight",
        "base_model.model.visual.blocks.0.attn.proj.lora_A.weight",
        "base_model.model.visual.blocks.0.attn.proj.lora_B.weight",
    ]
    assert infer_target_modules(normalized) == ["attn.proj", "q_proj"]


def test_accepts_dcp_loader_that_already_removed_model_wrapper() -> None:
    state = {
        "base_model.model.q_proj.lora_A.weight": object(),
        "base_model.model.q_proj.lora_B.weight": object(),
    }
    assert normalize_lora_state_dict(state) == state


@pytest.mark.parametrize(
    "state,match",
    [
        ({"x.lora_A.weight": object()}, "prefix"),
        ({"model.base_model.x.weight": object()}, "non-LoRA"),
    ],
)
def test_rejects_non_adapter_fsdp_state(state: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_lora_state_dict(state)


def test_rejects_unpaired_lora_tensors() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        infer_target_modules({"base_model.model.q_proj.lora_A.weight": object()})
