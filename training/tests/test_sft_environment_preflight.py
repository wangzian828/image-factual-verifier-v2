from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "verify_qwen35_sft_environment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen35_sft_environment", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_parses_exact_versions_and_visible_gpus() -> None:
    assert MODULE.parse_expected_versions(
        ["torch=2.10.0", "flash-attn=2.8.3"]
    ) == {"torch": "2.10.0", "flash-attn": "2.8.3"}
    assert MODULE.parse_visible_gpu_ids("0, 1,7") == [0, 1, 7]

    with pytest.raises(ValueError, match="duplicate expected package"):
        MODULE.parse_expected_versions(["torch=2.10.0", "torch=2.9.0"])
    with pytest.raises(ValueError, match="physical integer GPU ids"):
        MODULE.parse_visible_gpu_ids("GPU-abcd")


def test_preflight_parses_gpu_inventory() -> None:
    assert MODULE.parse_gpu_inventory(
        "0, NVIDIA A100-SXM4-40GB, 40960\n"
        "7, NVIDIA A100-SXM4-40GB, 40960\n"
    ) == {
        0: {
            "physical_index": 0,
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_total_mib": 40960,
        },
        7: {
            "physical_index": 7,
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_total_mib": 40960,
        },
    }


def test_preflight_reads_qwen35_hybrid_context_contract(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "max_position_embeddings": 262144,
                    "num_hidden_layers": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.read_model_contract(tmp_path)

    assert result["max_position_embeddings"] == 262144
    assert result["num_hidden_layers"] == 4
    assert result["layer_type_counts"] == {
        "full_attention": 1,
        "linear_attention": 3,
    }
