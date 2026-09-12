from __future__ import annotations

import json
from pathlib import Path

import pytest

from ifv_training.io import sha256_file, write_json
from training.scripts.h20.attest_merged_lora_serving import attest


def _fixture(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    for directory in (base, adapter, merged):
        directory.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (merged / "model-1.safetensors").write_bytes(b"merged")

    checkpoint = tmp_path / "checkpoint.json"
    write_json(
        checkpoint,
        {
            "schema_version": "ifv-qwen-checkpoint-manifest-v1",
            "training_method": "lora",
            "base_model": {"id_or_path": str(base)},
            "checkpoint": {"path": str(adapter)},
            "artifacts": [
                {
                    "scope": "model",
                    "path": name,
                    "sha256": sha256_file(adapter / name),
                }
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                )
            ],
        },
    )
    runtime = tmp_path / "runtime.json"
    write_json(
        runtime,
        {
            "schema_version": "ifv-qwen-serving-profile-v1",
            "profile_id": "policy-r1",
            "model_path": str(merged),
            "engine_model_path": str(merged),
            "deployment_mode": "full_model",
            "base_url": "http://127.0.0.1:8901/v1",
            "wire_api": "chat_completions",
            "multimodal": True,
            "context_length": 131072,
        },
    )
    merge = merged / "merge-export.json"
    write_json(
        merge,
        {
            "schema_version": "ifv-merged-lora-serving-export-v1",
            "passed": True,
            "source": {
                "base_model": str(base),
                "adapter": str(adapter),
                "adapter_config_sha256": sha256_file(
                    adapter / "adapter_config.json"
                ),
                "adapter_weights_sha256": sha256_file(
                    adapter / "adapter_model.safetensors"
                ),
            },
            "model_path": str(merged),
            "model_artifacts": {
                "model-1.safetensors": {
                    "sha256": sha256_file(merged / "model-1.safetensors")
                }
            },
        },
    )
    numeric = tmp_path / "numeric.json"
    write_json(
        numeric,
        {
            "schema_version": "ifv-merged-lora-audit-v1",
            "passed": True,
            "base_model": str(base),
            "adapter": str(adapter),
            "merged_model": str(merged),
            "samples": [{"module": "language", "passed": True}],
        },
    )
    return {
        "runtime_profile_path": runtime,
        "checkpoint_manifest_path": checkpoint,
        "merge_report_path": merge,
        "numeric_audit_path": numeric,
    }


def test_attest_emits_checkpoint_bound_merged_serving_profile(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    output = tmp_path / "output"

    result = attest(**inputs, output_dir=output)

    assert result["passed"] is True
    profile = json.loads(
        (output / "serving-profile.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        inputs["checkpoint_manifest_path"].read_text(encoding="utf-8")
    )
    assert profile["deployment_mode"] == "merged_lora"
    assert profile["model_path"] == checkpoint["checkpoint"]["path"]
    assert profile["engine_model_path"] == result["engine_model"]
    assert profile["checkpoint_manifest_sha256"] == sha256_file(
        inputs["checkpoint_manifest_path"]
    )


def test_attest_rejects_changed_merged_weight(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    merged = Path(
        json.loads(
            inputs["runtime_profile_path"].read_text(encoding="utf-8")
        )["model_path"]
    )
    (merged / "model-1.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="merged_artifacts_intact"):
        attest(**inputs, output_dir=tmp_path / "output")
