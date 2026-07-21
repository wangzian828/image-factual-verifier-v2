from __future__ import annotations

import json
from pathlib import Path

from ifv_training.checkpoints import (
    build_checkpoint_manifest,
    build_serving_profile,
)
from ifv_training.io import write_json
from ifv_training.manifests import environment_manifest


def test_checkpoint_and_serving_manifests(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    write_json(checkpoint / "adapter_config.json", {"r": 16})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    write_json(checkpoint / "args.json", {"model": "Qwen/Qwen3-VL-8B"})
    write_json(checkpoint / "trainer_state.json", {"global_step": 25})
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    write_json(
        dataset_manifest,
        {"dataset_version": "ifv-ms-swift-policy-v1"},
    )

    checkpoint_manifest_path = tmp_path / "checkpoint-manifest.json"
    checkpoint_manifest = build_checkpoint_manifest(
        checkpoint_dir=checkpoint,
        dataset_manifest_path=dataset_manifest,
        output_path=checkpoint_manifest_path,
        base_model_id="Qwen/Qwen3-VL-8B",
        model_revision="main",
        processor_revision="main",
        method="lora",
    )
    serving_path = tmp_path / "serving-profile.json"
    serving = build_serving_profile(
        output_path=serving_path,
        profile_id="student-qwen-local",
        model_path="/models/qwen-ifv",
        engine="vllm",
        port=8899,
        tensor_parallel_size=2,
        dtype="bfloat16",
        context_length=16384,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        thinking_enabled=False,
        checkpoint_manifest_path=checkpoint_manifest_path,
    )

    assert checkpoint_manifest["checkpoint"]["global_step"] == 25
    assert checkpoint_manifest["checkpoint"]["optimizer_state_available"] is True
    assert checkpoint_manifest["checkpoint"]["scheduler_state_available"] is True
    assert checkpoint_manifest["checkpoint"]["rng_state_available"] is True
    assert checkpoint_manifest["training_method"] == "lora"
    assert serving["base_url"] == "http://127.0.0.1:8899/v1"
    assert serving["tool_call_parser"] == "qwen3_coder"
    assert serving["thinking_enabled"] is False
    assert len(serving["checkpoint_manifest_sha256"]) == 64
    assert json.loads(serving_path.read_text())["profile_id"] == "student-qwen-local"


def test_environment_manifest_records_framework_and_lock(tmp_path: Path) -> None:
    manifest = environment_manifest(tmp_path)

    assert manifest["framework_release"]["version"] == "4.4.1"
    assert manifest["framework_release"]["git_tag"] == "v4.4.1"
    assert len(manifest["framework_release"]["git_commit"]) == 40
    assert len(manifest["package_lock"]["sha256"]) == 64
