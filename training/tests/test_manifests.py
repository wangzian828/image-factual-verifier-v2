from __future__ import annotations

import json
from pathlib import Path

from ifv_training.checkpoints import (
    _component_candidates,
    _optimizer_state_files,
    build_checkpoint_manifest,
    build_serving_profile,
)
from ifv_training import manifests as manifests_module
from ifv_training.io import write_json
from ifv_training.manifests import (
    cached_dataset_manifest,
    environment_manifest,
    verify_cached_dataset,
)


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

    assert manifest["framework_release"]["version"] == "4.4.2"
    assert manifest["framework_release"]["git_tag"] == "v4.4.2"
    assert len(manifest["framework_release"]["git_commit"]) == 40
    assert len(manifest["package_lock"]["sha256"]) == 64
    assert "flash-attn" in manifest["packages"]
    assert "flash-linear-attention" in manifest["packages"]
    assert "causal-conv1d" in manifest["packages"]


def test_full_checkpoint_component_candidates_include_merger_bias_and_projection() -> None:
    names = {
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.norm.weight",
        "model.visual.blocks.0.norm1.weight",
    }

    candidates = _component_candidates(names, "aligner")

    assert "model.visual.merger.linear_fc1.bias" in candidates
    assert "model.visual.merger.linear_fc1.weight" in candidates


def test_fsdp_optimizer_shards_are_recognized(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    optimizer = checkpoint / "optimizer_0"
    optimizer.mkdir(parents=True)
    (optimizer / ".metadata").write_bytes(b"metadata")
    (optimizer / "__0_0.distcp").write_bytes(b"state")

    files = _optimizer_state_files(checkpoint)

    assert [path.name for path in files] == [".metadata", "__0_0.distcp"]


def test_cached_dataset_manifest_records_profile_source_mtime_and_shape(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    cache = tmp_path / "cache-v2"
    (cache / "train").mkdir(parents=True)
    (cache / "val").mkdir()
    (cache / "train" / "data.arrow").write_bytes(b"train")
    (cache / "val" / "data.arrow").write_bytes(b"validation")
    source = tmp_path / "train.jsonl"
    source.write_text('{"row": 1}\n', encoding="utf-8")
    (cache / "source-dataset-fingerprints.tsv").write_text(
        "channel\tsplit\tpath\tbytes\tmtime_ns\tsha256\n"
        f"planning\ttrain\t{source}\t{source.stat().st_size}\t"
        f"{source.stat().st_mtime_ns}\t"
        f"{manifests_module.sha256_file(source)}\n",
        encoding="utf-8",
    )
    write_json(
        cache / "cache-profile.json",
        {
            "schema_version": "ifv-cached-dataset-profile-v1",
            "cache_id": cache.name,
            "sft_profile": {"id": "profile.env"},
        },
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        manifests_module,
        "_dataset_shape",
        lambda path: {
            "rows": 3 if path.name == "train" else 2,
            "columns": ["messages", "images"],
        },
    )

    result = cached_dataset_manifest(cache)

    assert result["schema_version"] == "ifv-cached-dataset-manifest-v2"
    assert result["metadata"]["train_rows"] == 3
    assert result["metadata"]["validation_rows"] == 2
    fingerprint = result["metadata"]["source_dataset_fingerprints"][0]
    assert fingerprint["mtime_ns"] == source.stat().st_mtime_ns
    assert result["metadata"]["cache_profile"]["sft_profile"]["id"] == "profile.env"


def test_cached_dataset_verifier_fails_closed_on_source_or_artifact_drift(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    cache = tmp_path / "cache-contract"
    train = cache / "train"
    validation = cache / "val"
    train.mkdir(parents=True)
    validation.mkdir()
    (train / "data.arrow").write_bytes(b"train")
    (validation / "data.arrow").write_bytes(b"validation")
    source = tmp_path / "source.jsonl"
    source.write_text('{"row": 1}\n', encoding="utf-8")
    source_fingerprint = {
        "channel": "planning",
        "split": "train",
        "path": str(source),
        "bytes": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "sha256": manifests_module.sha256_file(source),
    }
    artifacts = []
    for path in sorted(cache.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path.relative_to(cache)),
                    "bytes": path.stat().st_size,
                    "sha256": manifests_module.sha256_file(path),
                }
            )
    write_json(
        cache / "dataset-manifest.json",
        {
            "schema_version": "ifv-cached-dataset-manifest-v2",
            "kind": "ms-swift-cached-dataset",
            "dataset_version": cache.name,
            "metadata": {
                "cache_layout": {"train": "train", "validation": "val"},
                "train_rows": 3,
                "validation_rows": 2,
                "train_columns": ["messages"],
                "validation_columns": ["messages"],
                "source_dataset_fingerprints": [source_fingerprint],
            },
            "artifacts": artifacts,
        },
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        manifests_module,
        "_dataset_shape",
        lambda path: {
            "rows": 3 if path.name == "train" else 2,
            "columns": ["messages"],
        },
    )

    accepted = verify_cached_dataset(
        cache_dir=cache,
        train_dir=train,
        validation_dir=validation,
    )
    assert accepted["passed"] is True

    source.write_text('{"row": 2, "changed": true}\n', encoding="utf-8")
    rejected_source = verify_cached_dataset(
        cache_dir=cache,
        train_dir=train,
        validation_dir=validation,
    )
    assert rejected_source["passed"] is False
    assert any(
        error.startswith("source_dataset_")
        for error in rejected_source["errors"]
    )

    (train / "data.arrow").write_bytes(b"drift")
    rejected_cache = verify_cached_dataset(
        cache_dir=cache,
        train_dir=train,
        validation_dir=validation,
    )
    assert rejected_cache["passed"] is False
    assert any(
        error.startswith("cache_artifact_")
        for error in rejected_cache["errors"]
    )
