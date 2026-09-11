from __future__ import annotations

import pytest

from ifv_training.io import load_json, sha256_file, write_json, write_jsonl
from ifv_training.psd_initialization import verify_initialization
from ifv_training.psd_repair import PSDModelRoles


def fixture(tmp_path, *, adapter=False):
    base = tmp_path / "base"
    base.mkdir()
    checkpoint = tmp_path / "adapter" if adapter else base
    checkpoint.mkdir(exist_ok=True)
    weight = checkpoint / ("adapter_model.safetensors" if adapter else "model.safetensors")
    weight.write_bytes(b"synthetic model bytes")
    manifest = tmp_path / "checkpoint.json"
    write_json(manifest, {"schema_version": "ifv-qwen-checkpoint-manifest-v1",
        "checkpoint": {"path": str(checkpoint)},
        "artifacts": [{"scope": "model", "path": weight.name, "sha256": sha256_file(weight)}]})
    profile = tmp_path / "profile.json"
    write_json(profile, {"schema_version": "ifv-qwen-serving-profile-v1", "engine": "vllm",
        "wire_api": "chat_completions", "model_path": str(checkpoint), "engine_model_path": str(base),
        "base_url": "http://127.0.0.1:8901/v1", "profile_id": "policy",
        "checkpoint_manifest_sha256": sha256_file(manifest)})
    roles = PSDModelRoles("gemini", "hint", "qwen_local", "policy", str(checkpoint),
                          sha256_file(manifest), "qwen_local", "policy", str(checkpoint)).record()
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, [{"model_roles": roles}])
    datums = tmp_path / "datum-manifest.json"
    write_json(datums, {"source": {"targets": str(targets), "targets_sha256": sha256_file(targets)}})
    return {"datum_manifest_path": datums, "serving_profile_path": profile,
            "checkpoint_manifest_path": manifest, "model_path": str(base),
            "adapter_path": str(checkpoint) if adapter else None}, weight


@pytest.mark.parametrize("adapter", [False, True])
def test_initialization_matches_frozen_teacher_for_full_and_adapter(tmp_path, adapter):
    args, _ = fixture(tmp_path, adapter=adapter)
    assert verify_initialization(**args)["passed"]


def test_later_round_must_load_previous_adapter(tmp_path):
    args, _ = fixture(tmp_path, adapter=True)
    args["adapter_path"] = None
    with pytest.raises(ValueError, match="round-start adapter"):
        verify_initialization(**args)


def test_wrong_base_model_fails(tmp_path):
    args, _ = fixture(tmp_path)
    args["model_path"] = str(tmp_path / "wrong-base")
    with pytest.raises(ValueError, match="base model differs"):
        verify_initialization(**args)


def test_changed_checkpoint_bytes_fail(tmp_path):
    args, weight = fixture(tmp_path)
    weight.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="model bytes changed"):
        verify_initialization(**args)


def test_changed_targets_fail(tmp_path):
    args, _ = fixture(tmp_path)
    (tmp_path / "targets.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="targets changed"):
        verify_initialization(**args)


def test_rebound_targets_from_different_policy_fail(tmp_path):
    args, _ = fixture(tmp_path)
    target_path = tmp_path / "targets.jsonl"
    import json
    row = json.loads(target_path.read_text())
    for key in ("frozen_self_teacher", "trainable_student"):
        row["model_roles"][key]["checkpoint_manifest_sha256"] = "b" * 64
    write_jsonl(target_path, [row])
    datum = load_json(args["datum_manifest_path"])
    datum["source"]["targets_sha256"] = sha256_file(target_path)
    write_json(args["datum_manifest_path"], datum)
    with pytest.raises(ValueError, match="actual student initialization"):
        verify_initialization(**args)
