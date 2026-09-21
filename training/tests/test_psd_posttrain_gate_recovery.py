import json
from pathlib import Path

import pytest

from scripts.server.recover_psd_posttrain_gate import digest, recover, stat_identity


def fixture(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"ready"}\n', encoding="utf-8")
    adapter = tmp_path / "checkpoint/adapter-export"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    checkpoint = adapter.parent
    manifest_hash = digest(manifest)
    checks = {name: {"passed": True} for name in ("training_production_gate",
        "initialization_gate_passed", "optimizer_step_completed", "resumable_training_state",
        "output_artifacts_intact", "weight_artifacts_changed", "checkpoint_changed")}
    checks.update(training_dataset_manifest_bound={"passed": False, "actual": manifest_hash,
        "expected": ""}, dataset_hash_present={"passed": False})
    completion = {"passed": False, "checks": checks}
    identity = stat_identity(manifest)
    gate = {"manifest": {"path": str(manifest), "identity": identity}}
    ready = {"identity": {"files": {str(manifest.resolve()): identity}},
        "payload": {"datum_manifest": str(manifest)}}
    profile = {"passed_production_gate": True,
        "steps": {"complete": True, "last_observed_epoch": 5, "last": 640},
        "loss": {"count": 640, "last": 1.0}, "resources": {"passed": True}}
    return completion, manifest, gate, ready, profile, checkpoint, adapter


def test_recovers_only_missing_small_manifest_binding(tmp_path):
    values = fixture(tmp_path)
    completion, result = recover(completion=values[0], manifest_path=values[1],
        input_gate=values[2], ready=values[3], profile=values[4], checkpoint=values[5], adapter=values[6])
    assert completion["passed"] is True
    assert completion["recovery"]["large_payload_hashing"] is False
    assert result["passed"] is True


def test_refuses_any_additional_failed_gate(tmp_path):
    values = list(fixture(tmp_path))
    values[0]["checks"]["resumable_training_state"] = {"passed": False}
    with pytest.raises(ValueError, match="recoverable pair"):
        recover(completion=values[0], manifest_path=values[1], input_gate=values[2],
            ready=values[3], profile=values[4], checkpoint=values[5], adapter=values[6])
