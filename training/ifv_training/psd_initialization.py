"""Bind the actual Swift student initialization to the targets' frozen teacher."""
from __future__ import annotations

from pathlib import Path

from .io import load_json, load_jsonl, sha256_file
from .psd import _validated_model_roles
from .psd_topk import _validate_attestation


def verify_initialization(*, datum_manifest_path, serving_profile_path,
                          checkpoint_manifest_path, model_path, adapter_path=None):
    manifest = load_json(datum_manifest_path)
    source = manifest["source"]
    targets_path = Path(source["targets"])
    if sha256_file(targets_path) != source["targets_sha256"]:
        raise ValueError("PSD source targets changed after materialization")
    profile, checkpoint, checkpoint_sha = _validate_attestation(
        serving_profile_path=serving_profile_path, checkpoint_manifest_path=checkpoint_manifest_path)
    base = profile.get("engine_model_path") or checkpoint
    expected_adapter = checkpoint if base != checkpoint else None
    if Path(model_path).resolve() != Path(base).resolve():
        raise ValueError("Swift student base model differs from frozen teacher")
    actual_adapter = str(Path(adapter_path).resolve()) if adapter_path else None
    if actual_adapter != (str(Path(expected_adapter).resolve()) if expected_adapter else None):
        raise ValueError("Swift student must load the exact round-start adapter")
    targets = load_jsonl(targets_path)
    if not targets:
        raise ValueError("PSD source targets are empty")
    for row in targets:
        roles = _validated_model_roles(row.get("model_roles"))
        teacher, student = roles["frozen_self_teacher"], roles["trainable_student"]
        if (teacher["round_start_checkpoint"] != checkpoint or student["initial_checkpoint"] != checkpoint
                or teacher["checkpoint_manifest_sha256"] != checkpoint_sha
                or student["checkpoint_manifest_sha256"] != checkpoint_sha
                or teacher["model"] != profile["profile_id"]):
            raise ValueError("PSD target policy differs from actual student initialization")
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    model_files = [item for item in checkpoint_manifest.get("artifacts", []) if item.get("scope") == "model"]
    if not model_files:
        raise ValueError("round-start manifest lacks model artifact hashes")
    root = Path(checkpoint).resolve()
    for item in model_files:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != item["sha256"]:
            raise ValueError("round-start model bytes changed")
    return {"schema_version": "ifv-psd-initialization-gate-v1", "passed": True,
            "model": str(Path(base).resolve()), "adapter": actual_adapter,
            "checkpoint_manifest_sha256": checkpoint_sha, "targets_sha256": source["targets_sha256"],
            "targets": len(targets), "new_round_optimizer": "fresh unless explicit --resume_from_checkpoint"}
