"""Recover a PSD completion blocked only by a missing small-manifest digest.

The stat-only dataset gate intentionally avoids hashing the 1.3 GB datum file.
The legacy round closer nevertheless requires the SHA256 of the 2.7 KB datum
manifest.  This recovery binds that small manifest after the optimizer run and
reuses every other already-computed completion check without rereading model,
checkpoint, or datum payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EXPECTED_MISSING_CHECKS = frozenset({
    "training_dataset_manifest_bound",
    "dataset_hash_present",
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat_identity(path: Path) -> dict:
    value = path.stat()
    return {"path": str(path.resolve()), "device": value.st_dev, "inode": value.st_ino,
        "bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _same_stat(expected: dict, actual: dict) -> bool:
    return all(int(expected.get(key, -1)) == int(actual[key])
        for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns"))


def recover(*, completion: dict, manifest_path: Path, input_gate: dict,
            ready: dict, profile: dict, checkpoint: Path, adapter: Path) -> tuple[dict, dict]:
    failed = {name for name, value in (completion.get("checks") or {}).items()
        if value.get("passed") is not True}
    if completion.get("passed") is not False or failed != EXPECTED_MISSING_CHECKS:
        raise ValueError(f"completion failures are not the recoverable pair: {sorted(failed)}")
    manifest_path = manifest_path.resolve()
    manifest_hash = digest(manifest_path)  # Deliberately only the small JSON manifest.
    actual = str(completion["checks"]["training_dataset_manifest_bound"].get("actual") or "")
    if manifest_hash != actual:
        raise ValueError("small datum manifest digest does not match checkpoint binding")
    gate_manifest = input_gate.get("manifest") or {}
    if Path(str(gate_manifest.get("path") or "")).resolve() != manifest_path:
        raise ValueError("input gate points at a different datum manifest")
    if not _same_stat(gate_manifest.get("identity") or {}, stat_identity(manifest_path)):
        raise ValueError("datum manifest stat identity changed after training")
    ready_payload = ready.get("payload") or ready
    if Path(str(ready_payload.get("datum_manifest") or "")).resolve() != manifest_path:
        raise ValueError("ready receipt points at a different datum manifest")
    ready_identity = ((ready.get("identity") or {}).get("files") or {}).get(str(manifest_path)) or {}
    if not _same_stat(ready_identity, stat_identity(manifest_path)):
        raise ValueError("ready receipt manifest stat identity changed")
    steps, loss, resources = profile.get("steps") or {}, profile.get("loss") or {}, profile.get("resources") or {}
    if not (profile.get("passed_production_gate") is True
            and steps.get("complete") is True and float(steps.get("last_observed_epoch") or 0) >= 5
            and int(steps.get("last") or 0) == 640 and int(loss.get("count") or 0) == 640
            and math.isfinite(float(loss.get("last"))) and resources.get("passed") is True):
        raise ValueError("training profile is not a complete finite five-epoch run")
    adapter_file = adapter / "adapter_model.safetensors"
    if not checkpoint.is_dir() or not adapter_file.is_file() or adapter_file.stat().st_size <= 0:
        raise ValueError("final checkpoint or adapter export is missing")

    recovered = json.loads(json.dumps(completion))
    recovered["checks"]["training_dataset_manifest_bound"] = {
        "passed": True, "actual": actual, "expected": manifest_hash}
    recovered["checks"]["dataset_hash_present"] = {"passed": True, "sha256": manifest_hash}
    recovered["passed"] = all(value.get("passed") is True
        for value in recovered["checks"].values())
    recovered["recovery"] = {
        "schema_version": "ifv-psd-small-manifest-binding-recovery-v1",
        "reason": "stat-only input gate omitted the small datum-manifest sha256",
        "manifest": stat_identity(manifest_path), "manifest_sha256": manifest_hash,
        "large_payload_hashing": False, "reused_other_completion_checks": True,
    }
    if not recovered["passed"]:
        raise ValueError("recovered completion still has a failed check")
    result = {"passed": True, "profile": "", "completion": "",
        "checkpoint": str(checkpoint.resolve()), "adapter": str(adapter.resolve()),
        "formal_training": True, "epochs": 5, "recovery": recovered["recovery"]}
    return recovered, result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("completion", "manifest", "input-gate", "ready", "profile",
                 "checkpoint", "adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    recovered, result = recover(completion=load(args.completion),
        manifest_path=args.manifest, input_gate=load(args.input_gate), ready=load(args.ready),
        profile=load(args.profile), checkpoint=args.checkpoint, adapter=args.adapter)
    output = args.output.resolve()
    completion_path = output / "completion.json"
    atomic_json(completion_path, recovered)
    result["profile"] = str(args.profile.resolve())
    result["completion"] = str(completion_path)
    atomic_json(output / "result.json", result)
    print(json.dumps({"passed": True, "completion": str(completion_path),
        "large_payload_hashing": False}))


if __name__ == "__main__":
    main()
