"""Prepare the frozen next-1000 runtime release using hardlinks and no hashing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production1000x4-20260920-v2"
SELECTION = RUN / "selection"
SOURCE = ROOT / "data/psd-candidate-pool-4000-20260914-v3/train/runtime-release"


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_stat_identities(output):
    path = output / "asset-identities.jsonl"
    if path.exists():
        raise FileExistsError("runtime stat identities already exist")
    records = {}
    for row in rows(output / "runtime_input/cases.jsonl"):
        image = (output / "runtime_input" / row["image_path"]).resolve()
        image.relative_to((output / "runtime_input/assets").resolve())
        record = {**identity(image), "declared_image_sha256": row["image_sha256"]}
        previous = records.setdefault(str(image), record)
        if previous != record:
            raise ValueError("one runtime asset has inconsistent frozen identities")
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for record in sorted(records.values(), key=lambda item: item["path"]):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    atomic_json(output / "asset-identities-receipt.json", {
        "schema_version": "ifv-runtime-assets-stat-v1", "assets": len(records),
        "large_payload_hashing": False, "identities": identity(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stat-identities-only", action="store_true")
    args = parser.parse_args()
    output = SELECTION / "runtime-release"
    if args.stat_identities_only:
        if not output.is_dir():
            raise FileNotFoundError(output)
        write_stat_identities(output)
        return
    if output.exists():
        raise FileExistsError("next-1000 runtime release already exists")
    runtime = output / "runtime_input"
    private = output / "evaluator_private"
    assets = runtime / "assets"
    assets.mkdir(parents=True)
    private.mkdir(parents=True)
    benchmark = SELECTION / "benchmark.jsonl"
    os.link(benchmark, runtime / "cases.jsonl")
    selected_assets = set()
    for row in rows(benchmark):
        relative = Path(row["image_path"])
        if relative.is_absolute() or relative.parts[:1] != ("assets",) or ".." in relative.parts:
            raise ValueError("selected runtime image path escaped assets")
        selected_assets.add(relative)
    for relative in sorted(selected_assets):
        source = (SOURCE / "runtime_input" / relative).resolve()
        source.relative_to((SOURCE / "runtime_input/assets").resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    policy_source = SOURCE / "evaluator_private/source_access_policy.json"
    shutil.copyfile(policy_source, private / policy_source.name)
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    policy = source_manifest["source_access_policy"]
    manifest = {"schema_version": "ifv-image-only-benchmark-release-v0.3",
        "release_id": "ifv-psd-next1000x4-20260920-v2",
        "release_stage": "psd_training_experimental_first_step",
        "training_prohibited": False, "runtime_contract_version": "ifv-image-only-runtime-v1",
        "input_mode": "image_only", "decision_policy_version": "reinspect-v2",
        "runtime_contract": {"allowed_keys": ["case_id", "image_path", "image_sha256"],
            "private_keys_absent": True}, "artifacts": {"agent_input": "runtime_input/cases.jsonl"},
        "source_access_policy": {"active": True,
            "path": "evaluator_private/source_access_policy.json", "sha256": policy["sha256"]}}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "receipt.json", {"schema_version": "ifv-psd-runtime-release-stat-v1",
        "cases": 1000, "assets": len(selected_assets), "hardlinked_assets": True,
        "large_payload_hashing": False, "benchmark_identity": identity(runtime / "cases.jsonl"),
        "source_release": str(SOURCE)})
    write_stat_identities(output)


if __name__ == "__main__":
    os.umask(0o077)
    main()
