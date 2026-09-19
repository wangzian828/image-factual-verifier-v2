"""Launch a separate Pro retry wave for the six failed first-100 samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza/external-projects/wanglinhaotrain-20260915")
PROJECT = ROOT / "project"
SOURCE = PROJECT / "outputs/friend-pro-first100-fix-20260918"
OUTPUT = PROJECT / "outputs/friend-pro-first100-failed6-retry-20260919"
CONFIG = OUTPUT / "config.json"
MANIFEST = OUTPUT / "manifest.json"
PROCESS = OUTPUT / "process.json"
RUNNER = PROJECT / "src_linux_fallback/run_antigravity_batch.py"
BASE_CONFIG = PROJECT / "outputs/friend-pro-campaign-20260916/config-wave-01.json"


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs():
    receipt = load(SOURCE / "receipt.json")
    progress = load(SOURCE / "progress.json")
    selection = load(SOURCE / "selection.json")
    if receipt.get("status") != "completed" or receipt.get("total") != 19:
        raise RuntimeError("source first-100 retry is not frozen and complete")
    failed = [row["sample_id"] for row in progress["results"] if row.get("state") == "failed"]
    expected = [
        "sample-0005-6608a63f", "sample-0009-db2576f4", "sample-0023-e1cb3067",
        "sample-0027-56c24550", "sample-0029-f962a103", "sample-0031-8d90fe5e",
    ]
    if failed != expected:
        raise RuntimeError(f"failed sample set changed: {failed}")
    by_id = {row["sample_id"]: row for row in selection["sample_ids"]}
    samples = [by_id[sample_id] for sample_id in failed]
    if len(samples) != 6:
        raise RuntimeError("expected exactly six failed samples")
    return receipt, samples


def prepare():
    if OUTPUT.exists():
        raise RuntimeError("retry output already exists")
    receipt, samples = frozen_inputs()
    manifest = {"schema_version": 1, "sample_count": len(samples), "samples": samples}
    base = load(BASE_CONFIG)
    relative_manifest = MANIFEST.relative_to(PROJECT).as_posix()
    relative_run = (OUTPUT / "wave-00").relative_to(PROJECT).as_posix()
    config = {**base,
        "run_id": "friend-pro-first100-failed6-retry-20260919-wave-00",
        "manifest": relative_manifest,
        "run_root": relative_run,
        "expected_manifest_samples": 6,
        "resume_compatible_config_hashes": [],
        "campaign_protocol": {
            **base.get("campaign_protocol", {}),
            "authorization": "2026-09-19: rerun the six failed first-100 Pro samples",
            "retry_wave": "first100-failed6-wave-00",
            "retry_selection": "exact six failed source receipts; successes excluded",
            "original_outputs_read_only": True,
        },
    }
    save(MANIFEST, manifest)
    save(CONFIG, config)
    save(OUTPUT / "binding.json", {
        "source_receipt": str(SOURCE / "receipt.json"),
        "source_receipt_sha256": sha(SOURCE / "receipt.json"),
        "source_progress_sha256": sha(SOURCE / "progress.json"),
        "sample_ids": [row["sample_id"] for row in samples],
        "model": "gemini-3.1-pro-preview",
        "concurrency": 1,
        "max_attempts": 2,
        "source_succeeded_preserved": receipt["succeeded"],
        "time": time.time(),
    })


def worker():
    command = [sys.executable, "-u", str(RUNNER), "--config", str(CONFIG), "--execute"]
    raise SystemExit(subprocess.call(command, cwd=PROJECT, env=os.environ.copy()))


def start():
    prepare()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is unavailable")
    log_path = OUTPUT / "runner.log"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=os.environ.copy(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    save(PROCESS, {"pid": process.pid, "command": command, "log": str(log_path),
        "model": "gemini-3.1-pro-preview", "sample_count": 6,
        "concurrency": 1, "max_attempts": 2, "time": time.time()})
    print(json.dumps({"pid": process.pid, "sample_count": 6, "concurrency": 1}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start", "worker"))
    {"start": start, "worker": worker}[parser.parse_args().mode]()
