"""Take over completed teacher shards without hashing large PSD artifacts.

The v76 owner is paused while its four GPU children finish.  This controller
then restores the exact serving replicas, writes one gzip cache directly, and
materializes/attests the bank with stat-only v2 stage bindings.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
OUT = FINAL / "teacher-topk-preservation-dedup-v74"
DEDUP = FINAL / "preservation-dedup-v74/bank"
TARGETS = DEDUP / "targets/targets.jsonl"
TARGET_MANIFEST = DEDUP / "targets/manifest.json"
SNAPSHOT = ROOT / "training-artifacts/psd-contract-audit-20260916-v10/snapshot"
ROLLOUT_GATE = ROUND / "rollout-gate.json"
DEPLOY = ROOT / "training-artifacts/psd-lightweight-topk-handoff-20260920-v78"
CODE = DEPLOY / "code"


def owner_module():
    path = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_shards(owner) -> None:
    processes = owner.load(OUT / "score-processes.json")["processes"]
    while any(alive(int(row["pid"])) for row in processes):
        counts = []
        for index in range(4):
            path = OUT / f"gpu-{index}/teacher_topk_cache.jsonl"
            counts.append(sum(1 for _ in path.open("rb")) if path.is_file() else 0)
        atomic_json(DEPLOY / "progress.json", {"phase": "waiting_teacher_shards",
            "counts": counts, "total": sum(counts), "expected": owner.load(TARGET_MANIFEST)["counts"]["targets"]})
        time.sleep(15)
    for index in range(4):
        manifest = owner.load(OUT / f"gpu-{index}/manifest.json")
        if manifest.get("status") != "ready_for_materialization" or manifest["counts"]["remaining"]:
            raise RuntimeError(f"teacher shard {index} did not finish cleanly")


def restore_serving(owner, parent_pid: int) -> None:
    if alive(parent_pid):
        os.kill(parent_pid, signal.SIGKILL)
    backends = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in backends]
    errors = []
    for index in range(4):
        try:
            receipt = owner.spawn(backends[index]["command"], environments[index],
                DEPLOY / f"restored-backend-{index}.log")
            owner.save(SERVICE / f"replica-{index}.json", receipt)
        except Exception as error:
            errors.append({"gpu": index, "error_type": type(error).__name__})
    guard = owner.load(SERVICE / "guard.json")
    os.kill(guard["pid"], signal.SIGCONT)
    if errors:
        atomic_json(DEPLOY / "restore-errors.json", errors)
        raise RuntimeError("serving restore failed")
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
                if len(json.loads(response.read())["replicas"]) == 4:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("restored serving did not become healthy")


def merge_gzip(owner) -> Path:
    destination = OUT / "topk-cache/teacher_topk_cache.jsonl.gz"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    seen = set()
    with temporary.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as target:
            for index in range(4):
                with (OUT / f"gpu-{index}/teacher_topk_cache.jsonl").open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        target_id = str(json.loads(line).get("target_id") or "")
                        if not target_id or target_id in seen:
                            raise ValueError("teacher cache has a missing or duplicate target_id")
                        seen.add(target_id)
                        target.write(line)
        raw.flush()
        os.fsync(raw.fileno())
    expected = owner.load(TARGET_MANIFEST)["counts"]["targets"]
    if len(seen) != expected:
        raise ValueError("teacher cache coverage is incomplete")
    os.replace(temporary, destination)
    atomic_json(destination.parent / "receipt.json", {"schema_version": "ifv-psd-topk-stat-v1",
        "targets": len(seen), "cache": str(destination), "large_payload_hashing": False})
    return destination


def materialize(owner, cache: Path) -> str:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd import materialize_psd_topk_cache
    from ifv_training.psd_datums import build_sparse_topk_package
    from ifv_training.psd_materialization import completed_package
    from ifv_training.psd_preflight import verify_psd_training_input
    from scripts.run_psd_round import attest
    from types import SimpleNamespace

    resolved = OUT / "resolved-targets-stat-v78"
    result = completed_package(output_dir=resolved, input_files=[TARGETS, cache],
        build=lambda destination: materialize_psd_topk_cache(targets_path=TARGETS,
            cache_path=cache, output_dir=destination, topk=20))
    if result["status"] != "ready_for_training":
        raise RuntimeError("teacher cache materialization failed")
    datums = OUT / "datums-stat-v78"
    result = completed_package(output_dir=datums, input_files=[resolved / "targets.jsonl"],
        build=lambda destination: build_sparse_topk_package(targets_path=resolved / "targets.jsonl",
            output_dir=destination, topk=20, max_sequence_length=131072, require_both_kinds=True))
    if result["status"] != "ready_for_trainer":
        raise RuntimeError("PSD datum materialization failed")
    gate = verify_psd_training_input(datums_path=datums / "datums.jsonl",
        manifest_path=datums / "manifest.json", expected_topk=20, max_context=131072)
    if not gate["passed"]:
        raise RuntimeError("PSD datum preflight failed")
    ready = attest(SimpleNamespace(output=OUT / "attested-stat-v78", rollout_gate=ROLLOUT_GATE,
        datums=datums / "datums.jsonl", datum_manifest=datums / "manifest.json", snapshot=SNAPSHOT))
    atomic_json(OUT / "result-stat-v78.json", {"status": "ready_for_training", "topk": 20,
        "targets": owner.load(TARGET_MANIFEST)["counts"]["targets"], "ready": ready["ready"],
        "large_payload_hashing": False, "new_agent_or_provider_calls": False})
    # Reproducible scoring shards and the resolved copy are no longer needed.
    for path in [OUT / "target-shards", *(OUT / f"gpu-{index}" for index in range(4)),
                 resolved, OUT / ".resolved-targets-stat-v78-stage"]:
        if path.exists():
            shutil.rmtree(path)
    return ready["ready"]


def execute(parent_pid: int) -> None:
    owner = owner_module()
    try:
        wait_for_shards(owner)
        restore_serving(owner, parent_pid)
        atomic_json(DEPLOY / "state.json", {"phase": "teacher_complete_serving_restored"})
        cache = merge_gzip(owner)
        ready = materialize(owner, cache)
        atomic_json(OUT / "state-stat-v78.json", {"phase": "ready_for_training", "ready": ready,
            "formal_training": False, "large_payload_hashing": False})
        atomic_json(DEPLOY / "state.json", {"phase": "ready_for_training", "ready": ready})
    except Exception as error:
        atomic_json(DEPLOY / "state.json", {"phase": "failed_requires_fix",
            "error_type": type(error).__name__, "message": str(error)[:1000]})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "execute"))
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "execute":
        execute(args.parent_pid)
        return
    DEPLOY.mkdir(parents=True, exist_ok=True)
    if (DEPLOY / "state.json").exists() or (DEPLOY / "process.json").exists():
        raise FileExistsError("lightweight top-k handoff already launched")
    owner = owner_module()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute",
        "--parent-pid", str(args.parent_pid)]
    receipt = owner.spawn(command, os.environ.copy(), DEPLOY / "run.log")
    owner.save(DEPLOY / "process.json", receipt)
    atomic_json(DEPLOY / "state.json", {"phase": "launched", "pid": receipt["pid"]})
    print(json.dumps({"pid": receipt["pid"], "output": str(DEPLOY)}))


if __name__ == "__main__":
    main()
