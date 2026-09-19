"""Score the frozen 400x8 PSD package on four GPUs and attest its training bank.

The target JSONL is streamed once into deterministic target-id shards.  Each
GPU owns one Transformers frozen-teacher process.  Agent/provider inference is
never invoked, and the serving replicas are restored before CPU materialization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
DEDUP = FINAL / "preservation-dedup-v74/bank"
TARGETS = DEDUP / "targets/targets.jsonl"
TARGET_MANIFEST = DEDUP / "targets/manifest.json"
SNAPSHOT = ROOT / "training-artifacts/psd-contract-audit-20260916-v10/snapshot"
ROLLOUT_GATE = ROUND / "rollout-gate.json"
OUT = FINAL / "teacher-topk-preservation-dedup-v74"
DEPLOY = ROOT / "training-artifacts/psd-preserve-dedup-20260920-v73"
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


def split_targets(owner) -> list[Path]:
    manifest = owner.load(TARGET_MANIFEST)
    expected = manifest["counts"]["targets"]
    shards = OUT / "target-shards"
    receipt = shards / "receipt.json"
    paths = [shards / f"targets-{index}.jsonl" for index in range(4)]
    source_sha = owner.sha(TARGETS)
    if receipt.is_file():
        saved = owner.load(receipt)
        if (saved.get("source_sha256") != source_sha or saved.get("targets") != expected
                or saved.get("shards") != {str(path): owner.sha(path) for path in paths}):
            raise ValueError("Frozen PSD target shards changed")
        return paths
    if shards.exists() and any(shards.iterdir()):
        raise ValueError("Uncommitted target shards require inspection")
    shards.mkdir(parents=True, exist_ok=True)
    temporary = [path.with_suffix(".jsonl.tmp") for path in paths]
    handles = [path.open("xb") for path in temporary]
    counts = [0, 0, 0, 0]
    seen = set()
    try:
        with TARGETS.open("rb") as source:
            for raw in source:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                target_id = str(row.get("target_id") or "")
                if not target_id or target_id in seen:
                    raise ValueError("Frozen PSD targets contain a missing/duplicate target_id")
                seen.add(target_id)
                index = int(hashlib.sha256(target_id.encode()).hexdigest(), 16) % 4
                handles[index].write(raw)
                counts[index] += 1
        if len(seen) != expected:
            raise ValueError("Frozen PSD target count changed")
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for source, destination in zip(temporary, paths):
            os.replace(source, destination)
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()
    atomic_json(receipt, {"schema_version": "ifv-psd-topk-shards-v1",
        "source": str(TARGETS), "source_sha256": source_sha, "targets": expected,
        "counts": counts, "shards": {str(path): owner.sha(path) for path in paths}})
    return paths


def scoring_environment(index: int) -> dict[str, str]:
    runtime = ROOT / "envs/h20-qwen35-128k"
    cache = OUT / f"gpu-{index}/cache"
    env = dict(os.environ)
    env.update(PATH=str(runtime / "bin") + ":" + env["PATH"], CUDA_VISIBLE_DEVICES=str(index),
        CUDA_HOME=str(runtime), PYTHONPATH=str(CODE) + ":" + str(CODE / "training"),
        PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(ROOT / "tmp"),
        HF_HOME=str(ROOT / "cache/huggingface"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(ROOT / "cache/torch"),
        TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        XDG_CACHE_HOME=str(cache), OMP_NUM_THREADS="4")
    return env


def score_shard(index: int) -> None:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd_topk import collect_psd_topk_cache
    target = OUT / "target-shards" / f"targets-{index}.jsonl"
    result = collect_psd_topk_cache(targets_path=target,
        serving_profile_path=SNAPSHOT / "serving-profile.json",
        checkpoint_manifest_path=SNAPSHOT / "checkpoint-manifest.json",
        output_dir=OUT / f"gpu-{index}", topk=20, retries=3,
        backend="transformers", device="cuda:0")
    if result["status"] != "ready_for_materialization":
        raise RuntimeError(f"GPU {index} teacher scoring incomplete: {result['status']}")


def merge_cache(owner) -> Path:
    destination = OUT / "topk-cache/teacher_topk_cache.jsonl"
    receipt = OUT / "topk-cache/receipt.json"
    sources = [OUT / f"gpu-{index}/teacher_topk_cache.jsonl" for index in range(4)]
    source_hashes = {str(path): owner.sha(path) for path in sources}
    expected = owner.load(TARGET_MANIFEST)["counts"]["targets"]
    if receipt.is_file():
        saved = owner.load(receipt)
        if (saved.get("sources") != source_hashes or saved.get("cache_sha256") != owner.sha(destination)):
            raise ValueError("Merged frozen-teacher cache changed")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    seen = set()
    with temporary.open("xb") as target:
        for path in sources:
            with path.open("rb") as source:
                for raw in source:
                    if not raw.strip():
                        continue
                    target_id = str(json.loads(raw).get("target_id") or "")
                    if not target_id or target_id in seen:
                        raise ValueError("Teacher cache contains a missing/duplicate target_id")
                    seen.add(target_id)
                    target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    if len(seen) != expected:
        raise ValueError("Teacher cache coverage is incomplete")
    os.replace(temporary, destination)
    atomic_json(receipt, {"schema_version": "ifv-psd-merged-topk-cache-v1",
        "sources": source_hashes, "targets": len(seen), "cache_sha256": owner.sha(destination)})
    return destination


def materialize(owner, cache: Path) -> None:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd import materialize_psd_topk_cache
    from ifv_training.psd_datums import build_sparse_topk_package
    from ifv_training.psd_materialization import completed_package
    from ifv_training.psd_preflight import verify_psd_training_input
    from scripts.run_psd_round import attest
    from types import SimpleNamespace
    resolved = OUT / "resolved-targets"
    result = completed_package(output_dir=resolved, input_files=[TARGETS, cache],
        build=lambda destination: materialize_psd_topk_cache(targets_path=TARGETS,
            cache_path=cache, output_dir=destination, topk=20))
    if result["status"] != "ready_for_training":
        raise RuntimeError("Teacher cache materialization failed")
    datums = OUT / "datums"
    result = completed_package(output_dir=datums, input_files=[resolved / "targets.jsonl"],
        build=lambda destination: build_sparse_topk_package(targets_path=resolved / "targets.jsonl",
            output_dir=destination, topk=20, max_sequence_length=131072, require_both_kinds=True))
    if result["status"] != "ready_for_trainer":
        raise RuntimeError("PSD datum materialization failed")
    gate = verify_psd_training_input(datums_path=datums / "datums.jsonl",
        manifest_path=datums / "manifest.json", expected_topk=20, max_context=131072)
    if not gate["passed"]:
        raise RuntimeError("PSD datum preflight failed")
    ready = attest(SimpleNamespace(output=OUT / "attested", rollout_gate=ROLLOUT_GATE,
        datums=datums / "datums.jsonl", datum_manifest=datums / "manifest.json",
        snapshot=SNAPSHOT))
    atomic_json(OUT / "result.json", {"status": "ready_for_training", "topk": 20,
        "targets": owner.load(TARGET_MANIFEST)["counts"]["targets"], "ready": ready["ready"],
        "new_agent_or_provider_calls": False})
    atomic_json(OUT / "state.json", {"phase": "ready_for_training", "ready": ready["ready"],
        "formal_training": False})


def execute() -> None:
    owner = owner_module()
    owner.verify_export()
    if owner.load(FINAL / "preservation-dedup-v74/state.json")["phase"] != "requires_frozen_teacher_topk":
        raise ValueError("Deduplicated package is not waiting for teacher scoring")
    shards = split_targets(owner)
    backends = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in backends]
    guard = owner.load(SERVICE / "guard.json")
    owner.checked(guard)
    stopped = []
    try:
        os.kill(guard["pid"], signal.SIGSTOP)
        try:
            replicas = owner.http("http://127.0.0.1:19025/health")["replicas"]
            if not all(row["inflight"] == 0 for row in replicas):
                raise RuntimeError("Owned inference gateway has active requests")
        except OSError:
            pass
        for _ in range(90):
            busy = 0.0
            for index in range(4):
                with urllib.request.urlopen(f"http://127.0.0.1:{19002 + index}/metrics", timeout=5) as response:
                    text = response.read().decode()
                values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                    if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
                if len(values) != 2:
                    raise RuntimeError("Missing vLLM drain metrics")
                busy += sum(values)
            if busy == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Owned inference did not drain")
        with ThreadPoolExecutor(max_workers=4) as pool:
            def stop(index):
                owner.stop(backends[index])
                stopped.append(index)
            list(pool.map(stop, range(4)))
        atomic_json(OUT / "state.json", {"phase": "frozen_teacher_topk", "formal_training": False,
            "target_shards": [str(path) for path in shards]})
        runtime = ROOT / "envs/h20-qwen35-128k/bin/python"
        processes = []
        for index in range(4):
            log = (OUT / f"gpu-{index}/score.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("ab")
            command = [str(runtime), "-u", str(Path(__file__).resolve()), "score-shard", str(index)]
            process = subprocess.Popen(command, env=scoring_environment(index), cwd=CODE,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, start_new_session=True)
            processes.append((process, handle, command))
        atomic_json(OUT / "score-processes.json", {"processes": [
            {"pid": process.pid, "command": command} for process, _, command in processes]})
        failures = []
        for index, (process, handle, _) in enumerate(processes):
            code = process.wait()
            handle.close()
            if code:
                failures.append({"gpu": index, "returncode": code})
        if failures:
            atomic_json(OUT / "score-failures.json", failures)
            raise RuntimeError("Frozen-teacher shard scoring failed")
    finally:
        restore_errors = []
        for index in sorted(stopped):
            try:
                receipt = owner.spawn(backends[index]["command"], environments[index],
                    OUT / f"restored-backend-{index}.log")
                owner.save(SERVICE / f"replica-{index}.json", receipt)
            except Exception as error:
                restore_errors.append({"gpu": index, "error_type": type(error).__name__})
        os.kill(guard["pid"], signal.SIGCONT)
        if restore_errors:
            atomic_json(OUT / "restore-errors.json", restore_errors)
            raise RuntimeError("Owned inference restore failed")
        owner.verify_export()
    cache = merge_cache(owner)
    materialize(owner, cache)


def launch() -> None:
    owner = owner_module()
    if OUT.exists():
        state = OUT / "state.json"
        if state.is_file() and owner.load(state).get("phase") == "ready_for_training":
            print(json.dumps(owner.load(state)))
            return
        raise ValueError("Teacher top-k output already exists; resume with execute after inspection")
    OUT.mkdir(parents=True)
    receipt = owner.spawn([sys.executable, "-u", str(Path(__file__).resolve()), "execute"],
        os.environ.copy(), OUT / "run.log")
    owner.save(OUT / "process.json", receipt)
    atomic_json(OUT / "state.json", {"phase": "launched", "formal_training": False})
    print(json.dumps({"pid": receipt["pid"], "output": str(OUT)}))


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "execute", "score-shard"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.mode == "launch":
        launch()
    elif args.mode == "execute":
        execute()
    else:
        if args.index not in range(4):
            raise ValueError("score-shard requires GPU index 0..3")
        score_shard(args.index)


if __name__ == "__main__":
    main()
