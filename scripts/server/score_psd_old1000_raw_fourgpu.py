"""Score only selected old1000 targets with frozen SFT3 raw Transformers top-20.

Owns all four GPUs while scoring. Checked SFT3 service receipts are restored
afterward; the inference guard is resumed only after model identity verifies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
PRE = ROOT / "training-artifacts/psd-combined-sft3-preprocess-20260923-v1"
SHARDS = PRE / "old1000-score-shards-v1"
OUT = PRE / "old1000-teacher-raw-v1"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
SNAPSHOT = ROOT / "training-artifacts/psd-contract-audit-20260916-v10/snapshot"
CODE = ROOT / "training-artifacts/psd-old1000-infrastructure-resume-20260923-v193/code"
MODEL = ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
CHECKPOINT_SHA = "d5517f95d69ade6f72a4da5289254c0386603dc6c3550457b1e3ee16397cb98a"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _stat(path: Path) -> dict[str, int]:
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns}


def _owner():
    path = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_service_owner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("trusted service owner unavailable")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def _preflight() -> dict:
    manifest = _load(SHARDS / "manifest.json")
    if (manifest.get("status") != "frozen_raw_teacher_scoring_pending"
            or manifest.get("checkpoint_manifest_sha256") != CHECKPOINT_SHA
            or manifest.get("counts") != {"repair": 972, "preserve": 972}
            or len(manifest.get("shards", {})) != 4):
        raise ValueError("old1000 teacher shard gate not passed")
    for raw, identity in manifest["shards"].items():
        if _stat(Path(raw)) != identity:
            raise ValueError("teacher shard changed after preparation")
    profile = _load(SNAPSHOT / "serving-profile.json")
    if profile.get("model_path") != str(MODEL) or profile.get("checkpoint_manifest_sha256") != CHECKPOINT_SHA:
        raise ValueError("raw teacher profile does not bind frozen SFT3")
    for q in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmd = q.read_bytes().replace(b"\0", b" ")
        except (OSError, PermissionError):
            continue
        if b"run_psd_old1000_repair.py" in cmd and b"python -u -" not in cmd:
            raise RuntimeError("stopped old1000 repair owner reappeared")
    return manifest


def _models_healthy(timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    last = "service unavailable"
    while time.monotonic() < deadline:
        try:
            for port in range(19002, 19006):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                    cards = json.load(response)["data"]
                if not any(card.get("root") == str(MODEL) for card in cards):
                    raise RuntimeError(f"port {port} serves wrong model")
            with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
                health = json.load(response)
            if len(health.get("replicas", [])) != 4 or health.get("reject_corrupted_responses") is not True:
                raise RuntimeError("isolating gateway is not healthy")
            return
        except Exception as error:  # noqa: BLE001 - bounded cold-start wait
            last = f"{type(error).__name__}: {error}"
            time.sleep(5)
    raise RuntimeError(f"frozen SFT3 serving failed to recover: {last}")


def _environment(index: int) -> dict[str, str]:
    runtime = ROOT / "envs/h20-qwen35-128k"
    cache = OUT / f"gpu-{index}/cache"
    env = dict(os.environ)
    env.update(PATH=str(runtime / "bin") + ":" + env["PATH"],
        CUDA_VISIBLE_DEVICES=str(index), CUDA_HOME=str(runtime),
        PYTHONPATH=str(CODE) + ":" + str(CODE / "training"),
        PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(ROOT / "tmp"),
        HF_HOME=str(ROOT / "cache/huggingface"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(ROOT / "cache/torch"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        XDG_CACHE_HOME=str(cache), OMP_NUM_THREADS="4")
    return env


def _drain_owned_requests(timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        busy = 0.0
        for port in range(19002, 19006):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
                metrics = response.read().decode("utf-8", "replace")
            values = [float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
            if len(values) != 2:
                raise RuntimeError(f"port {port} lacks vLLM drain metrics")
            busy += sum(values)
        if busy == 0:
            return
        time.sleep(2)
    raise RuntimeError("owned inference did not drain before raw teacher scoring")


def score_shard(index: int) -> None:
    if index not in range(4):
        raise ValueError("scoring GPU index must be 0..3")
    _preflight()
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from ifv_training.psd_topk import collect_psd_topk_cache
    result = collect_psd_topk_cache(targets_path=SHARDS / f"targets-{index}.jsonl",
        serving_profile_path=SNAPSHOT / "serving-profile.json",
        checkpoint_manifest_path=SNAPSHOT / "checkpoint-manifest.json",
        output_dir=OUT / f"gpu-{index}", topk=20, retries=3,
        backend="transformers", device="cuda:0")
    if result.get("status") != "ready_for_materialization":
        raise RuntimeError(f"GPU {index} raw teacher scoring incomplete: {result.get('status')}")


def execute() -> None:
    owner = _owner()
    manifest = _preflight()
    _models_healthy(timeout=15)
    backend = [_load(SERVICE / f"replica-{i}.json") for i in range(4)]
    environments = [owner.checked(item) for item in backend]
    guard = _load(SERVICE / "guard.json")
    owner.checked(guard)
    stopped: list[int] = []
    try:
        os.kill(guard["pid"], signal.SIGSTOP)
        with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
            health = json.load(response)
        if any(replica.get("inflight") != 0 for replica in health.get("replicas", [])):
            raise RuntimeError("owned inference gateway still has active requests")
        _drain_owned_requests()
        with ThreadPoolExecutor(max_workers=4) as pool:
            def stop(index: int) -> None:
                owner.stop(backend[index])
                stopped.append(index)
            list(pool.map(stop, range(4)))
        _save(OUT / "state.json", {"phase": "raw_teacher_scoring", "shard_counts": manifest["shard_counts"],
                                   "formal_training_started": False})
        runtime = ROOT / "envs/h20-qwen35-128k/bin/python"
        processes = []
        for index in range(4):
            log = OUT / f"gpu-{index}/score.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("ab")
            command = [str(runtime), "-u", str(Path(__file__).resolve()), "score-shard", str(index)]
            process = subprocess.Popen(command, env=_environment(index), cwd=CODE,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, start_new_session=True)
            processes.append((process, handle, command))
        _save(OUT / "score-processes.json", {"processes": [
            {"pid": proc.pid, "command": command} for proc, _, command in processes]})
        failures = []
        for index, (process, handle, _) in enumerate(processes):
            code = process.wait()
            handle.close()
            if code:
                failures.append({"gpu": index, "returncode": code})
        if failures:
            _save(OUT / "score-failures.json", failures)
            raise RuntimeError("four-GPU raw teacher shard scoring failed")
        _save(OUT / "state.json", {"phase": "raw_teacher_scoring_complete_restoring_services",
                                   "formal_training_started": False})
    except Exception as error:
        _save(OUT / "state.json", {"phase": "failed_requires_fix", "error_type": type(error).__name__,
                                   "message": str(error)[:500], "formal_training_started": False})
        raise
    finally:
        restore_errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            def restore(index: int) -> None:
                try:
                    receipt = owner.spawn(backend[index]["command"], environments[index],
                        OUT / f"restored-backend-{index}.log")
                    owner.save(SERVICE / f"replica-{index}.json", receipt)
                except Exception as error:  # noqa: BLE001 - preserve every restore failure
                    restore_errors.append({"gpu": index, "error": str(error)[:400]})
            list(pool.map(restore, sorted(stopped)))
        if restore_errors:
            _save(OUT / "restore-errors.json", restore_errors)
            raise RuntimeError("frozen SFT3 service restore failed; guard remains stopped")
        _models_healthy()
        os.kill(guard["pid"], signal.SIGCONT)
    for index in range(4):
        report = _load(OUT / f"gpu-{index}/manifest.json")
        if report.get("status") != "ready_for_materialization":
            raise RuntimeError(f"teacher shard {index} manifest not complete")
    _save(OUT / "state.json", {"phase": "raw_teacher_scoring_complete",
                                "targets": 1944, "services_restored": True,
                                "formal_training_started": False})


def launch() -> None:
    manifest = _preflight()
    if OUT.exists():
        raise FileExistsError("raw teacher scoring output already exists; inspect before resuming")
    _models_healthy(timeout=15)
    OUT.mkdir(parents=True, exist_ok=False)
    owner = _owner()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute"]
    receipt = owner.spawn(command, os.environ.copy(), OUT / "controller.log")
    owner.save(OUT / "process.json", receipt)
    _save(OUT / "state.json", {"phase": "launched", "shard_counts": manifest["shard_counts"],
                               "formal_training_started": False})
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
        score_shard(args.index)


if __name__ == "__main__":
    main()
