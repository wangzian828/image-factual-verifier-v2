"""Run the attested small PSD bank for five epochs on DP4 and restore serving."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-lightweight-topk-handoff-20260920-v78"
CODE = DEPLOY / "code"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
MODEL = ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
SNAPSHOT = ROOT / "training-artifacts/psd-contract-audit-20260916-v10/snapshot"
EXPERIMENT = "psd-smallbank4095-dp4-5epoch-20260920-v1"


def wait_for_idle_serving(timeout_seconds: int = 600) -> list[dict]:
    """Wait through the expected cold-start window after the resume gate."""
    deadline = time.monotonic() + timeout_seconds
    last_error = "serving health unavailable"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=10) as response:
                replicas = json.loads(response.read())["replicas"]
            if len(replicas) != 4:
                last_error = f"expected four serving replicas, got {len(replicas)}"
            elif all(row["inflight"] == 0 for row in replicas):
                return replicas
            else:
                last_error = "owned inference has active requests"
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, KeyError,
                TypeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(5)
    raise RuntimeError(f"owned serving did not become idle: {last_error}")


def wait_for_restored_sft3(timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "frozen SFT3 serving unavailable"
    while time.monotonic() < deadline:
        try:
            for port in range(19002, 19006):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                    models = json.loads(response.read())["data"]
                if not any(row.get("root") == str(MODEL) for row in models):
                    raise RuntimeError(f"port {port} serves wrong model")
            with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=5) as response:
                health = json.loads(response.read())
            if (len(health.get("replicas", [])) != 4
                    or health.get("reject_corrupted_responses") is not True):
                raise RuntimeError("isolating gateway is not healthy")
            return
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, KeyError,
                TypeError, ValueError, RuntimeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(5)
    raise RuntimeError(f"frozen SFT3 service restoration failed: {last_error}")


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


def execute(ready_path: Path, gate_path: Path, output: Path) -> None:
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    from scripts.run_psd_round import load_ready
    from ifv_training.artifact_receipts import artifact_identity
    owner = owner_module()
    ready = load_ready(ready_path)
    gate = owner.load(gate_path)
    if gate.get("passed") is not True or gate.get("adapter_bitwise_equal") is not True:
        raise ValueError("native DP4 resume gate has not passed")
    datums = Path(ready["datums"]).resolve()
    replicas = wait_for_idle_serving()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "binding.json", {"ready": str(ready_path),
        "ready_identity": artifact_identity(ready_path), "datums": str(datums),
        "datums_identity": artifact_identity(datums), "gate": str(gate_path),
        "experiment": EXPERIMENT, "epochs": 5, "global_batch": 32,
        "large_payload_hashing": False, "formal_training": True})
    backends = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in backends]
    guard = owner.load(SERVICE / "guard.json")
    owner.checked(guard)
    stopped = []
    try:
        os.kill(guard["pid"], signal.SIGSTOP)
        for _ in range(90):
            busy = 0.0
            for index in range(4):
                with urllib.request.urlopen(f"http://127.0.0.1:{19002 + index}/metrics", timeout=5) as response:
                    metrics = response.read().decode()
                values = [float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                    if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
                if len(values) != 2:
                    raise RuntimeError("missing vLLM drain metrics")
                busy += sum(values)
            if busy == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("owned inference did not drain")
        def stop(index):
            owner.stop(backends[index])
            stopped.append(index)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stop, range(4)))
        runtime = ROOT / "envs/h20-qwen35-128k"
        env = dict(os.environ)
        env.update(PATH=str(runtime / "bin") + ":" + env["PATH"], CUDA_HOME=str(runtime),
            CUDA_VISIBLE_DEVICES="0,1,2,3", IFV_ALLOWED_GPU_IDS="0,1,2,3",
            PYTHONPATH=str(CODE) + ":" + str(CODE / "training"), PYTHONDONTWRITEBYTECODE="1",
            IFV_TRAINING_DATA_ROOT=str(ROOT), IFV_MODEL_ID=str(MODEL),
            IFV_PSD_PROFILE_MODE="production", IFV_PSD_ROUND_READY=str(ready_path),
            IFV_PSD_SERVING_PROFILE=str(SNAPSHOT / "serving-profile.json"),
            IFV_PSD_ROUND_START_MANIFEST=str(SNAPSHOT / "checkpoint-manifest.json"),
            FLASH_ATTENTION_DETERMINISTIC="1", OMP_NUM_THREADS="4", MAX_JOBS="8",
            TMPDIR=str(ROOT / "tmp"), HF_HOME=str(ROOT / "cache/huggingface"),
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(ROOT / "cache/torch"),
            TRITON_CACHE_DIR=str(ROOT / "cache/psd-smallbank-v1/triton"),
            TORCH_EXTENSIONS_DIR=str(ROOT / "cache/psd-smallbank-v1/extensions"),
            FLASHINFER_WORKSPACE_BASE=str(ROOT / "cache/psd-flashinfer"),
            XDG_CACHE_HOME=str(ROOT / "cache/psd-smallbank-v1"), WANDB_DISABLED="true",
            MASTER_PORT="29547")
        command = ["bash", str(CODE / "training/scripts/train/run_psd_topk.sh"),
            str(CODE / "training/configs/models/qwen3.5-9b.env"),
            str(CODE / "training/configs/psd/qwen3.5-lora-r32-h20-dp4-128k.env"),
            str(datums), EXPERIMENT]
        atomic_json(output / "state.json", {"phase": "formal_training", "command": command})
        with (output / "run.log").open("xb") as log:
            process = subprocess.Popen(command, env=env, cwd=CODE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
            atomic_json(output / "training-process.json", {"pid": process.pid, "command": command})
            code = process.wait()
        if code:
            raise RuntimeError(f"formal PSD training failed with exit {code}")
        profile = owner.load(ROOT / "logs" / EXPERIMENT / "profile.json")
        completion = owner.load(ROOT / "logs" / EXPERIMENT / "round-output/completion.json")
        if profile.get("passed_production_gate") is not True or completion.get("passed") is not True:
            raise RuntimeError("formal PSD post-training gate failed")
        checkpoints = sorted((ROOT / "checkpoints" / EXPERIMENT).glob("**/checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]))
        if not checkpoints or not (checkpoints[-1] / "adapter-export/adapter_model.safetensors").is_file():
            raise RuntimeError("formal PSD adapter export is missing")
        atomic_json(output / "result.json", {"passed": True, "profile": str(ROOT / "logs" / EXPERIMENT / "profile.json"),
            "completion": str(ROOT / "logs" / EXPERIMENT / "round-output/completion.json"),
            "checkpoint": str(checkpoints[-1]), "adapter": str(checkpoints[-1] / "adapter-export"),
            "formal_training": True, "epochs": 5})
        atomic_json(output / "state.json", {"phase": "formal_training_complete", "passed": True,
            "checkpoint": str(checkpoints[-1]), "adapter": str(checkpoints[-1] / "adapter-export")})
    except Exception as error:
        atomic_json(output / "state.json", {"phase": "failed_requires_fix",
            "error_type": type(error).__name__, "message": str(error)[:1000], "formal_training": True})
        raise
    finally:
        restore_errors = []
        for index in sorted(stopped):
            try:
                receipt = owner.spawn(backends[index]["command"], environments[index],
                    output / f"restored-backend-{index}.log")
                owner.save(SERVICE / f"replica-{index}.json", receipt)
            except Exception as error:
                restore_errors.append({"gpu": index, "error_type": type(error).__name__})
        if restore_errors:
            atomic_json(output / "restore-errors.json", restore_errors)
            raise RuntimeError("owned serving restore failed")
        wait_for_restored_sft3()
        os.kill(guard["pid"], signal.SIGCONT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    execute(args.ready.resolve(), args.gate.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
