"""Validate the small PSD run, then launch the frozen next 1000x4 collection."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v88"
CODE = DEPLOY / "code"
TRAIN_DEPLOY = DEPLOY
TRAIN_STATE = TRAIN_DEPLOY / "training-chain-state.json"
RUN = ROOT / "runs/psd-production1000x4-20260920-v2"
SELECTION = RUN / "selection"
FORMAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1/formal-training-smallbank4095-v1"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def load(path: Path):
    return json.loads(path.read_text())


def diagnostics(training_state: dict) -> dict:
    result = training_state.get("training") or {}
    profile_path, completion_path = Path(result["profile"]), Path(result["completion"])
    profile, completion = load(profile_path), load(completion_path)
    loss = profile.get("loss") or {}
    parallel = profile.get("parallelism") or {}
    resources = profile.get("resources") or {}
    checks = completion.get("checks") or {}
    required_completion = (
        "training_production_gate", "initialization_gate_passed",
        "optimizer_step_completed", "resumable_training_state",
        "output_artifacts_intact", "weight_artifacts_changed", "checkpoint_changed",
    )
    command = str((profile.get("launch") or {}).get("command") or "")
    required_flags = ("--lora_rank 32", "--lora_alpha 32", "--lora_dropout 0.0",
        "--learning_rate 4e-5", "--adam_beta1 0.9", "--adam_beta2 0.95",
        "--adam_epsilon 1e-12", "--weight_decay 0", "--num_train_epochs 5")
    report = {
        "schema_version": "ifv-psd-smallbank-training-diagnostics-v1",
        "profile": str(profile_path), "completion": str(completion_path),
        "formal_training": result.get("formal_training") is True,
        "production_gate": profile.get("passed_production_gate") is True,
        "steps_complete": (profile.get("steps") or {}).get("complete") is True,
        "five_epochs": float((profile.get("steps") or {}).get("last_observed_epoch") or 0) >= 5.0,
        "finite_loss": bool(loss.get("count")) and math.isfinite(float(loss.get("last"))),
        "global_batch_32": int(parallel.get("world_size") or 0)
            * int(parallel.get("per_device_train_batch_size") or 0)
            * int(parallel.get("gradient_accumulation_steps") or 0) == 32,
        "four_gpu_resources": set((resources.get("summary") or {}).get(
            "selected_physical_gpu_ids") or []) == {0, 1, 2, 3},
        "completion_checks": all((checks.get(name) or {}).get("passed") is True
            for name in required_completion),
        "hyperparameters": all(flag in command for flag in required_flags),
        "adapter": str(result.get("adapter") or ""),
        "checkpoint": str(result.get("checkpoint") or ""),
        "large_payload_hashing": False,
    }
    adapter = Path(report["adapter"]) / "adapter_model.safetensors"
    report["adapter_present"] = adapter.is_file() and adapter.stat().st_size > 0
    report["passed"] = all(value is True for key, value in report.items()
        if key not in {"large_payload_hashing"} and isinstance(value, bool))
    return report


def environment() -> dict:
    from dotenv import dotenv_values
    env = {**os.environ, **{key: value for key, value in
        dotenv_values(ROOT / "private/runtime.env").items() if value is not None}}
    present = lambda *keys: any(str(env.get(key, "")).strip() for key in keys)
    if not all((present("SERPER_API_KEY", "SERPER_KEY_ID"),
            present("JINA_API_KEY", "JINA_API_KEYS"),
            present("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            present("BAIDU_OCR_API_KEY"), present("BAIDU_OCR_SECRET_KEY"))):
        raise RuntimeError("authorized Agent tool credentials are incomplete")
    env.update(QWEN35_LOCAL_BASE_URL="http://127.0.0.1:19025/v1",
        QWEN35_LOCAL_MODEL="ifv-qwen3.5-9b-sft-3084",
        QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET="8192",
        QWEN_UNIFIED_JUDGMENT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_JUDGMENT_THINKING_TOKEN_BUDGET="8192",
        AGENT_LLM_REQUEST_TIMEOUT_SECONDS="1230", AGENT_STAGE_REQUEST_TIMEOUT_SECONDS="1260",
        AGENT_LLM_REQUEST_MAX_RETRIES="0", TOOL_CACHE_ENABLED="0",
        IFV_CAPTURE_POLICY_TOKENS="1", IFV_POLICY_TOPK="20", OMP_NUM_THREADS="1",
        PYTHONPATH=str(CODE) + ":" + str(CODE / "training"), TMPDIR=str(ROOT / "tmp"))
    return env


def launch_collection(report: dict) -> list[dict]:
    with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=10) as response:
        health = json.loads(response.read())
    if len(health.get("replicas") or []) != 4:
        raise RuntimeError("four-replica serving was not restored after PSD training")
    episodes = RUN / "episodes"
    if episodes.exists():
        raise FileExistsError("next 1000x4 collection already exists")
    episodes.mkdir()
    base = [sys.executable, "-u", str(CODE / "scripts/collect_psd_rollouts_experimental.py"),
        "--train-cases", str(SELECTION / "train-cases.jsonl"),
        "--runtime-stat-identities", str(SELECTION / "runtime-release/asset-identities.jsonl"),
        "--benchmark", str(SELECTION / "runtime-release/runtime_input/cases.jsonl"),
        "--source-access-policy", str(SELECTION / "runtime-release/evaluator_private/source_access_policy.json"),
        "--profile", "student-qwen3.5-local", "--concurrency", "4",
        "--rollouts-per-case", "4", "--base-sampling-seed", "41000", "--timeout", "3000",
        "--shard-count", "8"]
    processes = []
    env = environment()
    for index in range(8):
        command = [*base, "--shard-index", str(index), "--output-dir", str(episodes / f"shard-{index:02d}")]
        log = (RUN / f"collection-shard-{index:02d}.log").open("xb")
        process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
        log.close()
        processes.append({"shard_index": index, "pid": process.pid, "command": command})
    atomic_json(RUN / "collection-process.json", {"processes": processes,
        "training_diagnostics": str(DEPLOY / "training-diagnostics.json"),
        "storage_policy": "single-canonical-trace-stat-v1", "large_payload_hashing": False,
        "shards": 8, "concurrency_per_shard": 4, "total_concurrency": 32})
    selection = load(SELECTION / "selection.json")
    selection.update(status="collection_running", collection_started=True,
        collection_pids=[row["pid"] for row in processes], training_diagnostics_passed=report["passed"])
    atomic_json(SELECTION / "selection.json", selection)
    return processes


def main() -> None:
    os.umask(0o077)
    state_path = DEPLOY / "posttrain-state.json"
    while True:
        state = load(TRAIN_STATE)
        if state.get("phase") == "failed_requires_fix":
            raise RuntimeError("small-bank PSD chain failed")
        if state.get("phase") == "formal_training_complete_requires_diagnostics":
            break
        atomic_json(state_path, {"phase": "waiting_smallbank_training"})
        time.sleep(30)
    report = diagnostics(state)
    atomic_json(DEPLOY / "training-diagnostics.json", report)
    if not report["passed"]:
        raise RuntimeError("small-bank PSD training diagnostics failed")
    processes = launch_collection(report)
    atomic_json(state_path, {"phase": "next1000_collection_running", "processes": processes})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(DEPLOY / "posttrain-state.json", {"phase": "failed_requires_fix",
            "error_type": type(error).__name__, "message": str(error)[:1000]})
        raise
