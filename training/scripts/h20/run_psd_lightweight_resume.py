"""Resume the formal PSD round with bounded, resumable local postprocessing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production400x8-20260917-v6"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260917-v4"
DEPLOY = ROOT / "training-artifacts/psd-lightweight-candidates-20260917-v51"
CODE = DEPLOY / "code"
PREFETCH = RUN / "source-review-prefetch-v2-auto-retry"
SERVICE = ROOT / "inference/psd-sft3084-20260916"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def prepare_command() -> list[str]:
    binding = load(RUN / "binding.json")
    return [sys.executable, "-u", str(CODE / "scripts/run_psd_round.py"), "prepare",
        "--run-dir", str(RUN / "episodes"), "--benchmark", binding["benchmark"],
        "--train-cases", binding["train_cases"], "--private-gold", binding["private_gold"],
        "--source-access-policy", binding["source_access_policy"], "--snapshot", binding["snapshot"],
        "--output", str(ROUND), "--round-index", "1", "--attempts", "6",
        "--case-concurrency", "40", "--source-review-concurrency", "4",
        "--postprocess-workers", "16", "--candidate-workers", "16",
        "--reuse-completed-source-reviews",
        "--source-review-prefetch", str(PREFETCH), "--expected-rollouts-per-case", "8",
        "--task-source-selection", "longest_failed", "--repair-mode", "slate",
        "--judge-model", "gemini-3.1-pro-preview", "--teacher-device", "cuda:0", "--defer-topk"]


def current_status() -> str:
    if (ROUND / "ready.json").exists():
        return "ready_for_training"
    progress = ROUND / "search/progress.json"
    if progress.exists():
        return str(load(progress).get("status") or "")
    postprocess = RUN / "episodes/psd-postprocess-progress.json"
    if postprocess.exists():
        return "postprocess_" + str(load(postprocess).get("status") or "")
    return ""


def worker() -> None:
    retryable = {"", "postprocess_running", "paused_search_requires_resume"}
    for attempt in range(1, 33):
        save(CONTROL / "state.json", {"phase": "formal_prepare", "attempt": attempt,
            "status_before": current_status(), "time": time.time()})
        with (CONTROL / f"prepare-attempt-{attempt}.log").open("xb") as log:
            process = subprocess.run(prepare_command(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        status = current_status()
        save(CONTROL / "state.json", {"phase": "formal_prepare_result", "attempt": attempt,
            "returncode": process.returncode, "status": status, "time": time.time()})
        if process.returncode != 0:
            raise RuntimeError("lightweight formal PSD prepare exited nonzero; inspect bound log")
        if status in {"requires_frozen_teacher_topk", "ready_for_training"}:
            save(CONTROL / "state.json", {"phase": status, "attempt": attempt,
                "training_started": False, "time": time.time()})
            return
        if status not in retryable:
            raise RuntimeError(f"formal PSD prepare stopped in non-retryable status {status!r}")
        time.sleep(min(900, 60 * attempt))
    raise RuntimeError("formal PSD prepare exhausted controller retry budget")


def launch() -> None:
    if CONTROL.exists():
        raise RuntimeError("lightweight PSD controller already exists")
    state = load(DEPLOY / "stage-state.json")
    if (state.get("deployment_ready_not_live") is not True
            or state.get("commit") != "50e089d"):
        raise RuntimeError("lightweight PSD deployment is not validated")
    old_owners = []
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_psd_round.py prepare" in command or "handoff_psd_review_resume_perf.py worker" in command:
            old_owners.append(command)
    if old_owners:
        raise RuntimeError("another formal PSD prepare owner is still live")
    CONTROL.mkdir(parents=True, exist_ok=False)
    owner = module("psd_epoch3_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "time": time.time()})
    print(json.dumps({"pid": receipt["pid"], "mode": "lightweight_resume"}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker"))
    args = parser.parse_args()
    {"launch": launch, "worker": worker}[args.mode]()
