"""Adopt formal PSD prepare only if its first source-review pass has errors.

The active CODE45 pass is never interrupted.  If it completes with no pending
source reviews, this helper exits without changing ownership.  If it pauses on
transport/format tail items, the helper stops the sleeping controller by its
exact receipt and resumes the same bound output with validated CODE47, whose
only production change parallelizes local validation of already-saved reviews.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production400x8-20260917-v6"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
OLD = ROOT / "runs/psd-formal-prepare-controller-20260917-v1"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260917-v2"
DEPLOY = ROOT / "training-artifacts/psd-review-resume-handoff-20260917-v48"
CODE = ROOT / "training-artifacts/psd-review-resume-perf-20260917-v47/code"
PREFETCH = RUN / "source-review-prefetch-v2-auto-retry"


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


def alive(receipt: dict) -> bool:
    stat = Path("/proc", str(receipt["pid"]), "stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"


def prepare_command() -> list[str]:
    binding = load(RUN / "binding.json")
    return [sys.executable, "-u", str(CODE / "scripts/run_psd_round.py"), "prepare",
        "--run-dir", str(RUN / "episodes"), "--benchmark", binding["benchmark"],
        "--train-cases", binding["train_cases"], "--private-gold", binding["private_gold"],
        "--source-access-policy", binding["source_access_policy"], "--snapshot", binding["snapshot"],
        "--output", str(ROUND), "--round-index", "1", "--attempts", "6",
        "--case-concurrency", "40", "--source-review-concurrency", "4",
        "--source-review-prefetch", str(PREFETCH), "--expected-rollouts-per-case", "8",
        "--task-source-selection", "longest_failed", "--repair-mode", "slate",
        "--judge-model", "gemini-3.1-pro-preview", "--teacher-device", "cuda:0", "--defer-topk"]


def current_status() -> str:
    if (ROUND / "ready.json").exists():
        return "ready_for_training"
    progress = ROUND / "search/progress.json"
    if progress.exists():
        return str(load(progress).get("status") or "")
    summary = ROUND / "source-reviews/summary.json"
    if summary.exists() and load(summary).get("pending"):
        return "paused_source_review_requires_resolution"
    return ""


def worker() -> None:
    owner = module("psd_epoch3_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    old_receipt = load(OLD / "process.json")
    summary_path = ROUND / "source-reviews/summary.json"
    while not summary_path.exists():
        if not alive(old_receipt):
            raise RuntimeError("original formal controller exited before source-review summary")
        owner.checked(old_receipt)
        save(CONTROL / "state.json", {"phase": "waiting_for_first_review_pass", "time": time.time()})
        time.sleep(10)
    summary = load(summary_path)
    if not summary.get("pending"):
        save(CONTROL / "state.json", {"phase": "no_handoff_needed", "pending": 0,
            "original_controller_unchanged": True, "time": time.time()})
        return
    while True:
        state = load(OLD / "state.json")
        if state.get("phase") == "formal_prepare_result":
            if state.get("returncode") != 0 or state.get("status") != "paused_source_review_requires_resolution":
                raise RuntimeError("original controller did not reach the expected retryable boundary")
            owner.checked(old_receipt)
            os.killpg(old_receipt["pid"], signal.SIGSTOP)
            stopped_state = load(OLD / "state.json")
            if stopped_state != state:
                os.killpg(old_receipt["pid"], signal.SIGCONT)
                continue
            os.killpg(old_receipt["pid"], signal.SIGTERM)
            os.killpg(old_receipt["pid"], signal.SIGCONT)
            for _ in range(120):
                if not alive(old_receipt):
                    break
                time.sleep(.5)
            else:
                raise RuntimeError("original formal controller did not stop at handoff boundary")
            break
        if not alive(old_receipt):
            raise RuntimeError("original controller exited outside the retryable handoff boundary")
        owner.checked(old_receipt)
        time.sleep(5)
    save(CONTROL / "state.json", {"phase": "adopted_paused_source_review",
        "pending": summary["pending"], "old_controller_stopped_at_safe_boundary": True,
        "time": time.time()})

    retryable = {"", "paused_source_review_requires_resolution", "paused_search_requires_resume"}
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
            raise RuntimeError("resumed formal PSD prepare exited nonzero; inspect bound log")
        if status in {"requires_frozen_teacher_topk", "ready_for_training"}:
            save(CONTROL / "state.json", {"phase": status, "attempt": attempt,
                "training_started": False, "time": time.time()})
            return
        if status not in retryable:
            raise RuntimeError(f"resumed formal PSD prepare stopped in non-retryable status {status!r}")
        time.sleep(min(900, 60 * attempt))
    raise RuntimeError("resumed formal PSD prepare exhausted controller retry budget")


def launch() -> None:
    if CONTROL.exists():
        raise RuntimeError("review-resume handoff already exists")
    if load(CODE.parent / "stage-state.json").get("deployment_ready_not_live") is not True:
        raise RuntimeError("CODE47 is not validated")
    CONTROL.mkdir(parents=True, exist_ok=False)
    owner = module("psd_epoch3_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    old = load(OLD / "process.json")
    env = owner.checked(old)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "time": time.time()})
    print(json.dumps({"pid": receipt["pid"], "mode": "conditional_handoff"}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker"))
    args = parser.parse_args()
    {"launch": launch, "worker": worker}[args.mode]()
