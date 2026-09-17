"""Advance the completed 400x8 source bank through PSD repair preparation.

This controller captures the already-authorized runtime environment without
persisting credentials, waits for the independent collector and reviewer to
drain, runs the bound quote-repair regression, and resumes the formal repair
stage until it either needs frozen-teacher top-k or requires inspection.  It
does not stop inference services, score top-k, or start the optimizer.
"""
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
PREP = ROOT / "runs/psd-pilot400-preparation-20260915-v1"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
DEPLOY = ROOT / "training-artifacts/psd-formal-prepare-autopilot-20260917-v46"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260917-v1"
CODE = ROOT / "training-artifacts/psd-abstention-quote-repair-20260917-v45/code"
QUOTE = DEPLOY / "run_psd_quote_repair_canary.py"
PREFETCH = RUN / "source-review-prefetch-v2-auto-retry"
SECOND_EPISODE = "main-06143--52becd82--r002"


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


def wait_owned(owner, receipt: dict, label: str) -> None:
    while alive(receipt):
        owner.checked(receipt)
        save(CONTROL / "state.json", {"phase": f"waiting_for_{label}", "time": time.time()})
        time.sleep(15)


def require_collection_complete() -> None:
    progress = load(RUN / "collection-progress.json")
    state = load(RUN / "state.json")
    verification = load(RUN / "collection-verification.json")
    if (progress.get("completed") != 3200 or progress.get("canonical") != 3200
            or progress.get("normal_errors") != 0 or progress.get("pending_infrastructure") != 0
            or state.get("phase") != "collection_complete_requires_source_checker"
            or verification.get("passed") is not True):
        raise RuntimeError("formal source collection did not complete cleanly")


def require_prefetch_drained() -> None:
    progress = load(PREFETCH / "progress.json")
    if (progress.get("phase") != "prefetch_complete_requires_full_collection_gate"
            or progress.get("available") != 3200 or progress.get("active") != 0
            or progress.get("queued") != 0):
        raise RuntimeError("source-review prefetch did not drain cleanly")


def quote_repair_canary() -> dict:
    for attempt in range(1, 5):
        output = CODE.parent / f"real-quote-repair-canary-v2-attempt-{attempt}"
        if output.exists():
            result = output / "result.json"
            if result.exists() and load(result).get("passed") is True:
                return load(result)
            continue
        command = [sys.executable, "-u", str(QUOTE), "--output", str(output),
                   "--episode", SECOND_EPISODE]
        with (CONTROL / f"quote-canary-attempt-{attempt}.log").open("xb") as log:
            process = subprocess.run(command, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        if process.returncode == 0 and (output / "result.json").exists():
            result = load(output / "result.json")
            if result.get("passed") is True:
                return result
        time.sleep(60 * attempt)
    raise RuntimeError("bound quote-repair canary exhausted retry budget")


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
    wait_owned(owner, load(RUN / "process.json"), "source_collection")
    require_collection_complete()
    wait_owned(owner, load(RUN / "source-review-prefetch-process.json"), "source_review_prefetch")
    require_prefetch_drained()
    save(CONTROL / "state.json", {"phase": "running_quote_repair_canary", "time": time.time()})
    quote = quote_repair_canary()
    save(CONTROL / "quote-repair-canary.json", quote)

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
            raise RuntimeError("formal PSD prepare exited nonzero; inspect bound log")
        if status in {"requires_frozen_teacher_topk", "ready_for_training"}:
            save(CONTROL / "state.json", {"phase": status, "attempt": attempt,
                "training_started": False, "time": time.time()})
            return
        if status not in retryable:
            raise RuntimeError(f"formal PSD prepare stopped in non-retryable status {status!r}")
        time.sleep(min(900, 60 * attempt))
    raise RuntimeError("formal PSD prepare exhausted controller retry budget")


def launch() -> None:
    if (CONTROL / "process.json").exists() or ROUND.exists():
        raise RuntimeError("formal prepare controller or output already exists")
    CONTROL.mkdir(parents=True, exist_ok=False)
    owner = module("psd_epoch3_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    reviewer = load(RUN / "source-review-prefetch-process.json")
    env = owner.checked(reviewer)
    env.update(GEMINI_MAX_INFLIGHT_REQUESTS="16", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "time": time.time()})
    print(json.dumps({"pid": receipt["pid"], "output": str(ROUND)}, ensure_ascii=False))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker"))
    args = parser.parse_args()
    {"launch": launch, "worker": worker}[args.mode]()
