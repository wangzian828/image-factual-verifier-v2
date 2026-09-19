"""Stage and launch the pending-only PSD resume scheduler (v64)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
OLD = ROOT / "training-artifacts/psd-resume-tail-20260919-v63"
DEPLOY = ROOT / "training-artifacts/psd-resume-scope-20260919-v64"
CODE = DEPLOY / "code"
OVERLAYS = DEPLOY / "overlays"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-resume-scope-v7"
PREVIOUS_CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-resume-tail-v6"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
FILES = (
    "scripts/run_psd_feedback_canary.py",
    "scripts/server/deploy_psd_resume_scope_v64.py",
    "training/tests/test_psd_canary_pipeline.py",
)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def controller():
    result = module("psd_resume_scope_v64_controller",
        CODE / "training/scripts/h20/run_psd_lightweight_resume.py")
    result.DEPLOY = DEPLOY
    result.CODE = CODE
    result.CONTROL = CONTROL
    result.ROUTE = DEPLOY / "external-model-route.json"
    return result


def alive(pid):
    stat = Path("/proc", str(pid), "stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"


def stage():
    if CODE.exists():
        raise RuntimeError("immutable v64 snapshot already exists")
    if not (OLD / "code").is_dir():
        raise RuntimeError("v63 source snapshot is missing")
    for relative in FILES:
        if not (OVERLAYS / relative).is_file():
            raise RuntimeError("missing v64 overlay: " + relative)
    shutil.copytree(OLD / "code", CODE,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for relative in FILES:
        destination = CODE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OVERLAYS / relative, destination)
    shutil.copy2(OLD / "external-model-route.json", DEPLOY / "external-model-route.json")
    tests = (
        "training/tests/test_psd_canary_pipeline.py",
        "training/tests/test_psd_slate_feedback.py",
        "training/tests/test_psd_slate_pipeline.py",
        "training/tests/test_psd_infrastructure_retry.py",
        "training/tests/test_psd_case_pool.py",
        "training/tests/test_psd_lightweight_resume_controller.py",
    )
    env = {**os.environ, "PYTHONPATH": str(CODE) + os.pathsep + str(CODE / "training")}
    with (DEPLOY / "tests.log").open("x") as log:
        result = subprocess.call([sys.executable, "-m", "pytest", "-q",
            *[str(CODE / name) for name in tests]], cwd=CODE, env=env,
            stdout=log, stderr=log)
    save(DEPLOY / "stage-state.json", {
        "deployment_ready_not_live": result == 0,
        "tests_returncode": result,
        "source_snapshot": str(OLD / "code"),
        "source_and_completed_results_preserved": True,
        "terminal_episode_rescan": False,
        "terminal_manifest_check": True,
        "attempt_budget_reset": False,
        "repair_model": "gemini-3.7-flash",
        "thinking_level": "high",
        "overlays": {name: sha(OVERLAYS / name) for name in FILES},
        "time": time.time(),
    })
    if result:
        raise RuntimeError("v64 regression tests failed")
    print(json.dumps({"tests_returncode": 0, "deployment_ready_not_live": True}))


def worker():
    controller().worker()


def start():
    state = load(DEPLOY / "stage-state.json")
    if state.get("deployment_ready_not_live") is not True:
        raise RuntimeError("v64 deployment is not validated")
    if CONTROL.exists():
        raise RuntimeError("v64 controller already exists")
    previous = load(PREVIOUS_CONTROL / "process.json")
    if alive(previous["pid"]):
        raise RuntimeError("previous controller is still alive")
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_psd_round.py prepare" in command:
            raise RuntimeError("another formal prepare process is active")
    launch = controller()
    owner = module("psd_resume_scope_v64_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    external, checks = launch.external_environment()
    env.update(external)
    CONTROL.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u",
        str(CODE / "scripts/server/deploy_psd_resume_scope_v64.py"), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "external_checks": checks,
        "repair_model": "gemini-3.7-flash", "thinking_level": "high",
        "terminal_episode_rescan": False, "attempt_budget_reset": False,
        "time": time.time()})
    print(json.dumps({"controller_pid": receipt["pid"]}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "start", "worker"))
    mode = parser.parse_args().mode
    {"stage": stage, "start": start, "worker": worker}[mode]()
