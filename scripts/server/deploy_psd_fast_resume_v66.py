"""Deploy the argument-compatible fast PSD resume without rehashing inputs."""
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
OLD = ROOT / "training-artifacts/psd-fast-resume-20260919-v65"
DEPLOY = ROOT / "training-artifacts/psd-fast-resume-20260919-v66"
CODE = DEPLOY / "code"
OVERLAYS = DEPLOY / "overlays"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-fast-resume-v9"
PREVIOUS_CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-fast-resume-v8"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
FILES = (
    "training/scripts/h20/run_psd_lightweight_resume.py",
    "training/tests/test_psd_lightweight_resume_controller.py",
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def alive(pid):
    stat = Path("/proc", str(pid), "stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"


def prepare_processes():
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if ("run_psd_round.py prepare" in command
                or "run_psd_feedback_canary.py --source" in command):
            found.append((int(cmdline.parent.name), command))
    return found


def controller():
    result = module("psd_fast_resume_v66_controller",
        CODE / "training/scripts/h20/run_psd_lightweight_resume.py")
    result.DEPLOY = DEPLOY
    result.CODE = CODE
    result.CONTROL = CONTROL
    result.ROUTE = DEPLOY / "external-model-route.json"
    return result


def stage():
    if CODE.exists():
        raise RuntimeError("immutable v66 code snapshot already exists")
    for relative in FILES:
        if not (OVERLAYS / relative).is_file():
            raise RuntimeError("missing v66 overlay: " + relative)
    # Hard-link unchanged code so this compatibility-only deployment does not
    # duplicate another repository snapshot. Changed files are unlinked first.
    shutil.copytree(OLD / "code", CODE, copy_function=os.link,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for relative in FILES:
        destination = CODE / relative
        destination.unlink()
        shutil.copy2(OVERLAYS / relative, destination)
    shutil.copy2(OLD / "external-model-route.json", DEPLOY / "external-model-route.json")
    for name in ("fast-resume-inputs.json", "selected-candidates-offset-index.json"):
        shutil.copy2(OLD / name, DEPLOY / name)
    tests = (
        "training/tests/test_psd_canary_pipeline.py",
        "training/tests/test_psd_slate_feedback.py",
        "training/tests/test_psd_slate_pipeline.py",
        "training/tests/test_psd_infrastructure_retry.py",
        "training/tests/test_psd_case_pool.py",
        "training/tests/test_psd_lightweight_resume_controller.py",
    )
    env = {**os.environ, "PYTHONPATH": str(CODE) + os.pathsep + str(CODE / "training"),
        "PYTHONDONTWRITEBYTECODE": "1"}
    with (DEPLOY / "tests.log").open("x") as log:
        result = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            *[str(CODE / name) for name in tests]], cwd=CODE, env=env,
            stdout=log, stderr=log)
    save(DEPLOY / "stage-state.json", {
        "deployment_ready_not_live": result == 0,
        "tests_returncode": result,
        "source_snapshot": str(OLD / "code"),
        "shared_unchanged_files": True,
        "full_hash_repeated": False,
        "attempt_budget_reset": False,
        "overlays": {name: sha(OVERLAYS / name) for name in FILES},
        "inherited_receipt_sha256": sha(DEPLOY / "fast-resume-inputs.json"),
        "inherited_index_sha256": sha(DEPLOY / "selected-candidates-offset-index.json"),
        "time": time.time(),
    })
    if result:
        raise RuntimeError("v66 regression tests failed")
    print(json.dumps({"tests_returncode": 0, "deployment_ready_not_live": True}))


def worker():
    controller().worker()


def start():
    state = load(DEPLOY / "stage-state.json")
    if state.get("deployment_ready_not_live") is not True:
        raise RuntimeError("v66 deployment is not validated")
    previous = load(PREVIOUS_CONTROL / "process.json")
    if alive(previous["pid"]):
        raise RuntimeError("previous controller is still alive")
    if CONTROL.exists() or prepare_processes():
        raise RuntimeError("another formal prepare owner exists")
    launch = controller()
    owner = module("psd_fast_resume_v66_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    external, checks = launch.external_environment()
    env.update(external)
    CONTROL.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(DEPLOY / "deploy_psd_fast_resume_v66.py"), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "external_checks": checks,
        "full_hash_each_resume": False, "attempt_budget_reset": False,
        "time": time.time()})
    print(json.dumps({"controller_pid": receipt["pid"]}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "start", "worker"))
    {"stage": stage, "start": start, "worker": worker}[parser.parse_args().mode]()
