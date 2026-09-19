"""Restart validated fast-resume code against its original attestation paths."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path("/volume/ybo/wza")
CODE_DEPLOY = ROOT / "training-artifacts/psd-fast-resume-20260919-v66"
ATTESTATION_DEPLOY = ROOT / "training-artifacts/psd-fast-resume-20260919-v65"
DEPLOY = ROOT / "training-artifacts/psd-fast-resume-20260919-v67"
CODE = CODE_DEPLOY / "code"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-fast-resume-v10"
PREVIOUS_CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-fast-resume-v9"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
EXPECTED_CONTROLLER_SHA256 = "61aa5f3ee727e8016a4b702695ee1b13a1201ae5b6678fde3a61111caaeb9011"


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


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
    result = module("psd_fast_resume_v67_controller",
        CODE / "training/scripts/h20/run_psd_lightweight_resume.py")
    result.DEPLOY = ATTESTATION_DEPLOY
    result.CODE = CODE
    result.CONTROL = CONTROL
    result.ROUTE = ATTESTATION_DEPLOY / "external-model-route.json"
    return result


def start():
    if (DEPLOY / "start-state.json").exists():
        raise RuntimeError("v67 restart already exists")
    previous = load(PREVIOUS_CONTROL / "process.json")
    if alive(previous["pid"]):
        raise RuntimeError("previous controller is still alive")
    if CONTROL.exists() or prepare_processes():
        raise RuntimeError("another formal prepare owner exists")
    controller_path = CODE / "training/scripts/h20/run_psd_lightweight_resume.py"
    digest = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    if digest != EXPECTED_CONTROLLER_SHA256:
        raise RuntimeError("validated controller code changed")
    receipt = ATTESTATION_DEPLOY / "fast-resume-inputs.json"
    index = ATTESTATION_DEPLOY / "selected-candidates-offset-index.json"
    if not receipt.is_file() or not index.is_file():
        raise RuntimeError("original fast-resume attestation is missing")
    launch = controller()
    owner = module("psd_fast_resume_v67_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    external, checks = launch.external_environment()
    env.update(external)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(DEPLOY / "restart_psd_fast_resume_v67.py"), "worker"]
    process = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", process)
    save(CONTROL / "state.json", {"phase": "launched", "external_checks": checks,
        "code": str(CODE), "attestation": str(ATTESTATION_DEPLOY),
        "full_hash_repeated": False, "attempt_budget_reset": False,
        "time": time.time()})
    save(DEPLOY / "start-state.json", {"controller_pid": process["pid"],
        "controller_sha256": digest, "code_reused": True,
        "attestation_reused": True, "time": time.time()})
    print(json.dumps({"controller_pid": process["pid"]}))


def worker():
    controller().worker()


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start", "worker"))
    {"start": start, "worker": worker}[parser.parse_args().mode]()
