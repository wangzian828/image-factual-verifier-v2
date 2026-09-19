"""Automatic handoff: teacher finalization -> DP4 resume gate -> five epochs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v88"
CODE = DEPLOY / "code"
TAKEOVER = DEPLOY / "state.json"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
GATE_NAME = "dp4-resume-gate-smallbank4095-v1"
TRAIN_OUT = FINAL / "formal-training-smallbank4095-v1"


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def load(path):
    return json.loads(path.read_text())


def main():
    os.umask(0o077)
    state_path = DEPLOY / "training-chain-state.json"
    while True:
        state = load(TAKEOVER)
        if state.get("phase") == "failed_requires_fix":
            raise RuntimeError("teacher finalization failed")
        if state.get("phase") == "ready_for_training":
            ready = Path(state["ready"])
            break
        atomic_json(state_path, {"phase": "waiting_teacher_finalization"})
        time.sleep(15)
    gate_out = ready.parent / GATE_NAME
    atomic_json(state_path, {"phase": "dp4_resume_gate", "ready": str(ready)})
    command = [str(ROOT / "envs/h20-qwen35-128k/bin/python"), "-u",
        str(CODE / "scripts/server/run_psd_dp4_resume_gate.py"), "execute",
        "--output-name", GATE_NAME, "--deployment", DEPLOY.name, "--ready", str(ready)]
    with (DEPLOY / "dp4-gate.log").open("xb") as log:
        subprocess.run(command, cwd=CODE, env={**os.environ,
            "PYTHONPATH": str(CODE) + ":" + str(CODE / "training")},
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
    gate = load(gate_out / "result.json")
    if gate.get("passed") is not True:
        raise RuntimeError("DP4 resume gate did not pass")
    atomic_json(state_path, {"phase": "formal_training", "ready": str(ready), "gate": str(gate_out / "result.json")})
    command = [str(ROOT / "envs/h20-qwen35-128k/bin/python"), "-u",
        str(CODE / "scripts/server/run_psd_smallbank_training.py"),
        "--ready", str(ready), "--gate", str(gate_out / "result.json"), "--output", str(TRAIN_OUT)]
    with (DEPLOY / "formal-training.log").open("xb") as log:
        subprocess.run(command, cwd=CODE, env={**os.environ,
            "PYTHONPATH": str(CODE) + ":" + str(CODE / "training")},
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
    result = load(TRAIN_OUT / "result.json")
    if result.get("passed") is not True:
        raise RuntimeError("formal training result did not pass")
    atomic_json(state_path, {"phase": "formal_training_complete_requires_diagnostics",
        "ready": str(ready), "gate": str(gate_out / "result.json"), "training": result})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(DEPLOY / "training-chain-state.json", {"phase": "failed_requires_fix",
            "error_type": type(error).__name__, "message": str(error)[:1000]})
        raise
