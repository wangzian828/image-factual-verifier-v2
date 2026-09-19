"""Stage and automatically hand off to stat-bound indexed PSD resume (v65)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production400x8-20260917-v6"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
SEARCH = ROUND / "search-gemini37-flash-high"
OLD = ROOT / "training-artifacts/psd-resume-scope-20260919-v64"
DEPLOY = ROOT / "training-artifacts/psd-fast-resume-20260919-v65"
CODE = DEPLOY / "code"
OVERLAYS = DEPLOY / "overlays"
CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-fast-resume-v8"
PREVIOUS_CONTROL = ROOT / "runs/psd-formal-prepare-controller-20260919-resume-scope-v7"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
RECEIPT = DEPLOY / "fast-resume-inputs.json"
INDEX = DEPLOY / "selected-candidates-offset-index.json"
FILES = (
    "scripts/run_psd_feedback_canary.py",
    "scripts/server/deploy_psd_fast_resume_v65.py",
    "training/scripts/h20/run_psd_lightweight_resume.py",
    "training/tests/test_psd_canary_pipeline.py",
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
    result = module("psd_fast_resume_v65_controller",
        CODE / "training/scripts/h20/run_psd_lightweight_resume.py")
    result.DEPLOY = DEPLOY
    result.CODE = CODE
    result.CONTROL = CONTROL
    result.ROUTE = DEPLOY / "external-model-route.json"
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


def stage():
    if CODE.exists():
        raise RuntimeError("immutable v65 snapshot already exists")
    if not (OLD / "code").is_dir():
        raise RuntimeError("v64 source snapshot is missing")
    for relative in FILES:
        if not (OVERLAYS / relative).is_file():
            raise RuntimeError("missing v65 overlay: " + relative)
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
        "full_hash_each_resume": False,
        "selection_full_scan_each_resume": False,
        "attempt_budget_reset": False,
        "repair_model": "gemini-3.7-flash",
        "thinking_level": "high",
        "overlays": {name: sha(OVERLAYS / name) for name in FILES},
        "time": time.time(),
    })
    if result:
        raise RuntimeError("v65 regression tests failed")
    print(json.dumps({"tests_returncode": 0, "deployment_ready_not_live": True}))


def build_index(selected):
    entries, offset, seen = [], 0, set()
    with selected.open("rb") as source:
        for input_index, raw in enumerate(source):
            row = json.loads(raw)
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in seen:
                raise RuntimeError("selected candidates are not uniquely indexed")
            seen.add(case_id)
            entries.append({"case_id": case_id, "input_index": input_index,
                "offset": offset, "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()})
            offset += len(raw)
    if len(entries) != 383 or offset != selected.stat().st_size:
        raise RuntimeError("selected candidate index coverage changed")
    save(INDEX, {"schema_version": "ifv-psd-selected-offset-index-v1",
        "selected_path": str(selected.resolve()), "entries": entries})


def attest():
    if RECEIPT.exists() or INDEX.exists():
        raise RuntimeError("v65 attestation already exists")
    if prepare_processes():
        raise RuntimeError("cannot attest while a prepare process is active")
    prepared = ROUND / "source-bank/prepared.json"
    preparation = load(prepared)["preparation"]
    candidates = ROUND / "source-bank/candidates/repair_candidates.jsonl"
    preservation = ROUND / "source-bank/candidates/preservation_candidates.jsonl"
    selected = SEARCH / "task-source-selection/selected_candidates.jsonl"
    selection = SEARCH / "task-source-selection/selection.json"
    selection_stage = SEARCH / ".task-source-selection-stage/state.json"
    stage_state = load(selection_stage)
    build_index(selected)
    paths = [prepared, candidates, preservation,
        Path(preparation["benchmark"]), Path(preparation["case_split"]),
        Path(preparation["source_access_policy"]), Path(preparation["private_gold"]),
        RUN / "snapshot/serving-profile.json", RUN / "snapshot/checkpoint-manifest.json",
        selected, selection, selection_stage, INDEX]
    paths.extend(selection.parent / relative
        for relative in stage_state["payload"]["files"])
    datums = ROUND / "source-bank/datums/datums.jsonl"
    if datums.exists():
        paths.append(datums)
    paths = list(dict.fromkeys(Path(path).resolve() for path in paths))
    files = {}
    for path in paths:
        path = path.resolve()
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        keys = ("st_size", "st_mtime_ns", "st_ino", "st_dev")
        if any(getattr(before, key) != getattr(after, key) for key in keys):
            raise RuntimeError("fast-resume input changed while hashing: " + str(path))
        files[str(path)] = {"sha256": digest, "size": after.st_size,
            "mtime_ns": after.st_mtime_ns, "inode": after.st_ino, "device": after.st_dev}
    marker = load(SEARCH / "inputs.json")
    for name, digest in marker["identity"]["inputs"].items():
        if files.get(str(Path(name).resolve()), {}).get("sha256") != digest:
            raise RuntimeError("fast-resume input differs from frozen search binding")
    if files[str(candidates.resolve())]["sha256"] != stage_state["identity"]["inputs"][str(candidates.resolve())]:
        raise RuntimeError("selection input differs from frozen stage")
    for relative, digest in stage_state["payload"]["files"].items():
        output = (selection.parent / relative).resolve()
        if files.get(str(output), {}).get("sha256") != digest:
            raise RuntimeError("selection output differs from frozen stage")
    save(RECEIPT, {"schema_version": "ifv-psd-fast-resume-inputs-v1",
        "files": files, "full_hash_completed_once": True,
        "future_validation": "size_mtime_inode_device_plus_indexed_row_sha256",
        "time": time.time()})
    print(json.dumps({"attested_files": len(files), "indexed_cases": 383}))


def worker():
    controller().worker()


def start():
    state = load(DEPLOY / "stage-state.json")
    if state.get("deployment_ready_not_live") is not True:
        raise RuntimeError("v65 deployment is not validated")
    if not RECEIPT.is_file() or not INDEX.is_file():
        raise RuntimeError("v65 fast-resume attestation is missing")
    if CONTROL.exists():
        raise RuntimeError("v65 controller already exists")
    previous = load(PREVIOUS_CONTROL / "process.json")
    if alive(previous["pid"]):
        raise RuntimeError("previous controller is still alive")
    if prepare_processes():
        raise RuntimeError("another formal prepare process is active")
    launch = controller()
    owner = module("psd_fast_resume_v65_owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    env = owner.checked(load(SERVICE / "gateway.json"))
    external, checks = launch.external_environment()
    env.update(external)
    CONTROL.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u",
        str(CODE / "scripts/server/deploy_psd_fast_resume_v65.py"), "worker"]
    receipt = owner.spawn(command, env, CONTROL / "controller.log")
    save(CONTROL / "process.json", receipt)
    save(CONTROL / "state.json", {"phase": "launched", "external_checks": checks,
        "repair_model": "gemini-3.7-flash", "thinking_level": "high",
        "full_hash_each_resume": False, "attempt_budget_reset": False,
        "time": time.time()})
    print(json.dumps({"controller_pid": receipt["pid"]}))


def handoff():
    previous = load(PREVIOUS_CONTROL / "process.json")
    parent = previous["pid"]
    if not alive(parent) or "psd-resume-scope-20260919-v64" not in " ".join(previous["command"]):
        raise RuntimeError("v64 controller identity changed")
    save(DEPLOY / "handoff-state.json", {"phase": "waiting_for_v64_attempt",
        "previous_pid": parent, "time": time.time()})
    # Freeze only the outer retry owner.  Its current child remains alive and
    # finishes normally, but the owner cannot launch another attempt in the
    # narrow gap before the handoff observes that completion.
    os.kill(parent, signal.SIGSTOP)
    old_owner_stopped = True
    try:
        while prepare_processes():
            time.sleep(1)
        if not alive(parent):
            raise RuntimeError("v64 controller exited during handoff")
        save(DEPLOY / "handoff-state.json", {"phase": "attesting_once",
            "previous_pid": parent, "time": time.time()})
        attest()
    except BaseException:
        if old_owner_stopped and alive(parent):
            os.kill(parent, signal.SIGCONT)
        raise
    os.kill(parent, signal.SIGTERM)
    os.kill(parent, signal.SIGCONT)
    old_owner_stopped = False
    for _ in range(100):
        if not alive(parent):
            break
        time.sleep(.1)
    else:
        raise RuntimeError("v64 controller did not stop")
    if prepare_processes():
        raise RuntimeError("prepare process appeared during v65 handoff")
    start()
    save(DEPLOY / "handoff-state.json", {"phase": "v65_started",
        "previous_pid": parent, "new_control": str(CONTROL), "time": time.time()})


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "attest", "start", "worker", "handoff"))
    {"stage": stage, "attest": attest, "start": start,
     "worker": worker, "handoff": handoff}[parser.parse_args().mode]()
