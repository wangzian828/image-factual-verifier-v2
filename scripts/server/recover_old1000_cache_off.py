"""Reversibly replace the idle SFT3 cache-on service for old-1000 PSD repair.

Only serving mode changes: same frozen SFT3 model, four GPUs, gateway and
sampling contract. Preserve the original process receipts and environments in
memory; never print or persist credentials. The interrupted repair owner must
already have exited. On failure, attempt to restore the original service.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
RUN = ROOT / "runs/psd-production1000x4-20260918-v1"
OUTPUT = RUN / "processing-lightweight-v1/repair-search-v1"
SOURCE = ROOT / "training-artifacts/prefix-cache-hybrid-gate-20260921-v127"
DEPLOY = ROOT / "training-artifacts/psd-old1000-cache-off-recovery-run-20260922-v161"
MODEL = ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
GATEWAY_ALIAS = "ifv-qwen3.5-9b-sft-3084"
BACKEND_ALIAS = "ifv-psd-sft3084"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def owner_module():
    source = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_trusted_owner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("trusted SFT3 process owner unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_on_from_original(command: list[str]) -> list[str]:
    if (command.count("--no-enable-prefix-caching") != 1
            or "--enable-prefix-caching" in command
            or command.count("--mamba-cache-mode") != 1
            or command[command.index("--mamba-cache-mode") + 1] != "none"
            or len(command) < 4 or command[2] != "serve"
            or Path(command[3]).resolve() != MODEL.resolve()):
        raise ValueError("frozen cache-off SFT3 reference differs from expected model")
    result = list(command)
    result.remove("--no-enable-prefix-caching")
    result[result.index("--mamba-cache-mode") + 1] = "align"
    result.append("--enable-prefix-caching")
    return result


def json_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=7) as response:
        return json.load(response)


def live_gateway() -> tuple[int, list[str], dict[str, str], str]:
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = [part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if ("scripts.server.psd_qwen_gateway:app" in command
                and "--port" in command
                and command[command.index("--port") + 1] == "19025"):
            matches.append((int(path.name), command))
    if len(matches) != 1:
        raise RuntimeError("expected one owned 19025 gateway")
    pid, command = matches[0]
    if os.getpgid(pid) != pid:
        raise RuntimeError("gateway is not process-group owner")
    environment = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
    return pid, command, environment, os.readlink(f"/proc/{pid}/cwd")


def process_alive(pid: int) -> bool:
    path = Path("/proc") / str(pid) / "stat"
    try:
        return path.read_text().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def stop_gateway(pid: int) -> None:
    if not process_alive(pid):
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(120):
        if not process_alive(pid):
            return
        time.sleep(0.5)
    raise RuntimeError("owned gateway did not stop")


def spawn_gateway(command: list[str], environment: dict[str, str], cwd: str,
                  label: str) -> int:
    with (DEPLOY / f"gateway-{label}.log").open("xb") as log:
        process = subprocess.Popen(command, env=environment, cwd=cwd,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    save(DEPLOY / f"gateway-{label}.json", {"pid": process.pid,
         "command": command, "cwd": cwd, "environment_persisted": False})
    return process.pid


def wait_ready(mode: str, timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            for index in range(4):
                values = json_url(f"http://127.0.0.1:{19002 + index}/v1/models")["data"]
                if not any(row.get("id") == BACKEND_ALIAS
                           and Path(row.get("root", "")).resolve() == MODEL.resolve()
                           for row in values):
                    raise ValueError("wrong SFT3 replica model identity")
            values = json_url("http://127.0.0.1:19025/v1/models")["data"]
            if not any(row.get("id") == GATEWAY_ALIAS
                       and Path(row.get("root", "")).resolve() == MODEL.resolve()
                       for row in values):
                raise ValueError("wrong SFT3 gateway model identity")
            health = json_url("http://127.0.0.1:19025/health")
            if (health.get("prefix_cache_policy") != mode
                    or len(health.get("replicas") or []) != 4
                    or not all(row.get("healthy") is True for row in health["replicas"])):
                raise ValueError("four-card gateway has not reached requested cache mode")
            return
        except (OSError, ValueError, KeyError, TypeError) as error:
            last = type(error).__name__
        time.sleep(5)
    raise RuntimeError("SFT3 cache-mode switch readiness timed out: " + last)


def stop_live_replicas(owner: Any, receipts: list[dict[str, Any]]) -> None:
    alive = []
    for receipt in receipts:
        if process_alive(int(receipt["pid"])):
            owner.checked(receipt)
            alive.append(receipt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(owner.stop, alive))


def main() -> None:
    if DEPLOY.exists():
        raise RuntimeError("cache-off recovery directory already exists")
    latest = load(OUTPUT / "latest-process.json")
    if process_alive(int(latest["pid"])):
        raise RuntimeError("old-1000 repair owner must exit before service switch")
    owner = owner_module()
    guard = load(SERVICE / "guard.json")
    owner.checked(guard)
    originals = [load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in originals]
    references = [load(SOURCE / f"original-replica-{index}.json") for index in range(4)]
    for current, reference in zip(originals, references):
        if current["command"] != cache_on_from_original(reference["command"]):
            raise RuntimeError("current replica is not the attested SFT3 cache-on command")
    gateway_pid, gateway_command, gateway_env, gateway_cwd = live_gateway()
    if gateway_env.get("IFV_PREFIX_CACHE_MODE") != "case_isolated":
        raise RuntimeError("gateway is not the inspected case-isolated service")
    health = json_url("http://127.0.0.1:19025/health")
    if (len(health.get("replicas") or []) != 4
            or any(int(row.get("inflight", 0)) for row in health["replicas"])):
        raise RuntimeError("gateway still has active requests")
    DEPLOY.mkdir(parents=True)
    save(DEPLOY / "original-receipts.json", {"replicas": originals,
         "gateway_pid": gateway_pid, "gateway_command": gateway_command,
         "gateway_cwd": gateway_cwd, "guard_pid": guard["pid"],
         "environment_persisted": False})
    new_replicas: list[dict[str, Any]] = []
    new_gateway_pid: int | None = None
    stopped_gateway = False
    paused_guard = False
    stopped_replicas = False
    try:
        os.kill(int(guard["pid"]), signal.SIGSTOP)
        paused_guard = True
        stopped_gateway = True
        stop_gateway(gateway_pid)
        stopped_replicas = True
        stop_live_replicas(owner, originals)
        for index, (reference, environment) in enumerate(zip(references, environments)):
            receipt = owner.spawn(reference["command"], environment,
                                  DEPLOY / f"cache-off-replica-{index}.log")
            new_replicas.append(receipt)
            save(DEPLOY / f"cache-off-replica-{index}.json", receipt)
            save(SERVICE / f"replica-{index}.json", receipt)
        new_gateway_env = dict(gateway_env)
        new_gateway_env["IFV_PREFIX_CACHE_MODE"] = "request_isolated"
        new_gateway_pid = spawn_gateway(gateway_command, new_gateway_env,
                                        gateway_cwd, "cache-off")
        wait_ready("request_isolated")
        os.kill(int(guard["pid"]), signal.SIGCONT)
        paused_guard = False
        save(DEPLOY / "state.json", {"phase": "complete_cache_off", "replicas": 4,
             "gateway_pid": new_gateway_pid, "guard_resumed": True,
             "large_payload_hashing": False, "time": time.time()})
    except BaseException as error:
        recovery_errors = []
        try:
            if new_gateway_pid is not None:
                stop_gateway(new_gateway_pid)
            if new_replicas:
                stop_live_replicas(owner, new_replicas)
            if stopped_replicas:
                stop_live_replicas(owner, originals)
                for index, (receipt, environment) in enumerate(zip(originals, environments)):
                    restored = owner.spawn(receipt["command"], environment,
                        DEPLOY / f"restore-cache-on-replica-{index}.log")
                    save(SERVICE / f"replica-{index}.json", restored)
                    save(DEPLOY / f"restore-cache-on-replica-{index}.json", restored)
            if stopped_gateway:
                if process_alive(gateway_pid):
                    stop_gateway(gateway_pid)
                spawn_gateway(gateway_command, gateway_env, gateway_cwd, "restore-cache-on")
            if stopped_replicas or stopped_gateway:
                wait_ready("case_isolated")
        except BaseException as restore_error:
            recovery_errors.append(type(restore_error).__name__)
        if paused_guard:
            os.kill(int(guard["pid"]), signal.SIGCONT)
        save(DEPLOY / "state.json", {"phase": "failed_requires_fix",
             "error_type": type(error).__name__, "restore_errors": recovery_errors,
             "guard_resumed": True, "time": time.time()})
        raise


if __name__ == "__main__":
    main()
