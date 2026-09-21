"""Reversibly run the guarded four-replica Qwen hybrid APC gate.

Launch this only after formal Agent inference has finished and the original SFT
service is idle.  The owner always restores the exact cache-off replica and
gateway commands it observed before the probe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from scripts.server.control_corrected_agent_eval import (
    GATEWAY_PORT,
    MODEL_ALIAS,
    PORTS,
    ROOT,
    SERVICE,
    _gateway_process,
    _url_json,
    atomic_json,
    owner_module,
    wait_for_idle_gateway,
    wait_for_services,
)
from scripts.server.run_prefix_cache_canary import cache_on_command, probe_image


BLOCK_SIZE = 528
UNSAFE_WINDOW = 16
EXPECTED_SFT_ROOT = (
    ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
)


def active_agent_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    markers = (
        "run_agent_test_rollout.py",
        "control_corrected_agent_eval.py",
        "resume_corrected_agent_eval.py",
    )
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = [
                part.decode(errors="surrogateescape")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(any(value.endswith(marker) for value in command) for marker in markers):
            matches.append({"pid": int(entry.name), "marker": next(
                marker
                for marker in markers
                if any(value.endswith(marker) for value in command)
            )})
    return matches


def validate_original_replicas(receipts: list[dict[str, Any]]) -> Path:
    roots: list[Path] = []
    for index, receipt in enumerate(receipts):
        command = list(receipt.get("command") or [])
        if len(command) < 4 or command[2] != "serve":
            raise ValueError(f"replica {index} command is not vLLM serve")
        if "--no-enable-prefix-caching" not in command:
            raise ValueError(f"replica {index} is not explicitly cache-off")
        if "--mamba-cache-mode" not in command:
            raise ValueError(f"replica {index} has no mamba cache mode")
        mode = command[command.index("--mamba-cache-mode") + 1]
        if mode != "none":
            raise ValueError(f"replica {index} mamba cache mode is {mode}")
        roots.append(Path(command[3]).resolve())
    if len(set(roots)) != 1:
        raise ValueError(f"original replicas expose mixed roots: {roots}")
    if roots[0] != EXPECTED_SFT_ROOT.resolve():
        raise ValueError(f"probe must start from restored SFT service, got {roots[0]}")
    return roots[0]


def wait_replicas(model_root: Path, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            for port in PORTS:
                cards = _url_json(f"http://127.0.0.1:{port}/v1/models")["data"]
                matches = [row for row in cards if row.get("id") == MODEL_ALIAS]
                if len(matches) != 1:
                    raise ValueError(f"port {port} has no unique model alias")
                exposed = Path(str(matches[0].get("root") or "")).resolve()
                if exposed != model_root.resolve():
                    raise ValueError(f"port {port} exposes {exposed}")
            return
        except (OSError, KeyError, TypeError, ValueError) as error:
            last = f"{type(error).__name__}: {error}"
        time.sleep(5)
    raise RuntimeError(f"replicas did not become ready: {last}")


def stop_gateway(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            state = (proc / "stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            return
        if state == "Z":
            # Gateways launched by this controller remain as zombies until the
            # direct parent reaps them.  A zombie owns no socket or GPU work and
            # must not be mistaken for a live gateway during restoration.
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            return
        time.sleep(0.5)
    raise RuntimeError("gateway did not stop after SIGTERM")


def validate_source_snapshot(source_code: Path) -> None:
    """Reject partial overlays before stopping the live service."""

    required = (
        "scripts/__init__.py",
        "scripts/server/control_corrected_agent_eval.py",
        "scripts/server/psd_qwen_gateway.py",
        "scripts/server/run_prefix_cache_hybrid_gate.py",
        "src/integrations/llm/prefix_cache.py",
        "src/orchestrator/pipeline.py",
    )
    missing = [relative for relative in required if not (source_code / relative).is_file()]
    if missing:
        raise ValueError(f"source snapshot is incomplete: {missing}")


def start_gateway(
    *,
    command: list[str],
    environment: dict[str, str],
    cwd: str,
    log_path: Path,
) -> dict[str, Any]:
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "command": command,
        "started_unix": time.time(),
    }


def canary_gateway_command(command: list[str], source_code: Path) -> list[str]:
    result = list(command)
    if "--app-dir" not in result:
        raise ValueError("gateway command has no --app-dir")
    result[result.index("--app-dir") + 1] = str(source_code)
    return result


def wait_canary_gateway(timeout_seconds: int = 180) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            health = _url_json(f"http://127.0.0.1:{GATEWAY_PORT}/health")
            replicas = health.get("replicas") or []
            if (
                len(replicas) == 4
                and all(row.get("healthy") is True for row in replicas)
                and health.get("prefix_cache_policy") == "case_isolated"
                and health.get("prefix_cache_block_size") == BLOCK_SIZE
                and health.get("prefix_cache_unsafe_window") == UNSAFE_WINDOW
            ):
                return health
            last = json.dumps(health, ensure_ascii=False)[:1000]
        except (OSError, TypeError, ValueError) as error:
            last = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise RuntimeError(f"case-isolated gateway did not become ready: {last}")


def attest_block_size(log_paths: list[Path]) -> None:
    expected = f"Setting attention block size to {BLOCK_SIZE} tokens"
    mismatches = [str(path) for path in log_paths if expected not in path.read_text(errors="ignore")]
    if mismatches:
        raise ValueError(f"vLLM block-size attestation missing: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--keep-cache-on-if-passed", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    deploy = args.deploy.resolve()
    source_code = args.source_code.resolve()
    deploy.mkdir(parents=True, exist_ok=True)
    validate_source_snapshot(source_code)
    atomic_json(
        deploy / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "command": [sys.executable, *sys.argv],
            "started_unix": time.time(),
            "commit": args.source_commit,
        },
    )
    running = active_agent_processes()
    if running:
        raise RuntimeError(f"formal Agent inference is still active: {running}")
    wait_for_idle_gateway()
    owner = owner_module()
    guard = owner.load(SERVICE / "guard.json")
    owner.checked(guard)
    originals = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    for index, receipt in enumerate(originals):
        owner.save(deploy / f"original-replica-{index}.json", receipt)
    environments = [owner.checked(receipt) for receipt in originals]
    model_root = validate_original_replicas(originals)
    gateway_pid, gateway_command, gateway_environment, gateway_cwd = _gateway_process()
    gateway_public_model = (
        gateway_environment.get("PSD_PUBLIC_MODEL_ALIAS") or MODEL_ALIAS
    )
    guard_paused = False
    guard_resumed = False
    original_gateway_stopped = False
    original_replicas_stop_attempted = False
    canary_gateway: dict[str, Any] | None = None
    canary_replicas: list[dict[str, Any]] = []
    restored = False
    cache_on_kept = False
    probe_error: BaseException | None = None
    restore_errors: list[dict[str, str]] = []
    try:
        atomic_json(deploy / "state.json", {"phase": "draining_and_switching"})
        os.kill(int(guard["pid"]), signal.SIGSTOP)
        guard_paused = True
        stop_gateway(gateway_pid)
        original_gateway_stopped = True
        original_replicas_stop_attempted = True
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(owner.stop, originals))

        log_paths: list[Path] = []
        for index, (receipt, environment) in enumerate(zip(originals, environments)):
            command = cache_on_command(list(receipt["command"]))
            canary_environment = dict(environment)
            canary_environment["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"
            log_path = deploy / f"cache-on-replica-{index}.log"
            log_paths.append(log_path)
            started = owner.spawn(command, canary_environment, log_path)
            canary_replicas.append(started)
            owner.save(deploy / f"cache-on-replica-{index}.json", started)
        wait_replicas(model_root)
        attest_block_size(log_paths)

        canary_environment = dict(gateway_environment)
        canary_environment.update(
            PYTHONPATH=str(source_code)
            + (":" + canary_environment["PYTHONPATH"] if canary_environment.get("PYTHONPATH") else ""),
            IFV_PREFIX_CACHE_MODE="case_isolated",
            IFV_PREFIX_CACHE_BLOCK_SIZE=str(BLOCK_SIZE),
            IFV_PREFIX_CACHE_UNSAFE_WINDOW=str(UNSAFE_WINDOW),
        )
        command = canary_gateway_command(gateway_command, source_code)
        canary_gateway = start_gateway(
            command=command,
            environment=canary_environment,
            cwd=str(source_code),
            log_path=deploy / "cache-on-gateway.log",
        )
        atomic_json(deploy / "cache-on-gateway.json", canary_gateway)
        wait_canary_gateway()

        atomic_json(deploy / "state.json", {"phase": "running_hybrid_gate"})
        probe_output = deploy / "probe"
        probe_command = [
            str(ROOT / "envs/h20-qwen35-vllm-0181/bin/python"),
            str(source_code / "scripts/server/probe_prefix_cache_hybrid_safety.py"),
            "--model",
            gateway_public_model,
            "--output",
            str(probe_output),
            "--image",
            str(probe_image()),
            "--block-size",
            str(BLOCK_SIZE),
            "--unsafe-window",
            str(UNSAFE_WINDOW),
        ]
        with (deploy / "probe.log").open("xb") as log:
            completed = subprocess.run(
                probe_command,
                cwd=source_code,
                env=canary_environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"hybrid probe exited {completed.returncode}")
        verdict = owner.load(probe_output / "verdict.json")
        if verdict.get("passed") is not True:
            raise RuntimeError("hybrid probe rejected cache-on serving")
        atomic_json(
            deploy / "state.json",
            {"phase": "gate_passed_restoring", "verdict": verdict},
        )
    except BaseException as error:
        probe_error = error
        atomic_json(
            deploy / "state.json",
            {
                "phase": "gate_failed_restoring",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    finally:
        keep_cache_on = (
            args.keep_cache_on_if_passed
            and probe_error is None
            and canary_gateway is not None
            and len(canary_replicas) == 4
        )
        if keep_cache_on:
            try:
                for index, receipt in enumerate(canary_replicas):
                    owner.checked(receipt)
                    owner.save(SERVICE / f"replica-{index}.json", receipt)
                wait_for_services(model_root)
                cache_on_kept = True
                restored = True
            except BaseException as error:
                keep_cache_on = False
                restore_errors.append(
                    {"stage": "keep_cache_on", "error": type(error).__name__}
                )
        if canary_gateway is not None and not keep_cache_on:
            try:
                stop_gateway(int(canary_gateway["pid"]))
            except BaseException as error:
                restore_errors.append({"stage": "stop_canary_gateway", "error": type(error).__name__})
        if canary_replicas and not keep_cache_on:
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(owner.stop, canary_replicas))
            except BaseException as error:
                restore_errors.append({"stage": "stop_canary_replicas", "error": type(error).__name__})
        restored_replicas: list[dict[str, Any]] = []
        if keep_cache_on:
            restored_replicas = list(canary_replicas)
        elif original_replicas_stop_attempted:
            # A parallel stop can fail after some siblings have already exited.
            # Reconcile any surviving original process before restoring all four.
            for receipt in originals:
                try:
                    owner.checked(receipt)
                except (FileNotFoundError, ProcessLookupError, AssertionError):
                    continue
                try:
                    owner.stop(receipt)
                except BaseException as error:
                    restore_errors.append(
                        {"stage": "reconcile_original_replica", "error": type(error).__name__}
                    )
            for index, (receipt, environment) in enumerate(zip(originals, environments)):
                try:
                    started = owner.spawn(
                        receipt["command"], environment, deploy / f"restore-replica-{index}.log"
                    )
                    restored_replicas.append(started)
                    owner.save(SERVICE / f"replica-{index}.json", started)
                except BaseException as error:
                    restore_errors.append({"stage": f"restore_replica_{index}", "error": type(error).__name__})
        else:
            try:
                for receipt in originals:
                    owner.checked(receipt)
                    restored_replicas.append(receipt)
            except BaseException as error:
                restore_errors.append({"stage": "verify_original_replicas", "error": type(error).__name__})
        if len(restored_replicas) == 4 and not keep_cache_on:
            try:
                wait_replicas(model_root)
                if original_gateway_stopped:
                    restored_gateway = start_gateway(
                        command=gateway_command,
                        environment=gateway_environment,
                        cwd=gateway_cwd,
                        log_path=deploy / "restore-gateway.log",
                    )
                    atomic_json(deploy / "restore-gateway.json", restored_gateway)
                wait_for_services(model_root)
                restored = True
            except BaseException as error:
                restore_errors.append({"stage": "restore_gateway", "error": type(error).__name__})
        if guard_paused:
            try:
                os.kill(int(guard["pid"]), signal.SIGCONT)
                guard_resumed = True
            except BaseException as error:
                restore_errors.append({"stage": "resume_guard", "error": type(error).__name__})
        state_path = deploy / "state.json"
        state = owner.load(state_path) if state_path.is_file() else {}
        if restore_errors or not restored:
            state["phase"] = "failed_requires_fix"
        elif cache_on_kept:
            state["phase"] = "complete_cache_candidate_service_active"
        elif probe_error is not None:
            state["phase"] = "complete_cache_rejected_service_restored"
        else:
            state["phase"] = "complete_cache_candidate_service_restored"
        state["service_restored"] = restored and not cache_on_kept
        state["service_ready"] = restored
        state["cache_on_active"] = cache_on_kept
        state["guard_resumed"] = guard_resumed
        state["restore_errors"] = restore_errors
        atomic_json(state_path, state)
    if restore_errors or not restored:
        raise RuntimeError("cache-off service restoration failed")
    if probe_error is not None:
        raise probe_error


if __name__ == "__main__":
    main()
