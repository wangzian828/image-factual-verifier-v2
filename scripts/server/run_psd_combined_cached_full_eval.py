"""Serve the finished combined PSD adapter and run the frozen full Agent test.

This is a separate, reversible GPU owner.  It never starts while formal training
owns the cards.  A fresh merged-model vLLM process on each card is probed with
the case-isolated hybrid prefix-cache gate before any formal case is dispatched.
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

# The trusted process owner starts children from its own deployment directory.
# Keep this entry point importable even when the caller did not set PYTHONPATH.
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from scripts.server import control_corrected_agent_eval as agent
from scripts.server import run_prefix_cache_hybrid_gate as cache_gate
from scripts.server.run_prefix_cache_canary import cache_on_command, probe_image


ROOT = Path("/volume/ybo/wza")
TRAIN = ROOT / "training-artifacts/psd-combined-formal-training-20260923-v1"
TRAIN_OWNER = (
    ROOT / "runs/psd-combined-smallbank-old1000-sft3-20260923-v1"
    / "attested-v12/formal-training-launch-v1/process.json"
)
SFT_ROOT = ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
MERGED_ROOT = ROOT / "exports/qwen35-psd-combined-merged-20260924-v1/model"
DEPLOY = ROOT / "training-artifacts/psd-combined-cached-full-eval-20260924-v1"
LAUNCH = ROOT / "training-artifacts/psd-combined-cached-full-eval-launch-20260924-v1"
EVAL_OUTPUT = ROOT / "evaluation/qwen35-psd-combined-agent-full1526-selfextract-20260924-v1"
GATEWAY_CODE = ROOT / "training-artifacts/corrected-threeway-cache-guarded-20260921-v137/code"
PROFILE_MODEL = "ifv-qwen3.5-9b-sft3084-psd-combined-selfextract"


def live_process(pid: int, marker: str) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        if (proc / "stat").read_text().split()[2] == "Z":
            return False
        return marker in (proc / "cmdline").read_bytes().decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return False


def training_result() -> dict[str, Any]:
    result = agent.load(TRAIN / "result.json")
    profile = agent.load(Path(result["profile"]))
    completion = agent.load(Path(result["completion"]))
    adapter = Path(result["adapter"]).resolve()
    checkpoint = Path(result["checkpoint"]).resolve()
    if (
        result.get("passed") is not True
        or result.get("formal_training") is not True
        or result.get("epochs") != 5
        or profile.get("passed_production_gate") is not True
        or completion.get("passed") is not True
        or adapter != checkpoint / "adapter-export"
        or not (adapter / "adapter_model.safetensors").is_file()
        or not (adapter / "adapter_config.json").is_file()
        or not str(checkpoint).startswith(
            str(ROOT / "checkpoints/psd-combined-smallbank-old1000-sft3-dp4-5epoch-20260923-v1") + "/"
        )
    ):
        raise ValueError("combined formal training is not attested and complete")
    receipt = agent.load(TRAIN_OWNER)
    if live_process(int(receipt["pid"]), "run_psd_combined_training.py"):
        raise RuntimeError("formal training owner still owns the GPUs")
    return result


def frozen_cases() -> int:
    cases = [str(row["case_id"]) for row in agent.rows(agent.BENCHMARK)]
    if len(cases) != agent.EXPECTED_RUNNABLE or len(set(cases)) != len(cases):
        raise ValueError("frozen benchmark is not 1526 unique runnable cases")
    for path in (agent.MANIFEST, agent.PRIVATE_GOLD):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not set(agent.SMOKE_CASES) <= set(cases):
        raise ValueError("protocol smoke cases are not in the frozen cohort")
    return len(cases)


def candidate_command(original: list[str], merged_root: Path) -> list[str]:
    if len(original) < 4 or original[2] != "serve":
        raise ValueError("unrecognized original vLLM command")
    if Path(original[3]).resolve() != SFT_ROOT.resolve():
        raise ValueError("original replica is not frozen SFT3")
    if "--served-model-name" not in original:
        raise ValueError("original replica has no model identity")
    if original[original.index("--served-model-name") + 1] != agent.MODEL_ALIAS:
        raise ValueError("original replica has an unexpected internal alias")
    result = cache_on_command(original)
    result[3] = str(merged_root.resolve())
    if "--enable-prefix-caching" not in result:
        raise AssertionError("candidate does not enable prefix cache")
    if result[result.index("--mamba-cache-mode") + 1] != "align":
        raise AssertionError("candidate does not use aligned Mamba cache")
    return result


def candidate_environment(original: dict[str, str], deploy: Path, index: int) -> dict[str, str]:
    if index not in range(4):
        raise ValueError("replica index must be 0..3")
    environment = dict(original)
    environment["VLLM_CACHE_ROOT"] = str(deploy / f"aot-cache-replica-{index}")
    environment["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"
    return environment


def gateway_environment(original: dict[str, str]) -> dict[str, str]:
    environment = dict(original)
    previous = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(GATEWAY_CODE) + (":" + previous if previous else "")
    environment.update(
        PSD_PUBLIC_MODEL_ALIAS=PROFILE_MODEL,
        IFV_PREFIX_CACHE_MODE="case_isolated",
        IFV_PREFIX_CACHE_BLOCK_SIZE=str(cache_gate.BLOCK_SIZE),
        IFV_PREFIX_CACHE_UNSAFE_WINDOW=str(cache_gate.UNSAFE_WINDOW),
    )
    return environment


def attest_new_cache(verdict: dict[str, Any]) -> None:
    if verdict.get("passed") is not True:
        raise RuntimeError("new model hybrid prefix-cache probe failed")
    if verdict.get("block_size") != 528 or verdict.get("unsafe_window") != 16:
        raise RuntimeError("new model prefix-cache geometry differs from the safe gate")
    rows = verdict.get("boundary_rows") or []
    if len(rows) < 17 or not all(row.get("cache_behavior_passed") for row in rows):
        raise RuntimeError("new model cache boundary gate is incomplete")
    multimodal = verdict.get("incremental_multimodal") or {}
    stress = verdict.get("stress") or {}
    if multimodal.get("cache_hit_observed") is not True:
        raise RuntimeError("new model did not demonstrate a real multimodal cache hit")
    if stress.get("corrupted_delta") != 0 or stress.get("errors") or stress.get("healthy") != 32:
        raise RuntimeError("new model stress gate detected corruption or missing traffic")


def merge_model(adapter: Path, deploy: Path) -> None:
    script = CODE / "training/scripts/h20/merge_lora_for_serving.py"
    command = [
        sys.executable, str(script), "--base-model", str(SFT_ROOT),
        "--adapter", str(adapter), "--output", str(MERGED_ROOT), "--no-large-hashes",
    ]
    with (deploy / "merge.log").open("xb") as log:
        subprocess.run(command, cwd=CODE, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    record = agent.load(MERGED_ROOT / "merge-export.json")
    if (
        record.get("passed") is not True
        or record.get("large_payload_hashing") is not False
        or Path(record["source"]["base_model"]).resolve() != SFT_ROOT.resolve()
        or Path(record["source"]["adapter"]).resolve() != adapter.resolve()
        or record.get("adapter_parameter_count", 0) <= 0
        or not (MERGED_ROOT / "config.json").is_file()
    ):
        raise RuntimeError("merged model source binding failed")
    agent.atomic_json(deploy / "merge-binding.json", {
        "merged_model": agent.stat_identity(MERGED_ROOT / "config.json"),
        "adapter": agent.stat_identity(adapter / "adapter_model.safetensors"),
        "base": agent.stat_identity(SFT_ROOT / "config.json"),
        "large_payload_hashing": False,
    })


def probe_new_service(deploy: Path, environment: dict[str, str]) -> dict[str, Any]:
    output = deploy / "probe"
    command = [
        str(ROOT / "envs/h20-qwen35-vllm-0181/bin/python"),
        str(GATEWAY_CODE / "scripts/server/probe_prefix_cache_hybrid_safety.py"),
        "--model", PROFILE_MODEL, "--output", str(output),
        "--image", str(probe_image()), "--block-size", str(cache_gate.BLOCK_SIZE),
        "--unsafe-window", str(cache_gate.UNSAFE_WINDOW),
    ]
    with (deploy / "probe.log").open("xb") as log:
        subprocess.run(command, cwd=GATEWAY_CODE, env=environment,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    verdict = agent.load(output / "verdict.json")
    attest_new_cache(verdict)
    return verdict


def preflight() -> tuple[dict[str, Any], Any, list[dict[str, Any]], list[dict[str, str]]]:
    result = training_result()
    frozen_cases()
    cache_gate.validate_source_snapshot(GATEWAY_CODE)
    if cache_gate.active_agent_processes():
        raise RuntimeError("another Agent inference owner is active")
    agent.wait_for_idle_gateway()
    owner = agent.owner_module()
    guard = owner.load(agent.SERVICE / "guard.json")
    owner.checked(guard)
    originals = [owner.load(agent.SERVICE / f"replica-{i}.json") for i in range(4)]
    environments = [owner.checked(row) for row in originals]
    for row in originals:
        candidate_command(list(row["command"]), MERGED_ROOT)
    agent.wait_for_services(SFT_ROOT)
    return result, owner, originals, environments


def execute() -> None:
    result, owner, originals, environments = preflight()
    if DEPLOY.exists() or EVAL_OUTPUT.exists():
        raise FileExistsError("eval owner or output already exists; inspect before resuming")
    DEPLOY.mkdir(parents=True, exist_ok=False)
    agent.atomic_json(DEPLOY / "process.json", {
        "pid": os.getpid(), "command": [sys.executable, *sys.argv], "started_unix": time.time(),
    })
    agent.atomic_json(DEPLOY / "state.json", {"phase": "merging", "training": str(TRAIN / "result.json")})
    merge_model(Path(result["adapter"]), DEPLOY)

    guard = owner.load(agent.SERVICE / "guard.json")
    owner.checked(guard)
    gateway_pid, original_gateway_command, original_gateway_env, original_gateway_cwd = agent._gateway_process()
    original_gateway_receipt = {
        "pid": gateway_pid,
        "pgid": os.getpgid(gateway_pid),
        "command": original_gateway_command,
        "observed_unix": time.time(),
    }
    for index, row in enumerate(originals):
        agent.atomic_json(DEPLOY / f"original-replica-{index}.json", row)
    agent.atomic_json(DEPLOY / "original-gateway.json", original_gateway_receipt)

    guard_paused = False
    original_gateway_stop_attempted = False
    originals_stop_attempted = False
    candidates: list[dict[str, Any]] = []
    new_gateway: dict[str, Any] | None = None
    restore_errors: list[dict[str, str]] = []
    try:
        agent.atomic_json(DEPLOY / "state.json", {"phase": "switching_to_new_model"})
        os.kill(int(guard["pid"]), signal.SIGSTOP)
        guard_paused = True
        original_gateway_stop_attempted = True
        cache_gate.stop_gateway(gateway_pid)
        originals_stop_attempted = True
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(owner.stop, originals))
        logs: list[Path] = []
        for index, (receipt, environment) in enumerate(zip(originals, environments)):
            cache_dir = DEPLOY / f"aot-cache-replica-{index}"
            cache_dir.mkdir(parents=True, exist_ok=False)
            command = candidate_command(list(receipt["command"]), MERGED_ROOT)
            log = DEPLOY / f"new-replica-{index}.log"
            started = owner.spawn(command, candidate_environment(environment, DEPLOY, index), log)
            candidates.append(started)
            logs.append(log)
            owner.save(DEPLOY / f"new-replica-{index}.json", started)
        cache_gate.wait_replicas(MERGED_ROOT)
        cache_gate.attest_block_size(logs)
        gw_env = gateway_environment(original_gateway_env)
        gw_command = cache_gate.canary_gateway_command(original_gateway_command, GATEWAY_CODE)
        new_gateway = cache_gate.start_gateway(
            command=gw_command, environment=gw_env, cwd=str(GATEWAY_CODE),
            log_path=DEPLOY / "new-gateway.log",
        )
        owner.save(DEPLOY / "new-gateway.json", new_gateway)
        cache_gate.wait_canary_gateway()
        agent.wait_for_services(MERGED_ROOT)
        verdict = probe_new_service(DEPLOY, gw_env)
        for index, started in enumerate(candidates):
            owner.checked(started)
            owner.save(agent.SERVICE / f"replica-{index}.json", started)
        owner.save(agent.SERVICE / "gateway.json", new_gateway)
        agent.atomic_json(DEPLOY / "state.json", {
            "phase": "new_model_cache_gate_passed", "model": str(MERGED_ROOT),
            "cache_verdict": str(DEPLOY / "probe/verdict.json"),
            "case_isolated": True, "guard_paused": True,
        })
        inference = agent.evaluate_model(
            deploy=DEPLOY, source_code=CODE, key="sft3-psd-combined",
            profile_model=PROFILE_MODEL, model_root=MERGED_ROOT, output=EVAL_OUTPUT,
        )
        agent.atomic_json(DEPLOY / "state.json", {
            "phase": "inference_complete_restoring_sft3", "inference": inference,
            "cache_verdict": str(DEPLOY / "probe/verdict.json"),
        })
    except BaseException as error:
        state = agent.load(DEPLOY / "state.json")
        state.update(
            phase="failed_restoring_sft3", error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        agent.atomic_json(DEPLOY / "state.json", state)
        raise
    finally:
        if new_gateway is not None:
            try:
                cache_gate.stop_gateway(int(new_gateway["pid"]))
            except BaseException as error:
                restore_errors.append({"stage": "stop_new_gateway", "error": type(error).__name__})
        if candidates:
            try:
                with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
                    list(pool.map(owner.stop, candidates))
            except BaseException as error:
                restore_errors.append({"stage": "stop_new_replicas", "error": type(error).__name__})
        if originals_stop_attempted:
            for index, (original, environment) in enumerate(zip(originals, environments)):
                try:
                    try:
                        owner.checked(original)
                        restored = original
                    except (FileNotFoundError, ProcessLookupError, AssertionError):
                        restored = owner.spawn(original["command"], environment,
                                               DEPLOY / f"restore-sft-{index}.log")
                    owner.save(agent.SERVICE / f"replica-{index}.json", restored)
                except BaseException as error:
                    restore_errors.append({"stage": f"restore_sft_{index}", "error": type(error).__name__})
        if original_gateway_stop_attempted:
            try:
                if live_process(gateway_pid, "scripts.server.psd_qwen_gateway:app"):
                    owner.save(agent.SERVICE / "gateway.json", original_gateway_receipt)
                else:
                    restarted = cache_gate.start_gateway(
                        command=original_gateway_command, environment=original_gateway_env,
                        cwd=original_gateway_cwd, log_path=DEPLOY / "restore-gateway.log",
                    )
                    owner.save(agent.SERVICE / "gateway.json", restarted)
            except BaseException as error:
                restore_errors.append({"stage": "restore_gateway", "error": type(error).__name__})
        if originals_stop_attempted and not restore_errors:
            try:
                agent.wait_for_services(SFT_ROOT)
            except BaseException as error:
                restore_errors.append({"stage": "verify_sft_service", "error": type(error).__name__})
        if guard_paused:
            try:
                os.kill(int(guard["pid"]), signal.SIGCONT)
            except BaseException as error:
                restore_errors.append({"stage": "resume_guard", "error": type(error).__name__})
        state_path = DEPLOY / "state.json"
        state = agent.load(state_path) if state_path.is_file() else {}
        state["sft_service_restored"] = originals_stop_attempted and not restore_errors
        state["guard_resumed"] = guard_paused and not any(
            row["stage"] == "resume_guard" for row in restore_errors
        )
        state["restore_errors"] = restore_errors
        if restore_errors:
            state["phase"] = "failed_requires_service_repair"
        elif state.get("phase") == "inference_complete_restoring_sft3":
            state["phase"] = "inference_complete_sft3_restored"
        agent.atomic_json(state_path, state)
        if restore_errors:
            raise RuntimeError("new-model evaluation service restoration failed")


def launch() -> None:
    preflight()
    if LAUNCH.exists() or DEPLOY.exists() or EVAL_OUTPUT.exists():
        raise FileExistsError("combined full-eval owner already exists; inspect before relaunch")
    LAUNCH.mkdir(parents=True, exist_ok=False)
    owner = agent.owner_module()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "execute"]
    receipt = owner.spawn(command, os.environ.copy(), LAUNCH / "owner.log")
    owner.save(LAUNCH / "process.json", receipt)
    owner.save(LAUNCH / "state.json", {
        "phase": "launched", "training": str(TRAIN / "result.json"),
        "evaluation": str(EVAL_OUTPUT), "merged_model": str(MERGED_ROOT),
        "prefix_cache_gate_required": True,
    })
    print(json.dumps({"pid": receipt["pid"], "deploy": str(DEPLOY)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "launch", "execute"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "plan":
        cache_gate.validate_source_snapshot(GATEWAY_CODE)
        print(json.dumps({
            "phase": "waiting_for_formal_training", "runnable": frozen_cases(),
            "formal_denominator": agent.FORMAL_DENOMINATOR,
            "candidate": str(MERGED_ROOT), "prefix_cache": "new_model_gate_required",
        }))
    elif args.mode == "preflight":
        result, _, _, _ = preflight()
        print(json.dumps({
            "ready": True, "adapter": result["adapter"], "runnable": agent.EXPECTED_RUNNABLE,
            "formal_denominator": agent.FORMAL_DENOMINATOR,
            "candidate": str(MERGED_ROOT), "prefix_cache": "case_isolated_hybrid_gate_required",
        }))
    elif args.mode == "launch":
        launch()
    else:
        execute()


if __name__ == "__main__":
    main()
