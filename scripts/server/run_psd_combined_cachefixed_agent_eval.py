"""Run the frozen combined-PSD Agent cohort on the already-gated new service.

The v3 output is retained as evidence of cross-image perception-cache reuse.
This owner creates fresh v4 outputs and never switches the serving weights.
Gemini judging is deferred so provider availability cannot block Agent work.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from scripts.server import control_corrected_agent_eval as agent
from scripts.server import run_psd_combined_cached_full_eval as previous


ROOT = Path("/volume/ybo/wza")
SOURCE_DEPLOY = ROOT / "training-artifacts/psd-combined-cached-full-eval-20260924-v3"
SOURCE_OUTPUT = ROOT / "evaluation/qwen35-psd-combined-agent-full1526-selfextract-20260924-v3"
PERCEPTION_PROBE = ROOT / "training-artifacts/psd-combined-perception-probe-20260924-v1/result.json"
DEPLOY = ROOT / "training-artifacts/psd-combined-cachefixed-agent-eval-20260924-v4"
LAUNCH = ROOT / "training-artifacts/psd-combined-cachefixed-agent-eval-launch-20260924-v4"
FULL_OUTPUT = ROOT / "evaluation/qwen35-psd-combined-agent-full1526-selfextract-20260924-v4"


def output_for(name: str) -> Path:
    return FULL_OUTPUT.with_name(FULL_OUTPUT.name + "-" + name)


def preflight() -> dict:
    previous.training_result()
    previous.frozen_cases()
    if not SOURCE_OUTPUT.is_dir() or not SOURCE_DEPLOY.is_dir():
        raise RuntimeError("contaminated source evidence must remain present")
    probe = agent.load(PERCEPTION_PROBE)
    if (
        probe.get("different_results") is not True
        or [row.get("status") for row in probe.get("records", [])] != ["success", "success"]
    ):
        raise RuntimeError("cross-image cache-free perception probe did not pass")
    environment = agent.runtime_environment(
        CODE, previous.PROFILE_MODEL, prefix_cache_mode="case_isolated"
    )
    if environment["TOOL_CACHE_ENABLED"] != "0" or environment["PERCEPTION_CACHE_ENABLED"] != "0":
        raise RuntimeError("both independent tool caches must be disabled")
    if previous.live_process(2567752, "run_psd_combined_cached_full_eval.py"):
        raise RuntimeError("contaminated evaluation owner still exists")
    for pid in (2573275, 2598025, 2598192, 2601223):
        if (Path("/proc") / str(pid) / "cmdline").exists() and (
            Path("/proc") / str(pid) / "cmdline"
        ).read_bytes():
            raise RuntimeError(f"contaminated owner child still exists: {pid}")
    previous.agent.wait_for_services(previous.MERGED_ROOT)
    previous.attest_new_cache(agent.load(SOURCE_DEPLOY / "probe/verdict.json"))
    previous.attest_new_gateway_health(previous.cache_gate.wait_canary_gateway())
    return {
        "model": str(previous.MERGED_ROOT),
        "cache_gate": str(SOURCE_DEPLOY / "probe/verdict.json"),
        "perception_probe": str(PERCEPTION_PROBE),
        "perception_cache_enabled": False,
        "tool_cache_enabled": False,
        "judge_submission": "deferred_for_provider_recovery",
        "runnable": agent.EXPECTED_RUNNABLE,
        "formal_denominator": agent.FORMAL_DENOMINATOR,
    }


def execute() -> None:
    binding = preflight()
    if DEPLOY.exists() or FULL_OUTPUT.exists() or any(
        output_for(name).exists() for name, _, _ in previous.ABLATIONS
    ):
        raise FileExistsError("cachefixed v4 owner or output already exists")
    DEPLOY.mkdir(parents=True, exist_ok=False)
    agent.atomic_json(DEPLOY / "binding.json", binding)
    agent.atomic_json(DEPLOY / "process.json", {
        "pid": os.getpid(), "command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    })
    try:
        agent.atomic_json(DEPLOY / "state.json", {"phase": "full_agent"})
        full = agent.evaluate_model(
            deploy=DEPLOY, source_code=CODE, key="sft3-psd-combined-cachefixed",
            profile_model=previous.PROFILE_MODEL,
            model_root=previous.MERGED_ROOT, output=FULL_OUTPUT,
            prefix_cache_mode="case_isolated", launch_stream_judge=False,
        )
        if full.get("phase") != "inference_complete" or full.get("success") != agent.EXPECTED_RUNNABLE:
            raise RuntimeError("Full Agent did not close all frozen runnable cases")
        variants = {}
        for name, flag, config in previous.ABLATIONS:
            agent.atomic_json(DEPLOY / "state.json", {
                "phase": "ablation_agent", "variant": name,
                "full_success": full["success"],
            })
            result = previous.evaluate_ablation(
                deploy=DEPLOY / name, output=output_for(name),
                name=name, flag=flag, config=config,
                launch_stream_judge=False,
            )
            if result.get("phase") != "inference_complete" or result.get("success") != agent.EXPECTED_RUNNABLE:
                raise RuntimeError(f"{name} Agent did not close all frozen runnable cases")
            variants[name] = result
        previous.agent.wait_for_services(previous.MERGED_ROOT)
        previous.attest_new_gateway_health(previous.cache_gate.wait_canary_gateway())
        agent.atomic_json(DEPLOY / "result.json", {
            "passed_agent": True, "full": full, "ablations": variants,
            "new_model_service_retained": True,
            "judge_pending": True,
        })
        agent.atomic_json(DEPLOY / "state.json", {"phase": "agent_complete_judge_pending"})
    except BaseException as exc:
        agent.atomic_json(DEPLOY / "state.json", {
            "phase": "held_requires_inspection", "error_type": type(exc).__name__,
            "error": str(exc)[:800], "new_model_service_retained": True,
        })
        raise


def launch() -> None:
    preflight()
    if LAUNCH.exists() or DEPLOY.exists() or FULL_OUTPUT.exists() or any(
        output_for(name).exists() for name, _, _ in previous.ABLATIONS
    ):
        raise FileExistsError("cachefixed v4 owner already launched")
    LAUNCH.mkdir(parents=True, exist_ok=False)
    receipt = agent.owner_module().spawn(
        [sys.executable, "-u", str(Path(__file__).resolve()), "execute"],
        {**os.environ, "PYTHONPATH": str(CODE) + ":" + str(CODE / "training")},
        LAUNCH / "owner.log",
    )
    agent.owner_module().save(LAUNCH / "process.json", receipt)
    agent.owner_module().save(LAUNCH / "state.json", {
        "phase": "launched", "output": str(FULL_OUTPUT),
        "new_model_service_retained": True,
    })
    print(receipt["pid"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "launch", "execute"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "preflight":
        print(preflight())
    elif args.mode == "launch":
        launch()
    else:
        execute()


if __name__ == "__main__":
    main()
