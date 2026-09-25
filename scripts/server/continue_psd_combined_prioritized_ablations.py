"""Safely finish the live no-web wave, then run evidence before retrieval.

The old v42 parent is intentionally stopped while its no-web attempt-0 child
continues.  This handoff preserves every durable case result, the original
retry budgets/seeds and the immutable v4 output directories.  It never
resumes the parent into its obsolete ablation order.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from scripts.server import control_corrected_agent_eval as agent
from scripts.server import run_psd_combined_cached_full_eval as previous
from scripts.server import run_psd_combined_cachefixed_agent_eval as cachefixed


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-combined-prioritized-ablations-20260925-v1"
LAUNCH = ROOT / "training-artifacts/psd-combined-prioritized-ablations-launch-20260925-v1"
OLD_PARENT = (2604100, 719379169)
OLD_CHILD = (2624116, 721010854)
ORDER = ("no-web-search", "no-evidence-inspection", "no-image-retrieval")
CONFIG = {name: (flag, config) for name, flag, config in previous.ABLATIONS}


def process_identity(pid: int) -> dict | None:
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().split()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[2] == "Z":
        return None
    return {"pid": pid, "state": fields[2], "ppid": int(fields[3]),
            "startticks": int(fields[21]), "command": command}


def attest_parent_child(*, child_required: bool) -> tuple[dict, dict | None]:
    parent = process_identity(OLD_PARENT[0])
    child = process_identity(OLD_CHILD[0])
    if (parent is None or parent["startticks"] != OLD_PARENT[1]
            or parent["state"] != "T"
            or "run_psd_combined_cachefixed_agent_eval.py execute" not in parent["command"]):
        raise RuntimeError("old v42 owner is not the verified stopped parent")
    if child_required and child is None:
        raise RuntimeError("no-web attempt-0 child vanished before handoff")
    if child is not None and (
        child["startticks"] != OLD_CHILD[1]
        or child["ppid"] != OLD_PARENT[0]
        or str(cachefixed.output_for("no-web-search") / "attempt-0") not in child["command"]
        or "src.eval.run_cases" not in child["command"]
    ):
        raise RuntimeError("no-web attempt-0 child identity changed")
    return parent, child


def no_web_attempt_complete() -> bool:
    output = cachefixed.output_for("no-web-search")
    manifest = output / "attempt-0/run_manifest.json"
    if not manifest.is_file() or not (output / "attempt-0/summary.json").is_file():
        return False
    status = agent.load(manifest).get("status")
    if status not in {"completed", "completed_with_errors"}:
        return False
    previous.attest_ablation_run_manifest(manifest, CONFIG["no-web-search"][1])
    return True


def retire_old_parent() -> None:
    parent, child = attest_parent_child(child_required=False)
    if child is not None or not no_web_attempt_complete():
        raise RuntimeError("cannot retire old parent while no-web attempt-0 is active")
    # The stopped parent would launch no-image next.  Terminating it while
    # stopped prevents any obsolete-order instruction from executing.
    os.kill(OLD_PARENT[0], signal.SIGKILL)
    for _ in range(50):
        if process_identity(OLD_PARENT[0]) is None:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("old parent did not exit after verified retirement")
    agent.atomic_json(DEPLOY / "handoff.json", {
        "old_parent": parent, "old_child": list(OLD_CHILD),
        "attempt_0_manifest": str(cachefixed.output_for("no-web-search") / "attempt-0/run_manifest.json"),
        "retired_unresumed": True, "old_outputs_preserved": True,
        "time_unix": time.time(),
    })


def finish_no_web() -> dict:
    name = "no-web-search"
    flag, config = CONFIG[name]
    output = cachefixed.output_for(name)
    smoke = agent.load(output / "smoke-tool-ablation-audit.json")
    if smoke.get("passed") is not True or smoke.get("tool_families") != config.to_manifest():
        raise RuntimeError("no-web smoke attestation failed")
    previous.attest_ablation_run_manifest(output / "attempt-0/run_manifest.json", config)
    expected = [str(row["case_id"]) for row in agent.rows(agent.BENCHMARK)]
    if len(expected) != agent.EXPECTED_RUNNABLE or len(set(expected)) != len(expected):
        raise RuntimeError("frozen cohort changed")
    environment = agent.runtime_environment(CODE, previous.PROFILE_MODEL,
                                            prefix_cache_mode="case_isolated")
    for attempt, concurrency in enumerate((28, 20, 12, 8)):
        selected = agent.successful(output)
        if set(selected) - set(expected):
            raise RuntimeError("no-web output contains an unexpected case")
        pending = [case for case in expected if case not in selected]
        if not pending:
            break
        label = f"attempt-{attempt}"
        if attempt == 0:
            # The original child already completed this wave in its own process.
            # Continue only with the still-missing cases in attempt-1.
            continue
        if (output / label).exists() or (output / f"{label}-cases.txt").exists():
            raise RuntimeError(f"retry output already exists: {label}")
        agent.atomic_json(DEPLOY / "state.json", {"phase": "no_web_retry",
                     "attempt": attempt, "pending": len(pending)})
        agent.run_attempt(deploy=DEPLOY, source_code=CODE, output=output,
                          name=label, cases=pending, concurrency=concurrency,
                          base_seed=2903 + attempt * 1000, environment=environment,
                          tool_ablation_flags=(flag,))
        previous.attest_ablation_run_manifest(output / label / "run_manifest.json", config)
    selected = agent.successful(output)
    missing = [case for case in expected if case not in selected]
    summary = {"schema_version": "ifv-corrected-self-extract-tool-ablation-summary-v1",
               "phase": "inference_complete" if not missing else "engineering_retry_budget_exhausted",
               "variant": name, "tool_families": config.to_manifest(),
               "profile_model": previous.PROFILE_MODEL, "success": len(selected),
               "expected_runnable": agent.EXPECTED_RUNNABLE,
               "formal_denominator": agent.FORMAL_DENOMINATOR,
               "failures_retained_in_denominator": True, "remaining": missing,
               "judge_streaming_concurrently": True, "large_payload_hashing": False,
               "priority_handoff": str(DEPLOY / "handoff.json")}
    if (output / "inference-summary.json").exists():
        raise FileExistsError(output / "inference-summary.json")
    agent.atomic_json(output / "inference-summary.json", summary)
    return summary


def execute() -> None:
    try:
        parent, child = attest_parent_child(child_required=False)
        agent.atomic_json(DEPLOY / "state.json", {"phase": "waiting_for_no_web_attempt_0",
                          "parent_pid": parent["pid"], "child_pid": child["pid"] if child else None})
        while child is not None:
            time.sleep(15)
            _, child = attest_parent_child(child_required=False)
        if not no_web_attempt_complete():
            raise RuntimeError("no-web child exited without complete attempt-0 receipts")
        retire_old_parent()
        previous.agent.wait_for_services(previous.MERGED_ROOT)
        previous.attest_new_gateway_health(previous.cache_gate.wait_canary_gateway())
        no_web = finish_no_web()
        if no_web["phase"] != "inference_complete" or no_web["success"] != agent.EXPECTED_RUNNABLE:
            raise RuntimeError("no-web did not close all frozen runnable cases")
        results = {"no-web-search": no_web}
        for name in ORDER[1:]:
            flag, config = CONFIG[name]
            agent.atomic_json(DEPLOY / "state.json", {"phase": "ablation_agent",
                              "variant": name, "no_web_success": no_web["success"]})
            result = previous.evaluate_ablation(
                deploy=DEPLOY / name, output=cachefixed.output_for(name),
                name=name, flag=flag, config=config, launch_stream_judge=False)
            if result["phase"] != "inference_complete" or result["success"] != agent.EXPECTED_RUNNABLE:
                raise RuntimeError(f"{name} did not close all frozen runnable cases")
            results[name] = result
        previous.agent.wait_for_services(previous.MERGED_ROOT)
        previous.attest_new_gateway_health(previous.cache_gate.wait_canary_gateway())
        agent.atomic_json(DEPLOY / "result.json", {"passed_agent": True,
                          "ablations": results, "new_model_service_retained": True,
                          "judge_pending": True})
        agent.atomic_json(DEPLOY / "state.json", {"phase": "agent_complete_judge_pending"})
    except BaseException as exc:
        agent.atomic_json(DEPLOY / "state.json", {"phase": "held_requires_inspection",
                          "error_type": type(exc).__name__, "error": str(exc)[:800],
                          "new_model_service_retained": True})
        raise


def launch() -> None:
    if ORDER != ("no-web-search", "no-evidence-inspection", "no-image-retrieval"):
        raise RuntimeError("ablation priority order changed")
    attest_parent_child(child_required=True)
    if DEPLOY.exists() or LAUNCH.exists() or any(
        cachefixed.output_for(name).exists() for name in ORDER[1:]
    ):
        raise FileExistsError("prioritized handoff was already launched")
    if not cachefixed.output_for("no-web-search").is_dir():
        raise FileNotFoundError("no-web output is missing")
    LAUNCH.mkdir(parents=True, exist_ok=False)
    DEPLOY.mkdir(parents=True, exist_ok=False)
    receipt = agent.owner_module().spawn(
        [sys.executable, "-u", str(Path(__file__).resolve()), "execute"],
        {**os.environ, "PYTHONPATH": str(CODE) + ":" + str(CODE / "training")},
        LAUNCH / "owner.log")
    agent.owner_module().save(LAUNCH / "process.json", receipt)
    print(receipt["pid"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "execute"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "launch":
        launch()
    else:
        execute()


if __name__ == "__main__":
    main()
