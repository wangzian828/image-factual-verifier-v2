"""Resume the corrected three-policy Agent evaluation without resampling successes.

The first SFT3 inference wave was intentionally interrupted for a prefix-cache
correctness canary.  This owner preserves every durable success and its judge
receipt, resumes only unfinished SFT3 cases with the original sampling seed,
then continues the already-frozen PSD and Base profiles.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from scripts.server.control_corrected_agent_eval import (
    BENCHMARK,
    EXPECTED_RUNNABLE,
    FORMAL_DENOMINATOR,
    ROOT,
    SMOKE_CASES,
    ServiceSession,
    atomic_json,
    evaluate_model,
    load,
    rows,
    run_attempt,
    restart_gateway,
    runtime_environment,
    successful,
    wait_for_services,
)


SFT_OUTPUT = (
    ROOT / "evaluation/qwen35-sft3084-agent-full1526-selfextract-20260921-v3"
)
SFT_MODEL_ROOT = (
    ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model"
)
SFT_PROFILE_MODEL = "ifv-qwen3.5-9b-sft3084-selfextract"


def process_command(pid: int) -> list[str] | None:
    try:
        return [
            part.decode(errors="surrogateescape")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def matching_judges(output: Path) -> list[int]:
    expected = str((output / "judge-gemini37-stream-v1").resolve())
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        command = process_command(int(entry.name))
        if not command or not any(
            value.endswith("stream_agent_judges.py") for value in command
        ):
            continue
        if "--output" not in command:
            continue
        actual = str(Path(command[command.index("--output") + 1]).resolve())
        if actual == expected:
            matches.append(int(entry.name))
    return sorted(matches)


def validate_judge_ownership(*, durable_successes: int, judges: list[int]) -> None:
    """Require live streaming ownership only while inference is unfinished."""

    if durable_successes < EXPECTED_RUNNABLE:
        if len(judges) != 1:
            raise RuntimeError(f"expected one preserved SFT judge, found {judges}")
        return
    if len(judges) > 1:
        raise RuntimeError(f"expected at most one completed-SFT judge, found {judges}")


def validate_resume_binding(output: Path) -> dict[str, Any]:
    binding = load(output / "binding.json")
    required = {
        "schema_version": "ifv-corrected-self-extract-agent-eval-v1",
        "key": "sft3",
        "profile_model": SFT_PROFILE_MODEL,
        "runnable_cases": EXPECTED_RUNNABLE,
        "formal_denominator": FORMAL_DENOMINATOR,
        "page_extract_provider": "qwen_local",
        "page_extract_model": SFT_PROFILE_MODEL,
        "page_extract_thinking_enabled": True,
        "page_extract_thinking_token_budget": 2048,
        "large_payload_hashing": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": binding.get(key)}
        for key, expected in required.items()
        if binding.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"resume binding mismatch: {mismatches}")
    for name in ("smoke-engineering-audit.json", "smoke-extraction-protocol-audit.json"):
        audit = load(output / name)
        if audit.get("passed") is not True:
            raise ValueError(f"resume requires passed {name}")
    return binding


def available_wave_name(output: Path, stem: str) -> str:
    """Choose a fresh ledger name without discarding an interrupted wave."""

    candidate = stem
    suffix = 2
    while (output / candidate).exists() or (output / f"{candidate}-cases.txt").exists():
        candidate = f"{stem}-{suffix}"
        suffix += 1
    return candidate


def validate_safe_cache_off(session: ServiceSession) -> None:
    cache_mode = os.environ.get("IFV_PREFIX_CACHE_MODE", "request_isolated").strip()
    for index, receipt in enumerate(session.originals):
        command = receipt.get("command") or []
        if "--mamba-cache-mode" not in command:
            raise ValueError(f"replica {index} has no mamba cache mode")
        mode = command[command.index("--mamba-cache-mode") + 1]
        if cache_mode == "case_isolated":
            if "--enable-prefix-caching" not in command or mode != "align":
                raise ValueError(
                    f"replica {index} is not validated cache-on/align"
                )
        elif cache_mode == "request_isolated":
            if "--no-enable-prefix-caching" not in command:
                raise ValueError(f"replica {index} is not explicitly cache-off")
            if mode != "none":
                raise ValueError(f"replica {index} mamba cache mode is {mode}")
        else:
            raise ValueError(f"unsupported IFV_PREFIX_CACHE_MODE={cache_mode!r}")


def resume_sft(
    *, deploy: Path, source_code: Path, output: Path
) -> dict[str, Any]:
    validate_resume_binding(output)
    expected = [str(row["case_id"]) for row in rows(BENCHMARK)]
    if len(expected) != EXPECTED_RUNNABLE or len(set(expected)) != EXPECTED_RUNNABLE:
        raise ValueError("frozen runnable cohort is not 1526 unique cases")
    environment = runtime_environment(source_code, SFT_PROFILE_MODEL)
    initial = successful(output)
    judges = matching_judges(output)
    validate_judge_ownership(durable_successes=len(initial), judges=judges)
    if not set(SMOKE_CASES) <= set(initial):
        raise ValueError("preserved output lost a protocol smoke success")
    if len(initial) < len(SMOKE_CASES):
        raise ValueError("preserved output contains too few successes")
    atomic_json(
        deploy / "resume-binding.json",
        {
            "schema_version": "ifv-corrected-agent-resume-v1",
            "output": str(output),
            "durable_successes_before_resume": len(initial),
            "judge_pid": judges[0] if judges else None,
            "judge_required_for_resume": len(initial) < EXPECTED_RUNNABLE,
            "original_attempt": "attempt-0",
            "original_base_sampling_seed": 2903,
            "large_payload_hashing": False,
        },
    )
    waves = (
        ("attempt-0-resume", 28, 2903),
        ("attempt-1", 20, 3903),
        ("attempt-2", 12, 4903),
        ("attempt-3", 8, 5903),
    )
    for stem, concurrency, seed in waves:
        selected = successful(output)
        pending = [case for case in expected if case not in selected]
        if not pending:
            break
        name = available_wave_name(output, stem)
        run_attempt(
            deploy=deploy,
            source_code=source_code,
            output=output,
            name=name,
            cases=pending,
            concurrency=concurrency,
            base_seed=seed,
            environment=environment,
        )
    selected = successful(output)
    missing = [case for case in expected if case not in selected]
    result = {
        "schema_version": "ifv-corrected-self-extract-inference-v1",
        "phase": "inference_complete" if not missing else "engineering_retry_budget_exhausted",
        "key": "sft3",
        "profile_model": SFT_PROFILE_MODEL,
        "success": len(selected),
        "expected_runnable": EXPECTED_RUNNABLE,
        "formal_denominator": FORMAL_DENOMINATOR,
        "failures_retained_in_denominator": True,
        "remaining": missing,
        "judge_streaming_concurrently": True,
        "resumed_successes_without_resampling": len(initial),
        "large_payload_hashing": False,
    }
    atomic_json(output / "inference-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    deploy = args.deploy.resolve()
    source_code = args.source_code.resolve()
    deploy.mkdir(parents=True, exist_ok=True)
    atomic_json(
        deploy / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "command": [sys.executable, *sys.argv],
            "started_unix": time.time(),
        },
    )
    profiles = (
        (
            "sft3-psd",
            "ifv-qwen3.5-9b-sft3084-psd-smallbank4095-selfextract",
            ROOT / "exports/qwen35-psd-smallbank4095-merged-20260921-v1/model",
            ROOT / "evaluation/qwen35-psd-smallbank4095-agent-full1526-selfextract-20260921-v3",
        ),
        (
            "base",
            "ifv-qwen3.5-9b-base-selfextract",
            ROOT / "models/Qwen3.5-9B-local",
            ROOT / "evaluation/qwen35-base-agent-full1526-selfextract-20260921-v3",
        ),
    )
    session = ServiceSession(deploy)
    completed: list[dict[str, Any]] = []
    try:
        # Keep maintenance traffic out of the formal serving window.  The
        # guard talks directly to replicas and therefore cannot participate in
        # the gateway's per-case cache domains or corruption quarantine.
        session.pause_guard()
        # The cache gate deliberately leaves its tested gateway active.
        # Refresh only that lightweight proxy from this immutable source
        # snapshot so transport and corruption-quarantine fixes apply without
        # restarting the four validated replicas or discarding their caches.
        restart_gateway(deploy, "resume-source", source_code=source_code)
        validate_safe_cache_off(session)
        wait_for_services(SFT_MODEL_ROOT)
        result = resume_sft(deploy=deploy, source_code=source_code, output=SFT_OUTPUT)
        completed.append(result)
        for key, profile_model, model_root, output in profiles:
            atomic_json(
                deploy / "state.json",
                {
                    "phase": "switching_model",
                    "key": key,
                    "completed_inference_profiles": [row["key"] for row in completed],
                },
            )
            session.switch(key, model_root)
            result = evaluate_model(
                deploy=deploy,
                source_code=source_code,
                key=key,
                profile_model=profile_model,
                model_root=model_root,
                output=output,
            )
            completed.append(result)
        atomic_json(
            deploy / "state.json",
            {
                "phase": "all_inference_complete_judges_continue_independently",
                "profiles": completed,
                "service_restored": False,
            },
        )
    finally:
        session.restore()
        state_path = deploy / "state.json"
        state = load(state_path) if state_path.is_file() else {}
        state["service_restored"] = True
        atomic_json(state_path, state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        deploy_arg = next(
            (
                Path(sys.argv[index + 1])
                for index, value in enumerate(sys.argv[:-1])
                if value == "--deploy"
            ),
            None,
        )
        if deploy_arg is not None:
            state_path = deploy_arg / "state.json"
            state = load(state_path) if state_path.is_file() else {}
            state.update(
                phase="failed_requires_fix",
                error_type=type(error).__name__,
                message=str(error)[:1000],
            )
            atomic_json(state_path, state)
        raise
