"""Validate PSD training, run an engineering smoke, then the frozen full Agent set.

This controller is intentionally inference-only.  It does not inspect labels or
run a judge, and it never selects cases from outcomes.  Smoke successes are
reused in the full cohort; later attempts contain only cases without a prior
terminal success.  The pre-PSD serving stack is restored before this process
exits, whether evaluation succeeds or fails.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-posttrain-full-eval-20260921-v107"
SOURCE_CODE = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v91/code"
TRAIN_STATE = ROOT / "training-artifacts/psd-lightweight-recovery-20260920-v91/training-chain-state.json"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
BENCHMARK = ROOT / "evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl"
SMOKE_CASES = ROOT / "runs/eval/qwen35-sft1028-agent-canary4-20260914/target-case-list.txt"
OUTPUT = ROOT / "evaluation/qwen35-psd-smallbank4095-agent-full1526-20260921-v1"
MODEL_ALIAS = "ifv-psd-sft3084"
PORTS = (19002, 19003, 19004, 19005)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat_identity(path: Path) -> dict:
    value = path.stat()
    return {"path": str(path.resolve()), "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def wait_for_training() -> dict:
    state_path = DEPLOY / "state.json"
    while True:
        if not TRAIN_STATE.is_file():
            atomic_json(state_path, {"phase": "waiting_smallbank_training"})
            time.sleep(10)
            continue
        state = load(TRAIN_STATE)
        if state.get("phase") == "failed_requires_fix":
            raise RuntimeError("small-bank PSD training failed")
        if state.get("phase") == "formal_training_complete_requires_diagnostics":
            return state
        atomic_json(state_path, {"phase": "waiting_smallbank_training",
            "observed_training_phase": state.get("phase")})
        time.sleep(30)


def training_diagnostics(training_state: dict) -> dict:
    # Reuse the already tested lightweight diagnostics.  It reads small JSON
    # receipts and stats one adapter file; it does not hash model payloads.
    from scripts.server.control_psd_posttrain_next1000 import diagnostics
    report = diagnostics(training_state)
    report["controller"] = "posttrain_full_eval"
    return report


def _remove_option(command: list[str], option: str, *, takes_value: bool) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == option:
            index += 2 if takes_value else 1
            continue
        result.append(command[index])
        index += 1
    return result


def adapter_command(command: list[str], adapter: Path) -> list[str]:
    """Convert one frozen SFT replica command into a PSD-LoRA Agent replica."""
    result = list(command)
    if len(result) < 4 or result[2] != "serve" or "--served-model-name" not in result:
        raise ValueError("unrecognized vLLM replica command")
    for option, takes_value in (
        ("--tool-parser-plugin", True), ("--enable-lora", False),
        ("--max-lora-rank", True), ("--enable-tower-connector-lora", False),
        ("--lora-modules", True),
    ):
        result = _remove_option(result, option, takes_value=takes_value)
    result[result.index("--served-model-name") + 1] = MODEL_ALIAS + "-base"
    if "--tool-call-parser" in result:
        result[result.index("--tool-call-parser") + 1] = "qwen3_coder"
    else:
        result += ["--tool-call-parser", "qwen3_coder"]
    result += ["--enable-lora", "--max-lora-rank", "32",
        "--enable-tower-connector-lora", "--lora-modules", f"{MODEL_ALIAS}={adapter}"]
    return result


def _url_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_services(adapter: Path, timeout_seconds: int = 1200) -> None:
    deadline = time.monotonic() + timeout_seconds
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            for port in PORTS:
                cards = _url_json(f"http://127.0.0.1:{port}/v1/models")["data"]
                matches = [card for card in cards if card.get("id") == MODEL_ALIAS]
                if len(matches) != 1 or Path(matches[0].get("root", "")).resolve() != adapter.resolve():
                    raise ValueError(f"port {port} does not expose the bound PSD adapter")
            health = _url_json("http://127.0.0.1:19025/health")
            if len(health.get("replicas") or []) != 4:
                raise ValueError("gateway does not see four replicas")
            return
        except (OSError, KeyError, TypeError, ValueError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(10)
    raise RuntimeError(f"PSD adapter serving readiness timed out: {last}")


def wait_for_idle_gateway(timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            replicas = _url_json("http://127.0.0.1:19025/health")["replicas"]
            if len(replicas) == 4 and all(int(row.get("inflight", 0)) == 0 for row in replicas):
                return
        except (OSError, KeyError, TypeError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError("serving gateway did not become idle")


def owner_module():
    import importlib.util
    path = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_adapter_serving(adapter: Path) -> dict:
    owner = owner_module()
    guard = owner.load(SERVICE / "guard.json")
    owner.checked(guard)
    originals = [owner.load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in originals]
    wait_for_idle_gateway()
    os.kill(guard["pid"], signal.SIGSTOP)
    stopped: list[int] = []
    adapter_receipts: list[dict] = []
    try:
        def stop(index: int) -> None:
            owner.stop(originals[index])
            stopped.append(index)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stop, range(4)))
        for index in range(4):
            receipt = owner.spawn(adapter_command(originals[index]["command"], adapter),
                environments[index], DEPLOY / f"adapter-replica-{index}.log")
            adapter_receipts.append(receipt)
            owner.save(DEPLOY / f"adapter-replica-{index}.json", receipt)
        wait_for_services(adapter)
        return {"owner": owner, "guard": guard, "originals": originals,
            "environments": environments, "adapter_receipts": adapter_receipts}
    except Exception:
        for receipt in adapter_receipts:
            try:
                owner.stop(receipt)
            except Exception:
                pass
        for index in sorted(stopped):
            receipt = owner.spawn(originals[index]["command"], environments[index],
                DEPLOY / f"failed-start-restore-{index}.log")
            owner.save(SERVICE / f"replica-{index}.json", receipt)
        os.kill(guard["pid"], signal.SIGCONT)
        raise


def restore_original_serving(context: dict) -> None:
    owner = context["owner"]
    errors: list[dict] = []
    for receipt in context["adapter_receipts"]:
        try:
            owner.stop(receipt)
        except Exception as error:
            errors.append({"stage": "stop_adapter", "error": type(error).__name__})
    for index, (original, environment) in enumerate(zip(context["originals"], context["environments"])):
        try:
            receipt = owner.spawn(original["command"], environment,
                DEPLOY / f"restored-sft-replica-{index}.log")
            owner.save(SERVICE / f"replica-{index}.json", receipt)
        except Exception as error:
            errors.append({"stage": "restore_sft", "gpu": index, "error": type(error).__name__})
    os.kill(context["guard"]["pid"], signal.SIGCONT)
    if errors:
        atomic_json(DEPLOY / "restore-errors.json", errors)
        raise RuntimeError("serving restoration failed")


def runtime_environment() -> dict:
    from dotenv import dotenv_values
    env = {**os.environ, **{key: value for key, value in
        dotenv_values(ROOT / "private/runtime.env").items() if value is not None}}
    present = lambda *keys: any(str(env.get(key, "")).strip() for key in keys)
    required = (present("SERPER_API_KEY", "SERPER_KEY_ID"),
        present("JINA_API_KEY", "JINA_API_KEYS"),
        present("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        present("BAIDU_OCR_API_KEY"), present("BAIDU_OCR_SECRET_KEY"))
    if not all(required):
        raise RuntimeError("authorized Agent tool credentials are incomplete")
    env.update(QWEN35_LOCAL_BASE_URL="http://127.0.0.1:19025/v1",
        QWEN35_LOCAL_MODEL=MODEL_ALIAS,
        QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET="8192",
        QWEN_UNIFIED_JUDGMENT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_JUDGMENT_THINKING_TOKEN_BUDGET="8192",
        AGENT_LLM_REQUEST_TIMEOUT_SECONDS="1230", AGENT_STAGE_REQUEST_TIMEOUT_SECONDS="1260",
        AGENT_LLM_REQUEST_MAX_RETRIES="0", TOOL_CACHE_ENABLED="0", OMP_NUM_THREADS="1",
        PYTHONPATH=str(SOURCE_CODE) + ":" + str(SOURCE_CODE / "training"), TMPDIR=str(ROOT / "tmp"))
    return env


def successful(directory: Path) -> dict[str, dict]:
    if not (directory / "run_results.jsonl").is_file():
        return {}
    result: dict[str, dict] = {}
    for row in rows(directory / "run_results.jsonl"):
        case = str(row.get("case_id") or "")
        if (row.get("status") == "success" and row.get("termination") == "success"
                and row.get("verdict") in {"real", "fake"} and row.get("trace_path")):
            if case in result:
                raise ValueError(f"duplicate successful case: {case}")
            result[case] = row
    return result


def run_attempt(directory: Path, cases: list[str], *, concurrency: int, seed: int,
                environment: dict, resume_from: Path | None = None) -> None:
    if directory.exists():
        if not (directory / "summary.json").is_file():
            raise RuntimeError(f"interrupted attempt is preserved: {directory}")
        return
    case_list = OUTPUT / f"{directory.name}-cases.txt"
    case_list.write_text("".join(case + "\n" for case in cases), encoding="utf-8")
    command = [sys.executable, "-m", "src.eval.run_cases", "--benchmark", str(BENCHMARK),
        "--profile", "student-qwen3.5-local", "--output-dir", str(directory),
        "--case-list", str(case_list), "--concurrency", str(min(concurrency, len(cases))),
        "--base-sampling-seed", str(seed), "--timeout", "3000",
        "--skip-preflight-image-hash-verification"]
    if resume_from is not None:
        command += ["--resume-from", str(resume_from)]
    atomic_json(DEPLOY / "state.json", {"phase": "agent_inference", "attempt": directory.name,
        "pending_at_start": len(cases), "concurrency": min(concurrency, len(cases))})
    with (OUTPUT / f"{directory.name}.log").open("xb") as log:
        subprocess.run(command, cwd=SOURCE_CODE, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=False)
    if not (directory / "summary.json").is_file():
        raise RuntimeError(f"Agent attempt did not finish normally: {directory}")


def _looks_garbled(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    if not text:
        return True
    longest = 1
    current = 1
    for before, after in zip(text, text[1:]):
        current = current + 1 if before == after else 1
        longest = max(longest, current)
    return longest >= 80


def smoke_anomaly_report(directory: Path, expected: list[str]) -> dict:
    selected = successful(directory)
    records: list[dict] = []
    for case in expected:
        row = selected.get(case)
        issues: list[str] = []
        if row is None:
            issues.append("no_terminal_success")
            records.append({"case_id": case, "issues": issues})
            continue
        report = str(row.get("fact_check_report") or "").strip()
        if _looks_garbled(report):
            issues.append("empty_or_garbled_report")
        if len(report) > 120000:
            issues.append("oversized_report")
        calls = int(row.get("total_tool_calls") or 0)
        if calls > 30:
            issues.append("tool_budget_exceeded")
        trace_path = directory / str(row["trace_path"])
        if not trace_path.is_file():
            issues.append("missing_trace")
        else:
            trace = load(trace_path)
            finishes = Counter()
            signatures = Counter()
            for step in ((trace.get("state") or {}).get("all_steps") or []):
                metadata = step.get("metadata") or {}
                if metadata.get("finish_reason"):
                    finishes[str(metadata["finish_reason"])] += 1
                if step.get("action_type") == "tool_call":
                    signatures[(str(step.get("tool_name") or ""),
                        json.dumps(step.get("tool_input"), sort_keys=True, ensure_ascii=False))] += 1
            if finishes.get("length") or finishes.get("abort"):
                issues.append("length_or_abort_finish")
            if signatures and max(signatures.values()) >= 8:
                issues.append("repeated_identical_tool_loop")
        records.append({"case_id": case, "issues": issues,
            "report_chars": len(report), "tool_calls": calls})
    result = {"schema_version": "ifv-psd-posttrain-smoke-anomaly-v1",
        "scope": "engineering anomalies only; no labels, correctness, or judge",
        "expected": len(expected), "terminal_successes": len(selected),
        "passed": all(not row["issues"] for row in records), "records": records}
    return result


def run_evaluation(adapter: Path) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    benchmark_rows = rows(BENCHMARK)
    expected = [str(row["case_id"]) for row in benchmark_rows]
    smoke_ids = [line.strip() for line in SMOKE_CASES.read_text().splitlines() if line.strip()]
    if len(expected) != 1526 or len(set(expected)) != 1526:
        raise ValueError("frozen runnable test cohort is not exactly 1526 unique cases")
    if len(smoke_ids) != 4 or not set(smoke_ids) <= set(expected):
        raise ValueError("frozen four-case smoke list is invalid")
    binding = {"schema_version": "ifv-psd-posttrain-full-eval-binding-v1",
        "benchmark": stat_identity(BENCHMARK), "smoke_case_list": stat_identity(SMOKE_CASES),
        "adapter_file": stat_identity(adapter / "adapter_model.safetensors"),
        "model_alias": MODEL_ALIAS, "runnable_cases": 1526, "formal_denominator": 1527,
        "output_tokens": 32768, "thinking_budget": 8192, "large_payload_hashing": False,
        "smoke_gate": "engineering_anomalies_only", "judge_submitted": False}
    binding_path = OUTPUT / "binding.json"
    if binding_path.exists() and load(binding_path) != binding:
        raise ValueError("posttrain evaluation binding changed")
    atomic_json(binding_path, binding)
    environment = runtime_environment()
    smoke = OUTPUT / "smoke"
    run_attempt(smoke, smoke_ids, concurrency=4, seed=1903, environment=environment)
    audit = smoke_anomaly_report(smoke, smoke_ids)
    atomic_json(OUTPUT / "smoke-anomaly-report.json", audit)
    if not audit["passed"]:
        atomic_json(DEPLOY / "state.json", {"phase": "smoke_failed_requires_inspection",
            "report": str(OUTPUT / "smoke-anomaly-report.json")})
        raise RuntimeError("PSD posttrain anomaly smoke did not pass")
    selected: dict[str, dict] = successful(smoke)
    attempts = [smoke]
    for attempt, concurrency in enumerate((32, 24, 16, 8)):
        pending = [case for case in expected if case not in selected]
        if not pending:
            break
        directory = OUTPUT / f"attempt-{attempt}"
        run_attempt(directory, pending, concurrency=concurrency, seed=2903 + attempt * 1000,
            environment=environment, resume_from=attempts[-1])
        overlap = set(selected) & set(successful(directory))
        if overlap:
            raise ValueError(f"successful case was resampled: {sorted(overlap)[0]}")
        selected.update(successful(directory))
        attempts.append(directory)
    missing = [case for case in expected if case not in selected]
    result = {"schema_version": "ifv-psd-posttrain-full-inference-v1",
        "phase": "inference_complete" if not missing else "engineering_retry_budget_exhausted",
        "success": len(selected), "expected_runnable": len(expected),
        "formal_denominator": 1527, "failures_retained_in_denominator": True,
        "remaining": missing, "attempt_directories": [str(path) for path in attempts],
        "smoke_reused": True, "judge_submitted": False, "large_payload_hashing": False}
    atomic_json(OUTPUT / "inference-summary.json", result)
    return result


def main() -> None:
    os.umask(0o077)
    training = wait_for_training()
    report = training_diagnostics(training)
    atomic_json(DEPLOY / "training-diagnostics.json", report)
    if not report.get("passed"):
        raise RuntimeError("small-bank PSD posttrain diagnostics failed")
    adapter = Path(report["adapter"]).resolve()
    context = None
    try:
        atomic_json(DEPLOY / "state.json", {"phase": "starting_psd_adapter_serving",
            "adapter": str(adapter)})
        context = start_adapter_serving(adapter)
        result = run_evaluation(adapter)
        atomic_json(DEPLOY / "state.json", {"phase": result["phase"],
            "success": result["success"], "expected_runnable": result["expected_runnable"],
            "output": str(OUTPUT), "service_restored": False})
    finally:
        if context is not None:
            restore_original_serving(context)
            state = load(DEPLOY / "state.json") if (DEPLOY / "state.json").is_file() else {}
            state["service_restored"] = True
            atomic_json(DEPLOY / "state.json", state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state = load(DEPLOY / "state.json") if (DEPLOY / "state.json").is_file() else {}
        state.update(phase="failed_requires_fix", error_type=type(error).__name__,
            message=str(error)[:1000])
        atomic_json(DEPLOY / "state.json", state)
        raise
