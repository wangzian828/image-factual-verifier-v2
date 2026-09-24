"""Run Base, SFT3 and SFT3+PSD Agent evaluations with self page extraction.

The controller keeps one GPU policy active at a time, but starts an independent
case-granular Gemini judge as soon as each durable Agent result appears.  A
judge from an earlier policy may therefore continue while the next policy is
being served.  Recovery is append-only: successful cases are never sampled
again, failures remain in the formal denominator, and neither traces nor model
weights are scanned or hashed.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
BENCHMARK = (
    ROOT
    / "evaluation/factcheck-formal1527-available1526-20260912"
    / "runtime-release/runtime_input/cases.jsonl"
)
MANIFEST = (
    ROOT
    / "data/factcheck-test-1527-filtered-frozen-20260909/test-manifest.jsonl"
)
PRIVATE_GOLD = (
    ROOT
    / "data/factcheck-test-1527-filtered-frozen-20260909"
    / "evaluator_private/private-gold-v1/private-gold.jsonl"
)
MODEL_ALIAS = "ifv-psd-sft3084"
PORTS = (19002, 19003, 19004, 19005)
GATEWAY_PORT = 19025
FORMAL_DENOMINATOR = 1527
EXPECTED_RUNNABLE = 1526
SMOKE_CASES = (
    "route-aware-hrc-final-3000-20260816-input:0647:"
    "baseline_main_generated_r013-generated_refuted_mutation-CD4-18-0000",
    "route-aware-hrc-final-3000-20260816-input:0743:"
    "baseline_main_generated_r015-pipeline_generated_supported-0007",
    "route-aware-hrc-final-3000-20260816-input:1026:"
    "generated_r004-generated_refuted_mutation-EF2-08-0000",
    "route-aware-hrc-final-3000-20260816-input:2649:"
    "web_r006-web_crawled_supported-web-backlog-00001",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stat_identity(path: Path) -> dict[str, object]:
    value = path.stat()
    return {
        "path": str(path.resolve()),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def owner_module():
    path = (
        ROOT
        / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    )
    spec = importlib.util.spec_from_file_location("psd_service_owner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen service owner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _url_json(url: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_idle_gateway(timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            replicas = _url_json(
                f"http://127.0.0.1:{GATEWAY_PORT}/health"
            )["replicas"]
            if len(replicas) == 4 and all(
                int(row.get("inflight", 0)) == 0 for row in replicas
            ):
                return
        except (OSError, KeyError, TypeError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError("serving gateway did not become idle")


def wait_for_services(model_root: Path, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            for port in PORTS:
                cards = _url_json(f"http://127.0.0.1:{port}/v1/models")["data"]
                matches = [card for card in cards if card.get("id") == MODEL_ALIAS]
                if len(matches) != 1:
                    raise ValueError(f"port {port} has no unique model alias")
                exposed = Path(str(matches[0].get("root") or "")).resolve()
                if exposed != model_root.resolve():
                    raise ValueError(f"port {port} exposes {exposed}")
            health = _url_json(f"http://127.0.0.1:{GATEWAY_PORT}/health")
            replicas = health.get("replicas") or []
            if len(replicas) != 4 or not all(
                row.get("healthy") is True for row in replicas
            ):
                raise ValueError("gateway does not see four healthy replicas")
            return
        except (OSError, KeyError, TypeError, ValueError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(10)
    raise RuntimeError(f"model serving readiness timed out: {last}")


def _gateway_process() -> tuple[int, list[str], dict[str, str], str]:
    matches: list[tuple[int, list[str]]] = []
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
        if (
            "scripts.server.psd_qwen_gateway:app" in command
            and "--port" in command
            and command[command.index("--port") + 1] == str(GATEWAY_PORT)
        ):
            matches.append((int(entry.name), command))
    if len(matches) != 1:
        raise RuntimeError(f"expected one gateway, found {len(matches)}")
    pid, command = matches[0]
    if os.getpgid(pid) != pid:
        raise RuntimeError("gateway is not its process-group owner")
    environment: dict[str, str] = {}
    for item in (Path(f"/proc/{pid}/environ")).read_bytes().split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode(errors="surrogateescape")] = value.decode(
                errors="surrogateescape"
            )
    return pid, command, environment, os.readlink(f"/proc/{pid}/cwd")


def gateway_command_for_source(command: list[str], source_code: Path) -> list[str]:
    result = list(command)
    if "--app-dir" not in result:
        raise ValueError("gateway command has no --app-dir")
    result[result.index("--app-dir") + 1] = str(source_code.resolve())
    return result


def restart_gateway(
    deploy: Path, label: str, source_code: Path | None = None
) -> None:
    pid, command, environment, cwd = _gateway_process()
    if source_code is not None:
        source_code = source_code.resolve()
        command = gateway_command_for_source(command, source_code)
        previous = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source_code) + (
            ":" + previous if previous else ""
        )
        cwd = str(source_code)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            break
        try:
            state = (proc / "stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("gateway did not stop")
    log_path = deploy / f"gateway-{label}.log"
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
    atomic_json(
        deploy / f"gateway-{label}.json",
        {
            "old_pid": pid,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid),
            "environment_persisted": False,
            "large_payload_hashing": False,
        },
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gateway exited with {process.returncode}")
        try:
            replicas = _url_json(
                f"http://127.0.0.1:{GATEWAY_PORT}/health"
            ).get("replicas") or []
            if len(replicas) == 4 and all(
                row.get("healthy") is True for row in replicas
            ):
                return
        except (OSError, TypeError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError("fresh gateway readiness timed out")


def normalized_command(command: list[str], model_root: Path) -> list[str]:
    result = list(command)
    if len(result) < 4 or result[2] != "serve":
        raise ValueError("unrecognized vLLM command")
    result[3] = str(model_root)
    if "--served-model-name" not in result:
        raise ValueError("vLLM command has no served model name")
    result[result.index("--served-model-name") + 1] = MODEL_ALIAS
    for option, takes_value in (
        ("--tool-parser-plugin", True),
        ("--enable-lora", False),
        ("--max-lora-rank", True),
        ("--enable-tower-connector-lora", False),
        ("--lora-modules", True),
    ):
        cleaned: list[str] = []
        index = 0
        while index < len(result):
            if result[index] == option:
                index += 2 if takes_value else 1
            else:
                cleaned.append(result[index])
                index += 1
        result = cleaned
    if "--tool-call-parser" in result:
        result[result.index("--tool-call-parser") + 1] = "qwen3_coder"
    else:
        result += ["--tool-call-parser", "qwen3_coder"]
    return result


class ServiceSession:
    def __init__(self, deploy: Path) -> None:
        self.deploy = deploy
        self.owner = owner_module()
        self.guard = self.owner.load(SERVICE / "guard.json")
        self.owner.checked(self.guard)
        self.originals = [
            self.owner.load(SERVICE / f"replica-{index}.json")
            for index in range(4)
        ]
        self.environments = [
            self.owner.checked(receipt) for receipt in self.originals
        ]
        self.active = list(self.originals)
        self.guard_paused = False

    def pause_guard(self) -> None:
        if not self.guard_paused:
            os.kill(int(self.guard["pid"]), signal.SIGSTOP)
            self.guard_paused = True

    def _live_receipt(self, index: int, receipt: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve an explicitly replaced replica without accepting command drift.

        A bounded cache-recovery operation may restart one replica while a long
        Agent profile is running.  The in-memory receipt then names the dead
        process, while the authoritative SERVICE receipt names its replacement.
        Accept that replacement only when it is alive and its complete command
        is byte-for-byte identical.  A dead receipt is already stopped; a live
        process with a different command remains a hard ownership error.
        """

        try:
            self.owner.checked(receipt)
            return receipt
        except (FileNotFoundError, ProcessLookupError):
            pass

        replacement = self.owner.load(SERVICE / f"replica-{index}.json")
        try:
            self.owner.checked(replacement)
        except (FileNotFoundError, ProcessLookupError):
            return None
        if replacement.get("command") != receipt.get("command"):
            raise RuntimeError(
                f"replica {index} replacement command does not match active owner"
            )
        return replacement

    def _stop_active(self) -> None:
        resolved = [
            live
            for index, receipt in enumerate(self.active)
            if (live := self._live_receipt(index, receipt)) is not None
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.owner.stop, resolved))
        self.active = []

    def switch(self, key: str, model_root: Path) -> None:
        if not model_root.is_dir() or not (model_root / "config.json").is_file():
            raise FileNotFoundError(model_root)
        wait_for_idle_gateway()
        self.pause_guard()
        self._stop_active()
        receipts: list[dict[str, Any]] = []
        try:
            for index, (original, environment) in enumerate(
                zip(self.originals, self.environments)
            ):
                command = normalized_command(original["command"], model_root)
                receipt = self.owner.spawn(
                    command,
                    environment,
                    self.deploy / f"{key}-replica-{index}.log",
                )
                receipts.append(receipt)
                self.owner.save(
                    self.deploy / f"{key}-replica-{index}.json", receipt
                )
            self.active = receipts
            wait_for_services(model_root)
            restart_gateway(self.deploy, key)
            wait_for_services(model_root)
        except Exception:
            self.active = receipts
            raise

    def restore(self) -> None:
        errors: list[dict[str, str]] = []
        if self.active:
            try:
                self._stop_active()
            except Exception as error:
                errors.append({"stage": "stop_active", "error": type(error).__name__})
        restored: list[dict[str, Any]] = []
        for index, (original, environment) in enumerate(
            zip(self.originals, self.environments)
        ):
            try:
                receipt = self.owner.spawn(
                    original["command"],
                    environment,
                    self.deploy / f"restore-sft-{index}.log",
                )
                restored.append(receipt)
                self.owner.save(SERVICE / f"replica-{index}.json", receipt)
            except Exception as error:
                errors.append(
                    {
                        "stage": f"restore_sft_{index}",
                        "error": type(error).__name__,
                    }
                )
        self.active = restored
        if len(restored) == 4:
            try:
                restart_gateway(self.deploy, "restored-sft")
                wait_for_services(Path(self.originals[0]["command"][3]))
            except Exception as error:
                errors.append(
                    {"stage": "restore_gateway", "error": type(error).__name__}
                )
        if self.guard_paused:
            os.kill(int(self.guard["pid"]), signal.SIGCONT)
            self.guard_paused = False
        if errors:
            atomic_json(self.deploy / "restore-errors.json", errors)
            raise RuntimeError("original SFT service restoration failed")


def runtime_environment(source_code: Path, profile_model: str) -> dict[str, str]:
    from dotenv import dotenv_values

    env = {
        **os.environ,
        **{
            key: value
            for key, value in dotenv_values(ROOT / "private/runtime.env").items()
            if value is not None
        },
    }
    present = lambda *keys: any(str(env.get(key, "")).strip() for key in keys)
    required = (
        present("SERPER_API_KEY", "SERPER_KEY_ID"),
        present("JINA_API_KEY", "JINA_API_KEYS"),
        present("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        present("BAIDU_OCR_API_KEY"),
        present("BAIDU_OCR_SECRET_KEY"),
    )
    if not all(required):
        raise RuntimeError("authorized Agent/judge credentials are incomplete")
    env.update(
        QWEN35_LOCAL_BASE_URL=f"http://127.0.0.1:{GATEWAY_PORT}/v1",
        QWEN35_LOCAL_MODEL=profile_model,
        BROWSE_EXTRACT_PROVIDER="qwen_local",
        BROWSE_EXTRACT_MODEL=profile_model,
        BROWSE_EXTRACT_BASE_URL=f"http://127.0.0.1:{GATEWAY_PORT}/v1",
        BROWSE_EXTRACT_API_KEY="none",
        BROWSE_EXTRACT_WIRE_API="chat_completions",
        BROWSE_EXTRACT_ENABLE_THINKING="1",
        BROWSE_EXTRACT_THINKING_TOKEN_BUDGET="2048",
        QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET="8192",
        QWEN_UNIFIED_JUDGMENT_MAX_OUTPUT_TOKENS="32768",
        QWEN_UNIFIED_JUDGMENT_THINKING_TOKEN_BUDGET="8192",
        AGENT_LLM_REQUEST_TIMEOUT_SECONDS="1230",
        AGENT_STAGE_REQUEST_TIMEOUT_SECONDS="1260",
        AGENT_LLM_REQUEST_MAX_RETRIES="0",
        BROWSE_EXTRACT_MAX_RETRIES="0",
        TOOL_CACHE_ENABLED="0",
        IFV_STREAM_RUN_RESULTS="1",
        OMP_NUM_THREADS="1",
        PYTHONPATH=str(source_code) + ":" + str(source_code / "training"),
        TMPDIR=str(ROOT / "tmp"),
    )
    return env


def successful(output: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    selected: dict[str, tuple[dict[str, Any], Path]] = {}
    for ledger in sorted(output.glob("*/run_results.jsonl")):
        directory = ledger.parent
        for row in rows(ledger):
            case_id = str(row.get("case_id") or "")
            if not (
                row.get("status") == "success"
                and row.get("termination") == "success"
                and row.get("verdict") in {"real", "fake"}
                and row.get("trace_path")
            ):
                continue
            if case_id in selected:
                raise ValueError(f"successful case was sampled twice: {case_id}")
            selected[case_id] = (row, directory)
    return selected


def _walk_dicts(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def extraction_protocol_report(
    selected: dict[str, tuple[dict[str, Any], Path]],
    smoke_cases: Iterable[str],
    expected_model: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_extracts = 0
    mismatches: list[dict[str, str]] = []
    for case_id in smoke_cases:
        item = selected.get(case_id)
        if item is None:
            records.append({"case_id": case_id, "status": "missing"})
            continue
        row, directory = item
        trace = load(directory / str(row["trace_path"]))
        extracts = [
            value
            for value in _walk_dicts(trace)
            if value.get("kind") == "page_extract"
        ]
        total_extracts += len(extracts)
        for extract in extracts:
            provider = str(extract.get("provider") or "")
            model = str(extract.get("model") or "")
            status = str(extract.get("status") or "")
            thinking_enabled = extract.get("thinking_enabled")
            thinking_budget = extract.get("thinking_token_budget")
            if (
                provider != "qwen_local"
                or model != expected_model
                or status != "success"
                or thinking_enabled is not True
                or thinking_budget != 2048
            ):
                mismatches.append(
                    {
                        "case_id": case_id,
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "thinking_enabled": str(thinking_enabled),
                        "thinking_token_budget": str(thinking_budget),
                    }
                )
        records.append(
            {
                "case_id": case_id,
                "status": "success",
                "page_extracts": len(extracts),
            }
        )
    missing = [row["case_id"] for row in records if row["status"] == "missing"]
    return {
        "schema_version": "ifv-self-extract-smoke-audit-v1",
        "passed": not missing and total_extracts > 0 and not mismatches,
        "expected_provider": "qwen_local",
        "expected_model": expected_model,
        "page_extracts": total_extracts,
        "missing": missing,
        "mismatches": mismatches,
        "records": records,
    }


def _looks_garbled(text: str) -> bool:
    if not text or "\ufffd" in text or "\x00" in text:
        return True
    longest = 1
    current = 1
    for before, after in zip(text, text[1:]):
        current = current + 1 if before == after else 1
        longest = max(longest, current)
    return longest >= 80


def anomaly_report(
    selected: dict[str, tuple[dict[str, Any], Path]],
    smoke_cases: Iterable[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case_id in smoke_cases:
        issues: list[str] = []
        item = selected.get(case_id)
        if item is None:
            records.append({"case_id": case_id, "issues": ["no_terminal_success"]})
            continue
        row, directory = item
        report = str(row.get("fact_check_report") or "").strip()
        if _looks_garbled(report):
            issues.append("empty_or_garbled_report")
        if len(report) > 120000:
            issues.append("oversized_report")
        if int(row.get("total_tool_calls") or 0) > 30:
            issues.append("tool_budget_exceeded")
        trace = load(directory / str(row["trace_path"]))
        finishes: Counter[str] = Counter()
        signatures: Counter[tuple[str, str]] = Counter()
        for step in ((trace.get("state") or {}).get("all_steps") or []):
            metadata = step.get("metadata") or {}
            if metadata.get("finish_reason"):
                finishes[str(metadata["finish_reason"])] += 1
            if step.get("action_type") == "tool_call":
                signatures[
                    (
                        str(step.get("tool_name") or ""),
                        json.dumps(
                            step.get("tool_args"),
                            sort_keys=True,
                            ensure_ascii=False,
                        ),
                    )
                ] += 1
        if finishes.get("length") or finishes.get("abort"):
            issues.append("length_or_abort_finish")
        if signatures and max(signatures.values()) >= 8:
            issues.append("repeated_identical_tool_loop")
        records.append({"case_id": case_id, "issues": issues})
    return {
        "schema_version": "ifv-engineering-smoke-audit-v1",
        "passed": all(not row["issues"] for row in records),
        "records": records,
    }


def launch_judge(
    *,
    deploy: Path,
    source_code: Path,
    output: Path,
    source_model: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    judge_output = output / "judge-gemini37-stream-v1"
    command = [
        sys.executable,
        str(source_code / "scripts/server/stream_agent_judges.py"),
        "--rollout-root",
        str(output),
        "--benchmark",
        str(BENCHMARK),
        "--manifest",
        str(MANIFEST),
        "--gold",
        str(PRIVATE_GOLD),
        "--output",
        str(judge_output),
        "--inference-summary",
        str(output / "inference-summary.json"),
        "--source-model",
        source_model,
        "--concurrency",
        "32",
    ]
    with (output / "judge-stream.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=source_code,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "source_model": source_model,
        "output": str(judge_output),
        "case_granular": True,
        "large_payload_hashing": False,
    }
    atomic_json(deploy / f"judge-{source_model}.json", receipt)
    return receipt


def run_attempt(
    *,
    deploy: Path,
    source_code: Path,
    output: Path,
    name: str,
    cases: list[str],
    concurrency: int,
    base_seed: int,
    environment: dict[str, str],
    tool_ablation_flags: tuple[str, ...] = (),
) -> None:
    directory = output / name
    if directory.exists():
        raise FileExistsError(directory)
    case_list = output / f"{name}-cases.txt"
    case_list.write_text("".join(case + "\n" for case in cases), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.eval.run_cases",
        "--benchmark",
        str(BENCHMARK),
        "--profile",
        "student-qwen3.5-local",
        "--output-dir",
        str(directory),
        "--case-list",
        str(case_list),
        "--concurrency",
        str(min(concurrency, len(cases))),
        "--base-sampling-seed",
        str(base_seed),
        "--timeout",
        "3000",
        "--skip-preflight-image-hash-verification",
        *tool_ablation_flags,
    ]
    atomic_json(
        deploy / "state.json",
        {
            "phase": "agent_inference",
            "model": environment["QWEN35_LOCAL_MODEL"],
            "attempt": name,
            "pending_at_start": len(cases),
            "concurrency": min(concurrency, len(cases)),
        },
    )
    with (output / f"{name}.log").open("xb") as log:
        result = subprocess.run(
            command,
            cwd=source_code,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            check=False,
        )
    atomic_json(
        output / f"{name}-process-result.json",
        {
            "returncode": result.returncode,
            "summary_present": (directory / "summary.json").is_file(),
            "partial_results_preserved": (directory / "run_results.jsonl").is_file(),
        },
    )


def evaluate_model(
    *,
    deploy: Path,
    source_code: Path,
    key: str,
    profile_model: str,
    model_root: Path,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    benchmark_rows = rows(BENCHMARK)
    expected = [str(row["case_id"]) for row in benchmark_rows]
    if len(expected) != EXPECTED_RUNNABLE or len(set(expected)) != EXPECTED_RUNNABLE:
        raise ValueError("frozen runnable cohort is not 1526 unique cases")
    if not set(SMOKE_CASES) <= set(expected):
        raise ValueError("protocol smoke cases are outside the frozen cohort")
    binding = {
        "schema_version": "ifv-corrected-self-extract-agent-eval-v1",
        "key": key,
        "profile_model": profile_model,
        "served_model_root": stat_identity(model_root / "config.json"),
        "benchmark": stat_identity(BENCHMARK),
        "manifest": stat_identity(MANIFEST),
        "private_gold": stat_identity(PRIVATE_GOLD),
        "runnable_cases": EXPECTED_RUNNABLE,
        "formal_denominator": FORMAL_DENOMINATOR,
        "page_extract_provider": "qwen_local",
        "page_extract_model": profile_model,
        "page_extract_thinking_enabled": True,
        "page_extract_thinking_token_budget": 2048,
        "judge_model": "gemini-3.7-flash",
        "judge_submission": "per_case_immediately_after_durable_agent_result",
        "output_tokens": 32768,
        "thinking_budget": 8192,
        "large_payload_hashing": False,
        "history_reused": False,
    }
    atomic_json(output / "binding.json", binding)
    (output / "protocol-smoke-cases.txt").write_text(
        "".join(case + "\n" for case in SMOKE_CASES), encoding="utf-8"
    )
    environment = runtime_environment(source_code, profile_model)
    launch_judge(
        deploy=deploy,
        source_code=source_code,
        output=output,
        source_model=profile_model,
        environment=environment,
    )

    for attempt in range(2):
        selected = successful(output)
        pending = [case for case in SMOKE_CASES if case not in selected]
        if not pending:
            break
        run_attempt(
            deploy=deploy,
            source_code=source_code,
            output=output,
            name=f"smoke-{attempt}",
            cases=pending,
            concurrency=4,
            base_seed=3903 + attempt * 1000,
            environment=environment,
        )
    selected = successful(output)
    engineering = anomaly_report(selected, SMOKE_CASES)
    extraction = extraction_protocol_report(selected, SMOKE_CASES, profile_model)
    atomic_json(output / "smoke-engineering-audit.json", engineering)
    atomic_json(output / "smoke-extraction-protocol-audit.json", extraction)
    if not engineering["passed"] or not extraction["passed"]:
        summary = {
            "phase": "smoke_failed_requires_fix",
            "engineering_passed": engineering["passed"],
            "extraction_protocol_passed": extraction["passed"],
            "success": len(selected),
            "expected_runnable": EXPECTED_RUNNABLE,
            "formal_denominator": FORMAL_DENOMINATOR,
        }
        atomic_json(output / "inference-summary.json", summary)
        raise RuntimeError(f"{key} corrected-protocol smoke failed")

    # Keep the four replicas busy without over-subscribing the long ReAct
    # sessions.  Engineering failures are retried by the progressively smaller
    # waves below, so the first pass does not need to saturate at 32 clients.
    for attempt, concurrency in enumerate((28, 20, 12, 8)):
        selected = successful(output)
        pending = [case for case in expected if case not in selected]
        if not pending:
            break
        run_attempt(
            deploy=deploy,
            source_code=source_code,
            output=output,
            name=f"attempt-{attempt}",
            cases=pending,
            concurrency=concurrency,
            base_seed=2903 + attempt * 1000,
            environment=environment,
        )
    selected = successful(output)
    missing = [case for case in expected if case not in selected]
    result = {
        "schema_version": "ifv-corrected-self-extract-inference-v1",
        "phase": (
            "inference_complete"
            if not missing
            else "engineering_retry_budget_exhausted"
        ),
        "key": key,
        "profile_model": profile_model,
        "success": len(selected),
        "expected_runnable": EXPECTED_RUNNABLE,
        "formal_denominator": FORMAL_DENOMINATOR,
        "failures_retained_in_denominator": True,
        "remaining": missing,
        "judge_streaming_concurrently": True,
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
    profiles = (
        (
            "sft3",
            "ifv-qwen3.5-9b-sft3084-selfextract",
            ROOT / "exports/h20-sft-merged4872-3epoch-step3084-20260915/model",
            ROOT
            / "evaluation/qwen35-sft3084-agent-full1526-selfextract-20260921-v3",
        ),
        (
            "sft3-psd",
            "ifv-qwen3.5-9b-sft3084-psd-smallbank4095-selfextract",
            ROOT / "exports/qwen35-psd-smallbank4095-merged-20260921-v1/model",
            ROOT
            / "evaluation/qwen35-psd-smallbank4095-agent-full1526-selfextract-20260921-v3",
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
        for key, profile_model, model_root, output in profiles:
            atomic_json(
                deploy / "state.json",
                {
                    "phase": "switching_model",
                    "key": key,
                    "model_root": str(model_root),
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
                    "phase": "profile_inference_complete_judge_continues",
                    "key": key,
                    "result": result,
                    "completed_inference_profiles": [row["key"] for row in completed],
                },
            )
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
