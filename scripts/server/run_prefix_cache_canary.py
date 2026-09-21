"""Run one reversible Qwen3.5 hybrid prefix-cache correctness canary.

This owner compares a frozen multimodal incremental request sequence with APC
disabled and enabled, then restores the original four replicas and the safe
request-isolated gateway.  It never edits weights or existing evaluation data.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.request


ROOT = Path("/volume/ybo/wza")
SERVICE = ROOT / "inference/psd-sft3084-20260916"
BENCHMARK = (
    ROOT
    / "evaluation/factcheck-formal1527-available1526-20260912"
    / "runtime-release/runtime_input/cases.jsonl"
)
MODEL_ALIAS = "ifv-psd-sft3084"
PORTS = (19002, 19003, 19004, 19005)
GATEWAY_PORT = 19025
PROBE_CASE = (
    "route-aware-hrc-final-3000-20260816-input:0647:"
    "baseline_main_generated_r013-generated_refuted_mutation-CD4-18-0000"
)
CASE_SALT = "ifv-case-v1-" + "A" * 43
OTHER_SALT = "ifv-case-v1-" + "B" * 43


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


def owner_module():
    path = (
        ROOT
        / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    )
    spec = importlib.util.spec_from_file_location("psd_service_owner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen service owner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_model(port: int, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            cards = request_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)["data"]
            matches = [card for card in cards if card.get("id") == MODEL_ALIAS]
            if len(matches) == 1:
                return
            last = "model alias missing"
        except Exception as error:  # readiness includes transient socket errors
            last = f"{type(error).__name__}: {error}"
        time.sleep(3)
    raise RuntimeError(f"model on port {port} did not become ready: {last}")


def metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=10
    ) as response:
        text = response.read().decode("utf-8")
    wanted = {
        "vllm:prefix_cache_queries_total": "queries",
        "vllm:prefix_cache_hits_total": "hits",
        "vllm:mm_cache_hits_total": "mm_hits",
    }
    result = {name: 0.0 for name in wanted.values()}
    for line in text.splitlines():
        for metric, name in wanted.items():
            if line.startswith(metric + "{"):
                result[name] = float(line.rsplit(" ", 1)[1])
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in sorted(before)}


def probe_image() -> Path:
    for raw in BENCHMARK.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row.get("case_id") == PROBE_CASE:
            path = (BENCHMARK.parent / str(row["image_path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
    raise KeyError(PROBE_CASE)


def initial_payload(image_path: Path, *, cache_salt: str) -> dict[str, Any]:
    suffix = image_path.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "model": MODEL_ALIAS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Inspect the supplied image. Use the inspect_evidence tool once "
                    "with a concise query, then wait for its result."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": "Find one concrete visual claim that should be checked.",
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_evidence",
                    "description": "Inspect one fixed evidence target.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "seed": 4242,
        "max_tokens": 1536,
        "chat_template_kwargs": {"enable_thinking": True},
        "vllm_xargs": {"ifv_thinking_budget": 512},
        "cache_salt": cache_salt,
        "return_token_ids": True,
    }


def growing_payload(initial: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(initial))
    message = dict(response["choices"][0]["message"])
    result["messages"].append(message)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        result["messages"].append(
            {
                "role": "tool",
                "tool_call_id": str(tool_calls[0]["id"]),
                "content": json.dumps(
                    {"status": "success", "evidence": "Fixed canary evidence."}
                ),
            }
        )
    else:
        result["messages"].append(
            {"role": "user", "content": "Continue with one final concise observation."}
        )
    return result


def response_signature(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    message = choice.get("message") or {}
    return {
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
        "tool_calls": message.get("tool_calls"),
        "token_ids": choice.get("token_ids") or response.get("token_ids"),
    }


def response_is_well_formed(response: dict[str, Any]) -> bool:
    """Reject malformed payloads, non-finite values, and invalid token IDs."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict) or not choice.get("finish_reason"):
            return False
        token_ids = choice.get("token_ids") or response.get("token_ids") or []
        if token_ids and (
            not isinstance(token_ids, list)
            or any(not isinstance(token, int) or token < 0 for token in token_ids)
        ):
            return False
        stack: list[Any] = [response]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, float) and not math.isfinite(current):
                return False
        return True
    except (KeyError, IndexError, TypeError):
        return False


def run_probe(port: int, archive: Path) -> dict[str, Any]:
    image_path = probe_image()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    primary = initial_payload(image_path, cache_salt=CASE_SALT)
    sequence: list[dict[str, Any]] = []
    for label, payload in (
        ("primary_first", primary),
        ("primary_repeat", primary),
    ):
        before = metrics(port)
        response = request_json(url, payload, timeout=300)
        after = metrics(port)
        sequence.append(
            {
                "label": label,
                "metrics": metric_delta(before, after),
                "signature": response_signature(response),
                "well_formed": response_is_well_formed(response),
                "response": response,
            }
        )
    grown = growing_payload(primary, sequence[0]["response"])
    before = metrics(port)
    grown_response = request_json(url, grown, timeout=300)
    after = metrics(port)
    sequence.append(
        {
            "label": "primary_growing_turn",
            "metrics": metric_delta(before, after),
            "signature": response_signature(grown_response),
            "well_formed": response_is_well_formed(grown_response),
            "response": grown_response,
        }
    )
    secondary = initial_payload(image_path, cache_salt=OTHER_SALT)
    for label, payload in (
        ("secondary_first", secondary),
        ("secondary_repeat", secondary),
    ):
        before = metrics(port)
        response = request_json(url, payload, timeout=300)
        after = metrics(port)
        sequence.append(
            {
                "label": label,
                "metrics": metric_delta(before, after),
                "signature": response_signature(response),
                "well_formed": response_is_well_formed(response),
                "response": response,
            }
        )
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        json.dump(sequence, handle, ensure_ascii=False)
    return {
        "archive": str(archive),
        "records": [
            {
                "label": row["label"],
                "metrics": row["metrics"],
                "signature": row["signature"],
                "well_formed": row["well_formed"],
            }
            for row in sequence
        ],
    }


def cache_on_command(command: list[str]) -> list[str]:
    result = list(command)
    result = [item for item in result if item != "--no-enable-prefix-caching"]
    if "--enable-prefix-caching" not in result:
        result.append("--enable-prefix-caching")
    if "--mamba-cache-mode" not in result:
        result += ["--mamba-cache-mode", "align"]
    else:
        result[result.index("--mamba-cache-mode") + 1] = "align"
    return result


def stop_receipts(owner: Any, receipts: list[dict[str, Any]]) -> None:
    if not receipts:
        return
    with ThreadPoolExecutor(max_workers=len(receipts)) as pool:
        list(pool.map(owner.stop, receipts))


def start_gateway(deploy: Path, source_code: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(source_code)
            + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "QWEN_REPLICA_BACKENDS": ",".join(
                f"http://127.0.0.1:{port}" for port in PORTS
            ),
            "QWEN_REPLICA_MODEL_ID": MODEL_ALIAS,
            "PSD_PUBLIC_MODEL_ALIAS": "ifv-qwen3.5-9b-sft-3084",
            "IFV_PREFIX_CACHE_MODE": "request_isolated",
            "AGENT_LLM_REQUEST_MAX_RETRIES": "0",
            "PSD_GATEWAY_DEADLINE_SECONDS": "1200",
            "AGENT_LLM_REQUEST_TIMEOUT_SECONDS": "1230",
            "AGENT_STAGE_REQUEST_TIMEOUT_SECONDS": "1260",
        }
    )
    for key in (
        "PSD_POLICY_LOGPROB_SEMANTICS",
        "PSD_RAW_WORKER_RECEIPTS",
        "PSD_RAW_WORKER_FILES",
        "PSD_DIAGNOSTIC_RETURN_TOKEN_IDS",
        "PSD_WIRE_CAPTURE_DIR",
    ):
        environment.pop(key, None)
    command = [
        str(ROOT / "envs/h20-qwen35-vllm-0181/bin/python"),
        "-m",
        "uvicorn",
        "scripts.server.psd_qwen_gateway:app",
        "--app-dir",
        str(source_code),
        "--host",
        "127.0.0.1",
        "--port",
        str(GATEWAY_PORT),
        "--log-level",
        "warning",
    ]
    with (deploy / "restored-gateway.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=source_code,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {"pid": process.pid, "pgid": os.getpgid(process.pid), "command": command}
    atomic_json(deploy / "restored-gateway.json", receipt)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"restored gateway exited with {process.returncode}")
        try:
            health = request_json(
                f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=5
            )
            replicas = health.get("replicas") or []
            if (
                len(replicas) == 4
                and all(row.get("healthy") is True for row in replicas)
                and health.get("prefix_cache_policy") == "request_isolated"
            ):
                return receipt
        except Exception:
            pass
        time.sleep(2)
    os.killpg(process.pid, signal.SIGTERM)
    raise RuntimeError("restored gateway readiness timed out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    deploy = args.deploy.resolve()
    source_code = args.source_code.resolve()
    deploy.mkdir(parents=True, exist_ok=True)
    state_path = deploy / "state.json"
    atomic_json(
        deploy / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "command": [sys.executable, *sys.argv],
            "started_unix": time.time(),
        },
    )
    owner = owner_module()
    guard = load(SERVICE / "guard.json")
    owner.checked(guard)
    originals = [load(SERVICE / f"replica-{index}.json") for index in range(4)]
    environments = [owner.checked(receipt) for receipt in originals]
    active = list(originals)
    guard_paused = False
    restored = False
    verdict: dict[str, Any] | None = None
    probe_error: BaseException | None = None
    restore_error: BaseException | None = None
    try:
        atomic_json(state_path, {"phase": "cache_off_reference"})
        reference = run_probe(PORTS[0], deploy / "cache-off-responses.json.gz")
        atomic_json(deploy / "cache-off-summary.json", reference)

        os.kill(int(guard["pid"]), signal.SIGSTOP)
        guard_paused = True
        stop_receipts(owner, active)
        active = []

        atomic_json(state_path, {"phase": "starting_cache_on_canary"})
        command = cache_on_command(originals[0]["command"])
        canary = owner.spawn(
            command,
            environments[0],
            deploy / "cache-on-replica.log",
        )
        owner.save(deploy / "cache-on-replica.json", canary)
        active = [canary]
        wait_model(PORTS[0])
        atomic_json(state_path, {"phase": "cache_on_probe"})
        observed = run_probe(PORTS[0], deploy / "cache-on-responses.json.gz")
        atomic_json(deploy / "cache-on-summary.json", observed)

        off = {row["label"]: row for row in reference["records"]}
        on = {row["label"]: row for row in observed["records"]}
        first_equal = (
            off["primary_first"]["signature"]
            == on["primary_first"]["signature"]
        )
        repeat_equal = (
            on["primary_first"]["signature"]
            == on["primary_repeat"]["signature"]
        )
        primary_repeat_hits = on["primary_repeat"]["metrics"]["hits"]
        growing_hits = on["primary_growing_turn"]["metrics"]["hits"]
        secondary_first_hits = on["secondary_first"]["metrics"]["hits"]
        secondary_repeat_hits = on["secondary_repeat"]["metrics"]["hits"]
        labels = [row["label"] for row in reference["records"]]
        all_outputs_match = all(
            off[label]["signature"] == on[label]["signature"] for label in labels
        )
        all_responses_well_formed = all(
            off[label]["well_formed"] and on[label]["well_formed"]
            for label in labels
        )
        verdict = {
            "schema_version": "ifv-qwen35-prefix-cache-canary-v1",
            "cache_on_command": command,
            "first_output_matches_cache_off": first_equal,
            "all_outputs_match_cache_off": all_outputs_match,
            "all_responses_well_formed": all_responses_well_formed,
            "same_scope_repeat_output_equal": repeat_equal,
            "same_scope_repeat_hits": primary_repeat_hits,
            "growing_turn_hits": growing_hits,
            "new_scope_first_hits": secondary_first_hits,
            "new_scope_repeat_hits": secondary_repeat_hits,
            "passed": bool(
                first_equal
                and all_outputs_match
                and all_responses_well_formed
                and repeat_equal
                and primary_repeat_hits > 0
                and growing_hits > 0
                and secondary_first_hits == 0
                and secondary_repeat_hits > 0
            ),
        }
        atomic_json(deploy / "verdict.json", verdict)
        atomic_json(state_path, {"phase": "probe_complete_restoring", "verdict": verdict})
    except BaseException as error:
        probe_error = error
        atomic_json(
            state_path,
            {
                "phase": "probe_failed_restoring",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    finally:
        try:
            try:
                stop_receipts(owner, active)
            except BaseException as error:
                restore_error = error
            restored_receipts: list[dict[str, Any]] = []
            try:
                for index, (original, environment) in enumerate(zip(originals, environments)):
                    receipt = owner.spawn(
                        original["command"],
                        environment,
                        deploy / f"restore-sft-{index}.log",
                    )
                    restored_receipts.append(receipt)
                    owner.save(SERVICE / f"replica-{index}.json", receipt)
                for port in PORTS:
                    wait_model(port)
                start_gateway(deploy, source_code)
                restored = True
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        finally:
            if guard_paused:
                try:
                    os.kill(int(guard["pid"]), signal.SIGCONT)
                except ProcessLookupError:
                    pass
                guard_paused = False
            final = load(state_path) if state_path.is_file() else {}
            if restore_error is not None:
                phase = "failed_requires_fix"
            elif probe_error is not None:
                phase = "canary_failed_service_restored"
            elif verdict and verdict.get("passed"):
                phase = "complete_cache_candidate"
            else:
                phase = "complete_cache_rejected"
            final.update(
                service_restored=restored,
                guard_resumed=True,
                phase=phase,
                restore_error_type=(
                    type(restore_error).__name__ if restore_error is not None else None
                ),
                restore_error=(str(restore_error) if restore_error is not None else None),
            )
            atomic_json(state_path, final)
    if restore_error is not None or not restored:
        raise RuntimeError("service restoration did not complete") from restore_error
    if probe_error is not None:
        raise probe_error


if __name__ == "__main__":
    main()
