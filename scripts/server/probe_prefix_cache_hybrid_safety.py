"""Probe an already-isolated cache-on Qwen gateway for hybrid APC safety.

This script is intentionally not a launcher.  A separate reversible owner must
start four fresh cache-on replicas with NaN metrics enabled, start the gateway
in ``case_isolated`` mode, run this probe, and restore the cache-off service.
Only compact summaries are written; prompts, images, and model responses are
not duplicated in the output directory.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import base64
import json
import math
from pathlib import Path
import secrets
import time
from typing import Any, Callable
import urllib.request


SALT_PREFIX = "ifv-case-v1-"
COUNTERS = {
    "prefix_queries": "vllm:prefix_cache_queries",
    "prefix_hits": "vllm:prefix_cache_hits",
    "corrupted": "vllm:corrupted_requests",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def new_salt() -> str:
    return SALT_PREFIX + secrets.token_urlsafe(32)


def request_json(
    url: str, payload: dict[str, Any] | None = None, *, timeout: float = 300
) -> tuple[dict[str, Any], dict[str, str]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
        headers = {key.lower(): value for key, value in response.headers.items()}
    return result, headers


def prometheus_counter(text: str, metric: str) -> float:
    """Sum one Prometheus counter across labels and naming variants."""

    accepted = {metric, metric + "_total"}
    total = 0.0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in accepted:
            total += float(line.rsplit(" ", 1)[1])
    return total


def metrics(ports: tuple[int, ...]) -> dict[str, float]:
    result = {key: 0.0 for key in COUNTERS}
    for port in ports:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=10
        ) as response:
            text = response.read().decode("utf-8")
        for key, metric in COUNTERS.items():
            result[key] += prometheus_counter(text, metric)
    return result


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in sorted(before)}


def token_count(base_url: str, model: str, messages: list[dict[str, Any]]) -> int:
    response, _ = request_json(
        base_url.rstrip("/") + "/tokenize",
        {"model": model, "messages": messages},
        timeout=60,
    )
    value = response.get("count")
    if type(value) is not int or value < 1:
        token_ids = response.get("tokens") or response.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"tokenize response has no count: {sorted(response)}")
        value = len(token_ids)
    return value


def shape_text_for_exact_count(
    count: Callable[[str], int], target: int, *, max_search: int = 256
) -> tuple[str, int]:
    """Find repeated neutral text whose rendered prompt has exactly target tokens."""

    cache: dict[int, int] = {}

    def measured(repetitions: int) -> int:
        if repetitions not in cache:
            cache[repetitions] = count("probe" + " x" * repetitions)
        return cache[repetitions]

    low, high = 0, max(1, target)
    while measured(high) < target:
        high *= 2
        if high > target * 8 + 8192:
            raise RuntimeError("could not bracket exact prompt length")
    while low < high:
        middle = (low + high) // 2
        if measured(middle) < target:
            low = middle + 1
        else:
            high = middle
    for repetitions in range(max(0, low - max_search), low + max_search + 1):
        observed = measured(repetitions)
        if observed == target:
            return "probe" + " x" * repetitions, observed
        if observed > target + max_search:
            break
    raise RuntimeError(f"could not construct exact {target}-token prompt")


def response_health(response: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        token_ids = choice.get("token_ids") or response.get("token_ids") or []
        stack: list[Any] = [response]
        finite = True
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, float) and not math.isfinite(current):
                finite = False
        zero_run = 0
        longest_zero_run = 0
        for token in token_ids:
            zero_run = zero_run + 1 if token == 0 else 0
            longest_zero_run = max(longest_zero_run, zero_run)
        content = message.get("content")
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        nonempty = bool(content or reasoning or message.get("tool_calls"))
        healthy = bool(
            choice.get("finish_reason")
            and finite
            and nonempty
            and bool(token_ids)
            and longest_zero_run < 4
            and all(type(token) is int and token >= 0 for token in token_ids)
        )
        return {
            "healthy": healthy,
            "finish_reason": choice.get("finish_reason"),
            "token_count": len(token_ids),
            "token_ids_present": bool(token_ids),
            "longest_zero_run": longest_zero_run,
            "finite": finite,
            "nonempty": nonempty,
        }
    except (KeyError, IndexError, TypeError):
        return {"healthy": False, "malformed": True}


def base_payload(model: str, messages: list[dict[str, Any]], salt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "seed": 9317,
        "max_tokens": 256,
        "thinking_token_budget": 64,
        "chat_template_kwargs": {"enable_thinking": True},
        "cache_salt": salt,
        "return_token_ids": True,
    }


def assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    message = dict(response["choices"][0]["message"])
    message.setdefault("role", "assistant")
    return message


def build_exact_messages(
    base_url: str,
    model: str,
    *,
    target: int,
    image_url: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    system = {"role": "system", "content": "Answer this cache safety probe briefly."}

    def messages(text: str) -> list[dict[str, Any]]:
        if image_url is None:
            content: Any = text
        else:
            content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ]
        return [system, {"role": "user", "content": content}]

    text, observed = shape_text_for_exact_count(
        lambda value: token_count(base_url, model, messages(value)), target
    )
    return messages(text), observed


def run_pair(
    *,
    base_url: str,
    model: str,
    ports: tuple[int, ...],
    messages: list[dict[str, Any]],
    block_size: int,
    unsafe_window: int,
    second_image_url: str | None = None,
) -> dict[str, Any]:
    salt = new_salt()
    url = base_url.rstrip("/") + "/v1/chat/completions"
    before = metrics(ports)
    first, first_headers = request_json(url, base_payload(model, messages, salt))
    middle = metrics(ports)
    first_prompt_tokens = (first.get("usage") or {}).get("prompt_tokens")
    if type(first_prompt_tokens) is not int:
        raise ValueError("first response omitted prompt_tokens")
    continuation: Any
    long_text = "Continue the independent probe." + " continuation" * (block_size + 64)
    if second_image_url is None:
        continuation = long_text
    else:
        continuation = [
            {"type": "image_url", "image_url": {"url": second_image_url}},
            {"type": "text", "text": long_text},
        ]
    grown_messages = [
        *messages,
        assistant_message(first),
        {"role": "user", "content": continuation},
    ]
    grown_count = token_count(base_url, model, grown_messages)
    if grown_count - first_prompt_tokens < block_size:
        raise ValueError("second request does not prefill at least one new block")
    second, second_headers = request_json(
        url, base_payload(model, grown_messages, salt)
    )
    after = metrics(ports)
    first_replica = first_headers.get("x-ifv-qwen-replica")
    second_replica = second_headers.get("x-ifv-qwen-replica")
    remainder = first_prompt_tokens % block_size
    expected_rotation = 0 < remainder <= unsafe_window
    second_metrics = delta(middle, after)
    return {
        "first_prompt_tokens": first_prompt_tokens,
        "first_remainder": remainder,
        "second_prompt_tokens": (second.get("usage") or {}).get("prompt_tokens"),
        "new_prefill_tokens": grown_count - first_prompt_tokens,
        "expected_rotation": expected_rotation,
        "same_replica": bool(first_replica and first_replica == second_replica),
        "replica": first_replica,
        "first_health": response_health(first),
        "second_health": response_health(second),
        "first_metrics": delta(before, middle),
        "second_metrics": second_metrics,
        "cache_behavior_passed": bool(
            second_metrics["prefix_hits"] == 0
            if expected_rotation
            else second_metrics["prefix_hits"] > 0
        ),
    }


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:19025")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--unsafe-window", type=int, default=16)
    parser.add_argument("--minimum-blocks", type=int, default=8)
    parser.add_argument("--ports", default="19002,19003,19004,19005")
    parser.add_argument("--stress-domains", type=int, default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ports = tuple(int(value) for value in args.ports.split(","))
    health, _ = request_json(args.base_url.rstrip("/") + "/health", timeout=30)
    if health.get("prefix_cache_policy") != "case_isolated":
        raise ValueError("probe requires a case-isolated gateway")
    if health.get("prefix_cache_block_size") != args.block_size:
        raise ValueError("gateway block-size attestation mismatch")
    if health.get("prefix_cache_unsafe_window") != args.unsafe_window:
        raise ValueError("gateway unsafe-window attestation mismatch")
    if len(health.get("replicas") or []) != 4 or not all(
        row.get("healthy") is True for row in health["replicas"]
    ):
        raise ValueError("probe requires four healthy replicas")

    started = time.time()
    boundary_rows: list[dict[str, Any]] = []
    for remainder in range(0, 21):
        target = args.minimum_blocks * args.block_size + remainder
        messages, observed = build_exact_messages(
            args.base_url, args.model, target=target
        )
        if observed != target:
            raise AssertionError("prompt shaper returned the wrong token count")
        row = run_pair(
            base_url=args.base_url,
            model=args.model,
            ports=ports,
            messages=messages,
            block_size=args.block_size,
            unsafe_window=args.unsafe_window,
        )
        row["target_remainder"] = remainder
        if row["first_remainder"] != remainder:
            raise AssertionError(
                f"generation prompt remainder changed: {row['first_remainder']} != {remainder}"
            )
        boundary_rows.append(row)
        atomic_json(output / "boundary-progress.json", boundary_rows)

    data_url = image_data_url(args.image.resolve())
    multimodal_target = args.minimum_blocks * args.block_size + args.unsafe_window + 8
    multimodal_messages, _ = build_exact_messages(
        args.base_url,
        args.model,
        target=multimodal_target,
        image_url=data_url,
    )
    multimodal = run_pair(
        base_url=args.base_url,
        model=args.model,
        ports=ports,
        messages=multimodal_messages,
        block_size=args.block_size,
        unsafe_window=args.unsafe_window,
        second_image_url=data_url,
    )

    stress_messages, _ = build_exact_messages(
        args.base_url,
        args.model,
        target=args.minimum_blocks * args.block_size + args.unsafe_window + 8,
    )
    stress_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(32, args.stress_domains)) as pool:
        futures = [
            pool.submit(
                run_pair,
                base_url=args.base_url,
                model=args.model,
                ports=ports,
                messages=stress_messages,
                block_size=args.block_size,
                unsafe_window=args.unsafe_window,
            )
            for _ in range(args.stress_domains)
        ]
        for future in as_completed(futures):
            stress_rows.append(future.result())

    final_health, _ = request_json(args.base_url.rstrip("/") + "/health", timeout=30)
    all_rows = [*boundary_rows, multimodal, *stress_rows]
    replicas = sorted({row.get("replica") for row in stress_rows if row.get("replica")})
    passed = bool(
        all(row["first_health"]["healthy"] for row in all_rows)
        and all(row["second_health"]["healthy"] for row in all_rows)
        and all(row["same_replica"] for row in all_rows)
        and all(row["cache_behavior_passed"] for row in all_rows)
        and all(row["first_metrics"]["corrupted"] == 0 for row in all_rows)
        and all(row["second_metrics"]["corrupted"] == 0 for row in all_rows)
        and len(replicas) == 4
        and multimodal["second_metrics"]["prefix_hits"] > 0
    )
    verdict = {
        "schema_version": "ifv-qwen35-hybrid-prefix-cache-gate-v1",
        "passed": passed,
        "block_size": args.block_size,
        "unsafe_window": args.unsafe_window,
        "boundary_rows": boundary_rows,
        "incremental_multimodal": multimodal,
        "stress": {
            "domains": len(stress_rows),
            "replicas": replicas,
            "healthy": sum(
                row["first_health"]["healthy"] and row["second_health"]["healthy"]
                for row in stress_rows
            ),
            "cache_behavior_passed": sum(
                row["cache_behavior_passed"] for row in stress_rows
            ),
            "corrupted_delta": sum(
                row["second_metrics"]["corrupted"] for row in stress_rows
            ),
        },
        "gateway_before": {
            key: health.get(key)
            for key in (
                "prefix_cache_policy",
                "prefix_cache_block_size",
                "prefix_cache_unsafe_window",
                "cache_salt_rotations",
                "cache_salt_rotation_reasons",
            )
        },
        "gateway_after": {
            key: final_health.get(key)
            for key in (
                "prefix_cache_policy",
                "prefix_cache_block_size",
                "prefix_cache_unsafe_window",
                "cache_salt_rotations",
                "cache_salt_rotation_reasons",
                "cache_domain_replicas",
            )
        },
        "elapsed_seconds": time.time() - started,
        "large_payloads_archived": False,
    }
    atomic_json(output / "verdict.json", verdict)
    if not passed:
        raise RuntimeError("hybrid prefix-cache safety gate rejected this configuration")


if __name__ == "__main__":
    main()
