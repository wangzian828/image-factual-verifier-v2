"""Measure current Gemini Interactions capacity with one reproducible load profile.

This is intentionally a provider-capacity probe rather than a semantic benchmark.
It disables client retries, records every attempted request, and reports the same
four fields used in rollout capacity checks: raw success rate, success rate after
excluding content blocks, successful returns per minute, and p90 end-to-end time.

The default ``mixed`` profile resembles the request shapes used by IFV: structured
text output, an initial native tool call, and image-conditioned structured output.
It does not include labels, private gold, source URLs, or other evaluator-only data.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import mimetypes
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.gemini import (  # noqa: E402
    GeminiInteractionsClient,
    GeminiInteractionsHTTPError,
    extract_function_calls,
    extract_text,
)


load_dotenv()


CONTENT_BLOCK_MARKERS = (
    "content block",
    "content_block",
    "blocked by",
    "safety",
    "policy violation",
    "harm category",
)
PROFILE_NAMES = frozenset({"mixed", "text_json", "image_json", "tool_call"})


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }


def _tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "capacity_probe_tool",
        "description": "Return one boolean field for an API capacity probe.",
        "parameters": _schema(),
    }


def _image_item(image_path: Path) -> dict[str, str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return {
        "type": "image",
        "mime_type": mime_type,
        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
    }


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.90) - 1)
    return ordered[index]


def _content_blocked(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in CONTENT_BLOCK_MARKERS)


def _request_kind(profile: str, sequence: int) -> str:
    if profile != "mixed":
        return profile
    # Keep a 1:1:2 text / tool / image blend. All requests remain one native
    # Interaction create call, which makes returned-request throughput comparable.
    return ("text_json", "tool_call", "image_json", "image_json")[sequence % 4]


def _request_kwargs(
    kind: str,
    *,
    image: Mapping[str, str] | None,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    if kind == "text_json":
        return {
            "input": "Return only the required JSON object with ok=true.",
            "response_format": response_format,
            "generation_config": generation_config,
        }
    if kind == "image_json":
        if image is None:
            raise ValueError("image_json load requires --image")
        return {
            "input": [
                dict(image),
                {
                    "type": "text",
                    "text": "Inspect this image and return only the required JSON object with ok=true.",
                },
            ],
            "response_format": response_format,
            "generation_config": generation_config,
        }
    if kind == "tool_call":
        return {
            "input": "Call capacity_probe_tool exactly once with ok=true.",
            "tools": [_tool()],
            "generation_config": generation_config,
        }
    raise ValueError(f"unsupported request kind: {kind}")


async def _run_one(
    sequence: int,
    *,
    client: GeminiInteractionsClient,
    model: str,
    profile: str,
    image: Mapping[str, str] | None,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
    store: bool,
) -> dict[str, Any]:
    kind = _request_kind(profile, sequence)
    started = time.monotonic()
    record: dict[str, Any] = {
        "sequence": sequence,
        "request_kind": kind,
        "status": "error",
    }
    try:
        kwargs = _request_kwargs(
            kind,
            image=image,
            response_format=response_format,
            generation_config=generation_config,
        )
        async with semaphore:
            payload = await client.create(model=model, store=store, **kwargs)
        if kind == "tool_call":
            if not extract_function_calls(payload):
                raise RuntimeError("tool-call probe returned no native function call")
        elif not extract_text(payload).strip():
            raise RuntimeError("structured probe returned no output text")
        record.update(
            {
                "status": "success",
                "interaction_status": payload.get("status"),
                "interaction_id": payload.get("id"),
            }
        )
    except Exception as exc:  # Per-request failures are benchmark output, not fatal.
        detail = str(exc)
        blocked = _content_blocked(detail)
        record.update(
            {
                "status": "content_blocked" if blocked else "error",
                "error_type": type(exc).__name__,
            }
        )
        if isinstance(exc, GeminiInteractionsHTTPError):
            record["http_status"] = exc.status_code
    record["elapsed_seconds"] = round(time.monotonic() - started, 4)
    return record


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).expanduser().resolve() if args.image else None
    requires_image = args.profile in {"mixed", "image_json"}
    if requires_image and (image_path is None or not image_path.is_file()):
        raise ValueError(f"profile {args.profile!r} requires an existing --image")
    image = _image_item(image_path) if image_path is not None else None

    # These values are read while the client and its process-wide gate are first
    # constructed. Set both deliberately so a nominal 256/512 benchmark is not
    # silently reduced by the normal 128-connection transport default.
    os.environ["GEMINI_MAX_INFLIGHT_REQUESTS"] = str(args.concurrency)
    os.environ["IFV_HTTPX_MAX_CONNECTIONS"] = str(args.concurrency)
    os.environ["IFV_HTTPX_MAX_KEEPALIVE_CONNECTIONS"] = "0"

    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": _schema(),
    }
    generation_config = {
        "thinking_level": args.thinking_level,
        "max_output_tokens": args.max_output_tokens,
    }
    config = {
        "schema_version": "ifv-gemini-concurrency-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "profile": args.profile,
        "request_count": args.request_count,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "max_retries": 0,
        "thinking_level": args.thinking_level,
        "max_output_tokens": args.max_output_tokens,
        "store": bool(args.store),
        "image_path": str(image_path) if image_path else None,
        "request_gate": args.concurrency,
        "httpx_max_connections": args.concurrency,
        "p90_definition": "90th percentile of all request end-to-end durations",
        "raw_success_definition": "successful validated provider response / all attempted requests",
        "success_excluding_content_blocks_definition": "successful validated provider response / (all attempts - content blocks)",
        "successful_returns_per_minute_definition": "successful validated provider responses / wall-clock minute",
    }
    _write_json(output_dir / "run-config.json", config)

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    async with GeminiInteractionsClient(
        timeout=args.timeout,
        max_retries=0,
    ) as client:
        tasks = [
            asyncio.create_task(
                _run_one(
                    sequence,
                    client=client,
                    model=args.model,
                    profile=args.profile,
                    image=image,
                    response_format=response_format,
                    generation_config=generation_config,
                    semaphore=semaphore,
                    store=bool(args.store),
                )
            )
            for sequence in range(args.request_count)
        ]
        with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                record = await task
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    elapsed = time.monotonic() - started

    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    error_types = Counter(
        str(record.get("error_type") or "unknown")
        for record in records
        if record.get("status") == "error"
    )
    successes = int(statuses.get("success", 0))
    content_blocks = int(statuses.get("content_blocked", 0))
    non_blocked = args.request_count - content_blocks
    all_latencies = [float(record["elapsed_seconds"]) for record in records]
    successful_latencies = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("status") == "success"
    ]
    summary = {
        **config,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(elapsed, 4),
        "status_counts": dict(statuses),
        "error_types": dict(error_types),
        "raw_success_rate": round(successes / args.request_count, 6),
        "success_rate_excluding_content_blocks": (
            round(successes / non_blocked, 6) if non_blocked else None
        ),
        "successful_returns_per_minute": (
            round(successes * 60.0 / elapsed, 4) if elapsed > 0 else None
        ),
        "p90_seconds": round(_p90(all_latencies) or 0.0, 4),
        "success_p90_seconds": round(_p90(successful_latencies) or 0.0, 4),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--profile", choices=sorted(PROFILE_NAMES), default="mixed")
    parser.add_argument("--image")
    parser.add_argument("--request-count", type=int, default=512)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="persist provider interactions, matching rollout and audit calls (default: true)",
    )
    args = parser.parse_args()
    if args.request_count < 1:
        parser.error("--request-count must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
