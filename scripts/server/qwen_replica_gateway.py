"""Load-aware gateway for independent OpenAI-compatible Qwen replicas.

This process is intentionally small: it exposes one local OpenAI-compatible
endpoint and forwards each non-streaming request to one healthy single-GPU
replica.  It is useful on hosts where a model fits on one GPU but the available
serving backend cannot continuously batch that model across several GPUs.

Agent requests have a heavy-tailed duration distribution. Blind round-robin
therefore accumulates a queue behind a replica that happened to receive several
long generations while other replicas drain. The gateway tracks requests that
have not yet returned and always selects a least-loaded replica, using a rotating
cursor only to break ties.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response


def _replica_urls() -> list[str]:
    values = [
        item.strip().rstrip("/")
        for item in os.environ.get("QWEN_REPLICA_BACKENDS", "").split(",")
        if item.strip()
    ]
    if not values:
        raise RuntimeError("QWEN_REPLICA_BACKENDS must list at least one replica URL")
    return values


REPLICA_URLS = _replica_urls()
PINNED_MODEL_ID = os.environ.get("QWEN_REPLICA_MODEL_ID", "").strip()
REQUEST_TIMEOUT_SECONDS = max(
    30.0,
    float(os.environ.get("QWEN_REPLICA_GATEWAY_TIMEOUT_SECONDS", "900")),
)
_counter = itertools.count()
_inflight = [0 for _ in REPLICA_URLS]


def _acquire_replica(excluded: set[int] | None = None) -> tuple[int, str]:
    """Reserve a least-loaded replica without yielding the event loop.

    Uvicorn runs this gateway in one event loop. Selection and increment contain
    no ``await``, so another request cannot observe a selected replica before its
    reservation is recorded.
    """

    excluded = excluded or set()
    candidates = [
        index for index in range(len(REPLICA_URLS)) if index not in excluded
    ]
    if not candidates:
        raise RuntimeError("no Qwen replica remains available for this request")
    minimum = min(_inflight[index] for index in candidates)
    start = next(_counter) % len(REPLICA_URLS)
    selected = next(
        index
        for offset in range(len(REPLICA_URLS))
        if (index := (start + offset) % len(REPLICA_URLS)) in candidates
        and _inflight[index] == minimum
    )
    _inflight[selected] += 1
    return selected, REPLICA_URLS[selected]


def _acquire_replica_at(index: int) -> tuple[int, str]:
    """Reserve one specific replica for a request-scoped sticky route."""

    if not 0 <= index < len(REPLICA_URLS):
        raise IndexError(f"Qwen replica index {index} is out of range")
    _inflight[index] += 1
    return index, REPLICA_URLS[index]


def _release_replica(index: int) -> None:
    if _inflight[index] <= 0:
        raise RuntimeError(f"Qwen replica {index} has no in-flight reservation")
    _inflight[index] -= 1


def _replica_loads() -> list[int]:
    return list(_inflight)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    transport = httpx.AsyncHTTPTransport(retries=0)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0)
    _.state.http = httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        trust_env=False,
    )
    try:
        yield
    finally:
        await _.state.http.aclose()


app = FastAPI(lifespan=_lifespan)


def _forward_headers(request: Request) -> dict[str, str]:
    excluded = {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    excluded = {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in excluded
    }


def _normalize_request_body(request: Request, payload: bytes) -> bytes:
    """Translate the public served model name for pinned Transformers workers."""

    if not PINNED_MODEL_ID:
        return payload
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return payload
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return payload
    if not isinstance(parsed, dict):
        return payload
    parsed["model"] = PINNED_MODEL_ID
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


async def _request_replica(
    request: Request,
    path: str,
    *,
    retry_transport_once: bool,
) -> httpx.Response:
    payload = _normalize_request_body(request, await request.body())
    headers = _forward_headers(request)
    attempts = len(REPLICA_URLS) if retry_transport_once else 1
    failures: list[str] = []
    excluded: set[int] = set()

    for _ in range(attempts):
        index, replica = _acquire_replica(excluded)
        try:
            response = await request.app.state.http.request(
                request.method,
                f"{replica}/{path.lstrip('/')}",
                content=payload,
                headers=headers,
            )
        except httpx.TransportError as exc:
            failures.append(f"{replica}: {type(exc).__name__}")
            excluded.add(index)
            continue
        finally:
            _release_replica(index)
        response.headers["x-ifv-qwen-replica"] = replica
        return response

    raise HTTPException(
        status_code=503,
        detail=(
            "all Qwen replicas failed before responding: "
            + ", ".join(failures)
        ),
    )


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    async def probe(replica: str) -> bool:
        try:
            response = await request.app.state.http.get(f"{replica}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    status = await asyncio.gather(*(probe(replica) for replica in REPLICA_URLS))
    if not any(status):
        raise HTTPException(status_code=503, detail="no Qwen replica is healthy")
    return {
        "status": "ok",
        "replicas": [
            {"url": replica, "healthy": healthy, "inflight": inflight}
            for replica, healthy, inflight in zip(
                REPLICA_URLS,
                status,
                _replica_loads(),
            )
        ],
    }


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(request: Request, path: str) -> Response:
    # Agent calls are non-streaming JSON. Retrying only transport failures before
    # a backend response is safe and avoids silently duplicating a completed
    # generation on another replica.
    response = await _request_replica(
        request,
        path,
        retry_transport_once=request.method in {"GET", "POST"},
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=_response_headers(response),
        media_type=response.headers.get("content-type"),
    )
