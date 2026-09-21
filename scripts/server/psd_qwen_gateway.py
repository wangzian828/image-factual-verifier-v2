"""Isolated PSD gateway: exact thinking-budget translation and no POST replay.

Run one uvicorn worker. The frozen evaluation gateway remains unchanged.
Closing the upstream connection on cancellation must still be GPU-verified.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from scripts.server import qwen_replica_gateway as base
from scripts.server.psd_wire_capture import WireCapture
from scripts.server.psd_raw_logprobs import SEMANTICS, validate_raw_worker_receipts
from src.integrations.llm.prefix_cache import (
    CASE_CACHE_SALT_PREFIX,
    CASE_ISOLATED,
    prefix_cache_mode,
    validate_case_cache_salt,
)


@dataclass
class CacheDomain:
    """One rollout's backend affinity and current safe vLLM cache domain."""

    replica_index: int
    effective_salt: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_tokens: int | None = None
    rotations: int = 0
    last_rotation_reason: str | None = None


def prefix_cache_safety_config() -> tuple[int, int]:
    """Return the attested hybrid block size and conservative hazard window."""

    raw_block_size = os.environ.get("IFV_PREFIX_CACHE_BLOCK_SIZE")
    if prefix_cache_mode() == CASE_ISOLATED and raw_block_size is None:
        raise ValueError(
            "case-isolated prefix caching requires an explicitly attested "
            "IFV_PREFIX_CACHE_BLOCK_SIZE"
        )
    block_size = int(raw_block_size or "528")
    unsafe_window = int(os.environ.get("IFV_PREFIX_CACHE_UNSAFE_WINDOW", "16"))
    if block_size < 1:
        raise ValueError("IFV_PREFIX_CACHE_BLOCK_SIZE must be positive")
    if not 0 <= unsafe_window < block_size:
        raise ValueError(
            "IFV_PREFIX_CACHE_UNSAFE_WINDOW must be in [0, block_size)"
        )
    return block_size, unsafe_window


def cache_routing_salt(payload: bytes) -> str | None:
    """Read the validated client cache domain without changing the payload."""

    if prefix_cache_mode() != CASE_ISOLATED:
        return None
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return validate_case_cache_salt(parsed.get("cache_salt"))


def sticky_replica_index(cache_salt: str, replica_count: int) -> int:
    """Map a private rollout salt to one stable backend without exposing it."""

    if replica_count < 1:
        raise ValueError("replica_count must be positive")
    digest = hashlib.blake2b(cache_salt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % replica_count


def _domain_for(request: Request, cache_salt: str) -> CacheDomain:
    domains = getattr(request.app.state, "cache_domains", None)
    if domains is None:
        domains = request.app.state.cache_domains = {}
    domain = domains.get(cache_salt)
    if domain is None:
        loads = base._replica_loads()
        minimum = min(loads)
        candidates = [index for index, load in enumerate(loads) if load == minimum]
        tie_break = sticky_replica_index(cache_salt, len(candidates))
        domain = CacheDomain(
            # Balance the first turn, then pin every later turn to that replica.
            replica_index=candidates[tie_break],
            effective_salt=cache_salt,
        )
        domains[cache_salt] = domain
    return domain


def _effective_payload(payload: bytes, domain: CacheDomain | None) -> bytes:
    if domain is None:
        return payload
    parsed = json.loads(payload)
    parsed["cache_salt"] = domain.effective_salt
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()


def _rotate_domain(request: Request, domain: CacheDomain | None, reason: str) -> None:
    if domain is None:
        return
    # Keep the original private salt as the sticky-routing key, but make every
    # later backend request miss any state whose correctness is uncertain.
    domain.effective_salt = validate_case_cache_salt(
        CASE_CACHE_SALT_PREFIX + secrets.token_urlsafe(32)
    )
    domain.rotations += 1
    domain.last_rotation_reason = reason
    request.app.state.cache_salt_rotations = (
        getattr(request.app.state, "cache_salt_rotations", 0) + 1
    )
    reasons = getattr(request.app.state, "cache_salt_rotation_reasons", None)
    if reasons is None:
        reasons = request.app.state.cache_salt_rotation_reasons = {}
    reasons[reason] = reasons.get(reason, 0) + 1


def _update_domain_after_response(
    request: Request,
    domain: CacheDomain | None,
    response: httpx.Response,
) -> None:
    if domain is None:
        return
    if response.status_code != 200:
        _rotate_domain(request, domain, "non_200_response")
        return
    try:
        raw_prompt_tokens = (response.json().get("usage") or {})["prompt_tokens"]
        if type(raw_prompt_tokens) is not int or raw_prompt_tokens < 1:
            raise ValueError("prompt_tokens must be a positive integer")
        prompt_tokens = raw_prompt_tokens
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ):
        _rotate_domain(request, domain, "missing_or_invalid_prompt_tokens")
        return
    domain.prompt_tokens = prompt_tokens
    block_size, unsafe_window = request.app.state.prefix_cache_safety
    remainder = prompt_tokens % block_size
    if 0 < remainder <= unsafe_window:
        # A later request must not restore a hybrid GDN checkpoint produced
        # just past the aligned block boundary.  Rotate only this rollout's
        # effective domain; its private routing key and GPU affinity stay fixed.
        _rotate_domain(request, domain, "hybrid_block_boundary")


def raw_teacher_binding():
    marker = os.environ.get('PSD_POLICY_LOGPROB_SEMANTICS', '')
    if not marker:
        return None
    if marker != SEMANTICS:
        raise ValueError('Unknown PSD teacher logprob semantics')
    receipts = json.loads(os.environ.get('PSD_RAW_WORKER_RECEIPTS', '[]'))
    files = json.loads(os.environ.get('PSD_RAW_WORKER_FILES', '{}'))
    validate_raw_worker_receipts(receipts, base.REPLICA_URLS, files)
    return marker


def public_alias():
    return os.environ.get('PSD_PUBLIC_MODEL_ALIAS', '').strip()


def normalize_tokenize_payload(payload):
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError('Expected tokenizer request object')
    allowed = {value for value in [public_alias(), base.PINNED_MODEL_ID] if value}
    if not allowed or parsed.get('model') not in allowed:
        raise ValueError('Tokenizer model must identify this frozen policy')
    if not ('messages' in parsed or 'prompt' in parsed):
        raise ValueError('Tokenizer request needs messages or prompt')
    parsed['model'] = base.PINNED_MODEL_ID
    # Do not change text, tools, images, template kwargs, order or token flags.
    # This is non-generative: no budget plugin, salt or sampling parameters.
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':')).encode()


def model_alias_payload(payload):
    alias = public_alias()
    if not alias or alias == base.PINNED_MODEL_ID:
        return payload
    parsed = json.loads(payload)
    rows = parsed.get('data', [])
    backend = [row for row in rows if row.get('id') == base.PINNED_MODEL_ID]
    if len(backend) != 1 or not backend[0].get('root'):
        raise ValueError('Cannot attest alias without a unique actual backend model')
    existing = [row for row in rows if row.get('id') == alias]
    if existing:
        if len(existing) != 1 or existing[0].get('root') != backend[0]['root']:
            raise ValueError('Public alias conflicts with backend weights')
        return payload
    parsed['data'] = rows + [{**backend[0], 'id': alias,
                             'ifv_backend_model_id': base.PINNED_MODEL_ID}]
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':')).encode()


def deadline_config():
    gateway = float(os.environ.get('PSD_GATEWAY_DEADLINE_SECONDS', '1200'))
    client = float(os.environ.get('AGENT_LLM_REQUEST_TIMEOUT_SECONDS', '1230'))
    stage = float(os.environ.get('AGENT_STAGE_REQUEST_TIMEOUT_SECONDS', '1260'))
    if not 0 < gateway < client < stage:
        raise ValueError('PSD requires gateway deadline < model client timeout < stage timeout')
    if int(os.environ.get('AGENT_LLM_REQUEST_MAX_RETRIES', '0')) != 0:
        raise ValueError('PSD model client retries must be disabled to avoid POST replay')
    return gateway, client, stage


def normalize_payload(payload):
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError('Expected JSON object')
    if parsed.get('stream'):
        raise ValueError('This PSD gateway only supports non-streaming Agent requests')
    if base.PINNED_MODEL_ID:
        parsed['model'] = base.PINNED_MODEL_ID
    xargs = dict(parsed.get('vllm_xargs') or {})
    if any(key.startswith('ifv_') for key in xargs):
        raise ValueError('PSD thinking arguments must come from the explicit public budget')
    budget = parsed.pop('thinking_token_budget', None)
    enabled = (parsed.get('chat_template_kwargs') or {}).get('enable_thinking', True)
    if budget is not None and enabled:
        if type(budget) is not int or not 1 <= budget <= 8192:
            raise ValueError('thinking_token_budget must be an integer in [1, 8192]')
        limit = parsed.get('max_completion_tokens', parsed.get('max_tokens'))
        if type(limit) is not int or limit <= budget + 1:
            raise ValueError('Output budget must leave room for reasoning closure and an answer')
        xargs['ifv_thinking_budget'] = budget
    elif enabled:
        raise ValueError('Enabled thinking requires an explicit budget in isolated PSD')
    if xargs:
        parsed['vllm_xargs'] = xargs
    # A reproduced hybrid-state prefix-cache failure emitted token 0 from the
    # first decoding step even with the budget disabled. The production-safe
    # default therefore gives every request a unique domain. A future canary
    # may opt into one validated, unguessable domain per rollout; missing or
    # malformed scopes fail before dispatch rather than sharing accidentally.
    if prefix_cache_mode() == CASE_ISOLATED:
        parsed['cache_salt'] = validate_case_cache_salt(parsed.get('cache_salt'))
    else:
        parsed['cache_salt'] = 'ifv-psd-isolated-' + uuid.uuid4().hex
    if os.environ.get('PSD_DIAGNOSTIC_RETURN_TOKEN_IDS') == '1':
        if not os.environ.get('PSD_WIRE_CAPTURE_DIR'):
            raise ValueError('Diagnostic token IDs require an explicit wire archive')
        parsed['return_token_ids'] = True  # Observation only, not a sampling change.
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':')).encode()


@asynccontextmanager
async def lifespan(app):
    raw_teacher_binding()
    gateway, client, stage = deadline_config()
    app.state.deadline = gateway
    app.state.timeout_contract = {'gateway': gateway, 'model_client': client, 'stage': stage}
    app.state.post_dispatches = 0
    app.state.cache_domains = {}
    app.state.sticky_dispatches = 0
    app.state.cache_salt_rotations = 0
    app.state.cache_salt_rotation_reasons = {}
    app.state.prefix_cache_safety = prefix_cache_safety_config()
    app.state.wire_capture = WireCapture.from_env()
    app.state.http = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=0),
        timeout=httpx.Timeout(gateway, connect=10.0), trust_env=False)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


async def disconnected(request):
    while not await request.is_disconnected():
        await asyncio.sleep(0.1)


async def _request_once_locked(request, path, payload, domain):
    payload = _effective_payload(payload, domain)
    capture = getattr(request.app.state, 'wire_capture', None)
    ticket = None
    if capture is not None and request.method == 'POST' and path == 'v1/chat/completions':
        try:
            ticket = capture.begin(payload)
        except (OSError, ValueError) as exc:
            raise HTTPException(507, 'Diagnostic capture unavailable before dispatch') from exc
    if domain is None:
        index, replica = base._acquire_replica()
    else:
        index, replica = base._acquire_replica_at(domain.replica_index)
        request.app.state.sticky_dispatches += 1
    if request.method == 'POST':
        request.app.state.post_dispatches = getattr(request.app.state, 'post_dispatches', 0) + 1
    upstream = asyncio.create_task(request.app.state.http.request(
        request.method, f'{replica}/{path.lstrip("/")}',
        content=payload, headers=base._forward_headers(request)))
    watcher = asyncio.create_task(disconnected(request))
    response_observed = False
    response = None
    try:
        done, _ = await asyncio.wait(
            [upstream, watcher], timeout=request.app.state.deadline,
            return_when=asyncio.FIRST_COMPLETED)
        if upstream in done:
            response = upstream.result()
            _update_domain_after_response(request, domain, response)
            response_observed = True
            if ticket is not None:
                try:
                    capture.response(ticket, response.content, status_code=response.status_code, replica=replica)
                    response.headers['x-ifv-capture-id'] = ticket.name
                except OSError:
                    # Preserve generation result even when diagnostic disk writing fails.
                    logging.exception('PSD raw response capture failed; generation response preserved')
            response.headers['x-ifv-qwen-replica'] = replica
            return response
        if watcher in done:
            watcher.result()
            raise HTTPException(499, 'Client disconnected; upstream request cancelled, not retried')
        raise HTTPException(504, 'PSD gateway deadline exceeded; not retried')
    except httpx.TimeoutException as exc:
        raise HTTPException(504, 'Upstream timeout; not retried') from exc
    except httpx.TransportError as exc:
        raise HTTPException(502, 'Upstream transport failure; not retried') from exc
    finally:
        if domain is not None and not response_observed:
            _rotate_domain(request, domain, "incomplete_or_uncertain_request")
        for task in [upstream, watcher]:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(upstream, watcher, return_exceptions=True)
        finally:
            # ``AsyncClient.request`` buffers the body, so callers can still
            # consume ``response.content`` after this close.  Explicitly
            # release the upstream response here: relying on response-object
            # collection left vLLM sockets in CLOSE_WAIT under concurrent
            # non-streaming Agent traffic and could strand the downstream
            # request after generation had already finished.
            if response is not None:
                try:
                    await response.aclose()
                except Exception:
                    logging.exception('PSD upstream response close failed')
            base._release_replica(index)
            if ticket is not None and not (ticket/'response-meta.json').exists() and not (ticket/'error.json').exists():
                try:
                    capture.error(ticket, 'request_cancelled_or_transport_failed', replica=replica)
                except OSError:
                    logging.exception('PSD diagnostic failure receipt could not be written')


async def request_once(request, path, payload):
    cache_salt = (
        cache_routing_salt(payload)
        if request.method == "POST" and path == "v1/chat/completions"
        else None
    )
    domain = _domain_for(request, cache_salt) if cache_salt is not None else None
    if domain is None:
        return await _request_once_locked(request, path, payload, None)
    # A rollout is sequential by protocol.  This lock fails safe if two
    # components accidentally issue concurrent requests under the same salt.
    async with domain.lock:
        return await _request_once_locked(request, path, payload, domain)


@app.get('/health')
async def health(request: Request):
    result = await base.health(request)
    domain_replicas = {str(index): 0 for index in range(len(base.REPLICA_URLS))}
    for domain in request.app.state.cache_domains.values():
        domain_replicas[str(domain.replica_index)] += 1
    result.update(protocol='isolated-psd-thinking-budget-v1',
                  timeout_contract=request.app.state.timeout_contract,
                  post_retries=0, gpu_boundary_validated=False,
                  prefix_cache_policy=prefix_cache_mode(),
                  public_model_alias=public_alias() or None,
                  tokenizer_endpoint='/tokenize',
                  diagnostic_token_ids=os.environ.get('PSD_DIAGNOSTIC_RETURN_TOKEN_IDS') == '1',
                  post_dispatches=request.app.state.post_dispatches,
                  sticky_dispatches=request.app.state.sticky_dispatches,
                  cache_domains=len(request.app.state.cache_domains),
                  cache_domain_replicas=domain_replicas,
                  cache_salt_rotations=request.app.state.cache_salt_rotations,
                  cache_salt_rotation_reasons=dict(
                      request.app.state.cache_salt_rotation_reasons
                  ),
                  prefix_cache_block_size=request.app.state.prefix_cache_safety[0],
                  prefix_cache_unsafe_window=request.app.state.prefix_cache_safety[1])
    capture = getattr(request.app.state, 'wire_capture', None)
    result['wire_capture'] = capture.status() if capture else {'enabled':False}
    return result


@app.post('/tokenize')
async def tokenize(request: Request):
    try:
        payload = normalize_tokenize_payload(await request.body())
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    response = await request_once(request, 'tokenize', payload)
    return Response(content=response.content, status_code=response.status_code,
                    headers=base._response_headers(response))


@app.api_route('/v1/{path:path}', methods=['POST', 'GET'])
async def proxy(request: Request, path: str):
    payload = await request.body()
    if request.method == 'POST':
        if path != 'chat/completions':
            raise HTTPException(400, 'PSD gateway only accepts chat completions POST')
        try:
            raw_teacher_binding()  # Do not relabel after an unbound backend swap.
            payload = normalize_payload(payload)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(400, str(exc)) from exc
    response = await request_once(request, 'v1/' + path, payload)
    body = response.content
    if request.method == 'POST' and response.status_code == 200:
        marker = raw_teacher_binding()
        if marker:
            parsed = json.loads(body)
            parsed['ifv_policy_logprobs'] = marker
            body = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    if request.method == 'GET' and path == 'models' and response.status_code == 200:
        try:
            body = model_alias_payload(body)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(502, str(exc)) from exc
    return Response(content=body, status_code=response.status_code,
                    headers=base._response_headers(response))
