"""Isolated PSD gateway: exact thinking-budget translation and no POST replay.

Run one uvicorn worker. The frozen evaluation gateway remains unchanged.
Closing the upstream connection on cancellation must still be GPU-verified.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from scripts.server import qwen_replica_gateway as base


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
    # first decoding step even with the budget disabled. Unique cache domains
    # bypass that unsafe reuse without changing images, messages, or sampling.
    parsed['cache_salt'] = 'ifv-psd-isolated-' + uuid.uuid4().hex
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':')).encode()


@asynccontextmanager
async def lifespan(app):
    gateway, client, stage = deadline_config()
    app.state.deadline = gateway
    app.state.timeout_contract = {'gateway': gateway, 'model_client': client, 'stage': stage}
    app.state.post_dispatches = 0
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


async def request_once(request, path, payload):
    index, replica = base._acquire_replica()
    if request.method == 'POST':
        request.app.state.post_dispatches = getattr(request.app.state, 'post_dispatches', 0) + 1
    upstream = asyncio.create_task(request.app.state.http.request(
        request.method, f'{replica}/{path.lstrip("/")}',
        content=payload, headers=base._forward_headers(request)))
    watcher = asyncio.create_task(disconnected(request))
    try:
        done, _ = await asyncio.wait(
            [upstream, watcher], timeout=request.app.state.deadline,
            return_when=asyncio.FIRST_COMPLETED)
        if upstream in done:
            response = upstream.result()
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
        for task in [upstream, watcher]:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(upstream, watcher, return_exceptions=True)
        finally:
            base._release_replica(index)


@app.get('/health')
async def health(request: Request):
    result = await base.health(request)
    result.update(protocol='isolated-psd-thinking-budget-v1',
                  timeout_contract=request.app.state.timeout_contract,
                  post_retries=0, gpu_boundary_validated=False,
                  prefix_cache_policy='unique_salt_per_request',
                  post_dispatches=request.app.state.post_dispatches)
    return result


@app.api_route('/v1/{path:path}', methods=['POST', 'GET'])
async def proxy(request: Request, path: str):
    payload = await request.body()
    if request.method == 'POST':
        if path != 'chat/completions':
            raise HTTPException(400, 'PSD gateway only accepts chat completions POST')
        try:
            payload = normalize_payload(payload)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(400, str(exc)) from exc
    response = await request_once(request, 'v1/' + path, payload)
    return Response(content=response.content, status_code=response.status_code,
                    headers=base._response_headers(response))
