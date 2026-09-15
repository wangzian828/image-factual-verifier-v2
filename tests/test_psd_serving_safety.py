from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from scripts.server.psd_thinking_state import ThinkingState, thinking_budget, update_slots


def params(budget=8192, max_tokens=32768):
    return NS(extra_args={'ifv_thinking_budget': budget}, max_tokens=max_tokens)


def test_thinking_boundary_closes_once_then_releases_answer():
    output = []
    state = ThinkingState(3, 99, output)
    assert not state.needs_close()
    output.extend([1, 2])
    assert not state.needs_close()
    output.append(3)
    assert state.needs_close()
    assert state.needs_close()  # No counter increment without a new emitted token.
    assert state.reasoning_tokens == 3
    output.extend([99, 10, 11, 12])
    assert not state.needs_close()
    assert state.reasoning_tokens == 3


def test_early_closure_and_reopening_text_are_untouched():
    output = [1, 99, 88, 3, 4, 5]
    state = ThinkingState(3, 99, output)
    assert not state.needs_close()
    assert state.reasoning_tokens == 1


def test_shrinking_output_list_refused():
    output = [1, 2]
    state = ThinkingState(3, 99, output)
    state.needs_close()
    output.clear()
    with pytest.raises(ValueError):
        state.needs_close()


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, '8192', 8193])
def test_invalid_budget_refused(budget):
    with pytest.raises(ValueError):
        thinking_budget(params(budget))


def test_output_headroom_and_opt_out():
    assert thinking_budget(params()) == 8192
    assert thinking_budget(NS(extra_args=None, max_tokens=1)) is None
    with pytest.raises(ValueError):
        thinking_budget(params(max_tokens=8193))


def test_slot_reuse_swap_compaction_and_disabled_request():
    first, second, replacement = object(), object(), object()
    slots = {0: first, 1: second}
    factory = lambda value, *_: value
    update_slots(slots, NS(batch_size=2, removed=[0],
                          added=[(0, replacement, None, [])],
                          moved=[(0, 1, NS(name='SWAP'))]), factory)
    assert slots == {0: second, 1: replacement}
    update_slots(slots, NS(batch_size=1, removed=[0], added=[],
                          moved=[(1, 0, NS(name='UNIDIRECTIONAL'))]), factory)
    assert slots == {0: replacement}
    update_slots(slots, NS(batch_size=1, removed=[], added=[(0, None, None, [])], moved=[]), factory)
    assert slots == {}


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv('QWEN_REPLICA_BACKENDS', 'http://replica-0,http://replica-1')
    module = importlib.import_module('scripts.server.psd_qwen_gateway')
    monkeypatch.setattr(module.base, 'REPLICA_URLS', ['http://replica-0', 'http://replica-1'])
    monkeypatch.setattr(module.base, '_inflight', [0, 0])
    monkeypatch.setattr(module.base, 'PINNED_MODEL_ID', '')
    return module


def test_gateway_preserves_messages_tools_images_and_sampling(gateway):
    data = {'model': 'sft', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,EXACT'}}]}],
        'tools': [{'type': 'function', 'function': {'name': 'search'}}],
        'temperature': 0.7, 'top_p': 0.8, 'seed': 42,
        'max_tokens': 32768, 'thinking_token_budget': 8192}
    result = json.loads(gateway.normalize_payload(json.dumps(data).encode()))
    for key in set(data) - {'thinking_token_budget'}:
        assert result[key] == data[key]
    assert result['vllm_xargs'] == {'ifv_thinking_budget': 8192}
    assert 'thinking_token_budget' not in result


@pytest.mark.parametrize('changes', [
    {'stream': True}, {'thinking_token_budget': None}, {'thinking_token_budget': True},
    {'max_tokens': 8192}, {'vllm_xargs': {'ifv_thinking_budget': 1}}])
def test_gateway_rejects_silent_budget_bypass(gateway, changes):
    value = {'max_tokens': 32768, 'thinking_token_budget': 8192, **changes}
    with pytest.raises(ValueError):
        gateway.normalize_payload(json.dumps(value).encode())


def test_non_thinking_request_untouched(gateway):
    value = {'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}}
    result = json.loads(gateway.normalize_payload(json.dumps(value).encode()))
    assert result.pop('cache_salt').startswith('ifv-psd-isolated-')
    assert result == value


def test_requests_cannot_share_a_corrupted_prefix_cache(gateway):
    value = {'max_tokens': 32768, 'thinking_token_budget': 8192, 'cache_salt': 'shared'}
    one = json.loads(gateway.normalize_payload(json.dumps(value).encode()))
    two = json.loads(gateway.normalize_payload(json.dumps(value).encode()))
    assert one.pop('cache_salt') != two.pop('cache_salt')
    assert one == two


def test_tokenizer_only_remaps_attested_alias(gateway, monkeypatch):
    monkeypatch.setenv('PSD_PUBLIC_MODEL_ALIAS', 'public-qwen3.5-policy')
    monkeypatch.setattr(gateway.base, 'PINNED_MODEL_ID', 'backend-policy')
    body = {'model': 'public-qwen3.5-policy', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '{ "original": " spacing " }'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,EXACT'}}]}],
        'tools': [{'type': 'function', 'function': {'name': 'verify'}}],
        'chat_template_kwargs': {'enable_thinking': True}, 'add_special_tokens': False}
    result = json.loads(gateway.normalize_tokenize_payload(json.dumps(body).encode()))
    assert result == {**body, 'model': 'backend-policy'}
    with pytest.raises(ValueError):
        gateway.normalize_tokenize_payload(json.dumps({**body, 'model': 'other'}).encode())
    with pytest.raises(ValueError):
        gateway.normalize_tokenize_payload(json.dumps({'model': 'backend-policy'}).encode())


def test_alias_card_keeps_actual_weight_root_and_backend_card(gateway, monkeypatch):
    monkeypatch.setenv('PSD_PUBLIC_MODEL_ALIAS', 'public-qwen3.5-policy')
    monkeypatch.setattr(gateway.base, 'PINNED_MODEL_ID', 'backend-policy')
    backend = {'id': 'backend-policy', 'root': '/weights/frozen', 'max_model_len': 131072}
    wire = json.dumps({'object': 'list', 'data': [backend]}).encode()
    result = json.loads(gateway.model_alias_payload(wire))
    assert result['data'][0] == backend
    assert result['data'][1] == {**backend, 'id': 'public-qwen3.5-policy', 'ifv_backend_model_id': 'backend-policy'}
    assert gateway.model_alias_payload(gateway.model_alias_payload(wire)) == gateway.model_alias_payload(wire)
    with pytest.raises(ValueError):
        gateway.model_alias_payload(json.dumps({'data': []}).encode())
    with pytest.raises(ValueError):
        gateway.model_alias_payload(json.dumps({'data': [backend, {'id': 'public-qwen3.5-policy', 'root': '/other'}]}).encode())


def test_tokenize_route_forwards_once_without_generation_options(gateway, monkeypatch):
    monkeypatch.setenv('PSD_PUBLIC_MODEL_ALIAS', 'public-qwen3.5-policy')
    monkeypatch.setattr(gateway.base, 'PINNED_MODEL_ID', 'backend-policy')
    seen = []
    async def once(request, path, payload):
        seen.append((path, json.loads(payload)))
        return httpx.Response(200, json={'tokens': [1, 2, 3], 'count': 3})
    monkeypatch.setattr(gateway, 'request_once', once)
    body = {'model': 'public-qwen3.5-policy', 'prompt': 'exact  spacing'}
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
            reply = await client.post('/tokenize', json=body)
            assert reply.status_code == 200 and reply.json()['tokens'] == [1, 2, 3]
            invalid = await client.post('/tokenize', json={**body, 'model': 'other'})
            assert invalid.status_code == 400
    asyncio.run(run())
    assert seen == [('tokenize', {**body, 'model': 'backend-policy'})]


def test_model_alias_route_recomputes_response_length(gateway, monkeypatch):
    monkeypatch.setenv('PSD_PUBLIC_MODEL_ALIAS', 'public-qwen3.5-policy')
    monkeypatch.setattr(gateway.base, 'PINNED_MODEL_ID', 'backend-policy')
    body = b'{"data":[{"id":"backend-policy","root":"/weights","max_model_len":131072}]}'
    async def once(request, path, payload):
        assert request.method == 'GET' and path == 'v1/models'
        return httpx.Response(200, content=body, headers={'content-length': str(len(body))})
    monkeypatch.setattr(gateway, 'request_once', once)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
            reply = await client.get('/v1/models')
            assert len(reply.json()['data']) == 2
            assert int(reply.headers['content-length']) == len(reply.content) > len(body)
    asyncio.run(run())


def test_timeout_order_and_no_client_retry(gateway, monkeypatch):
    monkeypatch.setenv('PSD_GATEWAY_DEADLINE_SECONDS', '1200')
    monkeypatch.setenv('AGENT_LLM_REQUEST_TIMEOUT_SECONDS', '1230')
    monkeypatch.setenv('AGENT_STAGE_REQUEST_TIMEOUT_SECONDS', '1260')
    monkeypatch.setenv('AGENT_LLM_REQUEST_MAX_RETRIES', '0')
    assert gateway.deadline_config() == (1200, 1230, 1260)
    monkeypatch.setenv('AGENT_LLM_REQUEST_TIMEOUT_SECONDS', '900')
    with pytest.raises(ValueError):
        gateway.deadline_config()
    monkeypatch.setenv('AGENT_LLM_REQUEST_TIMEOUT_SECONDS', '1230')
    monkeypatch.setenv('AGENT_LLM_REQUEST_MAX_RETRIES', '1')
    with pytest.raises(ValueError):
        gateway.deadline_config()


class Request:
    method = 'POST'
    headers = {'content-type': 'application/json'}

    def __init__(self, http, deadline=1, disconnected=False):
        self.app = NS(state=NS(http=http, deadline=deadline))
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


@pytest.mark.parametrize('error', [httpx.ReadTimeout, httpx.ConnectError])
def test_gateway_never_replays_transport_failure(gateway, error):
    class Http:
        calls = 0

        async def request(self, *args, **kwargs):
            self.calls += 1
            raise error('failure after dispatch is not proof of non-execution')

    http = Http()
    with pytest.raises(gateway.HTTPException):
        asyncio.run(gateway.request_once(Request(http), 'v1/chat/completions', b'{}'))
    assert http.calls == 1
    assert gateway.base._inflight == [0, 0]


@pytest.mark.parametrize('mode', ['disconnect', 'deadline', 'outer_cancel'])
def test_gateway_cancels_upstream_and_releases_reservation(gateway, mode):
    class Http:
        calls = 0
        cancelled = False

        async def request(self, *args, **kwargs):
            self.calls += 1
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    async def run():
        http = Http()
        request = Request(http, deadline=0.02, disconnected=mode == 'disconnect')
        task = asyncio.create_task(gateway.request_once(request, 'v1/chat/completions', b'{}'))
        if mode == 'outer_cancel':
            await asyncio.sleep(0.005)
            task.cancel()
        with pytest.raises((gateway.HTTPException, asyncio.CancelledError)):
            await task
        assert http.calls == 1 and http.cancelled
        assert gateway.base._inflight == [0, 0]
    asyncio.run(run())


def test_gateway_success_returns_original_body(gateway):
    class Http:
        async def request(self, *args, **kwargs):
            return httpx.Response(200, content=b'{"exact":true}')
    result = asyncio.run(gateway.request_once(Request(Http()), 'v1/chat/completions', b'{}'))
    assert result.content == b'{"exact":true}'
    assert gateway.base._inflight == [0, 0]


def test_deadline_closes_real_upstream_socket_after_response_headers(gateway, monkeypatch):
    async def run():
        closed = asyncio.Event()
        responded = asyncio.Event()
        seen = []

        async def backend(reader, writer):
            try:
                header = await reader.readuntil(b'\r\n\r\n')
                seen.append(header)
                length = next(int(line.split(b':', 1)[1]) for line in header.split(b'\r\n')
                              if line.lower().startswith(b'content-length:'))
                await reader.readexactly(length)
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 9999\r\n\r\nx')
                await writer.drain()
                responded.set()
                assert await reader.read() == b''
                closed.set()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(backend, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(gateway.base, 'REPLICA_URLS', [f'http://127.0.0.1:{port}'])
        monkeypatch.setattr(gateway.base, '_inflight', [0])
        try:
            # Allow lazy HTTP backend imports on a cold/shared server before the
            # deliberate body stall. Assert headers really arrived before expiry.
            async with httpx.AsyncClient(trust_env=False, timeout=15.0) as http:
                with pytest.raises(gateway.HTTPException) as exc:
                    await gateway.request_once(Request(http, deadline=5.0), 'v1/chat/completions', b'{}')
                assert exc.value.status_code == 504
                assert responded.is_set()
                await asyncio.wait_for(closed.wait(), 2)
                assert len(seen) == 1
                assert gateway.base._inflight == [0]
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())
