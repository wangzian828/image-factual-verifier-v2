import asyncio
import hashlib
import json

import pytest

from ifv_training.psd_gemini_judge import _request
from ifv_training.psd_slate import propose_slate


class Client:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {'id': 'test', 'status': 'completed', 'model': kwargs['model'],
                'outputs': [{'type': 'text', 'text': '{"accepted": false}'}]}


def activate(monkeypatch, tmp_path):
    root = tmp_path / 'search'
    root.mkdir(exist_ok=True)
    path = tmp_path / 'route.json'
    path.write_text(json.dumps({'schema_version': 'ifv-psd-external-model-route-v1',
        'from_model': 'gemini-3.7-flash', 'to_model': 'gemini-3.6-flash',
        'search_root': str(root.resolve()), 'thinking_level': 'high',
        'preserve_completed_responses': True}))
    monkeypatch.setenv('IFV_PSD_EXTERNAL_ROUTE', str(path))
    monkeypatch.setenv('IFV_PSD_EXTERNAL_ROUTE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    return root, path


def activate_direct(monkeypatch, tmp_path):
    root = tmp_path / 'search'
    root.mkdir(exist_ok=True)
    path = tmp_path / 'route.json'
    path.write_text(json.dumps({'schema_version': 'ifv-psd-external-model-route-v2',
        'mode': 'direct_model', 'request_model': 'gemini-3.7-flash',
        'model': 'gemini-3.7-flash', 'search_root': str(root.resolve()),
        'thinking_level': 'high', 'preserve_completed_responses': True}))
    monkeypatch.setenv('IFV_PSD_EXTERNAL_ROUTE', str(path))
    monkeypatch.setenv('IFV_PSD_EXTERNAL_ROUTE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    return root, path


def request(client, cache, packet=None, model='gemini-3.7-flash'):
    return asyncio.run(_request(client, packet or {}, prompt='prompt', schema={},
                               model=model, cache_dir=cache))


def test_completed_old_response_and_new_actual_model_both_resume(monkeypatch, tmp_path):
    client = Client()
    cache = tmp_path / 'search' / 'case' / 'judge-cache'
    old = request(client, cache)
    old_files = {p: p.read_bytes() for p in cache.glob('*.json')}
    activate(monkeypatch, tmp_path)
    assert request(client, cache) == old
    new = request(client, cache, {'new': True})
    assert new[1]['request_binding']['model'] == 'gemini-3.6-flash'
    assert new[1]['model_route']['from_model'] == 'gemini-3.7-flash'
    assert request(client, cache, {'new': True}) == new
    assert [c['model'] for c in client.calls] == ['gemini-3.7-flash', 'gemini-3.6-flash']
    assert all(c['generation_config']['thinking_level'] == 'high' for c in client.calls)
    assert all(p.read_bytes() == raw for p, raw in old_files.items())


def test_route_does_not_change_other_jobs_or_uncached_probes(monkeypatch, tmp_path):
    root, _ = activate(monkeypatch, tmp_path)
    client = Client()
    request(client, tmp_path / 'unrelated')
    request(client, None)
    request(client, root / 'case', model='gemini-3.1-pro-preview')
    assert [c['model'] for c in client.calls] == [
        'gemini-3.7-flash', 'gemini-3.7-flash', 'gemini-3.1-pro-preview']


def test_direct_route_uses_gemini37_and_preserves_completed_cache(monkeypatch, tmp_path):
    root, _ = activate_direct(monkeypatch, tmp_path)
    client = Client()
    cache = root / 'case'
    first = request(client, cache)
    second = request(client, cache, {'new': True})
    assert first[1]['request_binding']['model'] == 'gemini-3.7-flash'
    assert second[1]['request_binding']['model'] == 'gemini-3.7-flash'
    assert second[1]['model_route']['mode'] == 'direct_model'
    assert [c['model'] for c in client.calls] == ['gemini-3.7-flash', 'gemini-3.7-flash']


def test_route_tamper_fails_before_request(monkeypatch, tmp_path):
    root, path = activate(monkeypatch, tmp_path)
    path.write_text(path.read_text() + ' ')
    client = Client()
    with pytest.raises(ValueError, match='changed after launch'):
        request(client, root / 'case')
    assert not client.calls


def test_malformed_completed_old_cache_is_not_resampled(monkeypatch, tmp_path):
    class Malformed(Client):
        async def create(self, **kwargs):
            response = await super().create(**kwargs)
            response['outputs'][0]['text'] = 'not json'
            return response
    client = Malformed()
    cache = tmp_path / 'search' / 'case'
    with pytest.raises(ValueError):
        request(client, cache)
    activate(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        request(client, cache)
    assert len(client.calls) == 1


def test_hint_author_comes_from_actual_request_provenance(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    async def actual(*args, **kwargs):
        return {'hints': [{'position': 0, 'hint': 'Check the unresolved relation.'}]}, {
            'request_binding': {'model': 'gemini-3.6-flash'}}
    monkeypatch.setattr(judge, '_request', actual)
    hints, _ = asyncio.run(propose_slate(None,
        public_context={'observed': 'tool error', 'decision_map': {0: 0}}, previous={},
        passing_positions=[], failed_position=0, model='gemini-3.7-flash', cache_dir=None))
    assert hints[0].model == 'gemini-3.6-flash'
