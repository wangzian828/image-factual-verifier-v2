from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest

from ifv_training.io import load_json, sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training import psd_source_review as source
from scripts import prefetch_psd_source_reviews as script
from scripts.review_psd_sources import review_sources
from test_psd_source_review import source_trace, artifact, Client


@pytest.fixture
def bank(tmp_path, monkeypatch):
    run = tmp_path/'production/episodes'
    image = tmp_path/'image.png'
    image.write_bytes(b'synthetic media fixture')
    inputs = {'benchmark': tmp_path/'public.jsonl', 'train_cases': tmp_path/'split.jsonl',
              'private_gold': tmp_path/'gold.jsonl'}
    write_jsonl(inputs['benchmark'], [{'case_id': 'a', 'image_path': 'image.png', 'image_sha256': sha256_file(image)}])
    write_jsonl(inputs['train_cases'], [{'case_id': 'a', 'split': 'train'}])
    write_jsonl(inputs['private_gold'], [{'case_id': 'a', 'factual_status': 'supported'}])
    policy = tmp_path/'access.json'; write_json(policy, {})
    snapshot = run.parent/'snapshot'
    write_json(snapshot/'serving-profile.json', {'profile_id': 'student'})
    write_json(snapshot/'checkpoint-manifest.json', {'frozen': True})
    binding = {**{k: str(v) for k,v in inputs.items()}, 'source_access_policy': str(policy),
        'snapshot': str(snapshot), 'formal_source_collection': True, 'rollouts_per_case': 8,
        'temperature': .7, 'seed': 0, 'case_ids': ['a'], 'slots': 8, 'policy_revision': 'frozen-revision',
        'files': {str(p): sha256_file(p) for p in [*inputs.values(), policy]}}
    write_json(run.parent/'binding.json', binding)
    manifest = {'status': 'running', 'benchmark': {'sha256': sha256_file(inputs['benchmark'])},
        'git_commit': 'frozen-revision', 'agent': {'model': 'student', 'provider': 'qwen_local',
        'base_url': 'http://localhost/v1', 'timeout_seconds': 3000., 'rollouts_per_case': 8, 'base_sampling_seed': 0}}
    write_json(run/'run_manifest.json', manifest)
    args = dict(run_dir=run, **inputs, model='test', output=tmp_path/'prefetch')
    scope = script.load_scope(**{k:v for k,v in args.items() if k != 'output'})
    trace_template = source_trace()
    trace_template['state']['runtime_case']['image_sha256'] = sha256_file(image)
    expected = artifact(trace_template, status='fail')
    monkeypatch.setattr(source, 'review_images', lambda *a, **kw: ([], expected['media']))
    results = []

    def complete(index, *, publish=True):
        episode, spec = list(scope['expected'].items())[index]
        trace = copy.deepcopy(trace_template); trace['image_id'] = episode
        trace['state']['all_steps'][-1]['output'] += ' fixture rollout '+str(index)
        slot = run/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
        identity = {'version': 'fixture', 'inputs': {**{k:spec[k] for k in ('case_id','episode_id','sampling_seed')},
            'model': 'student', 'provider': 'qwen_local', 'base_url': 'http://localhost/v1',
            'temperature': .7, 'timeout': 3000., 'image_sha256': sha256_file(image)}}
        save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [{'status': 'completed'}]})
        save_bound(slot/'result.json', identity=identity, payload=trace)
        path = run/'traces'/(episode+'.json')
        if publish: write_json(path, trace)
        results.append({'case_id': 'a', 'episode_id': episode, 'trace_path': str(path.relative_to(run))})
        write_jsonl(run/'run_results.jsonl', results)
        return episode, slot, path, identity

    return args, scope, complete, Client(expected['decision']), manifest


def test_incremental_prefetch_resumes_without_rejudging(bank):
    args, scope, complete, client, _ = bank
    complete(0)
    first = asyncio.run(script.prefetch(**args, client=client))
    assert first['counts'] == {'fail': 1} and first['available'] == 1 and first['expected'] == 8
    assert not first['training_started'] and not first['full_collection_admitted']
    asyncio.run(script.prefetch(**args, client=client)); assert client.calls == 1
    complete(1)
    assert asyncio.run(script.prefetch(**args, client=client))['counts'] == {'fail': 2}
    assert client.calls == 2


def test_formal_gate_rejects_partial_and_reuses_exact_responses_when_complete(bank):
    args, scope, complete, client, manifest = bank
    for i in range(8): complete(i)
    assert asyncio.run(script.prefetch(**args, client=client))['reviewed'] == 8
    formal = {**args, 'output': args['output'].parent/'formal', 'prefetch': args['output'], 'client': client}
    with pytest.raises(ValueError, match='not completed'): asyncio.run(review_sources(**formal))
    manifest['status'] = 'completed'; write_json(args['run_dir']/'run_manifest.json', manifest)
    result = asyncio.run(review_sources(**formal))
    assert result['counts']['fail'] == 8 and result['pending'] == 0 and client.calls == 8


def test_unpublished_and_unfinished_slots_never_call_judge(bank):
    args, scope, complete, client, _ = bank
    episode, slot, path, identity = complete(0, publish=False)
    assert asyncio.run(script.prefetch(**args, client=client))['attempted'] == 0
    write_json(path, load_bound(slot/'result.json', identity=identity))
    save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [{'status': 'running'}]})
    assert asyncio.run(script.prefetch(**args, client=client))['attempted'] == 0
    assert client.calls == 0


@pytest.mark.parametrize('field', ['sampling_seed', 'model', 'temperature', 'image_sha256'])
def test_wrong_slot_binding_is_pending_not_policy_failure(bank, field):
    args, scope, complete, client, _ = bank
    _, slot, _, identity = complete(0)
    trace = load_bound(slot/'result.json', identity=identity)
    identity['inputs'][field] = 'wrong'
    save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [{'status': 'completed'}]})
    save_bound(slot/'result.json', identity=identity, payload=trace)
    assert asyncio.run(script.prefetch(**args, client=client))['counts'] == {'pending_error': 1}
    assert client.calls == 0


def test_transport_errors_are_bounded_and_secret_free(bank):
    from src.integrations.gemini import GeminiInteractionsHTTPError
    args, _, complete, _, _ = bank; complete(0)
    class FailingClient:
        calls = 0
        async def create(self, **kwargs):
            self.calls += 1
            raise GeminiInteractionsHTTPError(429, 'do-not-record-secret', 'private-url', retry_attempts=2)
    client = FailingClient()
    for _ in range(2):
        assert asyncio.run(script.prefetch(**args, client=client))['counts'] == {'pending_error': 1}
    assert client.calls == 1
    raw = next((args['output']/'records').glob('*.json')).read_text()
    assert 'do-not-record-secret' not in raw and 'private-url' not in raw and '429' in raw


def test_worker_persistence_failure_is_not_a_deadlock(bank, monkeypatch):
    args, _, complete, client, _ = bank; complete(0)
    original = script.save_bound
    def broken(path, **kwargs):
        if path.parent.name == 'records': raise OSError('synthetic disk full')
        return original(path, **kwargs)
    monkeypatch.setattr(script, 'save_bound', broken)
    with pytest.raises(OSError):
        asyncio.run(asyncio.wait_for(script.prefetch(**args, client=client), timeout=2))
    assert load_json(args['output']/'progress.json')['phase'] == 'interrupted_requires_inspection'
    assert client.calls == 0


def test_changed_model_or_frozen_input_rejected(bank):
    args, _, complete, client, _ = bank; complete(0)
    asyncio.run(script.prefetch(**args, client=client))
    with pytest.raises(ValueError, match='binding changed'):
        asyncio.run(script.prefetch(**{**args, 'model':'different'}, client=client))
    write_jsonl(args['train_cases'], [{'case_id':'a','split':'test'}])
    with pytest.raises(ValueError, match='differs'):
        asyncio.run(script.prefetch(**args, client=client))
    assert client.calls == 1


def test_live_client_receives_explicit_sixteen_request_gate(bank, monkeypatch):
    import src.integrations.gemini as gemini
    args, _, complete, client, _ = bank; complete(0)
    gates = []
    class FakeLive(Client):
        def __init__(self, **kwargs):
            gates.append(kwargs['request_gate'].limit)
            super().__init__(client.decision)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(gemini, 'GeminiInteractionsClient', FakeLive)
    assert asyncio.run(script.prefetch(**args, concurrency=16))['reviewed'] == 1
    assert gates == [16]


def test_follower_picks_up_new_completed_slots(bank):
    args, _, complete, client, _ = bank
    complete(0)
    async def run():
        async def publish():
            await asyncio.sleep(.05)
            complete(1)
        publisher = asyncio.create_task(publish())
        value = await script.prefetch(**args, client=client, follow=True, limit=2, poll_seconds=.01)
        await publisher
        return value
    assert asyncio.run(asyncio.wait_for(run(), timeout=3))['reviewed'] == 2
    assert client.calls == 2


def test_recovery_ancestor_cache_is_reused_but_unrelated_bank_rejected(bank):
    import shutil
    args, _, complete, client, manifest = bank
    for i in range(8): complete(i)
    asyncio.run(script.prefetch(**args, client=client)); assert client.calls == 8
    previous = args['run_dir'].parent
    new = previous.parent/'recovered'
    shutil.copytree(previous, new)
    binding = load_json(new/'binding.json')
    binding.update(snapshot=str(new/'snapshot'), reuse_run=str(previous))
    write_json(new/'binding.json', binding)
    updated = {**args, 'run_dir': new/'episodes', 'output': new/'prefetch', 'cache_source': args['output']}
    assert asyncio.run(script.prefetch(**updated, client=client))['reviewed'] == 8
    assert client.calls == 8
    cache = script.validate_prefetch_cache(new/'prefetch', **{k:v for k,v in updated.items()
        if k not in ('output','cache_source')})
    assert cache == args['output']/'judge-cache'
    binding['reuse_run'] = None; write_json(new/'binding.json', binding)
    with pytest.raises(ValueError, match='ancestor'):
        script.validate_prefetch_cache(args['output'], **{k:v for k,v in updated.items()
            if k not in ('output','cache_source')})
    assert client.calls == 8
