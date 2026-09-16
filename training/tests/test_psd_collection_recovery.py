import hashlib
import json
from pathlib import Path

import pytest

from ifv_training.psd_collection_recovery import import_prior_slots
from ifv_training.psd_repair_storage import save_bound, load_bound
from ifv_training.psd_infrastructure_retry import VERSION
from ifv_training.psd_infrastructure_retry import recovery_attempt_budget, retry_episode
from ifv_training.io import sha256_file, write_json


def source(tmp_path, *, nan=False, receipt=True):
    old = tmp_path/'original'
    episode = 'c--r000'
    slot = old/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
    runtime = slot/'attempt-001/traces/runtime/c/a'
    inputs = {'episode_id': episode, 'sampling_seed': 7, 'model': 'frozen-sft', 'base_url': 'http://policy/v1'}
    identity = {'version': 'ifv-psd-infrastructure-retry-v1', 'inputs': inputs, 'max_attempts': 3}
    error = ('RuntimeError: HTTP 400 Bad Request for http://policy/v1/chat/completions: '
             '{"error":{"message":"Out of range float values are not JSON compliant: nan",'
             '"type":"BadRequestError","code":400}}') if nan else 'ordinary wrong action'
    result = {'image_id': episode, 'verdict': 'fake', 'correct': False, 'error': error,
              'state': {'runtime_store': {'runtime_path': str(runtime)}}}
    save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [
        {'index': 1, 'directory': str(slot/'attempt-001'), 'status': 'completed'}]})
    save_bound(slot/'result.json', identity=identity, payload=result)
    path = old/'traces'/f'{episode}.json'
    path.parent.mkdir(); path.write_text(json.dumps(result))
    if receipt:
        context = runtime/'context'; context.mkdir(parents=True)
        (context/'req-000001.json').write_text(json.dumps({'status': 'error', 'error': error,
            'model': 'frozen-sft', 'stage': 'unified_react'}))
    return old, slot, result, identity


@pytest.mark.parametrize('nan', [False, True])
def test_only_proven_numerical_failure_uses_remaining_budget(tmp_path, nan):
    old, slot, result, identity = source(tmp_path, nan=nan)
    original = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
    out = tmp_path/'new'
    report = import_prior_slots(source=old, destination=out, episode_ids=['c--r000'], seeds=[7])
    assert report['retry_remaining'] == int(nan)
    new = out/'psd-infrastructure-attempts'/slot.name
    binding = {**identity, 'version': VERSION}
    state = load_bound(new/'retry-state.json', identity=binding)
    assert len(state['attempts']) == 1
    assert state['attempts'][0]['status'] == ('infrastructure_failed' if nan else 'completed')
    assert (new/'result.json').exists() == (not nan)
    if not nan:
        assert load_bound(new/'result.json', identity=binding) == result
    assert all(p.read_bytes() == content for p, content in original.items())


def test_model_or_tool_text_without_native_receipt_cannot_authorize_retry(tmp_path):
    old, _, _, _ = source(tmp_path, nan=True, receipt=False)
    with pytest.raises(ValueError, match='independent native'):
        import_prior_slots(source=old, destination=tmp_path/'new', episode_ids=['c--r000'], seeds=[7])
    assert not (tmp_path/'new/psd-infrastructure-attempts').exists()


def test_recovery_does_not_change_seed_or_reuse_destination(tmp_path):
    old, _, _, _ = source(tmp_path)
    with pytest.raises(ValueError, match='episode/seed'):
        import_prior_slots(source=old, destination=tmp_path/'new', episode_ids=['c--r000'], seeds=[8])
    args = dict(source=old, destination=tmp_path/'new', episode_ids=['c--r000'], seeds=[7])
    import_prior_slots(**args)
    with pytest.raises(ValueError, match='empty destination'):
        import_prior_slots(**args)


def exhausted_source_with_gate(tmp_path):
    old, slot, result, identity = source(tmp_path)
    identity['version'] = VERSION
    attempts = [{'index': i, 'directory': str(slot/f'attempt-{i:03d}'),
        'status': 'infrastructure_failed', 'reason': 'model_http_400_nonfinite_serialization'}
        for i in range(1, 4)]
    save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': attempts})
    (slot/'result.json').unlink()
    (old/'traces/c--r000.json').unlink()
    intervention = tmp_path/'intervention.json'; write_json(intervention, {'phase': 'ready_for_replay'})
    replay = tmp_path/'replay.json'
    rows = [{'gpu': g, 'source_index': s, 'repetition': r, 'status': 'completed',
             'finish_reason': 'tool_calls'} for g in range(4) for s in range(4) for r in range(4)]
    write_json(replay, {'diagnostic_only': True, 'completed': 64, 'results': rows})
    gate = {'episode_ids': ['c--r000'], 'evidence': {k: {'path': str(p), 'sha256': sha256_file(p)}
            for k, p in [('intervention', intervention), ('replay', replay)]}}
    return old, slot, identity, gate


def test_recovery_charges_original_failures_and_only_allows_two_more(tmp_path):
    import asyncio
    old, slot, identity, gate = exhausted_source_with_gate(tmp_path)
    original = (slot/'retry-state.json').read_bytes()
    dest = tmp_path/'new'
    report = import_prior_slots(source=old, destination=dest, episode_ids=['c--r000'], seeds=[7], numerical_recovery=gate)
    assert report['explicit_budget_extensions'] == 1
    target = dest/'psd-infrastructure-attempts'/slot.name
    assert recovery_attempt_budget(target, identity['inputs']) == 5
    calls = []
    async def generate(directory):
        calls.append(directory.name)
        return {'wrong_answer_is_still_accepted': True}
    async def no_sleep(_): pass
    asyncio.run(retry_episode(root=target, identity=identity['inputs'], generate=generate, sleep=no_sleep))
    assert calls == ['attempt-004']
    assert (slot/'retry-state.json').read_bytes() == original
    assert len(load_bound(target/'retry-state.json', identity={**identity,'max_attempts':5})['attempts']) == 4


@pytest.mark.parametrize('problem', ['incomplete', 'failed', 'wrong_case', 'changed_receipt'])
def test_invalid_recovery_gate_is_rejected_before_writes(tmp_path, problem):
    old, slot, identity, gate = exhausted_source_with_gate(tmp_path)
    replay = Path(gate['evidence']['replay']['path'])
    data = json.loads(replay.read_text())
    if problem == 'incomplete': data['results'].pop()
    elif problem == 'failed': data['results'][0]['status'] = 'invalid_logprob_detected'
    elif problem == 'wrong_case': gate['episode_ids'] = []
    else: data['changed'] = True
    write_json(replay, data)
    if problem != 'changed_receipt': gate['evidence']['replay']['sha256'] = sha256_file(replay)
    with pytest.raises(ValueError):
        import_prior_slots(source=old, destination=tmp_path/'new', episode_ids=['c--r000'], seeds=[7], numerical_recovery=gate)
    assert not (tmp_path/'new/psd-infrastructure-attempts').exists()


def completed_extended_source(tmp_path, *, packed=False):
    import asyncio
    old, slot, identity, gate = exhausted_source_with_gate(tmp_path)
    recovered = tmp_path/'recovered'
    import_prior_slots(source=old, destination=recovered, episode_ids=['c--r000'], seeds=[7], numerical_recovery=gate)
    target = recovered/'psd-infrastructure-attempts'/slot.name
    result = {'image_id': 'c--r000', 'ordinary_wrong_answer': True}
    async def generate(directory): return result
    async def no_sleep(_): pass
    asyncio.run(retry_episode(root=target, identity=identity['inputs'], generate=generate, sleep=no_sleep))
    write_json(recovered/'traces/c--r000.json', result)
    if packed:
        from scripts.server.compact_psd_completed_storage import pack_cache
        pack_cache(target/'result.json', identity={**identity, 'max_attempts': 5})
    return recovered, target, identity, result


@pytest.mark.parametrize('packed', [False, True])
def test_completed_extended_bank_survives_repeated_import_without_resampling(tmp_path, packed):
    import asyncio
    source_run, source_slot, identity, result = completed_extended_source(tmp_path, packed=packed)
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    for name in ('next', 'next_again'):
        dest = tmp_path/name
        report = import_prior_slots(source=source_run, destination=dest, episode_ids=['c--r000'], seeds=[7])
        assert report['explicit_budget_extensions'] == 0
        assert report['preserved_recovery_allowances'] == report['reused'] == 1
        assert report['records'][0]['spent_attempts'] == 4
        assert report['records'][0]['maximum_attempts'] == 5
        if packed and name == 'next':
            assert any(p.endswith('.json.gz') for p in report['records'][0]['originals'])
        target = dest/'psd-infrastructure-attempts'/source_slot.name
        assert recovery_attempt_budget(target, identity['inputs']) == 5
        assert load_bound(target/'retry-state.json', identity={**identity, 'max_attempts':5})['attempts'] == load_bound(source_slot/'retry-state.json', identity={**identity,'max_attempts':5})['attempts']
        async def forbidden(directory): raise AssertionError('Completed result must not generate again')
        assert asyncio.run(retry_episode(root=target, identity=identity['inputs'], generate=forbidden)) == result
        write_json(dest/'traces/c--r000.json', result)
        source_run, source_slot = dest, target
    assert all(p.read_bytes() == raw for p, raw in original.items())


@pytest.mark.parametrize('problem', ['missing_allowance', 'changed_evidence', 'erased_old_attempt', 'running', 'exhausted'])
def test_extended_bank_invalid_history_rejected_without_destination_writes(tmp_path, problem):
    recovered, slot, identity, result = completed_extended_source(tmp_path)
    binding = {**identity, 'max_attempts':5}
    state = load_bound(slot/'retry-state.json', identity=binding)
    if problem == 'missing_allowance':
        (slot/'recovery-allowance.json').unlink()
    elif problem == 'changed_evidence':
        allowance = json.loads((slot/'recovery-allowance.json').read_text())
        write_json(Path(allowance['evidence']['replay']['path']), {'changed':True})
    else:
        if problem == 'erased_old_attempt': state['attempts'][0]['reason'] = 'changed'
        elif problem == 'running': state['attempts'][-1]['status'] = 'running'
        else: state['attempts'][-1]['status'] = 'infrastructure_failed'
        save_bound(slot/'retry-state.json', identity=binding, payload=state)
    with pytest.raises(ValueError):
        import_prior_slots(source=recovered, destination=tmp_path/'new', episode_ids=['c--r000'], seeds=[7])
    assert not (tmp_path/'new').exists()
