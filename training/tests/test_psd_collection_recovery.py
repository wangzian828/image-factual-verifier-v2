import hashlib
import json
from pathlib import Path

import pytest

from ifv_training.psd_collection_recovery import import_prior_slots
from ifv_training.psd_repair_storage import save_bound, load_bound
from ifv_training.psd_infrastructure_retry import VERSION


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
