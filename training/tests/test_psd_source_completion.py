import asyncio
import copy
import hashlib

import pytest

from ifv_training.io import write_json
from ifv_training.psd_source_completion import source_completion_failure
from ifv_training.psd_infrastructure_retry import retry_episode, InfrastructureRetriesExhausted, VERSION
from ifv_training.psd_repair_storage import load_bound, save_bound


def complete_trace(verdict='fake'):
    report = {'headline': 'Synthetic test only', 'claim_under_review': 'A claim',
        'verdict_summary': 'A decision', 'key_findings': ['A finding'], 'evidence_summary': 'Evidence'}
    return {'image_id': 'case--r003', 'termination': 'success', 'verdict': verdict, 'correct': False,
        'judgment': {'verdict': verdict, 'confidence': .5, 'overall_assessment': 'Synthetic control',
                     'fact_check_report': report},
        'state': {'all_steps': [{'stage': 'unified_judgment', 'action_type': 'output', 'output': 'full report',
            'metadata': {'policy_token_capture': {'status': 'complete', 'prompt_token_ids': [1],
                'completion_token_ids': [2], 'completion_logprobs': [-.3]}}}]}}


async def no_sleep(_): pass


@pytest.mark.parametrize('kind', ['error', 'no_report', 'bad_report', 'no_judgment_step', 'no_capture', 'nan'])
def test_incomplete_source_is_retried_but_complete_wrong_answer_is_retained(tmp_path, kind):
    good = complete_trace(); bad = copy.deepcopy(good)
    if kind == 'error': bad.update(termination='error', error='unusable tool call')
    elif kind == 'no_report': bad['judgment']['fact_check_report'] = None
    elif kind == 'bad_report': bad['judgment']['fact_check_report']['key_findings'] = []
    elif kind == 'no_judgment_step': bad['state']['all_steps'][0]['stage'] = 'unified_react'
    elif kind == 'no_capture': bad['state']['all_steps'][0]['metadata'] = {}
    else: bad['state']['all_steps'][0]['metadata']['policy_token_capture']['completion_logprobs'] = [float('nan')]
    assert source_completion_failure(bad) and source_completion_failure(good) is None
    calls = []
    async def generate(directory):
        calls.append(directory)
        return bad if len(calls) == 1 else good
    args = dict(root=tmp_path, identity={}, generate=generate, sleep=no_sleep, validate_result=source_completion_failure)
    assert asyncio.run(retry_episode(**args)) == good
    assert asyncio.run(retry_episode(**args)) == good and len(calls) == 2
    identity = {'version': VERSION, 'inputs': {}, 'max_attempts': 3}
    ledger = load_bound(tmp_path/'retry-state.json', identity=identity)
    assert [a['status'] for a in ledger['attempts']] == ['trajectory_failed', 'completed']
    assert (calls[0]/'incomplete-result.json').exists() and not good['correct']


def test_retry_budget_does_not_reset_and_cached_bad_source_requires_migration(tmp_path):
    bad = {**complete_trace(), 'termination': 'error'}; calls = []
    async def generate(directory): calls.append(directory); return bad
    args = dict(root=tmp_path, identity={}, generate=generate, sleep=no_sleep, validate_result=source_completion_failure)
    for _ in range(2):
        with pytest.raises(InfrastructureRetriesExhausted): asyncio.run(retry_episode(**args))
    assert len(calls) == 3 and not (tmp_path/'result.json').exists()
    save_bound(tmp_path/'result.json', identity={'version': VERSION, 'inputs': {}, 'max_attempts': 3}, payload=bad)
    with pytest.raises(ValueError, match='migrate'): asyncio.run(retry_episode(**args))
    assert len(calls) == 3


def test_empty_failed_tool_and_wrong_semantic_claim_do_not_trigger_resampling():
    trace = complete_trace()
    step = copy.deepcopy(trace['state']['all_steps'][0])
    step.update(stage='unified_react', action_type='tool_call', tool_result={'status': 'error', 'items': []})
    step['metadata']['tool_success'] = False
    trace['state']['all_steps'].insert(0, step)
    assert source_completion_failure(trace) is None
    trace['judgment']['fact_check_report']['verdict_summary'] = 'unsupported semantic assertion'
    assert source_completion_failure(trace) is None


@pytest.mark.parametrize('incomplete', [False, True])
def test_import_charges_old_attempt_without_modifying_old_trace_or_resampling_wrong_answers(tmp_path, incomplete):
    from ifv_training.psd_collection_recovery import import_prior_slots
    old, new = tmp_path/'old', tmp_path/'new'; episode = 'case--r003'
    slot = old/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
    identity = {'version': VERSION, 'max_attempts': 3, 'inputs': {'episode_id': episode,
        'sampling_seed': 71, 'model': 'frozen', 'base_url': 'http://model/v1'}}
    result = complete_trace()
    if incomplete: result.update(termination='error', error='unusable tool call')
    save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [
        {'index': 1, 'directory': str(slot/'attempt-001'), 'status': 'completed'}]})
    save_bound(slot/'result.json', identity=identity, payload=result)
    write_json(old/'traces'/f'{episode}.json', result)
    originals = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
    report = import_prior_slots(source=old, destination=new, episode_ids=[episode], seeds=[71], require_complete=True)
    assert report['retry_remaining'] == int(incomplete) and report['reused'] == int(not incomplete)
    target = new/'psd-infrastructure-attempts'/slot.name
    if incomplete:
        ledger = load_bound(target/'retry-state.json', identity=identity)
        assert len(ledger['attempts']) == 1 and ledger['attempts'][0]['status'] == 'trajectory_failed'
        assert not (target/'result.json').exists()
        calls = []
        async def generate(directory): calls.append(directory); return complete_trace()
        assert asyncio.run(retry_episode(root=target, identity=identity['inputs'], generate=generate,
            validate_result=source_completion_failure, sleep=no_sleep))['termination'] == 'success'
        assert calls[0].name == 'attempt-002'
    else:
        assert (target/'result.json').samefile(slot/'result.json')
    assert all(p.read_bytes() == content for p,content in originals.items())
