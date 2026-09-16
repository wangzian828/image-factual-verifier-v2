import hashlib
import json

import pytest

from ifv_training.io import write_json, write_jsonl
from ifv_training.psd_collection import require_completed_collection, repairable_terminal_model_failure
from ifv_training.psd_repair_storage import save_bound
from types import SimpleNamespace
from src.eval.rollout import rollout_specs
from src.eval.result_records import run_result_record


def native_records():
    manifest = {'git_commit': 'frozen', 'agent': {'model': 'policy', 'rollouts_per_case': 8,
        'base_sampling_seed': 0, 'psd_sampling': {'temperature': .7, 'collector_sha256': 'a'*64}}}
    specs = rollout_specs([{}, {}], [SimpleNamespace(case_id=c) for c in ['a', 'b']],
        rollouts_per_case=8, base_sampling_seed=0, policy_revision='frozen', model='policy')
    rows = [run_result_record({'case_id': r['case_id']}, {'termination': 'error', 'error': 'ordinary failure'},
        trace_path='trace.json', metadata=None, **{k: r[k] for k in
            ('episode_id', 'prompt_group_id', 'rollout_index', 'sampling_seed')}) for r in specs]
    return rows, manifest


def bank(tmp_path):
    rows, manifest = native_records()
    manifest.update(status='completed_with_errors', benchmark={'sample_count': 2, 'episode_count': 16})
    manifest['agent']['psd_sampling']['infrastructure_retry'] = {
        'version': 'ifv-psd-infrastructure-retry-v2', 'slots': 16, 'unresolved': [], 'selection_by_answer': False}
    slots = []
    for row in rows:
        episode = row['episode_id']
        inputs = {'episode_id': episode, 'case_id': row['case_id'], 'sampling_seed': row['sampling_seed'],
                  'model': manifest['agent']['model'], 'base_url': 'http://policy/v1'}
        identity = {'inputs': inputs, 'version': 'ifv-psd-infrastructure-retry-v2', 'max_attempts': 3}
        slot = tmp_path/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
        save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [{'status': 'completed'}]})
        result = {'termination': 'error', 'error': 'ordinary policy failure', 'image_id': episode}
        save_bound(slot/'result.json', identity=identity, payload=result)
        row['trace_path'] = f'traces/{episode}.json'
        write_json(tmp_path/row['trace_path'], result)
        slots.append((slot, identity, row))
    write_jsonl(tmp_path/'run_results.jsonl', rows)
    return manifest, slots


def test_full_sampling_can_complete_with_policy_errors_without_mutating_manifest(tmp_path):
    manifest, _ = bank(tmp_path)
    before = json.dumps(manifest, sort_keys=True)
    result = require_completed_collection(tmp_path, manifest)
    assert result['passed'] and result['model_failures_retained'] == 16
    assert json.dumps(manifest, sort_keys=True) == before


@pytest.mark.parametrize('corruption', ['pending', 'missing_trace', 'missing_result', 'missing_slot', 'numerical'])
def test_collection_errors_cannot_hide_infrastructure_or_coverage_failures(tmp_path, corruption):
    manifest, slots = bank(tmp_path)
    slot, identity, row = slots[0]
    if corruption == 'pending':
        save_bound(slot/'retry-state.json', identity=identity, payload={'attempts': [{'status': 'running'}]})
    elif corruption == 'missing_trace':
        (tmp_path/row['trace_path']).unlink()
    elif corruption == 'missing_result':
        (slot/'result.json').unlink()
    elif corruption == 'missing_slot':
        manifest['agent']['psd_sampling']['infrastructure_retry']['unresolved'] = ['one']
    else:
        result = {'error': 'RuntimeError: HTTP 400 Bad Request for http://policy/v1/chat/completions: '
                  '{"error":{"type":"BadRequestError","code":400,"message":"Out of range float values are not JSON compliant: nan"}}'}
        save_bound(slot/'result.json', identity=identity, payload=result)
        write_json(tmp_path/row['trace_path'], result)
    with pytest.raises((ValueError, OSError)):
        require_completed_collection(tmp_path, manifest)


def test_no_blanket_relaxation_for_running_or_unattested_failed_runs(tmp_path):
    for status in ('failed', 'running', 'completed_with_errors'):
        with pytest.raises((ValueError, OSError)):
            require_completed_collection(tmp_path, {'status': status})


def failed_policy_trace(tmp_path):
    error = ('RuntimeError: Chat Completions returned an unusable response: choices=1, '
             'finish_reason=tool_calls, content_chars=0, reasoning_chars=332, reasoning_fallback_requested=False')
    write_json(tmp_path/'context/req-000002.json', {'status': 'error', 'error': error, 'stage': 'unified_react'})
    return {'termination': 'error', 'error': error, 'state': {
        'runtime_store': {'runtime_path': str(tmp_path)}, 'all_steps': [
            {'metadata': {'policy_token_capture': {'status': 'complete', 'completion_token_ids': [1],
                                                   'completion_logprobs': [-.1]}}},
            {'metadata': {'deterministic_segment_boundary': True}}]}}


def test_unusable_policy_output_can_seed_repair_but_is_not_task_success(tmp_path):
    trace = failed_policy_trace(tmp_path)
    assert repairable_terminal_model_failure(trace) == 'unusable_policy_output_with_captured_prefix'
    assert trace['termination'] == 'error'


@pytest.mark.parametrize('corruption', ['http', 'receipt', 'nonfinite', 'no_prefix'])
def test_policy_failure_adapter_does_not_hide_numerical_or_unbound_error(tmp_path, corruption):
    trace = failed_policy_trace(tmp_path)
    if corruption == 'http':
        trace['error'] = 'RuntimeError: HTTP 400 Bad Request'
    elif corruption == 'receipt':
        (tmp_path/'context/req-000002.json').unlink()
    elif corruption == 'nonfinite':
        trace['state']['all_steps'][0]['metadata']['policy_token_capture']['completion_logprobs'] = [float('nan')]
    else:
        trace['state']['all_steps'] = []
    assert repairable_terminal_model_failure(trace) is None
