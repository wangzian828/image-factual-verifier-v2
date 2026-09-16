import json

import pytest

from scripts.server.probe_psd_failed_slots import exhausted_requests
from scripts.server.psd_nan_metric_ab import candidate_environment


def fixture_slot(tmp_path, error='PolicyInfrastructureFailure: model_http_400_nonfinite_serialization'):
    slot = tmp_path / 'episodes/psd-infrastructure-attempts/slot'
    attempt = slot / 'attempt-003'
    context = attempt / 'traces/runtime/case/attempt/context/req-000004.json'
    context.parent.mkdir(parents=True)
    context.write_text(json.dumps({'status': 'error', 'error': error, 'request_id': 'req-000004'}))
    state = {'identity': {'max_attempts': 3, 'inputs': {'episode_id': 'case-r001'}},
             'payload': {'attempts': [{'index': i, 'status': 'infrastructure_failed',
                 'reason': 'model_http_400_nonfinite_serialization', 'directory': str(attempt)} for i in range(1, 4)]}}
    ledger = slot / 'retry-state.json'
    ledger.write_text(json.dumps(state))
    return ledger, context


@pytest.mark.parametrize('error', [
    'PolicyInfrastructureFailure: model_http_400_nonfinite_serialization',
    'RuntimeError: HTTP 400: Out of range float values are not JSON compliant: nan'])
def test_selects_native_numerical_receipt_without_editing_ledger(tmp_path, error):
    ledger, context = fixture_slot(tmp_path, error)
    before = ledger.read_bytes()
    selected = exhausted_requests(tmp_path)
    assert selected[0]['request_id'] == 'req-000004'
    assert selected[0]['archive'] == str(context.parent.parent)
    assert ledger.read_bytes() == before


def test_rejects_unbound_other_error(tmp_path):
    fixture_slot(tmp_path, 'RuntimeError: ordinary bad request')
    with pytest.raises(ValueError, match='one native numerical'):
        exhausted_requests(tmp_path)


def test_rejects_ambiguous_context(tmp_path):
    _, context = fixture_slot(tmp_path)
    context.with_name('req-000005.json').write_bytes(context.read_bytes())
    with pytest.raises(ValueError, match='one native numerical'):
        exhausted_requests(tmp_path)


def test_metric_ab_changes_only_one_switch_and_keeps_restart_control():
    for gpu in (1, 3):
        env = {'CUDA_VISIBLE_DEVICES': str(gpu), 'VLLM_COMPUTE_NANS_IN_LOGITS': '1',
               'TRITON_CACHE_DIR': '/unchanged', 'OTHER': 'untouched'}
        candidate = candidate_environment(env, gpu)
        expected = dict(env)
        if gpu == 1:
            expected['VLLM_COMPUTE_NANS_IN_LOGITS'] = '0'
        assert candidate == expected
        assert env['VLLM_COMPUTE_NANS_IN_LOGITS'] == '1'


def test_metric_ab_rejects_other_card_or_unexpected_original():
    with pytest.raises(ValueError):
        candidate_environment({'CUDA_VISIBLE_DEVICES': '0'}, 0)
    with pytest.raises(ValueError):
        candidate_environment({'CUDA_VISIBLE_DEVICES': '1'}, 1)
