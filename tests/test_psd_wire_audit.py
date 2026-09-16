import gzip
import hashlib
import json
import pytest
from scripts.server.psd_wire_audit import audit_wire_archive
from scripts.server.audit_psd_wire_capture import audit
from scripts.server.psd_wire_capture import WireCapture


def test_audit_retains_empty_length_as_anomaly(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    receipt = capture.begin(b'{"max_tokens":32768,"vllm_xargs":{"ifv_thinking_budget":8192}}')
    response = {'choices':[{'finish_reason':'length', 'message':{'content':None, 'reasoning_content':'!'*20}}],
                'usage':{'prompt_tokens':100, 'completion_tokens':32768}}
    capture.response(receipt, json.dumps(response).encode(), status_code=200, replica='local')
    result = audit(capture.root)
    assert result['completed'] == 1 and len(result['anomalies']) == 1
    assert result['anomalies'][0]['choices'][0]['reasoning_unique_chars'] == 1
    assert result['anomalies'][0]['request_budget']['vllm_xargs']['ifv_thinking_budget'] == 8192
    assert not result['psd_gate_passed']


def test_pending_and_failed_receipts_not_counted_as_success(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    first = capture.begin(b'{}'); capture.error(first, 'request_cancelled_or_transport_failed')
    capture.begin(b'{}')
    result = audit(capture.root)
    assert result['started'] == 2 and result['completed'] == 0
    assert len(result['pending']) == len(result['capture_failures']) == 1


def test_receipt_hash_mismatch_rejected(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    receipt = capture.begin(b'{}')
    path = receipt/'request-meta.json'
    meta = json.loads(path.read_text()); meta['sha256'] = 'wrong'; path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='receipt'):
        audit(capture.root)


def test_native_tool_args_syntax_and_success(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    for argument in ['{"query":"x"}', 'broken']:
        receipt = capture.begin(b'{}')
        response = {'choices':[{'finish_reason':'tool_calls', 'message':{
            'tool_calls':[{'function':{'name':'search','arguments':argument}}]}}]}
        capture.response(receipt,json.dumps(response).encode(),status_code=200,replica='local')
    result = audit(capture.root)
    assert result['completed'] == 2 and len(result['anomalies']) == 1
    assert result['anomalies'][0]['choices'][0]['malformed_argument_json'] == 1


class Tokenizer:
    def token_to_id(self, value):
        return 2

    def decode(self, value, **kwargs):
        return '[{"name":"search","arguments":{}}]'


def ticket(root, name, request, *, error=None, finish='stop'):
    directory = root / name
    directory.mkdir()
    sides = {'request': request}
    if error is None:
        sides['response'] = {'choices': [{'token_ids': [1, 2, 3], 'finish_reason': finish}]}
    for side, value in sides.items():
        raw = json.dumps(value).encode()
        (directory / (side + '.json.gz')).write_bytes(gzip.compress(raw))
        (directory / (side + '-meta.json')).write_text(json.dumps(
            {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'status_code': 200}))
    if error:
        (directory / 'error.json').write_text(json.dumps({'kind': error}))
    return directory


def aux():
    return {'messages': [{'role': 'system', 'content': 'frozen perception'}]}


def run(root):
    return audit_wire_archive(root, tokenizer=Tokenizer(), perception_prompt='frozen perception')


def test_records_auxiliary_transport_failures_without_calling_them_policy_failures(tmp_path):
    ticket(tmp_path, 'failed', aux(), error='request_cancelled_or_transport_failed')
    ticket(tmp_path, 'successful', aux())
    result = run(tmp_path)
    assert result['requests'] == 2 and result['responses'] == 1
    assert result['auxiliary_transport_failures'] == 1
    assert result['auxiliary_failure_evidence'][0]['identical_request_successes'] == ['successful']
    assert (tmp_path / 'failed/error.json').exists()


@pytest.mark.parametrize('kind', ['response_capture_limit', 'disk_full', 'request_cancelled_or_transport_failed'])
def test_native_policy_errors_are_never_waived(tmp_path, kind):
    ticket(tmp_path, 'failed', {'logprobs': True, 'top_logprobs': 20}, error=kind)
    with pytest.raises(ValueError, match='policy or capture'):
        run(tmp_path)


def test_auxiliary_capture_error_is_not_a_transport_retry(tmp_path):
    ticket(tmp_path, 'failed', aux(), error='response_capture_limit')
    ticket(tmp_path, 'successful', aux())
    with pytest.raises(ValueError, match='policy or capture'):
        run(tmp_path)


def test_unknown_prompt_or_missing_success_cannot_pass(tmp_path):
    ticket(tmp_path, 'failed', aux(), error='request_cancelled_or_transport_failed')
    with pytest.raises(ValueError, match='no successful identical'):
        run(tmp_path)
    ticket(tmp_path, 'other', {'messages': [{'role': 'system', 'content': 'different'}]})
    with pytest.raises(ValueError, match='Unknown'):
        run(tmp_path)


def test_required_policy_action_keeps_raw_single_call_check(tmp_path):
    ticket(tmp_path, 'policy', {'logprobs': True, 'top_logprobs': 20, 'tool_choice': 'required'}, finish='tool_calls')
    assert run(tmp_path)['required_single_calls'] == 1
