import json

import pytest

from scripts.server.audit_psd_wire_capture import audit
from scripts.server.psd_wire_capture import WireCapture


def test_audit_retains_empty_length_as_anomaly(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    ticket = capture.begin(b'{"max_tokens":32768,"vllm_xargs":{"ifv_thinking_budget":8192}}')
    response = {'choices':[{'finish_reason':'length', 'message':{'content':None, 'reasoning_content':'!'*20}}],
                'usage':{'prompt_tokens':100, 'completion_tokens':32768}}
    capture.response(ticket, json.dumps(response).encode(), status_code=200, replica='local')
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
    ticket = capture.begin(b'{}')
    path = ticket/'request-meta.json'
    meta = json.loads(path.read_text()); meta['sha256'] = 'wrong'; path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='receipt'):
        audit(capture.root)


def test_native_tool_args_syntax_and_success(tmp_path):
    capture = WireCapture(tmp_path/'wire', min_free_bytes=0)
    for argument in ['{"query":"x"}', 'broken']:
        ticket = capture.begin(b'{}')
        response = {'choices':[{'finish_reason':'tool_calls', 'message':{
            'tool_calls':[{'function':{'name':'search','arguments':argument}}]}}]}
        capture.response(ticket,json.dumps(response).encode(),status_code=200,replica='local')
    result = audit(capture.root)
    assert result['completed'] == 2 and len(result['anomalies']) == 1
    assert result['anomalies'][0]['choices'][0]['malformed_argument_json'] == 1
