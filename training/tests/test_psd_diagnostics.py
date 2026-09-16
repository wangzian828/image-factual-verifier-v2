import gzip
import json

from ifv_training.psd_diagnostics import exception_snapshot, persist_exception


def test_provider_error_text_and_unrelated_locals_are_never_persisted(tmp_path):
    api_key = 'DO-NOT-EXPOSE-CREDENTIAL'
    try:
        raise ValueError('Authorization: Bearer ' + api_key)
    except Exception as error:
        record = persist_exception(tmp_path, error)
    with gzip.open(record['path'], 'rt') as stream:
        body = stream.read()
    assert api_key not in body and 'Authorization' not in body
    value = json.loads(body)
    assert value['error_type'] == 'ValueError' and value['frames']
    assert value['selected_model_context'] == []
    assert value['not_a_training_target'] and value['exception_message_omitted']


def test_allowlist_is_bound_to_module_and_function_and_preserves_json_order():
    namespace = {'__name__': 'ifv_training.psd_slate'}
    exec('def capture_target():\n'
         '    request = {"z": "{\\"z\\":1,\\"a\\":2}", "a": 1}\n'
         '    private_key = "DO-NOT-EXPOSE"\n'
         '    raise ValueError(private_key)\n', namespace)
    try:
        namespace['capture_target']()
    except Exception as error:
        snapshot = exception_snapshot(error)
    local = snapshot['selected_model_context'][0]['locals']
    assert list(local['request']) == ['z', 'a']
    assert local['request']['z'] == '{"z":1,"a":2}'
    assert 'DO-NOT-EXPOSE' not in json.dumps(snapshot)
