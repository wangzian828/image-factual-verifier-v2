import pytest

from scripts.server.probe_psd_nan_stream import invalid_logprobs


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf')])
def test_nonfinite_or_null_logprobs_cannot_pass(value):
    payload = {'choices': [{'logprobs': {'content': [
        {'token': 'token_id:0', 'logprob': -1.0,
         'top_logprobs': [{'token': 'token_id:10', 'logprob': value}]}]}}]}
    assert invalid_logprobs(payload)


def test_empty_chunks_and_finite_masked_logprobs_are_not_nan():
    assert not invalid_logprobs({'choices': [{'logprobs': None}]})
    assert not invalid_logprobs({'choices': [{'logprobs': {'content': [
        {'token': 'token_id:10', 'logprob': -9999.0, 'top_logprobs': []}]}}]})


def test_eager_ablation_only_changes_execution_mode_on_owned_gpu():
    from scripts.server.psd_eager_diagnostic import eager_command, ROOT
    command = ['python', 'vllm', 'serve', str(ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'),
        '--port', '19005', '--served-model-name', 'ifv-psd-sft3084', '--max-model-len', '131072',
        '--mamba-cache-mode', 'none', '--mm-processor-cache-gb', '0']
    assert eager_command(command) == [*command, '--enforce-eager']
    wrong = command[:]
    wrong[wrong.index('--port')+1] = '19003'
    with pytest.raises(AssertionError):
        eager_command(wrong)
    other = command[:]
    other[other.index('--port')+1] = '19004'
    selected = eager_command(other, gpu=2, decode_only=True)
    import json
    assert selected[:-2] == other
    assert selected[-2] == '--compilation-config'
    assert json.loads(selected[-1]) == {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY'}
    other[other.index('--port')+1] = '19003'
    selected = eager_command(other, gpu=1, no_graph=True)
    assert selected[:-2] == other
    assert json.loads(selected[-1]) == {'mode': 3, 'cudagraph_mode': 'NONE'}
    with pytest.raises(AssertionError):
        eager_command(other,gpu=1,no_graph=True,decode_only=True)


@pytest.mark.parametrize('gpu',range(4))
def test_execution_matrix_rejects_accidentally_equivalent_modes(gpu):
    from scripts.server.probe_psd_execution_matrix import validate_mode
    import json
    command = ['--port',str(19002+gpu),'--no-enable-prefix-caching','--mamba-cache-mode','none']
    config = {1:{'mode':3,'cudagraph_mode':'NONE'},2:{'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'}}
    if gpu in config:
        command += ['--compilation-config',json.dumps(config[gpu])]
    if gpu==3:
        command += ['--enforce-eager']
    validate_mode(gpu,command)
    with pytest.raises(AssertionError):
        validate_mode(gpu,command+['--enforce-eager'] if gpu!=3 else command[:-1])


def test_candidate_promotion_preserves_non_execution_arguments():
    from scripts.server.promote_psd_execution_candidate import candidate_command,ROOT
    import json
    base=['python','vllm','serve',str(ROOT/'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'),
        '--max-model-len','131072','--mm-processor-cache-gb','0','--port','19002']
    eager=base+['--enforce-eager']
    decoded=candidate_command(eager,'decode')
    assert decoded[:-2]==base
    assert json.loads(decoded[-1])=={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'}
    assert candidate_command(decoded,'eager')==eager


def test_candidate_promotion_rejects_a_failed_diagnostic():
    from scripts.server.promote_psd_execution_candidate import gate
    class Fake:
        def load(self,path):
            rows=[{'gpu':2,'status':'completed','finish_reason':'tool_calls'} for _ in range(50)]
            rows[0]={'gpu':2,'status':'invalid_logprob_detected'}
            return {'completed':200,'results':rows}
    with pytest.raises(AssertionError):gate(Fake(),'decode')


def wire_request(tmp_path):
    import gzip, hashlib, json
    body = {'model': 'ifv-psd-sft3084', 'top_logprobs': 20, 'max_tokens': 32768,
            'messages': [{'content': 'unchanged', 'role': 'user'}],
            'tools': [{'z': 1, 'a': 2}], 'cache_salt': 'original',
            'vllm_xargs': {'ifv_thinking_budget': 8192}}
    raw = json.dumps(body).encode()
    (tmp_path / 'request.json.gz').write_bytes(gzip.compress(raw))
    (tmp_path / 'request-meta.json').write_text(json.dumps({
        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}))
    return body


def test_wire_diagnostic_preserves_actual_messages_and_order(tmp_path):
    from scripts.server.probe_psd_nan_stream import captured_wire_body
    original = wire_request(tmp_path)
    body = captured_wire_body(tmp_path)
    assert body['stream'] is True
    assert body['cache_salt'] != original['cache_salt']
    assert list(body['tools'][0]) == ['z', 'a']
    assert {k: v for k, v in body.items() if k not in ('stream', 'stream_options', 'cache_salt')} == {
        k: v for k, v in original.items() if k != 'cache_salt'}


def test_wire_diagnostic_refuses_unbound_bytes(tmp_path):
    from scripts.server.probe_psd_nan_stream import captured_wire_body
    wire_request(tmp_path)
    (tmp_path / 'request-meta.json').write_text('{"bytes":0,"sha256":"bad"}')
    with pytest.raises(ValueError, match='receipt'):
        captured_wire_body(tmp_path)
