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
