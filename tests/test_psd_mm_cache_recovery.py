import pytest

from scripts.server.psd_mm_cache_recovery import MODEL, cache_free_command


def command():
    return ['python', 'vllm', 'serve', str(MODEL), '--max-model-len', '131072',
        '--served-model-name', 'ifv-psd-sft3084', '--tool-call-parser', 'ifv_psd_qwen3_single',
        '--mamba-cache-mode', 'none', '--no-enable-prefix-caching', '--limit-mm-per-prompt',
        '{"image":32,"video":0}', '--max-num-seqs', '16']


def test_cache_workaround_changes_only_cache_setting():
    original = command()
    result = cache_free_command(original)
    assert result == original + ['--mm-processor-cache-gb', '0']
    assert original == command()
    assert cache_free_command(result) == result
    assert cache_free_command(original + ['--mm-processor-cache-gb', '4']) == result


def test_refuse_other_model_or_modified_budgets():
    for index, value in [(3, '/unrelated/model'), (5, '32768'), (7, 'unrelated')]:
        args = command()
        args[index] = value
        with pytest.raises(AssertionError):
            cache_free_command(args)
