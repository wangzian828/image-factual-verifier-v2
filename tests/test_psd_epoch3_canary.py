from scripts.server import psd_epoch3_canary as candidate


def test_policy_and_outputs_are_epoch3_and_separate():
    assert '3084' in str(candidate.EXPORT)
    assert '3084' in candidate.ALIAS and '3084' in candidate.BACKEND
    for output in (candidate.SERVICE,candidate.RUN,candidate.DEPLOY):
        assert output != candidate.EXPORT.parent
        assert candidate.EXPORT.parent not in output.parents
        assert 'checkpoints' not in output.parts


def test_model_command_changes_only_identity_and_plugin_location():
    original=['python','vllm','serve','old-model','--served-model-name','old-alias',
              '--no-enable-prefix-caching','--mamba-cache-mode','none',
              '--tool-call-parser','ifv_psd_qwen3_single','--tool-parser-plugin','old-plugin',
              '--max-model-len','131072','--limit-mm-per-prompt','{"image":32}']
    before=original[:]
    result=candidate.model_command(original)
    assert original==before
    assert result[3]==str(candidate.EXPORT.parent/'model')
    assert result[result.index('--served-model-name')+1]==candidate.BACKEND
    assert result[-4:]==original[-4:]
