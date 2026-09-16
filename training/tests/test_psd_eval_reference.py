import copy
import pytest
from scripts.server.psd_eval_reference import reference_body, reference_command, stop_if_present, deferred_teacher_command


def test_reference_preserves_semantics_and_only_varies_capture():
    original = {'max_tokens':32768, 'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,test'}}]}],
        'tools':[{'type':'function'}],'tool_choice':'required','parallel_tool_calls':False,'seed':0,'temperature':.7,
        'return_token_ids':True,'return_tokens_as_token_ids':True,'logprobs':True,'top_logprobs':20,
        'vllm_xargs':{'ifv_thinking_budget':8192},'cache_salt':'original'}
    saved = copy.deepcopy(original)
    for mode in ('plain','token_ids','top20'):
        result = reference_body(original,mode)
        for key in ('messages','tools','tool_choice','parallel_tool_calls','seed','temperature','max_tokens'):
            assert result[key] == original[key]
        assert 'vllm_xargs' not in result and 'cache_salt' not in result
        assert result['thinking_token_budget']==8192
        assert bool(result.get('logprobs')) == (mode=='top20')
        assert bool(result.get('return_token_ids')) == (mode!='plain')
    assert original == saved
    with pytest.raises(ValueError): reference_body(original,'invalid')


def test_reference_removes_psd_hooks_not_model_or_budget():
    old = ['python','vllm','serve','protected-model','--served-model-name','alias','--max-num-seqs','16',
        '--max-model-len','131072','--limit-mm-per-prompt','{"image":32}', '--tool-call-parser','ifv_psd_qwen3_single',
        '--worker-cls','RawTeacherWorker','--logits-processors','PSDThinkingBudget','--tool-parser-plugin','single.py',
        '--mm-processor-cache-gb','0','--mamba-cache-mode','none','--enforce-eager','--no-async-scheduling',
        '--no-enable-prefix-caching']
    result = reference_command(old)
    assert old[old.index('--max-num-seqs')+1]=='16'
    assert result[result.index('--max-num-seqs')+1]=='8'
    assert result[result.index('--tool-call-parser')+1]=='qwen3_coder'
    assert result[3]=='protected-model'
    assert '--worker-cls' not in result and '--enforce-eager' not in result
    assert result[result.index('--max-model-len')+1]=='131072'


def test_dead_candidate_restoration_and_pid_reuse_guard(tmp_path):
    from types import SimpleNamespace
    stops = []
    owner = SimpleNamespace(stop=lambda receipt: stops.append(receipt['pid']))
    receipt = {'pid':123,'command':['python','worker.py']}
    stop_if_present(owner,receipt,proc_root=tmp_path)
    directory=tmp_path/'123';directory.mkdir()
    (directory/'cmdline').write_bytes(b'')
    stop_if_present(owner,receipt,proc_root=tmp_path)
    assert not stops
    (directory/'cmdline').write_bytes(b'python\0worker.py\0')
    stop_if_present(owner,receipt,proc_root=tmp_path)
    assert stops == [123]
    (directory/'cmdline').write_bytes(b'python\0another.py\0')
    with pytest.raises(RuntimeError,match='PID was reused'):
        stop_if_present(owner,receipt,proc_root=tmp_path)
    assert stops == [123]


def test_deferred_teacher_uses_stock_worker_and_keeps_effective_contract(tmp_path):
    original = ['python', 'vllm', 'serve', 'protected-model', '--served-model-name', 'alias',
                '--max-num-seqs', '16', '--max-model-len', '131072', '--port', '19004',
                '--limit-mm-per-prompt', '{"image":32}', '--tool-call-parser', 'ifv_psd_qwen3_single',
                '--worker-cls', 'RawTeacherWorker', '--logits-processors', 'OldBudget',
                '--tool-parser-plugin', 'old.py', '--mm-processor-cache-gb', '0',
                '--enforce-eager', '--no-async-scheduling']
    result = deferred_teacher_command(original, tmp_path)
    assert '--worker-cls' not in result and '--enforce-eager' not in result
    assert '--no-async-scheduling' not in result
    assert result[3] == 'protected-model'
    for key, value in (('--max-num-seqs', '8'), ('--max-model-len', '131072'),
                       ('--port', '19004'), ('--limit-mm-per-prompt', '{"image":32}'),
                       ('--tool-call-parser', 'ifv_psd_qwen3_single'),
                       ('--logits-processors', 'scripts.server.psd_qwen_thinking:PSDThinkingBudget'),
                       ('--mm-processor-cache-gb', '0'), ('--mamba-cache-mode', 'none')):
        assert result.count(key) == 1
        assert result[result.index(key) + 1] == value
    assert '--no-enable-prefix-caching' in result
