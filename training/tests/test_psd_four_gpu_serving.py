import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'scripts/server/psd_four_gpu_serving.py'
    spec = importlib.util.spec_from_file_location('four_gpu_serving_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def template(m):
    return ['python', 'vllm', 'serve', str(m.MODEL), '--port', '19003',
            '--served-model-name', m.BACKEND, '--max-model-len', '131072',
            '--gpu-memory-utilization', '0.94', '--max-num-seqs', '16',
            '--tool-call-parser', 'ifv_psd_qwen3_single', '--mamba-cache-mode', 'none',
            '--no-enable-prefix-caching', '--limit-mm-per-prompt', '{"image":32,"video":0}']


@pytest.mark.parametrize('gpu', range(4))
def test_replica_clone_changes_only_port(gpu):
    m = module()
    original = template(m)
    before = copy.deepcopy(original)
    result = m.replica_command(original, gpu)
    assert original == before
    assert result[result.index('--port') + 1] == str(19002 + gpu)
    result[result.index('--port') + 1] = '19003'
    assert result == original


@pytest.mark.parametrize('option,value', [('--tool-call-parser', 'qwen3_coder'),
    ('--mamba-cache-mode', 'align'), ('--max-model-len', '32768'),
    ('--served-model-name', 'old-model'), ('--max-num-seqs', '8')])
def test_unvalidated_template_rejected(option, value):
    m = module()
    original = template(m)
    original[original.index(option) + 1] = value
    with pytest.raises(ValueError):
        m.replica_command(original, 0)


def test_full_gateway_no_duplicate_archive_and_same_safety_deadlines():
    m = module()
    previous = {'PSD_WIRE_CAPTURE_DIR': '/old/wire', 'PSD_DIAGNOSTIC_RETURN_TOKEN_IDS': '1',
                'PSD_GATEWAY_DEADLINE_SECONDS': '1200', 'AGENT_LLM_REQUEST_MAX_RETRIES': '0'}
    env = m.gateway_environment(previous)
    assert 'PSD_WIRE_CAPTURE_DIR' not in env
    assert 'PSD_DIAGNOSTIC_RETURN_TOKEN_IDS' not in env
    assert len(env['QWEN_REPLICA_BACKENDS'].split(',')) == 4
    assert env['PSD_GATEWAY_DEADLINE_SECONDS'] == '1200'
    assert env['AGENT_LLM_REQUEST_MAX_RETRIES'] == '0'
    assert previous['PSD_DIAGNOSTIC_RETURN_TOKEN_IDS'] == '1'
    assert m.CONCURRENCY == 40
