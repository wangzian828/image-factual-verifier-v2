import sys
from types import SimpleNamespace

import pytest

from scripts.server import run_psd_grouped_canary as grouped
from scripts.server import run_psd_runtime_gate as gate


@pytest.mark.parametrize('private_cache_value', [None, '1'])
def test_cache_free_launcher_overrides_both_independent_flags(monkeypatch, private_cache_value):
    private = {key: 'test-only' for key in (
        'SERPER_API_KEY', 'JINA_API_KEY', 'GEMINI_API_KEY',
        'BAIDU_OCR_API_KEY', 'BAIDU_OCR_SECRET_KEY')}
    if private_cache_value is not None:
        private.update(TOOL_CACHE_ENABLED=private_cache_value,
                       PERCEPTION_CACHE_ENABLED=private_cache_value)
    monkeypatch.setitem(sys.modules, 'dotenv', SimpleNamespace(dotenv_values=lambda _: private))
    monkeypatch.setenv('TOOL_CACHE_ENABLED', '1')
    monkeypatch.setenv('PERCEPTION_CACHE_ENABLED', '1')
    monkeypatch.setattr(gate, 'DISABLE_PERCEPTION_CACHE', True)
    env, checks = gate.environment()
    assert all(checks.values())
    assert env['TOOL_CACHE_ENABLED'] == env['PERCEPTION_CACHE_ENABLED'] == '0'
    assert env['QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET'] == '8192'
    assert env['QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS'] == '32768'


def test_contaminated_canary_refused_before_any_remote_preflight(monkeypatch, tmp_path):
    (tmp_path/'cache-contamination-hold').mkdir()
    monkeypatch.setattr(grouped, 'RUN', tmp_path)
    monkeypatch.setattr(grouped, 'helpers', lambda: pytest.fail('must not reach helper'))
    with pytest.raises(AssertionError, match='contaminated'):
        grouped.preflight()
