import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).parent / 'scripts/server'
sys.path.insert(0, str(SCRIPTS))
import benchmark_sft2056_fixed_tokens as bench


def test_fixed_budget_does_not_mutate_formal_request():
    source = {'max_tokens': 32768, 'thinking_token_budget': 8192,
              'messages': [{'role': 'user', 'content': 'fixture'}], 'tools': [{'x': 1}]}
    fixed = bench.fixed_body(source, 'test-salt')
    assert source['max_tokens'] == 32768 and 'min_tokens' not in source
    assert fixed['max_tokens'] == fixed['min_tokens'] == 1024
    assert fixed['ignore_eos'] is True and fixed['thinking_token_budget'] == 8192
    fixed['messages'][0]['content'] = 'changed'
    assert source['messages'][0]['content'] == 'fixture'


def test_metrics_aggregate_and_delta_without_gauges():
    a = 'vllm:x_sum{engine="0"} 4\nvllm:x_count{engine="0"} 2\nvllm:running 3\n'
    b = 'vllm:x_sum{engine="0"} 8\nvllm:x_sum{engine="1"} 2\nvllm:x_count 3\nvllm:running 0\n'
    assert bench.metric_delta(a, b) == {'vllm:x_sum': 6, 'vllm:x_count': 1}
