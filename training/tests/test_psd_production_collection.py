import math
import pytest
from scripts.server.run_psd_production_collection import compact_result, capture_stats
from src.eval.result_records import run_result_record


def test_compaction_preserves_result_record_without_retaining_native_history():
    result = {'verdict': 'fake', 'termination': 'success', 'judgment': {'fact_check_report': 'report'},
              'state': {'all_steps': [{'large_native_capture': [0]*100}], 'stage_timings': {'react': 2}}}
    kwargs = dict(trace_path='traces/example.json', metadata=None, episode_id='e',
                  prompt_group_id='g', rollout_index=0, sampling_seed=42)
    compact = compact_result(result)
    assert run_result_record({'case_id': 'c'}, compact, **kwargs) == run_result_record({'case_id': 'c'}, result, **kwargs)
    assert compact['state'] == {'stage_timings': {'react': 2}}
    assert result['state']['all_steps']


def trace(probability=-.1):
    return {'state': {'all_steps': [{'stage': 'unified_react', 'action_type': 'tool_call', 'metadata': {
        'tool_success': True, 'policy_token_capture': {'status': 'complete', 'prompt_token_ids': [1],
        'completion_token_ids': [2], 'completion_logprobs': [probability]}}}]}}


def test_native_capture_retained_without_claiming_frozen_teacher_scoring():
    assert capture_stats(trace()) == {'captures': 1, 'successful_tool_calls': 1}


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_nonfinite_selected_probability_blocks_online_extension(bad):
    with pytest.raises(ValueError):
        capture_stats(trace(bad))


def test_missing_capture_blocks_online_extension():
    value = trace()
    value['state']['all_steps'][0]['metadata']['policy_token_capture'] = {}
    with pytest.raises(ValueError):
        capture_stats(value)
