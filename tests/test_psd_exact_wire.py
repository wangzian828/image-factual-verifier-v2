import copy

import pytest

from scripts.server.probe_psd_exact_wire import variants, probe_plan


def payload():
    return {'tool_choice':'required','max_tokens':32768,'vllm_xargs':{'ifv_thinking_budget':8192},
            'temperature':.7,'messages':[{'role':'user','content':'exact  text'}],
            'tools':[{'type':'function','function':{'name':'x'}}],'seed':99,'cache_salt':'existing'}


def test_controls_preserve_original_except_declared_fields():
    original=payload(); before=copy.deepcopy(original)
    for label, body in variants(original):
        assert body.pop('return_token_ids') is True
        assert body['tool_choice'] == label
        body['tool_choice']='required'
        assert body == before
    assert original == before


@pytest.mark.parametrize('key,value',[('tool_choice','auto'),('max_tokens',8192),('temperature',0),('vllm_xargs',{})])
def test_protocol_drift_refused(key,value):
    original=payload();original[key]=value
    with pytest.raises(ValueError):variants(original)


def test_sampling_plan_is_finite_and_balanced():
    assert len(probe_plan(False)) == 4
    parallel = probe_plan(True)
    assert len(set(parallel)) == len(parallel) == 8
    for index in range(2):
        for choice in ['required','auto']:
            assert sum(i == index and c == choice for i, _, c in parallel) == 2
