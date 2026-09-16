"""Run inside the pinned vLLM environment; no weights or GPU needed."""
import copy

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from scripts.server.psd_single_tool_parser import PSDSingleToolParser


def request(choice='required', parallel=False):
    return ChatCompletionRequest(model='pinned',messages=[{'role':'user','content':'same'}],
        tools=[{'type':'function','function':{'name':'search','parameters':{
            'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}],
        tool_choice=choice,parallel_tool_calls=parallel,max_tokens=32768,temperature=.7)


def test_actual_parser_enforces_requested_single_call_only():
    old=Qwen3CoderToolParser.__new__(Qwen3CoderToolParser)
    new=PSDSingleToolParser.__new__(PSDSingleToolParser)
    original=request();before=original.model_dump()
    baseline=old.adjust_request(copy.deepcopy(original));adjusted=new.adjust_request(original)
    assert 'maxItems' not in baseline.structured_outputs.json
    assert adjusted.structured_outputs.json['maxItems'] == 1
    candidate=copy.deepcopy(adjusted.structured_outputs.json);candidate.pop('maxItems')
    assert candidate == baseline.structured_outputs.json
    for key in ['messages','tools','tool_choice','parallel_tool_calls','max_tokens','temperature']:
        assert adjusted.model_dump()[key] == before[key]
    import xgrammar
    assert xgrammar.Grammar.from_json_schema(adjusted.structured_outputs.json) is not None


@pytest.mark.parametrize('choice,parallel',[('auto',False),('none',False),('required',True)])
def test_other_paths_unchanged(choice,parallel):
    old=Qwen3CoderToolParser.__new__(Qwen3CoderToolParser)
    new=PSDSingleToolParser.__new__(PSDSingleToolParser)
    original=request(choice,parallel)
    assert new.adjust_request(copy.deepcopy(original)).model_dump() == old.adjust_request(original).model_dump()
