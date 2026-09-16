import copy
import json

import jsonschema
import pytest

from scripts.server.psd_single_tool_schema import single_call_schema


def schema():
    return {'type':'array','minItems':1,'items':{'type':'object','properties':{
        'name':{'const':'search'},'parameters':{'type':'object','properties':{'query':{'type':'string'}},
            'required':['query'],'additionalProperties':False}},'required':['name','parameters']}}


def test_exactly_one_call_and_arguments_unchanged():
    original=schema();before=copy.deepcopy(original);bounded=single_call_schema(original)
    call={'name':'search','parameters':{'query':'image event'}}
    jsonschema.validate([call],bounded)
    for calls in [[],[call,call],[{'name':'search','parameters':{}}]]:
        with pytest.raises(jsonschema.ValidationError):jsonschema.validate(calls,bounded)
    assert original == before and bounded.pop('maxItems') == 1 and bounded == original
    assert single_call_schema(json.dumps(original)) == single_call_schema(original)


@pytest.mark.parametrize('wrong',[{}, {'type':'object'}, {'type':'array','minItems':0,'items':{}},
                                {'type':'array','minItems':1,'items':{},'maxItems':8}])
def test_unexpected_schema_refused(wrong):
    with pytest.raises(ValueError):single_call_schema(wrong)
