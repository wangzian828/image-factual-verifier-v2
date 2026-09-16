import asyncio
import copy
import pytest

from ifv_training.psd_slate import SLATE_SCHEMA, bound_slate_schema, propose_slate, validate_slate_revision


def test_schema_matches_mechanical_edit_boundary_without_mutating_global():
    original=copy.deepcopy(SLATE_SCHEMA)
    result=bound_slate_schema({2:'retained',7:'replace'},7)
    hints=result['properties']['hints']
    assert hints['maxItems']==2
    assert hints['items']['properties']['position']['enum']==[2,7]
    assert SLATE_SCHEMA==original
    # Empty slate still allowed: do not force a spurious intervention.
    assert hints.get('minItems',0)==0
    with pytest.raises(ValueError):validate_slate_revision({2:'retained',7:'replace'},
        {2:'changed',7:'new'},passing_positions=[2],failed_position=7)


@pytest.mark.parametrize('position',[-1,25,True,'7'])
def test_non_native_position_fails_before_api(position):
    with pytest.raises(ValueError):bound_slate_schema({},position)


def test_first_proposal_cannot_request_an_unrelated_earlier_position(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    seen=[]
    async def request(client,packet,**kw):
        seen.append((packet,kw))
        return {'hints':[{'position':21,'hint':'Check that the remaining conclusion follows from observed evidence.'}]},{}
    monkeypatch.setattr(judge,'_request',request)
    hints,_=asyncio.run(propose_slate(None,public_context={'observed':'public evidence'},previous={},
        passing_positions=[],failed_position=21,model='judge',cache_dir=None,
        private_context={'private_reference':'secret-private-answer'}))
    assert set(hints)=={21}
    packet,kw=seen[0]
    assert kw['schema']['properties']['hints']['items']['properties']['position']['enum']==[21]
    assert 'change only position 21' in kw['prompt']
    assert 'secret-private-answer' not in str(seen)
