import asyncio
import copy
import pytest

from ifv_training.psd_slate import SLATE_SCHEMA, bound_slate_schema, propose_slate, validate_slate_revision


def test_schema_uses_observed_positions_not_diagnostic_anchor():
    original=copy.deepcopy(SLATE_SCHEMA)
    result=bound_slate_schema({2:'retained',7:'replace'},7,decision_positions=[2,3,4,6,7,24])
    hints=result['properties']['hints']
    assert hints['maxItems']==6
    assert hints['items']['properties']['position']['enum']==[2,3,4,6,7,24]
    assert SLATE_SCHEMA==original
    # Empty slate still allowed: do not force a spurious intervention.
    assert hints.get('minItems',0)==0
    with pytest.raises(ValueError):validate_slate_revision({2:'retained',7:'replace'},
        {2:'changed',7:'new'},passing_positions=[2],failed_position=7,decision_positions=[2,3,4,6,7,24])


@pytest.mark.parametrize('position',[-1,25,True,'7'])
def test_non_native_position_fails_before_api(position):
    with pytest.raises(ValueError):bound_slate_schema({},position,decision_positions=[0,7,24])


@pytest.mark.parametrize('positions',[[3],[4],[6],[3,6],[21],[3,21]])
def test_diagnostic_21_allows_earlier_or_multiple_interventions(monkeypatch,positions):
    import ifv_training.psd_gemini_judge as judge
    seen=[]
    async def request(client,packet,**kw):
        seen.append((packet,kw))
        return {'hints':[{'position':p,'hint':'Check that each inference follows from observed evidence.'}
                         for p in positions]},{}
    monkeypatch.setattr(judge,'_request',request)
    hints,_=asyncio.run(propose_slate(None,public_context={'observed':'public evidence',
        'decision_map':{3:6,4:8,6:12,21:42,24:43}},previous={},
        passing_positions=[],failed_position=21,model='judge',cache_dir=None,
        private_context={'private_reference':'secret-private-answer'}))
    assert set(hints)==set(positions)
    packet,kw=seen[0]
    assert kw['schema']['properties']['hints']['items']['properties']['position']['enum']==[3,4,6,21,24]
    assert 'change only position 21' not in kw['prompt']
    assert 'not an exclusive edit boundary' in kw['prompt']
    assert 'secret-private-answer' not in str(seen)


@pytest.mark.parametrize('bad',[
    [{'position':5,'hint':'Check the unresolved inference.'}],
    [{'position':3,'hint':'Use the exact query {secret}.'}],
    [{'position':3,'hint':'Check the evidence.'},{'position':3,'hint':'Check it again.'}],
])
def test_relaxed_anchor_does_not_relax_observed_positions_procedurality_or_uniqueness(monkeypatch,bad):
    import ifv_training.psd_gemini_judge as judge
    from ifv_training.psd_slate import SlateProposalRejected
    async def request(*a,**kw):return {'hints':bad},{}
    monkeypatch.setattr(judge,'_request',request)
    with pytest.raises(SlateProposalRejected):
        asyncio.run(propose_slate(None,public_context={'decision_map':{3:6,21:42}},previous={},
            passing_positions=[],failed_position=21,model='judge',cache_dir=None))


def test_absent_diagnostic_anchor_fails_before_provider(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    async def request(*a,**kw):pytest.fail('invalid map must fail before API')
    monkeypatch.setattr(judge,'_request',request)
    with pytest.raises(ValueError,match='absent'):
        asyncio.run(propose_slate(None,public_context={'decision_map':{3:6}},previous={},
            passing_positions=[],failed_position=21,model='judge',cache_dir=None))


def test_earlier_proposal_is_reserved_for_real_rerun_without_burning_budget(monkeypatch):
    import ifv_training.psd_gemini_judge as judge
    from ifv_training.psd_slate_search import propose_with_budget
    from ifv_training.psd_repair import _sha
    requests,state=[],{}
    async def request(*a,**kw):
        requests.append(kw)
        return {'hints':[{'position':3,'hint':'Verify the visual reading before relying on it.'}]},{'response_sha256':'test'}
    monkeypatch.setattr(judge,'_request',request)
    options=dict(state=state,round_index=0,budget=12,persist=lambda:None,judge=None,
        kwargs=dict(public_context={'decision_map':{3:6,21:42}},previous={},passing_positions=[],
            failed_position=21,model='judge',cache_dir=None))
    hints,_=asyncio.run(propose_with_budget(**options))
    assert set(hints)=={3} and len(requests)==len(state['proposals'])==1
    assert state['proposals'][0]['status']=='accepted'
    assert state['pending_proposal']['hints']['3']['audit']['passed']
    before=_sha(state)
    asyncio.run(propose_with_budget(**options))
    assert len(requests)==1 and _sha(state)==before
