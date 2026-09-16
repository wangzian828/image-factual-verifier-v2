import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('campaign', Path(__file__).parents[1] / 'scripts/server/friend_generation_campaign.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(state, path):
    return dict(state=state, status_path=path, status_sha256=path+'sha')


def test_first_success_preserved_and_failure_is_not_skipped():
    waves = [{'a':item('succeeded','original-a'), 'b':item('failed','original-b')},
             {'a':item('succeeded','retry-a'), 'b':item('succeeded','retry-b')}]
    result=m.aggregate(['a','b','c'],waves)
    assert result['success']==2 and result['unattempted_ids']==['c']
    assert result['selected']['a']['status_path']=='original-a'
    assert result['selected']['b']['status_path']=='retry-b'


def test_active_and_failed_not_success():
    result=m.aggregate(['a','b'],[{'a':item('failed','a'),'b':item('running','b')}])
    assert result['failed_ids']==['a'] and result['active_ids']==['b']
    assert result['success']==0


@pytest.mark.parametrize('field,value',[('config_hash','wrong'),('sample_id','other'),('state','unknown')])
def test_foreign_status_rejected(tmp_path,field,value):
    p=tmp_path/'sample-a/status.json';p.parent.mkdir()
    d=dict(sample_id='sample-a',config_hash='hash',state='failed');d[field]=value
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):m.load_statuses(tmp_path,['sample-a'],'hash')


def test_identity_change_refused(tmp_path):
    path=tmp_path/'identity.json';m.freeze(path,{'a':1});m.freeze(path,{'a':1})
    with pytest.raises(ValueError):m.freeze(path,{'a':2})


def test_pro_cannot_reuse_flash_thinking_or_other_model():
    expected={'includeThoughts':True,'thinkingBudget':-1}
    rows=[dict(event='request',model='gemini-3.1-pro-preview',images=[{'sha256':'image'}],
               function_responses=1,max_output_tokens=65535,thinking=expected),
          dict(event='response',generation_complete=True)]
    assert all(m.wire_checks(rows,'gemini-3.1-pro-preview','image',expected).values())
    rows[0]['thinking']={'thinkingBudget':4000}
    assert not m.wire_checks(rows,'gemini-3.1-pro-preview','image',expected)['thinking']
    rows[0]['model']='gemini-3.6-flash'
    assert not m.wire_checks(rows,'gemini-3.1-pro-preview','image',expected)['exact_model']


def test_model_route_is_exact_and_keeps_query():
    path='/v1beta/models/gemini-3.1-pro-preview-customtools:streamGenerateContent?alt=sse'
    assert m.pro_model_route(path)==path.replace('-customtools','')
    for wrong in ['/v1beta/models/gemini-3.6-flash:generateContent',
                  'https://evil.test'+path, '/v1beta/models/gemini-3.1-pro-preview-customtools-evil:generateContent']:
        assert m.pro_model_route(wrong)==wrong
