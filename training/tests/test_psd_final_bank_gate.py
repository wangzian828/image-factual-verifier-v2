import json
from pathlib import Path

import pytest

from scripts.server.run_psd_dp4_resume_gate import ready_inputs


def bank(tmp_path):
    root = tmp_path/'owned'
    out = root/'runs/final-bank';out.mkdir(parents=True)
    source = root/'runs/source/snapshot';source.mkdir(parents=True)
    datums = out/'search/datums';datums.mkdir(parents=True)
    files = {'datums':datums/'datums.jsonl', 'datum_manifest':datums/'manifest.json',
             'rollout_gate':out/'rollout-gate.json', 'serving_profile':source/'serving-profile.json',
             'checkpoint_manifest':source/'checkpoint-manifest.json'}
    for path in files.values():path.write_text('{}')
    files['serving_profile'].write_text(json.dumps({'base_url':'http://127.0.0.1:19025/v1'}))
    ready = {k:str(v) for k,v in files.items()}
    ready.update(model=str(root/'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'),adapter=None)
    path=out/'ready.json';path.write_text('{}')
    return root,path,ready


def test_final_gate_uses_attested_datums_snapshot_and_current_gateway(tmp_path):
    root,path,ready=bank(tmp_path)
    calls=[]
    def load(p):calls.append(p);return ready
    selected=ready_inputs(path,root=root,load_ready=load)
    assert calls==[path.resolve()]
    assert selected['datums']==Path(ready['datums'])
    assert selected['snapshot']==Path(ready['serving_profile']).parent
    assert selected['gateway']=='http://127.0.0.1:19025'
    assert selected['run']==path.parent


@pytest.mark.parametrize('problem',['old_gateway','other_model','adapter','outside_datums','missing','manifest_layout','outside_ready'])
def test_invalid_final_bank_rejected_before_gpu_lease(tmp_path,problem):
    root,path,ready=bank(tmp_path)
    if problem=='old_gateway':
        Path(ready['serving_profile']).write_text(json.dumps({'base_url':'http://127.0.0.1:19019/v1'}))
    elif problem=='other_model':ready['model']=str(root/'exports/old/model')
    elif problem=='adapter':ready['adapter']=str(root/'checkpoints/other')
    elif problem=='outside_datums':
        outside=tmp_path/'outside.jsonl';outside.write_text('{}');ready['datums']=str(outside)
    elif problem=='missing':Path(ready['datums']).unlink()
    elif problem=='manifest_layout':ready['datum_manifest']=ready['rollout_gate']
    else:path=tmp_path/'outside-ready.json'
    with pytest.raises(ValueError):ready_inputs(path,root=root,load_ready=lambda _:ready)


def test_attestation_failure_propagates_without_selecting_bank(tmp_path):
    root,path,ready=bank(tmp_path)
    def load(_):raise ValueError('Datum hashes changed')
    with pytest.raises(ValueError,match='hashes changed'):
        ready_inputs(path,root=root,load_ready=load)
