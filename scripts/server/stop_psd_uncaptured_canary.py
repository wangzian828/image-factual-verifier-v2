"""Stop only the owned epoch3 diagnostic missing training token capture."""
import importlib.util
import os
from pathlib import Path
import time

DEPLOY=Path('/volume/ybo/wza/training-artifacts/psd-epoch3-20260916-v1')
spec=importlib.util.spec_from_file_location('epoch3_owner',DEPLOY/'psd_epoch3_canary.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)

def main():
    receipt=owner.load(owner.RUN/'process.json')
    env=owner.checked(receipt)
    assert env.get('IFV_CAPTURE_POLICY_TOKENS','').lower() not in ('1','true','yes','on')
    assert receipt['command']==[str(owner.ROOT/'envs/h20-qwen35-128k/bin/python'),
        '-u',str(DEPLOY/'psd_epoch3_canary.py'),'--collect']
    marker=owner.RUN/'capture-preflight-hold.json'
    assert not marker.exists()
    owner.save(marker,{'reason':'collector did not request native exact policy token/logprob metadata',
        'training_started':False,'accepted_source_bank':False,'time':time.time(),
        'process':receipt,'all_artifacts_preserved':True})
    owner.stop(receipt)
    for _ in range(180):
        if all(r['inflight']==0 for r in owner.http('http://127.0.0.1:19017/health')['replicas']):
            print('Owned collector stopped; gateway drained; model/guard/weights retained',flush=True)
            return
        time.sleep(1)
    raise RuntimeError('Cancellation has not drained; do not relaunch')

if __name__=='__main__':main()
