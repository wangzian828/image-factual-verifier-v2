"""Promote a tested execution candidate for real repair, not a proof of NaN fix."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OUT=SERVICE/'validated-execution-candidate-v1'
OWNER=ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py'


def candidate_command(command,kind):
    result=list(command)
    assert result[2]=='serve' and result[3]==str(ROOT/'exports/h20-sft-merged4872-3epoch-step3084-20260915/model')
    if '--enforce-eager' in result:result.remove('--enforce-eager')
    if '--compilation-config' in result:
        i=result.index('--compilation-config');del result[i:i+2]
    if kind=='decode':return [*result,'--compilation-config',json.dumps({'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'})]
    assert kind=='eager'
    return [*result,'--enforce-eager']


def gate(o,kind):
    gpu=2 if kind=='decode' else 3
    paths=[SERVICE/f'nan-execution-matrix-part{part}-v1/summary.json' for part in (1,2)]
    paths.append(SERVICE/'nan-mixed-history-v1/summary.json')
    counts=[]
    for index,p in enumerate(paths):
        data=o.load(p)
        assert data['completed']==(200 if index<2 else 96)
        rows=[r for r in data['results'] if r['gpu']==gpu]
        assert len(rows)==(50 if index<2 else 24)
        assert all(r['status']=='completed' and r.get('finish_reason')=='tool_calls' for r in rows)
        counts.append(len(rows))
    return {'candidate':kind,'validated_gpu':gpu,'request_counts':counts,'files':{str(p):o.sha(p) for p in paths},
        'permanent_fix_proven':False,'purpose':'real_repair_validation_only','training_started':False}


def wait_ready(o,new,port):
    for _ in range(240):
        o.checked(new)
        try:
            model=o.http(f'http://127.0.0.1:{port}/v1/models')['data'][0]
            if model['id']=='ifv-psd-sft3084' and model['root']==new['command'][3]:
                assert model['max_model_len']==131072
                return
        except OSError:pass
        time.sleep(2)
    raise RuntimeError('Candidate readiness timeout; stop and diagnose')


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['launch','execute']);p.add_argument('--kind',choices=['decode','eager'],required=True)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('owner',OWNER);o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
    o.verify_export()
    validation=gate(o,args.kind)
    if args.mode=='launch':
        for name in ['nan-execution-matrix-part2-v1','nan-mixed-history-v1','nan-execution-matrix-restarts-v1']:
            receipt=o.load(SERVICE/name/'process.json');stat=Path(f'/proc/{receipt["pid"]}/stat')
            assert not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z'
        assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19019/health')['replicas'])
        OUT.mkdir(exist_ok=False)
        o.save(OUT/'binding.json',validation)
        receipt=o.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute','--kind',args.kind],os.environ.copy(),OUT/'run.log')
        o.save(OUT/'process.json',receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    guard=o.load(SERVICE/'guard.json');o.checked(guard)
    for gpu in range(4):
        old=o.load(SERVICE/f'replica-{gpu}.json');env=o.checked(old)
        assert env['CUDA_VISIBLE_DEVICES']==str(gpu)
        command=candidate_command(old['command'],args.kind)
        if command==old['command']:continue
        o.save(OUT/f'before-{gpu}.json',old)
        os.kill(guard['pid'],signal.SIGSTOP)
        try:
            assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19019/health')['replicas'])
            for _ in range(90):
                with urllib.request.urlopen(f'http://127.0.0.1:{19002+gpu}/metrics',timeout=5) as response:
                    values=[float(l.rsplit(' ',1)[1]) for l in response.read().decode().splitlines()
                        if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
                assert len(values)==2
                if sum(values)==0:break
                time.sleep(1)
            else:raise RuntimeError('Candidate backend did not drain')
            o.stop(old)
            new=o.spawn(command,env,OUT/f'backend-{gpu}.log')
            o.save(SERVICE/f'replica-{gpu}.json',new);o.save(OUT/f'replica-{gpu}.json',new)
            o.save(OUT/'state.json',{'phase':'waiting_candidate','gpu':gpu,'training_started':False})
        finally:os.kill(guard['pid'],signal.SIGCONT)
        wait_ready(o,new,19002+gpu)
    # Record exact live request bytes on the actual non-streaming repair path.
    gateway_path=SERVICE/'four-gpu-v7/gateway.json'
    old=o.load(gateway_path);env=o.checked(old)
    assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19019/health')['replicas'])
    assert not env.get('PSD_WIRE_CAPTURE_DIR')
    env.update(PSD_WIRE_CAPTURE_DIR=str(OUT/'wire'),PSD_WIRE_MAX_RESPONSE_BYTES=str(64*1024**2))
    o.save(OUT/'gateway-before.json',old);o.stop(old)
    new=o.spawn(old['command'],env,OUT/'gateway.log')
    o.save(gateway_path,new);o.save(OUT/'gateway.json',new)
    for _ in range(60):
        o.checked(new)
        try:
            health=o.http('http://127.0.0.1:19019/health')
            if health['wire_capture']['enabled'] and len(health['replicas'])==4:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Candidate diagnostic gateway did not become ready')
    o.save(OUT/'state.json',{**validation,'phase':'candidate_ready_for_real_repair_not_formal_training',
        'replicas':{str(g):o.load(SERVICE/f'replica-{g}.json') for g in range(4)},
        'gateway':new,'wire_capture':health['wire_capture'],'time':time.time()})


if __name__=='__main__':main()
