"""Use existing scoped CUDA libraries, then test a real prefill before ready."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/volume/ybo/wza')
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OUT=SERVICE/'nan-flashinfer-link-recovery-v2'


def main():
    os.umask(0o077)
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if sys.argv[1:]==['launch']:
        OUT.mkdir(exist_ok=False)
        r=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',r);print(json.dumps({'pid':r['pid'],'output':str(OUT)}));return
    assert sys.argv[1:]==['execute']
    old=owner.load(SERVICE/'replica-2.json')
    assert old==owner.load(SERVICE/'nan-flashinfer-toolchain-recovery-v1/replica-2.json')
    assert not Path(f'/proc/{old["pid"]}').exists()
    env=owner.checked(owner.load(SERVICE/'replica-0.json'))
    cuda=ROOT/'envs/h20-qwen35-128k';lib=cuda/'targets/x86_64-linux/lib'
    assert (lib/'libcudart.so').is_file() and (lib/'stubs/libcuda.so').is_file()
    env.update(CUDA_VISIBLE_DEVICES='2',FLASHINFER_WORKSPACE_BASE=str(ROOT/'cache/psd-flashinfer'),
        CUDA_HOME=str(cuda),CUDACXX=str(cuda/'bin/nvcc'),MAX_JOBS='8',
        CPLUS_INCLUDE_PATH=str(cuda/'targets/x86_64-linux/include'),
        LIBRARY_PATH=str(lib)+':'+str(lib/'stubs'),
        LD_LIBRARY_PATH=str(lib)+':'+env.get('LD_LIBRARY_PATH',''),PATH=str(cuda/'bin')+':'+env['PATH'])
    r=owner.spawn(old['command'],env,OUT/'backend.log')
    owner.save(SERVICE/'replica-2.json',r);owner.save(OUT/'replica-2.json',r)
    owner.save(OUT/'state.json',{'phase':'waiting_library_link_and_real_prefill','formal_training':False})
    for _ in range(300):
        owner.checked(r)
        try:
            card=owner.http('http://127.0.0.1:19004/v1/models')['data'][0]
            if card['root']==r['command'][3]:break
        except OSError:pass
        time.sleep(2)
    else:raise RuntimeError('No FlashInfer API readiness')
    spec=importlib.util.spec_from_file_location('probe',ROOT/'training-artifacts/psd-observed-positions-20260916-v14/probe_psd_nan_stream_v4.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    target=OUT/'real-prefill';target.mkdir(exist_ok=False)
    asyncio.run(probe.execute(target,2,wire_ticket=SERVICE/'validated-execution-candidate-v1/wire/1789551642580005259-b2b919425c11426eb34c440879f2f248'))
    result=owner.load(target/'state.json')
    assert result['status']=='completed' and result.get('finish_reason')=='tool_calls'
    owner.checked(r)
    log=(OUT/'backend.log').read_text()
    assert 'Using FlashInfer GDN prefill kernel' in log and 'compilation terminated' not in log and 'Falling back to Triton' not in log
    owner.save(OUT/'state.json',{'phase':'real_prefill_passed_ready_for_corpus','formal_training':False})
    owner.save(SERVICE/'nan-scheduler-gdn-ablation-v1/state.json',{'phase':'ready_for_ablation_not_production',
        'training_started':False,'flashinfer_recovery':str(OUT)})


if __name__=='__main__':main()
