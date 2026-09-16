"""Lease one owned GPU for exact rescoring, preserving all completed episodes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
DEPLOY=ROOT/'training-artifacts/psd-raw-teacher-20260916-v18'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
RUN=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/psd-raw-teacher-canary-v18'
OUT=RUN/'gpu-rescore-v1'


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('launch','execute'))
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if args.mode=='launch':
        assert owner.load(RUN/'state.json')['phase']=='requires_frozen_teacher_topk'
        probe=owner.load(SERVICE/'nan-raw-teacher-concurrency40-v1/summary.json')
        assert probe['completed']==120 and all(r['status']=='completed' and r['invalid_logprobs']==0
            and r['sentinel_logprobs']==0 for r in probe['results'])
        OUT.mkdir(exist_ok=False)
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    backend=owner.load(SERVICE/'replica-3.json');old_env=owner.checked(backend)
    guard=owner.load(SERVICE/'guard.json');owner.checked(guard)
    stopped=False
    try:
        os.kill(guard['pid'],signal.SIGSTOP)
        for _ in range(90):
            with urllib.request.urlopen('http://127.0.0.1:19005/metrics',timeout=5) as f:text=f.read().decode()
            numbers=[float(l.rsplit(' ',1)[1]) for l in text.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(numbers)==2
            if sum(numbers)==0:break
            time.sleep(1)
        else:raise RuntimeError('Owned GPU3 did not drain')
        owner.stop(backend);stopped=True
    finally:os.kill(guard['pid'],signal.SIGCONT)
    try:
        owner.verify_export()
        cuda=ROOT/'envs/h20-qwen35-128k';code=DEPLOY/'code'
        env={**os.environ,'PATH':str(cuda/'bin')+':'+os.environ['PATH'],'CUDA_HOME':str(cuda),
            'CUDA_VISIBLE_DEVICES':'3','PYTHONPATH':str(code)+':'+str(code/'training'),'PYTHONDONTWRITEBYTECODE':'1',
            'TMPDIR':str(ROOT/'tmp'),'HF_HOME':str(ROOT/'cache/huggingface'),'HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','TORCH_HOME':str(ROOT/'cache/torch'),
            'TRITON_CACHE_DIR':str(ROOT/'cache/psd-gpu-probe/triton'),
            'TORCH_EXTENSIONS_DIR':str(ROOT/'cache/psd-gpu-probe/extensions'),
            'XDG_CACHE_HOME':str(ROOT/'cache/psd-gpu-probe'),'OMP_NUM_THREADS':'4'}
        command=[str(cuda/'bin/python'),'-u',str(DEPLOY/'rebuild_psd_raw_bank.py'),'score']
        with (OUT/'score.log').open('x') as log:
            p=subprocess.Popen(command,env=env,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            owner.save(OUT/'score-process.json',{'pid':p.pid,'command':command});returncode=p.wait()
        owner.save(OUT/'state.json',{'phase':'rescored' if returncode==0 else 'requires_fix',
            'returncode':returncode,'formal_training':False})
    finally:
        if stopped:
            receipt=owner.spawn(backend['command'],old_env,OUT/'restored-backend.log')
            owner.save(SERVICE/'replica-3.json',receipt)
        owner.verify_export()


if __name__=='__main__':main()
