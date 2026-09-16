"""Lease owned idle GPU3 for real PSD update/resume; restore inference on exit."""
import importlib.util
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
DEPLOY=ROOT/'training-artifacts/psd-grounded-review-20260916-v15'
CODE=DEPLOY/'code'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
RUN=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/psd-grounded-review-canary-v15'
OUT=RUN/'gpu-update-gate-v1'
SNAPSHOT=ROOT/'training-artifacts/psd-contract-audit-20260916-v10/snapshot'


def main():
    global OUT, DEPLOY, CODE, RUN
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('launch','execute'))
    parser.add_argument('--output-name',default='gpu-update-gate-v1')
    parser.add_argument('--probe-script',default='psd_real_datum_gpu_smoke.py')
    parser.add_argument('--deterministic-attention',action='store_true')
    parser.add_argument('--deployment',default=DEPLOY.name)
    parser.add_argument('--run-name',default=RUN.name)
    args=parser.parse_args()
    assert Path(args.deployment).name==args.deployment and args.deployment.startswith('psd-')
    assert Path(args.run_name).name==args.run_name and args.run_name.startswith('psd-')
    DEPLOY=ROOT/'training-artifacts'/args.deployment
    CODE=DEPLOY/'code'
    RUN=RUN.parent/args.run_name
    assert Path(args.output_name).name==args.output_name and args.output_name.startswith('gpu-update-gate-')
    assert Path(args.probe_script).name==args.probe_script and args.probe_script.endswith('.py')
    OUT=RUN/args.output_name
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if args.mode=='launch':
        assert owner.load(RUN/'state.json')['phase']=='ready_for_trainer'
        OUT.mkdir(exist_ok=False)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'execute',
            '--output-name',args.output_name,'--probe-script',args.probe_script,
            '--deployment',args.deployment,'--run-name',args.run_name]
        if args.deterministic_attention:command+=['--deterministic-attention']
        receipt=owner.spawn(command,os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    owner.verify_export()
    assert owner.load(DEPLOY/'state.json')['deployment_ready_not_live']
    for relative,digest in owner.load(DEPLOY/'code-binding.json').items():
        assert owner.sha(CODE/relative)==digest,'Immutable training source changed'
    guard=owner.load(SERVICE/'guard.json');owner.checked(guard)
    backend=owner.load(SERVICE/'replica-3.json');original_env=owner.checked(backend)
    assert original_env['CUDA_VISIBLE_DEVICES']=='3'
    owner.save(OUT/'binding.json',{'backend':backend,'probe_sha256':owner.sha(DEPLOY/args.probe_script),
        'datums_sha256':owner.sha(RUN/'bank/datums/datums.jsonl'),'formal_training':False})
    stopped=False
    os.kill(guard['pid'],signal.SIGSTOP)
    try:
        assert all(r['inflight']==0 for r in owner.http('http://127.0.0.1:19019/health')['replicas'])
        for _ in range(90):
            metrics=urllib.request.urlopen('http://127.0.0.1:19005/metrics',timeout=5).read().decode()
            values=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines()
                if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(values)==2
            if sum(values)==0:break
            time.sleep(1)
        else:raise RuntimeError('Owned GPU3 did not drain')
        owner.stop(backend);stopped=True
    finally:os.kill(guard['pid'],signal.SIGCONT)
    try:
        cuda=ROOT/'envs/h20-qwen35-128k'
        env={**os.environ,'PATH':str(cuda/'bin')+':'+os.environ['PATH'],'CUDA_VISIBLE_DEVICES':'3',
            'CUDA_HOME':str(cuda),'PYTHONPATH':str(CODE)+':'+str(CODE/'training'),
            'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(ROOT/'tmp'),'OMP_NUM_THREADS':'4','MAX_JOBS':'8',
            'HF_HOME':str(ROOT/'cache/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'TORCH_HOME':str(ROOT/'cache/torch'),'TRITON_CACHE_DIR':str(ROOT/'cache/psd-gpu-probe/triton'),
            'TORCH_EXTENSIONS_DIR':str(ROOT/'cache/psd-gpu-probe/extensions'),
            'FLASHINFER_WORKSPACE_BASE':str(ROOT/'cache/psd-flashinfer'),
            'XDG_CACHE_HOME':str(ROOT/'cache/psd-gpu-probe')}
        cmd=[str(cuda/'bin/python'),'-u',str(DEPLOY/args.probe_script),
            '--datums',str(RUN/'bank/datums/datums.jsonl'),'--datum-manifest',str(RUN/'bank/datums/manifest.json'),
            '--serving-profile',str(SNAPSHOT/'serving-profile.json'),
            '--checkpoint-manifest',str(SNAPSHOT/'checkpoint-manifest.json'),
            '--output-dir',str(OUT/'probe')]
        if args.deterministic_attention:cmd+=['--deterministic-attention']
        owner.save(OUT/'state.json',{'phase':'real_gpu_optimizer_and_resume','formal_training':False})
        with (OUT/'probe.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            owner.save(OUT/'probe-process.json',{'pid':p.pid,'command':cmd})
            code=p.wait()
        owner.save(OUT/'state.json',{'phase':'probe_completed' if code==0 else 'probe_failed_requires_fix',
            'returncode':code,'formal_training':False})
    finally:
        if stopped:
            os.kill(guard['pid'],signal.SIGSTOP)
            try:
                restored=owner.spawn(backend['command'],original_env,OUT/'restored-backend.log')
                owner.save(SERVICE/'replica-3.json',restored)
            finally:os.kill(guard['pid'],signal.SIGCONT)
        owner.verify_export()


if __name__=='__main__':main()
