"""Roll owned replicas to verified pre-grammar capture after GPU resume gate exits."""
from concurrent.futures import ThreadPoolExecutor
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
DEPLOY=ROOT/'training-artifacts/psd-raw-teacher-20260916-v18'
CODE=DEPLOY/'code'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OUT=SERVICE/'raw-teacher-noasync-v1'
GATE=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/psd-grounded-review-canary-v15/dp4-resume-gate-v1'


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('launch','execute'))
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    assert owner.load(DEPLOY/'state.json')['deployment_ready_not_live']
    if args.mode=='launch':
        OUT.mkdir(exist_ok=False)
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    owner.save(OUT/'state.json',{'phase':'waiting_for_gpu_resume_gate','formal_training':False})
    previous=owner.load(GATE/'process.json')
    for _ in range(1800):
        path=Path(f"/proc/{previous['pid']}/cmdline")
        if not path.exists() or not path.read_bytes(): break
        if path.read_bytes().rstrip(b'\0').decode().split('\0')!=previous['command']: break
        time.sleep(1)
    else:raise RuntimeError('Training gate still active; no service changes made')
    for rel,digest in owner.load(DEPLOY/'code-binding.json').items():
        assert owner.sha(CODE/rel)==digest,'Candidate source changed'
    # Wait for the gate's finally-restored processes, not merely for a PID.
    for _ in range(480):
        try:
            for i in range(4):
                owner.http(f'http://127.0.0.1:{19002+i}/v1/models')
            break
        except OSError:time.sleep(1)
    else:raise RuntimeError('Restored services did not become ready')
    owner.verify_export()
    old=[owner.load(SERVICE/f'replica-{i}.json') for i in range(4)]
    envs=[owner.checked(receipt) for receipt in old]
    guard=owner.load(SERVICE/'guard.json');owner.checked(guard)
    owner.save(OUT/'before.json',{'backends':old,'code_binding_sha256':owner.sha(DEPLOY/'code-binding.json'),
        'old_nan_unresolved':True,'candidate_validation_required':True})
    os.kill(guard['pid'],signal.SIGSTOP)
    def restart(i):
        command=old[i]['command'][:]
        for flag in ('--worker-cls','--gdn-prefill-backend'):
            if flag in command:
                n=command.index(flag);del command[n:n+2]
        if '--async-scheduling' in command:command.remove('--async-scheduling')
        if '--no-async-scheduling' not in command:command.append('--no-async-scheduling')
        command+=['--worker-cls','scripts.server.psd_raw_teacher_worker.RawTeacherWorker','--gdn-prefill-backend','triton']
        assert '--enforce-eager' in command
        env=envs[i].copy();env.update(PYTHONPATH=str(CODE)+':'+env.get('PYTHONPATH',''),
            PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(ROOT/'cache/psd-raw-teacher'),
            FLASHINFER_WORKSPACE_BASE=str(ROOT/'cache/psd-flashinfer'))
        env.pop('PSD_NUMERICAL_DIAGNOSTIC_DIR',None)
        owner.stop(old[i])
        try:
            receipt=owner.spawn(command,env,OUT/f'backend-{i}.log')
        except Exception:
            receipt=owner.spawn(old[i]['command'],envs[i],OUT/f'restore-{i}.log')
            owner.save(SERVICE/f'replica-{i}.json',receipt)
            raise
        owner.save(SERVICE/f'replica-{i}.json',receipt);owner.save(OUT/f'replica-{i}.json',receipt)
    try:
        assert all(r['inflight']==0 for r in owner.http('http://127.0.0.1:19019/health')['replicas'])
        for _ in range(90):
            busy=0
            for i in range(4):
                with urllib.request.urlopen(f'http://127.0.0.1:{19002+i}/metrics',timeout=5) as f:text=f.read().decode()
                values=[float(l.rsplit(' ',1)[1]) for l in text.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
                assert len(values)==2;busy+=sum(values)
            if busy==0:break
            time.sleep(1)
        else:raise RuntimeError('Owned inference did not drain')
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(restart,range(4)))
    finally:os.kill(guard['pid'],signal.SIGCONT)
    owner.save(OUT/'state.json',{'phase':'loading_raw_teacher_candidate','formal_training':False})
    for _ in range(480):
        try:
            for i in range(4):
                owner.checked(owner.load(SERVICE/f'replica-{i}.json'))
                card=owner.http(f'http://127.0.0.1:{19002+i}/v1/models')['data'][0]
                assert card['root']==str(ROOT/'exports/h20-sft-merged4872-3epoch-step3084-20260915/model')
            owner.save(OUT/'state.json',{'phase':'ready_for_raw_probability_validation','formal_training':False,
                'sampling_method_changed':False,'async_scheduling':False});return
        except OSError:time.sleep(1)
    raise RuntimeError('Raw teacher startup failed; inspect per-replica logs')


if __name__=='__main__':main()
