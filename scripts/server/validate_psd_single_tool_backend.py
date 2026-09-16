"""Opt-in GPU1 serving candidate, retaining GPU2 baseline and every old artifact."""
from __future__ import annotations

import asyncio
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
SERVICE=ROOT/'inference/psd-sft2056-safety-20260916'
HELPERS=ROOT/'training-artifacts/psd-serving-safety-20260916'
DEPLOY=HELPERS/'single-tool-v1'
OUT=SERVICE/'single-tool-backend-v1'


def save(path,value):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(path)


def checked_process(path):
    receipt=json.loads(path.read_text());pid=receipt['pid']
    args=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    assert args==receipt['command'] and os.getpgid(pid)==pid
    env=dict(x.decode().split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if x)
    return receipt,env


def stop(receipt):
    current,_=checked_process(OUT/'stop-target.json')
    assert current==receipt
    os.killpg(receipt['pid'],signal.SIGTERM)
    for _ in range(120):
        p=Path(f'/proc/{receipt["pid"]}/stat')
        if not p.exists() or p.read_text().split(') ',1)[1][0]=='Z':return
        time.sleep(.5)
    raise RuntimeError('Owned backend did not stop gracefully')


def spawn(command,env,name):
    with (OUT/name).open('xb') as log:
        return subprocess.Popen(command,cwd=DEPLOY,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def ready(child):
    for _ in range(180):
        if child.poll() is not None:raise RuntimeError('Candidate backend exited')
        try:
            with urllib.request.urlopen('http://127.0.0.1:19003/v1/models',timeout=3) as f:card=json.load(f)['data'][0]
            assert card['id']=='ifv-psd-sft2056-safety' and card['max_model_len']==131072
            return
        except OSError:time.sleep(5)
    raise RuntimeError('Backend readiness deadline')


def main():
    assert not OUT.exists(),'Never repeat or overwrite a candidate'
    assert (SERVICE/'exact-wire-tool-choice-concurrent-v1/summary.json').is_file()
    old,env=checked_process(SERVICE/'replica-1.json')
    baseline,_=checked_process(SERVICE/'replica-2.json')
    guard,_=checked_process(SERVICE/'guard.json')
    assert env['CUDA_VISIBLE_DEVICES']=='1'
    command=baseline['command'][:]
    assert '--no-enable-prefix-caching' in command
    command[command.index('--port')+1]='19003'
    command[command.index('--tool-call-parser')+1]='ifv_psd_qwen3_single'
    command+=['--tool-parser-plugin',str(DEPLOY/'scripts/server/psd_single_tool_parser.py')]
    original_env=env.copy();env['PYTHONPATH']=str(DEPLOY)
    OUT.mkdir();save(OUT/'backend-before.json',old);save(OUT/'baseline.json',baseline)
    save(OUT/'guard-before.json',guard);save(OUT/'stop-target.json',old)
    paused=False;stopped=False;child=None
    try:
        os.kill(guard['pid'],signal.SIGSTOP);paused=True
        for _ in range(60):
            with urllib.request.urlopen('http://127.0.0.1:19003/metrics',timeout=5) as f:metrics=f.read().decode()
            active=[float(s.rsplit(' ',1)[1]) for s in metrics.splitlines()
                    if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(active)==2
            if sum(active)==0:break
            time.sleep(1)
        else:raise RuntimeError('GPU1 still busy; do not interrupt work')
        stop(old);stopped=True
        child=spawn(command,env,'backend.log')
        receipt={'pid':child.pid,'command':command,'started_at':time.time(),'replaces':old['pid'],
                 'prefix_caching':False,'single_tool_constraint':True}
        save(SERVICE/'replica-1.json',receipt);save(OUT/'candidate.json',receipt)
        save(OUT/'state.json',{'phase':'loading','psd_gate_passed':False})
        # Other three models remain resident: the unchanged guard cannot start
        # its all-GPU-free fallback while this one endpoint loads.
        os.kill(guard['pid'],signal.SIGCONT);paused=False
        ready(child)
    except BaseException as error:
        save(OUT/'failure.json',{'type':type(error).__name__,'time':time.time()})
        if child is not None and child.poll() is None:
            current=json.loads((OUT/'candidate.json').read_text());save(OUT/'stop-target.json',current);stop(current)
        if stopped:
            restored=spawn(old['command'],original_env,'restored-backend.log')
            save(SERVICE/'replica-1.json',{'pid':restored.pid,'command':old['command'],
                'started_at':time.time(),'restored_from':old['pid']})
            ready(restored)
        raise
    finally:
        if paused:os.kill(guard['pid'],signal.SIGCONT)
    spec=importlib.util.spec_from_file_location('candidate_exact_wire',HELPERS/'probe_psd_exact_wire_v3.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    probe.BACKEND_INDEX=1;probe.CONCURRENT=True;probe.OUT=OUT/'exact-wire'
    probe.prepare();save(OUT/'state.json',{'phase':'exact_wire_probe_running','psd_gate_passed':False})
    asyncio.run(probe.execute())
    save(OUT/'state.json',{'phase':'completed_requires_review','psd_gate_passed':False})


if __name__=='__main__':
    os.umask(0o077)
    if '--launch' in sys.argv:
        receipt=SERVICE/'single-tool-controller.json'
        with receipt.open('x') as f:json.dump({'phase':'launch_reserved'},f)
        command=[sys.executable,'-u',str(Path(__file__).resolve())]
        with (SERVICE/'single-tool-controller.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        save(receipt,{'pid':child.pid,'command':command,'time':time.time()})
        print(json.dumps({'pid':child.pid,'output':str(OUT)}))
    else:main()
