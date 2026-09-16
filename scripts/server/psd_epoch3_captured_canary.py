"""New immutable 4x8 run with native exact token/logprob capture enabled.

Reuse the verified epoch3 GPU; never silently relabel the uncaptured diagnostic.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT=Path('/volume/ybo/wza')
V1=ROOT/'training-artifacts/psd-epoch3-20260916-v1'
DEPLOY=ROOT/'training-artifacts/psd-epoch3-capture-20260916-v2'
CODE=DEPLOY/'code'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
CONTROL=SERVICE/'captured-v2'
RUN=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916'
PORT=19018


def owner():
    spec=importlib.util.spec_from_file_location('epoch3_owner_v2',V1/'psd_epoch3_canary.py')
    o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
    o.CODE=CODE;o.RUN=RUN;o.PORT=PORT
    # Runtime/serving plugin code stays v1; copied collector code gets a new
    # hash-bound identity. Do not change the old service or model receipts.
    original=o.preflight
    def preflight():
        h,binding=original()
        binding['native_policy_capture']={'enabled':True,'topk':20}
        binding['capture_gateway_code_sha256']={str(p.relative_to(DEPLOY)):o.sha(p)
            for p in (DEPLOY/'scripts/server').glob('*.py')}
        binding['capture_launcher_sha256']=o.sha(Path(__file__))
        return h,binding
    o.preflight=preflight
    return o


def capture_environment(o):
    env,checks=o.helper().environment()
    env.update(IFV_CAPTURE_POLICY_TOKENS='1',IFV_POLICY_TOPK='20')
    return env,checks


def launch():
    o=owner()
    assert not CONTROL.exists() and not RUN.exists() and not CODE.exists()
    assert (ROOT/'runs/psd-sft3084-canary4x8-20260916/capture-preflight-hold.json').exists()
    prior=o.load(ROOT/'runs/psd-sft3084-canary4x8-20260916/process.json')
    assert not Path(f'/proc/{prior["pid"]}/cmdline').exists() or not Path(f'/proc/{prior["pid"]}/cmdline').read_bytes()
    assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19017/health')['replicas'])
    with socket.socket() as s:s.bind(('127.0.0.1',PORT))
    o.verify_export()
    shutil.copytree(V1/'code',CODE,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for source,destination in (
        ('psd_collection.py','training/ifv_training/psd_collection.py'),
        ('collect_psd_rollouts.py','scripts/collect_psd_rollouts.py'),
        ('run_psd_round.py','scripts/run_psd_round.py')):
        shutil.copy2(DEPLOY/source,CODE/destination)
    old=o.load(SERVICE/'gateway.json');env=o.checked(old)
    command=old['command'][:]
    command[command.index('--app-dir')+1]=str(DEPLOY)
    command[command.index('--port')+1]=str(PORT)
    env.update(PYTHONPATH=str(DEPLOY),PSD_WIRE_CAPTURE_DIR=str(CONTROL/'wire'),
        PSD_WIRE_MAX_RESPONSE_BYTES=str(64*1024**2))
    CONTROL.mkdir()
    receipt=o.spawn(command,env,CONTROL/'gateway.log');o.save(CONTROL/'gateway.json',receipt)
    for _ in range(30):
        o.checked(receipt)
        try:
            if o.http(f'http://127.0.0.1:{PORT}/health')['status']=='ok':break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Native-capture gateway readiness deadline')
    o.grouped().prepare()
    env,checks=capture_environment(o);o.save(RUN/'credential-presence.json',checks)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--execute']
    receipt=o.spawn(command,env,RUN/'run.log');receipt['script_sha256']=o.sha(__file__)
    o.save(RUN/'process.json',receipt)
    o.save(SERVICE/'state.json',{'phase':'native_capture_canary32_running','run':str(RUN),
        'gateway':f'http://127.0.0.1:{PORT}','training_started':False,'time':time.time()})
    print(json.dumps({'run':str(RUN),'pid':receipt['pid'],'slots':32,'training_started':False}))


def execute():
    o=owner()
    try:
        assert os.environ.get('IFV_CAPTURE_POLICY_TOKENS')=='1' and os.environ.get('IFV_POLICY_TOPK')=='20'
        o.collect()
    except BaseException as error:
        o.save(CONTROL/'failure.json',{'type':type(error).__name__,'time':time.time(),'training_started':False})
        raise


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch',action='store_true');modes.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    launch() if args.launch else execute()
