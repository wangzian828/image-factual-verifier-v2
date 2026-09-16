"""New epoch-3 PSD identity: preserve weights, switch GPU1, collect 4x8.

This controller does not start production optimization or choose a larger pool.
All source images, checkpoints and previous diagnostics remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
OLD=ROOT/'inference/psd-sft2056-safety-20260916'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
DEPLOY=ROOT/'training-artifacts/psd-epoch3-20260916-v1'
CODE=DEPLOY/'code'
OLD_CODE=ROOT/'training-artifacts/psd-adjustments-20260915/code'
PREP=ROOT/'runs/psd-pilot400-preparation-20260915-v1'
RUN=ROOT/'runs/psd-sft3084-canary4x8-20260916'
EXPORT=ROOT/'exports/h20-sft-merged4872-3epoch-step3084-20260915/export.json'
EXPORT_SHA='1c342e73e6fc82bfa573e4435c67c2c38f26207030307313e7a1396a09a1850c'
ALIAS='ifv-qwen3.5-9b-sft-3084'
BACKEND='ifv-psd-sft3084'
PORT=19017


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def http(url):
    with urllib.request.urlopen(url,timeout=5) as response:return json.load(response)


def checked(receipt):
    pid=receipt['pid']
    args=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    assert args==receipt['command'] and os.getpgid(pid)==pid,'Process identity changed'
    return dict(x.decode().split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if x)


def stop(receipt):
    checked(receipt);os.killpg(receipt['pid'],signal.SIGTERM)
    for _ in range(120):
        p=Path(f'/proc/{receipt["pid"]}/stat')
        if not p.exists() or p.read_text().split(') ',1)[1][0]=='Z':return
        time.sleep(.5)
    raise RuntimeError('Owned process did not stop gracefully')


def spawn(command,env,log):
    with Path(log).open('xb') as f:
        p=subprocess.Popen(command,env=env,cwd=DEPLOY,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return {'pid':p.pid,'command':command,'time':time.time()}


def model_command(original):
    command=list(original)
    assert command[2]=='serve' and '--no-enable-prefix-caching' in command
    assert command[command.index('--mamba-cache-mode')+1]=='none'
    assert command[command.index('--tool-call-parser')+1]=='ifv_psd_qwen3_single'
    command[3]=str(EXPORT.parent/'model')
    command[command.index('--served-model-name')+1]=BACKEND
    command[command.index('--tool-parser-plugin')+1]=str(DEPLOY/'scripts/server/psd_single_tool_parser.py')
    return command


def verify_export(full=False):
    assert sha(EXPORT)==EXPORT_SHA
    exported=load(EXPORT)
    assert exported['passed'] and exported['global_step']==3084 and exported['epoch']==3
    assert exported['source_full_state_preserved'] and exported['optimizer_scheduler_rng_available']
    source=Path(exported['source_checkpoint']).resolve();source.relative_to(ROOT/'checkpoints')
    assert source.is_dir()
    for name,expected in exported['source_metadata_sha256'].items():
        assert sha(source/name)==expected,name
    if full:
        for name,record in exported['model_artifacts'].items():
            assert sha(EXPORT.parent/'model'/name)==record['sha256'],name
    return exported


def setup():
    assert not SERVICE.exists() and not RUN.exists() and not CODE.exists(),'Never overwrite a prior epoch3 run'
    for old_run in ('psd-single-tool4x2-20260916','psd-wire-diagnostic4x2-20260916'):
        state=load(ROOT/'runs'/old_run/'state.json')
        assert state['phase'].startswith('completed')
    assert load(ROOT/'runs/psd-single-tool4x2-20260916/runtime-acceptance.json')['runtime_gate_passed']
    exported=verify_export(full=True)
    with socket.socket() as s:s.bind(('127.0.0.1',PORT))
    old_backend=load(OLD/'replica-1.json');old_env=checked(old_backend)
    guard=load(OLD/'guard.json');guard_env=checked(guard)
    gateway=load(OLD/'wire-single-tool-v1/gateway-process.json');gateway_env=checked(gateway)
    assert all(r['inflight']==0 for r in http('http://127.0.0.1:19016/health')['replicas'])
    assert old_env['CUDA_VISIBLE_DEVICES']=='1'
    SERVICE.mkdir()
    save(SERVICE/'protected-sft.json',{'export':str(EXPORT),'export_sha256':EXPORT_SHA,
        'model_artifacts':exported['model_artifacts'],'source_checkpoint':exported['source_checkpoint'],
        'source_metadata_sha256':exported['source_metadata_sha256'],
        'policy':'read-only initialization; PSD outputs must not overlap either source directory',
        'full_weight_hashes_verified':True,'time':time.time()})
    shutil.copytree(OLD_CODE,CODE,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copy2(DEPLOY/'psd_collection.py',CODE/'training/ifv_training/psd_collection.py')
    save(SERVICE/'backend-before.json',old_backend);save(SERVICE/'guard-before.json',guard)
    backend=None;new_guard=None;paused=False;stopped=False;guard_stopped=False
    try:
        os.kill(guard['pid'],signal.SIGSTOP);paused=True
        for _ in range(90):
            with urllib.request.urlopen('http://127.0.0.1:19003/metrics',timeout=5) as f:metrics=f.read().decode()
            values=[float(s.rsplit(' ',1)[1]) for s in metrics.splitlines()
                if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(values)==2
            if sum(values)==0:break
            time.sleep(1)
        else:raise RuntimeError('GPU1 not drained')
        stop(old_backend);stopped=True
        env={**old_env,'PYTHONPATH':str(DEPLOY)}
        backend=spawn(model_command(old_backend['command']),env,SERVICE/'backend.log')
        save(SERVICE/'replica-1.json',backend)
        # Replace the old guard with an explicitly bound per-GPU model map.
        os.kill(guard['pid'],signal.SIGCONT);paused=False;stop(guard);guard_stopped=True
        command=guard['command'][:];command[1]=str(DEPLOY/'gpu_utilization_guard.py')
        command[command.index('--state-file')+1]=str(SERVICE/'idle-guard/state.json')
        command[command.index('--log-file')+1]=str(SERVICE/'idle-guard/samples.jsonl')
        command+=['--models',','.join(['ifv-psd-sft2056-safety',BACKEND,
                                       'ifv-psd-sft2056-safety','ifv-psd-sft2056-safety'])]
        new_guard=spawn(command,guard_env,SERVICE/'guard.log');save(SERVICE/'guard.json',new_guard)
        for _ in range(180):
            checked(backend)
            try:
                card=http('http://127.0.0.1:19003/v1/models')['data'][0]
                if card['id']==BACKEND and card['root']==str(EXPORT.parent/'model'):
                    assert card['max_model_len']==131072
                    break
            except OSError:pass
            time.sleep(5)
        else:raise RuntimeError('Epoch3 model readiness deadline')
    except BaseException:
        if new_guard is not None:stop(new_guard)
        if backend is not None:stop(backend)
        if stopped:
            restored=spawn(old_backend['command'],old_env,SERVICE/'rollback-backend.log')
            save(OLD/'replica-1.json',restored)
        if guard_stopped:
            restored=spawn(guard['command'],guard_env,SERVICE/'rollback-guard.log');save(OLD/'guard.json',restored)
        raise
    finally:
        if paused:os.kill(guard['pid'],signal.SIGCONT)
    command=gateway['command'][:]
    command[command.index('--app-dir')+1]=str(DEPLOY)
    command[command.index('--port')+1]=str(PORT)
    env={**gateway_env,'PYTHONPATH':str(DEPLOY),'QWEN_REPLICA_MODEL_ID':BACKEND,
        'PSD_PUBLIC_MODEL_ALIAS':ALIAS,'QWEN_REPLICA_BACKENDS':'http://127.0.0.1:19003',
        'PSD_WIRE_CAPTURE_DIR':str(SERVICE/'wire'),'PSD_DIAGNOSTIC_RETURN_TOKEN_IDS':'1'}
    receipt=spawn(command,env,SERVICE/'gateway.log');save(SERVICE/'gateway.json',receipt)
    for _ in range(30):
        checked(receipt)
        try:
            if http(f'http://127.0.0.1:{PORT}/health')['status']=='ok':break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Epoch3 gateway readiness deadline')
    save(SERVICE/'state.json',{'phase':'epoch3_serving_ready','training_started':False,'time':time.time()})


def helper():
    h=module('epoch3_runtime_environment',DEPLOY/'scripts/server/run_psd_runtime_gate.py')
    h.CODE=CODE;h.GATEWAY=f'http://127.0.0.1:{PORT}';h.POLICY_NAME=ALIAS
    h.DISABLE_PERCEPTION_CACHE=True
    return h


def preflight():
    verify_export()
    backend=load(SERVICE/'replica-1.json');checked(backend)
    assert backend['command']==model_command(load(SERVICE/'backend-before.json')['command'])
    health=http(f'http://127.0.0.1:{PORT}/health')
    assert [(r['url'],r['healthy']) for r in health['replicas']]==[('http://127.0.0.1:19003',True)]
    assert health['timeout_contract']=={'gateway':1200,'model_client':1230,'stage':1260}
    assert health['wire_capture']['budget_accounting']=='compressed_completed_plus_bounded_inflight'
    cards=http(f'http://127.0.0.1:{PORT}/v1/models')['data']
    assert {r['id'] for r in cards}=={ALIAS,BACKEND}
    assert all(r['root']==str(EXPORT.parent/'model') and r['max_model_len']==131072 for r in cards)
    hashes={}
    for p in sorted((CODE/'src').rglob('*.py')):
        relative=p.relative_to(CODE)
        assert p.read_text()==(ROOT/'image-factual-verifier-v2'/relative).read_text(),str(relative)
        hashes[str(relative)]=sha(p)
    payload=load(PREP/'prepared.json')['payload']
    binding={'source_sha256':hashes,'source_normalized_equal_to_frozen':True,
        'export_sha256':EXPORT_SHA,'prepared_sha256':sha(PREP/'prepared.json'),
        'policy_model_name':ALIAS,'pinned_backend_alias':BACKEND,'gateway':f'http://127.0.0.1:{PORT}',
        'perception_cache_disabled':True,'temperature':.7,'think_budget':8192,'output_budget':32768,
        'base_sampling_seed':0,'server_git_invoked':False,
        'code_sha256':{str(p.relative_to(CODE)):sha(p) for parent in ('scripts','training')
                       for p in (CODE/parent).rglob('*.py')},
        'deployment_sha256':{str(p.relative_to(DEPLOY)):sha(p) for p in (DEPLOY/'scripts/server').glob('*.py')},
        'benchmark':payload['benchmark'],'source_access_policy':payload['source_access_policy']}
    return helper(),binding


def grouped():
    g=module('epoch3_grouped',DEPLOY/'scripts/server/run_psd_grouped_canary.py')
    g.CODE=CODE;g.RUN=RUN;g.EXPORT=EXPORT;g.EXPORT_SHA=EXPORT_SHA;g.SERVICE=SERVICE
    g.CACHE_FREE=True;g.GATEWAY_PORT=PORT;g.TOOL_PARSER='ifv_psd_qwen3_single';g.ALIAS=ALIAS
    g.preflight=preflight
    return g


def collect():
    h,current=preflight();bound=load(RUN/'binding.json')
    assert all(current[k]==bound[k] for k in current if k not in ('benchmark','source_access_policy'))
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.eval import run_cases
    original=sys.argv[:]
    sys.argv=['check','--benchmark',bound['benchmark'],'--profile','student-qwen3.5-local',
              '--output-dir',str(RUN/'episodes')]
    parsed=run_cases._parse_args();workflow=PSDWorkflow(run_cases._workflow_config(parsed))
    o=workflow._get_orchestrator(validate_startup=False)
    assert not o.tool_cache.enabled and not o.cacheable_tools
    configs={s:o._stage_generation_config(s) for s in ('UNIFIED_REACT','UNIFIED_JUDGMENT')}
    assert all(c['temperature']==.7 and c['thinking_token_budget']==8192 for c in configs.values())
    save(RUN/'effective-stage-config.json',configs);sys.argv=original
    grouped().execute()
    verify_export()


def execute():
    setup();grouped().prepare()
    env,checks=helper().environment();save(RUN/'credential-presence.json',checks)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--collect']
    child=spawn(command,env,RUN/'run.log');save(RUN/'process.json',child)
    save(SERVICE/'state.json',{'phase':'epoch3_canary32_running','run':str(RUN),
        'training_started':False,'full_collection_started':False,'time':time.time()})
    print(json.dumps({'run':str(RUN),'slots':32,'pid':child['pid'],'training_started':False}),flush=True)


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for flag in ('launch','execute','collect'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.launch:
        receipt=DEPLOY/'controller.json'
        with receipt.open('x') as f:json.dump({'reserved':True},f)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--execute']
        child=spawn(command,os.environ.copy(),DEPLOY/'controller.log');child['script_sha256']=sha(__file__)
        save(receipt,child);print(json.dumps(child))
    else:
        try:collect() if args.collect else execute()
        except BaseException as e:
            save(DEPLOY/('collection-failure.json' if args.collect else 'failure.json'),
                 {'type':type(e).__name__,'time':time.time(),'training_started':False})
            raise


if __name__=='__main__':main()
