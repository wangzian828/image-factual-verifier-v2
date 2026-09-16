"""Launch an isolated capture gateway and 4 fixed train cases x 2 diagnostic slots."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
CODE=ROOT/'training-artifacts/psd-adjustments-20260915/code'
HELPERS=ROOT/'training-artifacts/psd-serving-safety-20260916'
SERVICE=ROOT/'inference/psd-sft2056-safety-20260916'
DEPLOY=HELPERS/'gateway-wire-v1'
CONTROL=SERVICE/'wire-diagnostic-v1'
RUN=ROOT/'runs/psd-wire-diagnostic4x2-20260916'
PORT=19013
BACKEND_INDEX=2
SINGLE_TOOL=False


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def helper():
    spec=importlib.util.spec_from_file_location('wire_gate_helper',HELPERS/'run_psd_runtime_gate_v4.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.RUN=RUN;h.GATEWAY=f'http://127.0.0.1:{PORT}';h.DISABLE_PERCEPTION_CACHE=True
    return h


def binding(h):
    value=h.preflight()
    canary=[json.loads(line) for line in (h.PREP/'training-canary-case-list.jsonl').open()]
    value.update(case_ids=[row['case_id'] for row in canary[:4]],concurrency=8,rollouts_per_case=2,
                 purpose='4x2 fixed real-Agent diagnostics with full wire capture; not PSD targets',
                 gateway_code_sha256={p.name:sha(p) for p in (DEPLOY/'scripts/server').glob('*.py')})
    with urllib.request.urlopen(h.GATEWAY+'/health',timeout=5) as response:health=json.load(response)
    assert [x['url'] for x in health['replicas']]==[f'http://127.0.0.1:{19002+BACKEND_INDEX}']
    assert health['wire_capture']['enabled']
    value['single_tool_constraint']=SINGLE_TOOL
    value['diagnostic_token_ids']=health.get('diagnostic_token_ids',False)
    assert value['diagnostic_token_ids']==SINGLE_TOOL
    return value


def execute():
    h=helper();frozen=json.loads((RUN/'binding.json').read_text());assert binding(h)==frozen
    assert os.environ.get('TOOL_CACHE_ENABLED')==os.environ.get('PERCEPTION_CACHE_ENABLED')=='0'
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    from src.eval import run_cases
    from scripts.collect_psd_rollouts import PSDWorkflow
    sys.argv=['psd-wire-diagnostic','--benchmark',frozen['benchmark'],
        '--source-access-policy',frozen['source_access_policy'],'--profile','student-qwen3.5-local',
        '--output-dir',str(RUN/'episodes'),'--concurrency','8','--rollouts-per-case','2',
        '--base-sampling-seed','0','--timeout','3000']
    for case in frozen['case_ids']:sys.argv+=['--case-id',case]
    parsed=run_cases._parse_args();run_cases.VerificationWorkflow=PSDWorkflow
    run_cases._git_commit=lambda:''
    workflow=PSDWorkflow(run_cases._workflow_config(parsed));o=workflow._get_orchestrator(validate_startup=False)
    assert not o.tool_cache.enabled and not o.cacheable_tools
    configs={stage:o._stage_generation_config(stage) for stage in ['UNIFIED_REACT','UNIFIED_JUDGMENT']}
    for config in configs.values():assert config['temperature']==.7 and config['thinking_token_budget']==8192
    save(RUN/'effective-stage-config.json',configs)
    save(RUN/'state.json',{'phase':'running','slots':8,'training_started':False,'psd_gate_passed':False,'time':time.time()})
    result=asyncio.run(run_cases._run_cases(parsed))
    save(RUN/'state.json',{'phase':'completed_requires_raw_review','summary':result,
        'training_started':False,'psd_gate_passed':False,'time':time.time()})


def launch():
    assert not CONTROL.exists() and not RUN.exists(),'Preserve prior diagnostic; never relaunch over it'
    backend=json.loads((SERVICE/f'replica-{BACKEND_INDEX}.json').read_text())
    actual=[x.decode() for x in Path(f'/proc/{backend["pid"]}/cmdline').read_bytes().split(b'\0') if x]
    assert actual==backend['command'] and '--no-enable-prefix-caching' in actual
    assert actual[actual.index('--mamba-cache-mode')+1]=='none'
    if SINGLE_TOOL:
        assert actual[actual.index('--tool-call-parser')+1]=='ifv_psd_qwen3_single'
        probe=json.loads((SERVICE/'single-tool-backend-v1/exact-wire/summary.json').read_text())
        assert len(probe['results'])==8 and all(r.get('finish_reason')=='tool_calls' and r.get('tool_calls')==1 for r in probe['results'])
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/model/tokenizer.json'))
        end_think=tokenizer.token_to_id('</think>')
        for result in probe['results']:
            if '-required-' not in result['label']:continue
            response=json.loads((SERVICE/'single-tool-backend-v1/exact-wire'/(result['label']+'-response.json')).read_text())
            tokens=response['choices'][0]['token_ids'];offset=tokens.index(end_think)+1
            calls=json.loads(tokenizer.decode(tokens[offset:],skip_special_tokens=True).strip())
            assert isinstance(calls,list) and len(calls)==1,'Check raw generation, not post-filtered tool_calls'
    old=json.loads((SERVICE/'gateway-no-apc.json').read_text())
    actual=[x.decode() for x in Path(f'/proc/{old["pid"]}/cmdline').read_bytes().split(b'\0') if x]
    assert actual==old['command']
    with socket.socket() as s:s.bind(('127.0.0.1',PORT))
    env=dict(x.decode().split('=',1) for x in Path(f'/proc/{old["pid"]}/environ').read_bytes().split(b'\0') if x)
    env.update(QWEN_REPLICA_BACKENDS=f'http://127.0.0.1:{19002+BACKEND_INDEX}',PSD_WIRE_CAPTURE_DIR=str(CONTROL/'wire'),PYTHONPATH=str(DEPLOY))
    if SINGLE_TOOL:env['PSD_DIAGNOSTIC_RETURN_TOKEN_IDS']='1'
    command=actual[:];command[command.index('--app-dir')+1]=str(DEPLOY);command[command.index('--port')+1]=str(PORT)
    CONTROL.mkdir();RUN.mkdir()
    with (CONTROL/'gateway.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=DEPLOY,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(CONTROL/'gateway-process.json',{'pid':child.pid,'command':command,'time':time.time()})
    h=helper()
    for _ in range(25):
        if child.poll() is not None:raise RuntimeError('Capture gateway exited')
        try:
            with urllib.request.urlopen(h.GATEWAY+'/health',timeout=1) as response:
                if json.load(response)['status']=='ok':break
        except OSError:time.sleep(1)
    else:raise RuntimeError('Gateway readiness timeout')
    save(RUN/'binding.json',binding(h))
    env,checks=h.environment();save(RUN/'credential-presence.json',checks)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--execute']
    if SINGLE_TOOL:command.append('--single-tool')
    with (RUN/'run.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(RUN/'process.json',{'pid':child.pid,'command':command,'script_sha256':sha(Path(__file__)),'time':time.time()})
    print(json.dumps({'diagnostic_pid':child.pid,'run':str(RUN),'capture':str(CONTROL/'wire'),'slots':8}),flush=True)


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--single-tool',action='store_true')
    args=parser.parse_args()
    if args.single_tool:
        SINGLE_TOOL=True;BACKEND_INDEX=1;PORT=19016
        DEPLOY=HELPERS/'gateway-wire-single-tool-v1'
        CONTROL=SERVICE/'wire-single-tool-v1'
        RUN=ROOT/'runs/psd-single-tool4x2-20260916'
    if args.execute:
        try:execute()
        except BaseException as error:
            save(RUN/'state.json',{'phase':'failed_requires_inspection','error_type':type(error).__name__,
                                  'time':time.time(),'psd_gate_passed':False})
            raise
    elif args.launch:launch()
    else:parser.error('Choose --launch or --execute')
