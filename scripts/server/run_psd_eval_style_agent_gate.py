"""Complete Agent gate using eval-style scheduling and deferred raw teacher.

Keep effective thinking budget and one-call contract, but remove the custom
raw-probability worker. No formal targets, no source resampling for training.
"""
import argparse
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from psd_eval_reference import reference_command, stop_if_present

ROOT=Path('/volume/ybo/wza')
CODE=ROOT/'training-artifacts/psd-capture-callsite-20260916-v21/code'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OUT=SERVICE/'eval-style-agent-v1'
RUN=ROOT/'runs/psd-eval-style-agent-gate-20260916-v1'
OLD=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916'
PORT=19024


def episodes(owner):
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    from ifv_training.psd_capture_semantics import install_capture_semantics, RAW_POLICY_LOGPROBS
    install_capture_semantics()
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.eval import run_cases
    from scripts.audit_real_trace import audit_trace, SourceAccessPolicy, _json_summary
    binding=owner.load(OLD/'binding.json')
    sys.argv=['eval-style-agent-gate','--benchmark',binding['benchmark'],
        '--source-access-policy',binding['source_access_policy'],'--profile','student-qwen3.5-local',
        '--output-dir',str(RUN/'episodes'),'--concurrency','2','--rollouts-per-case','1',
        '--base-sampling-seed','0','--timeout','3000']
    for case in binding['case_ids'][:2]:sys.argv+=['--case-id',case]
    run_cases.VerificationWorkflow=PSDWorkflow
    run_cases._git_commit=lambda:''
    owner.save(RUN/'state.json',{'phase':'complete_agent_running','formal_training':False})
    summary=asyncio.run(run_cases._run_cases(run_cases._parse_args()))
    paths=sorted((RUN/'episodes/traces').glob('*.json'));assert len(paths)==2
    audit=_json_summary([audit_trace(p,source_access_policy=SourceAccessPolicy.load(
        Path(binding['source_access_policy']))) for p in paths],strict_scheduler=True)
    owner.save(RUN/'strict-audit.json',audit)
    stats=[]
    for path in paths:
        trace=owner.load(path)
        report={'case':trace.get('case_id'),'error':bool(trace.get('error')),
            'termination':trace.get('termination'),'captures':0,'invalid_logprobs':0,'max_think_tokens':0}
        for step in trace['state']['all_steps']:
            meta=step.get('metadata',{})
            if step.get('stage') not in ('unified_react','unified_judgment') or meta.get('deterministic_segment_boundary'):continue
            if step.get('action_type') not in ('tool_call','output'):continue
            cap=meta.get('policy_token_capture',{})
            assert cap.get('status')=='complete' and cap.get('prompt_token_ids') and cap.get('completion_token_ids')
            assert cap.get('teacher_logprob_semantics') != RAW_POLICY_LOGPROBS,'Stock probabilities must remain pending for teacher scoring'
            ids=cap['completion_token_ids']
            assert len(ids)==len(cap['completion_logprobs'])
            report['invalid_logprobs']+=sum(not math.isfinite(x) for x in cap['completion_logprobs'])
            if 248069 in ids:report['max_think_tokens']=max(report['max_think_tokens'],ids.index(248069))
            report['captures']+=1
        stats.append(report)
    owner.save(RUN/'capture-audit.json',{'cases':stats,'teacher_top20':'pending_frozen_exact_token_scoring',
        'sampling_top20_not_training_targets':True,'summary':summary})
    passed=audit['passed'] and all(not s['error'] and s['termination']=='success' and s['captures']>=1
        and not s['invalid_logprobs'] and s['max_think_tokens']<=8192 for s in stats)
    owner.save(RUN/'state.json',{'phase':'complete_agent_passed_pending_teacher_scoring' if passed else 'requires_fix',
        'passed':passed,'cases':stats,'formal_training':False})
    if not passed:raise RuntimeError('Eval-style complete Agent gate failed; preserved original traces')


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('launch','execute','episodes'))
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if args.mode=='episodes':episodes(owner);return
    if args.mode=='launch':
        comparison=owner.load(SERVICE/'eval-reference-ab-v1/state.json')
        assert comparison['phase']=='reference_comparison_completed' and comparison['passed']==12
        OUT.mkdir(exist_ok=False);RUN.mkdir(exist_ok=False)
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'controller.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT),'run':str(RUN)}));return
    owner.verify_export()
    for rel,digest in owner.load(CODE.parent/'code-binding.json').items():assert owner.sha(CODE/rel)==digest
    old=owner.load(SERVICE/'replica-2.json');original_env=owner.checked(old)
    assert original_env['CUDA_VISIBLE_DEVICES']=='2'
    assert old['command'][old['command'].index('--worker-cls')+1]=='scripts.server.psd_raw_teacher_worker.RawTeacherWorker'
    guard=owner.load(SERVICE/'guard.json');owner.checked(guard)
    command=reference_command(old['command'])
    command[command.index('--tool-call-parser')+1]='ifv_psd_qwen3_single'
    command+=['--tool-parser-plugin',str(CODE/'scripts/server/psd_single_tool_parser.py'),
        '--logits-processors','scripts.server.psd_qwen_thinking:PSDThinkingBudget',
        '--mm-processor-cache-gb','0','--no-enable-prefix-caching','--mamba-cache-mode','none']
    with socket.socket() as s:s.bind(('127.0.0.1',PORT))
    owner.save(OUT/'binding.json',{'before':old,'candidate':command,'formal_training':False,
        'source_cases':owner.load(OLD/'binding.json')['case_ids'][:2],
        'teacher_probability_policy':'deferred exact-token frozen teacher; never stock grammar logprobs',
        'changes_from_eval_reference':['effective 8192 thinking closure','single-call grammar','disable faulty MM cache'],
        'code_binding_sha256':owner.sha(CODE.parent/'code-binding.json')})
    paused=False;stopped=False;backend=None;gateway=None
    try:
        os.kill(guard['pid'],signal.SIGSTOP);paused=True
        for port in (19019,19022):assert all(r['inflight']==0 for r in owner.http(f'http://127.0.0.1:{port}/health')['replicas'])
        for _ in range(90):
            metrics=urllib.request.urlopen('http://127.0.0.1:19004/metrics',timeout=5).read().decode()
            values=[float(s.rsplit(' ',1)[1]) for s in metrics.splitlines() if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(values)==2
            if not sum(values):break
            time.sleep(1)
        else:raise RuntimeError('Owned GPU2 did not drain')
        owner.stop(old);stopped=True
        cache=OUT/'cache'
        env={**original_env,'PYTHONPATH':str(CODE),'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(ROOT/'tmp'),
            'XDG_CACHE_HOME':str(cache),'TRITON_CACHE_DIR':str(cache/'triton'),'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'),
            'VLLM_CACHE_ROOT':str(cache/'vllm'),'FLASHINFER_WORKSPACE_BASE':str(cache/'flashinfer'),
            'TORCH_EXTENSIONS_DIR':str(cache/'extensions')}
        for key in list(env):
            if key.startswith('PSD_'):env.pop(key)
        backend=owner.spawn(command,env,OUT/'backend.log');owner.save(SERVICE/'replica-2.json',backend)
        owner.save(OUT/'replica.json',backend)
        os.kill(guard['pid'],signal.SIGCONT);paused=False
        owner.save(OUT/'state.json',{'phase':'loading_eval_style_candidate','formal_training':False})
        for _ in range(240):
            owner.checked(backend)
            try:
                if owner.http('http://127.0.0.1:19004/v1/models')['data'][0]['root']==command[3]:break
            except OSError:pass
            time.sleep(5)
        else:raise RuntimeError('Eval-style candidate readiness deadline')
        prior=owner.load(SERVICE/'four-gpu-v7/gateway.json');gateway_env=owner.checked(prior)
        gateway_command=prior['command'][:]
        gateway_command[gateway_command.index('--app-dir')+1]=str(CODE)
        gateway_command[gateway_command.index('--port')+1]=str(PORT)
        for key in list(gateway_env):
            if key.startswith(('PSD_RAW_','PSD_POLICY_LOGPROB','PSD_WIRE_','PSD_DIAGNOSTIC_')):gateway_env.pop(key)
        gateway_env.update(PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',QWEN_REPLICA_BACKENDS='http://127.0.0.1:19004')
        gateway=owner.spawn(gateway_command,gateway_env,OUT/'gateway.log');owner.save(OUT/'gateway.json',gateway)
        for _ in range(45):
            owner.checked(gateway)
            try:
                if owner.http(f'http://127.0.0.1:{PORT}/health')['replicas'][0]['healthy']:break
            except OSError:pass
            time.sleep(1)
        else:raise RuntimeError('Eval-style gateway readiness deadline')
        h=owner.helper();h.CODE=CODE;h.GATEWAY=f'http://127.0.0.1:{PORT}'
        run_env,checks=h.environment()
        run_env.update(IFV_CAPTURE_POLICY_TOKENS='1',IFV_POLICY_TOPK='20',PYTHONDONTWRITEBYTECODE='1')
        owner.save(RUN/'credential-presence.json',checks)
        owner.save(OUT/'state.json',{'phase':'complete_agent_validation','formal_training':False})
        with (RUN/'run.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'episodes'],env=run_env,
                cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            owner.save(RUN/'process.json',{'pid':child.pid,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'episodes']})
            code=child.wait()
        owner.save(OUT/'state.json',{'phase':'agent_validation_completed' if code==0 else 'agent_validation_failed',
            'returncode':code,'formal_training':False})
    except BaseException as error:
        owner.save(OUT/'state.json',{'phase':'requires_fix','error_type':type(error).__name__,'error':str(error),'formal_training':False})
        raise
    finally:
        try:
            if gateway is not None:stop_if_present(owner,gateway)
            if stopped:
                if not paused:os.kill(guard['pid'],signal.SIGSTOP);paused=True
                if backend is not None:stop_if_present(owner,backend)
                restored=owner.spawn(old['command'],original_env,OUT/'restored-backend.log')
                owner.save(SERVICE/'replica-2.json',restored)
        finally:
            if paused:os.kill(guard['pid'],signal.SIGCONT)
            owner.verify_export()


if __name__=='__main__':main()
