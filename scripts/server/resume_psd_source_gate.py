"""Resume the mixed-checker boundary fix only after the active prepare pass ends."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/volume/ybo/wza')
OLD=ROOT/'training-artifacts/psd-checker-feedback-20260917-v58'
DEPLOY=ROOT/'training-artifacts/psd-source-gate-20260917-v59'
CODE=DEPLOY/'code'
PREVIOUS=ROOT/'runs/psd-formal-prepare-controller-20260917-checker-feedback-v1'
CONTROL=ROOT/'runs/psd-formal-prepare-controller-20260917-checker-feedback-v2'
ROUND=ROOT/'runs/psd-production-round1-20260917-v1'
SEARCH=ROUND/'search-gemini37-flash-high'
OVERLAYS={'psd_slate_feedback.py':'training/ifv_training',
    'psd_slate_search.py':'training/ifv_training','run_psd_repair_driver.py':'scripts',
    'test_psd_slate_feedback.py':'training/tests','resume_psd_source_gate.py':'scripts/server'}


def load(path): return json.loads(path.read_text())


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def base():
    path=CODE/'scripts/server/deploy_psd_checker_feedback.py'
    spec=importlib.util.spec_from_file_location('psd_source_gate_base',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    result.CODE,result.DEPLOY,result.CONTROL=CODE,DEPLOY,CONTROL
    return result


def stage():
    if CODE.exists():raise RuntimeError('immutable snapshot already exists')
    shutil.copytree(OLD/'code',CODE,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name,parent in OVERLAYS.items():shutil.copy2(DEPLOY/name,CODE/parent/name)
    tests=['test_psd_slate_feedback.py','test_psd_slate_pipeline.py','test_psd_slate.py',
        'test_psd_slate_bound_positions.py','test_psd_candidate_binding.py',
        'test_psd_repair_storage.py','test_psd_model_route.py']
    with (DEPLOY/'tests.log').open('x') as log:
        result=subprocess.call([sys.executable,'-m','pytest','-q',
            *[str(CODE/'training/tests'/x) for x in tests]],cwd=CODE,env=base().environment(),stdout=log,stderr=log)
    if result:raise RuntimeError('source gate regression failed')
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    from ifv_training.psd_source_review import source_review_reference
    from ifv_training.psd_repair_verifier import verify_source_rollout_failure
    from ifv_training.psd_slate_feedback import checker_feedback
    real=[]
    for key in ('c98647673c0507d7','b175d5b9d098572d','9c07b3fa1ab53e20'):
        config=load(SEARCH/'repairs'/key/'run-inputs.json')['identity']
        trace=load(Path(config['trace']['path']));gold=load(Path(config['gold']['path']))
        candidate=load(Path(config['candidate']['path']));audit=load(Path(config['audit']['path']))
        review=source_review_reference(candidate['source'])
        proof=verify_source_rollout_failure(trace,gold=gold,source_task_review=review,
            source_audit=audit if audit.get('source_trace_canonical_sha256') else None)
        if not proof['passed']:raise RuntimeError('real source was not independently failed')
        feedback=checker_feedback(review,trace,source_failure=proof)
        if feedback['status']!='fail' or not feedback['independent_task_check_failed']:
            raise RuntimeError('mixed checker source still vetoed')
        real.append({'key':key,'semantic_status':review['decision']['status'],
            'independent_failure':True,'public_citations':len(feedback['cited_observations'])})
    save(DEPLOY/'stage-state.json',{'deployment_ready_not_live':True,'tests_returncode':0,
        'real_source_gate_checks':real,'provider_calls':0,'time':time.time()})
    print(json.dumps({'tests_returncode':0,'real_source_checks':len(real)}),flush=True)


def start():
    if not load(DEPLOY/'stage-state.json')['deployment_ready_not_live']:raise RuntimeError('unvalidated stage')
    if (DEPLOY/'handoff-process.json').exists():raise RuntimeError('handoff already started')
    b=base()
    receipt=b.owner().spawn([sys.executable,'-u',str(CODE/'scripts/server/resume_psd_source_gate.py'),
        'wait-boundary'],b.environment(),DEPLOY/'handoff.log')
    save(DEPLOY/'handoff-process.json',receipt)
    print(json.dumps({'handoff_pid':receipt['pid']}),flush=True)


def alive(pid):
    f=Path('/proc',str(pid),'stat')
    return f.exists() and f.read_text().split(') ',1)[1][0]!='Z'


def wait_boundary():
    previous=load(PREVIOUS/'process.json');b=base();owner=b.owner()
    expected=[sys.executable,'-u',str(OLD/'code/scripts/server/deploy_psd_checker_feedback.py'),'worker']
    if previous.get('command')!=expected:raise RuntimeError('unexpected previous owner')
    while True:
        state=load(PREVIOUS/'state.json')
        if state.get('phase') in {'requires_frozen_teacher_topk','ready_for_training'}:
            save(DEPLOY/'handoff-state.json',{'phase':'no_handoff_needed','previous_state':state});return
        if state.get('phase')=='formal_prepare_result':break
        if not alive(previous['pid']):raise RuntimeError('prepare died outside a verified boundary')
        save(DEPLOY/'handoff-state.json',{'phase':'waiting_current_pass_no_interruption','time':time.time()})
        time.sleep(5)
    frozen=False
    try:
        if alive(previous['pid']):
            owner.checked(previous);os.kill(previous['pid'],signal.SIGSTOP);frozen=True
        # Recheck after suspension. The in-flight child must already be gone.
        current=load(PREVIOUS/'state.json')
        if current!=state:raise RuntimeError('boundary changed before suspension')
        group=[]
        for entry in Path('/proc').glob('[0-9]*'):
            try:
                pid=int(entry.name)
                if os.getpgid(pid)!=previous['pid'] or not alive(pid):continue
                cmd=[x.decode() for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            tracker=(len(cmd)==3 and cmd[:2]==[sys.executable,'-c'] and
                re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\(\d+\)',cmd[2]))
            if cmd!=expected and not tracker:raise RuntimeError('active child remains; refusing handoff')
            group.append(pid)
        if CONTROL.exists():raise RuntimeError('new controller already exists')
        if group:
            os.killpg(previous['pid'],signal.SIGTERM)
            if frozen:os.kill(previous['pid'],signal.SIGCONT);frozen=False
            for _ in range(40):
                if not any(alive(pid) for pid in group):break
                time.sleep(.25)
            else:raise RuntimeError('old owner still alive')
        CONTROL.mkdir(exist_ok=False)
        receipt=owner.spawn([sys.executable,'-u',str(CODE/'scripts/server/resume_psd_source_gate.py'),
            'worker'],b.environment(),CONTROL/'controller.log')
        save(CONTROL/'process.json',receipt)
        save(DEPLOY/'handoff-state.json',{'phase':'switched_at_completed_pass','previous_state':state,
            'new_controller':str(CONTROL),'new_pid':receipt['pid'],'interrupted_rollouts':0,'time':time.time()})
        save(PREVIOUS/'source-gate-handoff.json',{'new_controller':str(CONTROL),'time':time.time()})
    finally:
        if frozen and alive(previous['pid']):os.kill(previous['pid'],signal.SIGCONT)


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','start','wait-boundary','worker'])
    mode=parser.parse_args().mode
    if mode=='stage':stage()
    elif mode=='start':start()
    elif mode=='wait-boundary':wait_boundary()
    else:base().launcher().worker()
