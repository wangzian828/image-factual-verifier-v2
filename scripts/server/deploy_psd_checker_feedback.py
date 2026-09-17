"""Bounded real canary and incremental PSD controller handoff; no service restart."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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

ROOT = Path('/volume/ybo/wza')
OLD = ROOT / 'training-artifacts/psd-flash36-incremental-20260917-v57'
DEPLOY = ROOT / 'training-artifacts/psd-checker-feedback-20260917-v58'
CODE = DEPLOY / 'code'
ROUND = ROOT / 'runs/psd-production-round1-20260917-v1'
SEARCH = ROUND / 'search-gemini37-flash-high'
CONTROL = ROOT / 'runs/psd-formal-prepare-controller-20260917-checker-feedback-v1'
PREVIOUS = ROOT / 'runs/psd-formal-prepare-controller-20260917-flash36-high-v1'
CANARY = SEARCH / 'checker-feedback-validation-v58'
OVERLAYS = {
    'psd_slate_feedback.py': 'training/ifv_training',
    'psd_slate_search.py': 'training/ifv_training',
    'psd_slate.py': 'training/ifv_training',
    'psd_repair_runtime.py': 'training/ifv_training',
    'run_psd_repair_driver.py': 'scripts',
    'test_psd_slate_feedback.py': 'training/tests',
    'test_psd_slate_pipeline.py': 'training/tests',
    'test_psd_slate.py': 'training/tests',
    'deploy_psd_checker_feedback.py': 'scripts/server',
}


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def launcher():
    result = module('checker_feedback_launcher', CODE / 'training/scripts/h20/run_psd_lightweight_resume.py')
    result.CODE, result.DEPLOY, result.CONTROL = CODE, DEPLOY, CONTROL
    result.ROUTE = OLD / 'external-model-route.json'
    result.INTERRUPTION_RECEIPT = DEPLOY / 'controlled-interruptions.json'
    return result


def owner():
    return module('checker_feedback_owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')


def environment():
    launch = launcher()
    env = owner().checked(load(launch.SERVICE / 'gateway.json'))
    external, _ = launch.external_environment()
    env.update(external)
    env.update(PYTHONPATH=str(CODE)+':'+str(CODE/'training'), PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(ROOT/'tmp'))
    return env


def stage():
    if CODE.exists(): raise RuntimeError('immutable code snapshot already exists')
    for name in OVERLAYS:
        if not (DEPLOY/name).is_file(): raise RuntimeError('missing overlay: '+name)
    shutil.copytree(OLD/'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, parent in OVERLAYS.items(): shutil.copy2(DEPLOY/name, CODE/parent/name)
    tests = [str(CODE/'training/tests'/name) for name in (
        'test_psd_slate_feedback.py', 'test_psd_slate_pipeline.py', 'test_psd_slate.py',
        'test_psd_slate_bound_positions.py', 'test_psd_candidate_binding.py',
        'test_psd_repair_storage.py', 'test_psd_model_route.py')]
    with (DEPLOY/'tests.log').open('x') as log:
        code = subprocess.call([sys.executable, '-m', 'pytest', '-q', *tests],
            cwd=CODE, env=environment(), stdout=log, stderr=log)
    save(DEPLOY/'stage-state.json', {'deployment_ready_not_live': code == 0,
        'repair_model':'gemini-3.6-flash', 'thinking_level':'high',
        'tests_returncode':code, 'source_runtime_unchanged':True,
        'overlays':{name:hashlib.sha256((DEPLOY/name).read_bytes()).hexdigest() for name in OVERLAYS}})
    print(json.dumps({'tests_returncode':code}), flush=True)
    if code: raise RuntimeError('tests failed')


async def canary_worker():
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from scripts.run_psd_repair_driver import _parser, _run
    from ifv_training.psd_case_pool import safe_case_error
    async def one(key):
        identity = load(SEARCH/'repairs'/key/'run-inputs.json')['identity']
        cli = []
        for action in _parser()._actions:
            name = action.dest
            if name not in identity or name in {'resume', 'help'}: continue
            value = identity[name]
            if isinstance(value, dict) and 'path' in value: value=value['path']
            if name=='output_dir': value=str(CANARY/key)
            if name=='repair_attempts': value=1
            if value is None: continue
            if isinstance(action, argparse._StoreTrueAction):
                if value: cli.append(action.option_strings[0])
            elif isinstance(action, argparse._StoreFalseAction):
                if not value: cli.append(action.option_strings[0])
            else: cli.extend([action.option_strings[0], str(value)])
        if (CANARY/key/'run-inputs.json').exists(): cli.append('--resume')
        try:
            result = await _run(_parser().parse_args(cli))
            save(CANARY/(key+'-result.json'), {'key':key,'result':result,'time':time.time()})
        except Exception as error:
            save(CANARY/(key+'-error.json'), {'key':key,**safe_case_error(error),'time':time.time()})
            raise
    await asyncio.gather(*(one(key) for key in ('4dc7eb5a06fe4f29','f6f1be3454263f76')))


def canary():
    state=load(DEPLOY/'stage-state.json')
    if not state['deployment_ready_not_live']: raise RuntimeError('unvalidated code')
    CANARY.mkdir(exist_ok=False)
    process=owner().spawn([sys.executable,'-u',str(CODE/'scripts/server/deploy_psd_checker_feedback.py'),
        'canary-worker'], environment(), DEPLOY/'canary.log')
    save(DEPLOY/'canary-process.json',process)
    print(json.dumps({'canary_pid':process['pid']}),flush=True)


def switch():
    if CONTROL.exists(): raise RuntimeError('new controller already exists')
    tests=load(DEPLOY/'stage-state.json')
    if not tests['deployment_ready_not_live']: raise RuntimeError('unvalidated code')
    results=[load(CANARY/(key+'-result.json')) for key in ('4dc7eb5a06fe4f29','f6f1be3454263f76')]
    if any(row['result'].get('complete_reruns',0)!=1 for row in results):
        raise RuntimeError('real full-rerun canary incomplete')
    previous=load(PREVIOUS/'process.json')
    manager=owner()
    manager.checked(previous)
    expected=[sys.executable,'-u',str(OLD/'code/training/scripts/h20/run_psd_lightweight_resume.py'),'worker']
    if previous.get('command')!=expected: raise RuntimeError('unexpected controller owner')
    group=[]
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            pid=int(entry.name)
            if os.getpgid(pid)!=previous['pid']: continue
            command=[p.decode() for p in (entry/'cmdline').read_bytes().split(b'\0') if p]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        tracker=(len(command)==3 and command[:2]==[sys.executable,'-c']
            and re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\(\d+\)',command[2]))
        if tracker:
            ppid=(entry/'stat').read_text().split(') ',1)[1].split()[1]
            parent=Path('/proc',ppid,'cmdline').read_bytes().replace(b'\0',b' ').decode()
            tracker=str(OLD/'code/scripts/run_psd_round.py')+' prepare' in parent
        if not (command==expected or tracker or (command[:4]==[sys.executable,'-u',
                str(OLD/'code/scripts/run_psd_round.py'),'prepare'] and str(ROUND) in command)):
            raise RuntimeError('unexpected member of prepare process group')
        group.append({'pid':pid,'command':command})
    # Save the exact owner/child scope before stopping only this prepare group.
    save(DEPLOY/'handoff.json',{'previous':previous,'group':group,'canary_results':results,
        'reason':'remove_localizer_veto_and_restore_checker_feedback',
        'gpu_services_unchanged':True,'source_and_completed_results_preserved':True,'time':time.time()})
    manager.checked(previous)
    os.killpg(previous['pid'],signal.SIGTERM)
    for _ in range(40):
        live=[]
        for row in group:
            p=Path('/proc',str(row['pid']),'stat')
            if p.exists() and p.read_text().split(') ',1)[1][0]!='Z': live.append(row['pid'])
        if not live: break
        time.sleep(.5)
    else: raise RuntimeError('old group still live')
    launch=launcher()
    save(launch.INTERRUPTION_RECEIPT,{'schema_version':'ifv-psd-controlled-interruptions-v1',
        'round':str(ROUND),'terminated_prepare_pids':[r['pid'] for r in group],
        'reason':'checker_feedback_bugfix','time':time.time()})
    launch.reconcile_controlled_interruptions()
    CONTROL.mkdir(exist_ok=False)
    process=manager.spawn([sys.executable,'-u',str(CODE/'scripts/server/deploy_psd_checker_feedback.py'),
        'worker'],environment(),CONTROL/'controller.log')
    save(CONTROL/'process.json',process)
    save(CONTROL/'state.json',{'phase':'launched','repair_model':'gemini-3.6-flash',
        'case_concurrency':40,'external_concurrency':16,'time':time.time()})
    save(PREVIOUS/'checker-feedback-handoff.json',{'new_controller':str(CONTROL),'time':time.time()})
    print(json.dumps({'controller_pid':process['pid']}),flush=True)


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['stage','canary','canary-worker','switch','worker'])
    mode=parser.parse_args().mode
    if mode=='stage': stage()
    elif mode=='canary': canary()
    elif mode=='canary-worker': asyncio.run(canary_worker())
    elif mode=='switch': switch()
    else: launcher().worker()
