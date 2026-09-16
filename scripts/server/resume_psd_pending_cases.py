"""One explicit engineering recovery, preserving completed episodes and budgets."""
import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import time
import json

ROOT=Path('/volume/ybo/wza')
DEPLOY=ROOT/'training-artifacts/psd-observed-positions-20260916-v14'
CODE=DEPLOY/'code'
RUN=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
CONTROL=RUN/'execution-recovery-v1'
CASES=['7996ba9752652f37','82897bdeb31b6ade','e79d4db478e45800']


def load_pending(case):
    from ifv_training.io import load_json,sha256_file
    from ifv_training.psd_repair import _sha
    from ifv_training.psd_repair_storage import load_bound
    config=load_json(case/'run-inputs.json')['identity']
    load_bound(case/'run-inputs.json',identity=config)
    for value in config.values():
        if isinstance(value,dict) and set(value)=={'path','sha256'}:
            assert sha256_file(Path(value['path']))==value['sha256'],'Bound input changed'
    state_file=load_json(case/'slate-state.json')
    assert state_file['identity']['inputs']==_sha(config)
    state=load_bound(case/'slate-state.json',identity=state_file['identity'])
    assert state['status']=='repairing' and state['pending_proposal']['round_index']==len(state['rounds'])
    assert config['repair_attempts']==6 and config['proposal_rounds']==12
    completed={str(case/'run-inputs.json'):sha256_file(case/'run-inputs.json')}
    for row in state['rounds']:
        for path,digest in row['files'].items():
            assert sha256_file(Path(path))==digest
            completed[path]=digest
    return config,state,completed


def restore_args(config):
    values={k:Path(v['path']) if isinstance(v,dict) and set(v)=={'path','sha256'} else v
        for k,v in config.items() if k!='continuation_policy_version'}
    values.update(output_dir=Path(values['output_dir']),resume=True,skip_auto_judge=False,generation_retries=1)
    return argparse.Namespace(**values)


async def execute(o):
    from scripts.run_psd_repair_driver import _run
    from ifv_training.psd_diagnostics import persist_exception
    o.save(CONTROL/'state.json',{'phase':'resuming_original_pending_cases','training_started':False})
    async def one(key):
        case=RUN/'search/repairs'/key
        config,state,before=load_pending(case)
        o.save(CONTROL/f'{key}-before.json',{'completed_files':before,'rounds':len(state['rounds']),
            'proposals':len(state.get('proposals',[])),'pending_round':state['pending_proposal']['round_index'],
            'budgets_unchanged':True,'incomplete_attempt_recovery_only':True})
        try:
            result=await _run(restore_args(config))
            o.save(CONTROL/f'{key}-result.json',result)
            return {'case':key,'result':result}
        except Exception as error:
            failure=persist_exception(CONTROL/f'{key}-private-diagnostics',error)
            o.save(CONTROL/f'{key}-failure.json',failure)
            return {'case':key,'error_type':type(error).__name__,'status':'held_requires_diagnosis'}
        finally:
            for path,digest in before.items():
                assert o.sha(path)==digest,'Completed source/episode/review mutated'
    results=await asyncio.gather(*(one(key) for key in CASES))
    o.save(CONTROL/'result.json',{'results':results,'training_started':False})
    o.save(CONTROL/'state.json',{'phase':'cases_finished_inspect_results','training_started':False,'time':time.time()})


def main():
    os.umask(0o077)
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['launch','execute']);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('observed',DEPLOY/'run_psd_observed_positions_gate.py')
    observed=importlib.util.module_from_spec(spec);spec.loader.exec_module(observed)
    original,o=observed.owner()
    observed.validate(o)
    if args.mode=='execute':
        asyncio.run(execute(o));return
    # This file deliberately does not change serving configuration, weights,
    # proposer protocol or any completed checker result.
    recovery=o.load(SERVICE/'validated-execution-candidate-v1/state.json')
    assert recovery['phase']=='candidate_ready_for_real_repair_not_formal_training'
    for path in [RUN/'process.json',RUN/'reviewed-case-continuation-v1/process.json']:
        stat=Path(f'/proc/{o.load(path)["pid"]}/stat')
        assert not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z'
    for key in CASES:load_pending(RUN/'search/repairs'/key)
    CONTROL.mkdir(exist_ok=False)
    o.save(CONTROL/'serving-binding.json',recovery)
    env,checks=original.capture_environment(o)
    env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),PYTHONPATH=str(CODE)+':'+str(CODE/'training'))
    o.save(CONTROL/'credential-presence.json',checks)
    receipt=o.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],env,CONTROL/'run.log')
    o.save(CONTROL/'process.json',receipt)
    print(json.dumps({'pid':receipt['pid'],'output':str(CONTROL)}))


if __name__=='__main__':main()
