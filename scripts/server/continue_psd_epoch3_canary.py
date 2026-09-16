"""Audit the fixed 32-slot epoch3 bank, then run checker and slate repair.

No automatic teacher load onto occupied GPUs; stop at the explicit top-k gate.
The hourly task performs the validated GPU handoff and production continuation.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path('/volume/ybo/wza')
PREVIOUS=ROOT/'training-artifacts/psd-epoch3-repair-20260916-v3'
DEPLOY=ROOT/'training-artifacts/psd-epoch3-repair-20260916-v4'
CODE=DEPLOY/'code'
CAPTURE=ROOT/'training-artifacts/psd-epoch3-capture-20260916-v2'
RUN=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916'
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OUT=RUN/'psd-round-v4'


def owner():
    p=CAPTURE/'psd_epoch3_captured_canary.py'
    spec=importlib.util.spec_from_file_location('capture_owner',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,m.owner()


def audit():
    m,o=owner()
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    from scripts.audit_real_trace import audit_trace, SourceAccessPolicy, _json_summary
    from ifv_training.psd_collection import verify_collection
    from ifv_training.io import load_jsonl
    from tokenizers import Tokenizer
    b=o.load(RUN/'binding.json')
    current=o.preflight()[1]
    for k in current:
        if k not in ('benchmark','source_access_policy'):assert current[k]==b[k],k
    slots=verify_collection(load_jsonl(RUN/'episodes/run_results.jsonl'),case_ids=b['case_ids'],
        manifest=o.load(RUN/'episodes/run_manifest.json'),expected_rollouts=8)
    paths=sorted((RUN/'episodes/traces').glob('*.json'));assert len(paths)==32
    report=_json_summary([audit_trace(p,source_access_policy=SourceAccessPolicy.load(Path(b['source_access_policy'])))
        for p in paths],strict_scheduler=True)
    o.save(RUN/'strict-audit-v4.json',report);assert report['passed']
    counts=Counter();cache=Counter();capture_count=0
    for p in paths:
        trace=o.load(p);assert not trace.get('error') and trace.get('termination')=='success'
        counts[trace['case_id']]+=1
        for step in trace['state']['all_steps']:
            metadata=step.get('metadata',{})
            if step.get('tool_name') in ('perceive_scene','ocr_with_position'):
                cache[str(metadata.get('cache_hit'))]+=1
            if (step.get('stage') not in ('unified_react','unified_judgment')
                    or step.get('action_type') not in ('tool_call','output')
                    or metadata.get('deterministic_segment_boundary')):continue
            cap=metadata.get('policy_token_capture',{})
            assert cap.get('status')=='complete' and cap.get('topk')==20
            assert cap.get('prompt_token_ids') and cap.get('completion_token_ids')
            assert len(cap['completion_token_ids'])==len(cap['completion_logprobs'])
            assert all(math.isfinite(x) for x in cap['completion_logprobs'])
            capture_count+=1
    assert set(counts)==set(b['case_ids']) and set(counts.values())=={8}
    assert cache and set(cache)=={'False'}
    tokenizer=Tokenizer.from_file(str(o.EXPORT.parent/'model/tokenizer.json'))
    from src.tools.perceive_scene import PERCEIVE_SCENE_PROMPT
    wire_audit=o.module('epoch3_wire_audit',DEPLOY/'psd_wire_audit.py')
    wire=wire_audit.audit_wire_archive(SERVICE/'captured-v2/wire',tokenizer=tokenizer,
                                     perception_prompt=PERCEIVE_SCENE_PROMPT)
    health=o.http('http://127.0.0.1:19018/health')
    assert all(r['inflight']==0 for r in health['replicas'])
    assert health['wire_capture']['completed']==wire['responses']
    assert health['wire_capture']['started']==wire['requests']
    assert health['wire_capture']['failed']==wire.get('auxiliary_transport_failures',0)
    assert wire['requests']==wire['responses']+wire.get('auxiliary_transport_failures',0)
    assert health['wire_capture']['outstanding_requests']==0
    result={'runtime_gate_passed':True,'full_psd_gate_passed':False,'training_started':False,
        'collection':slots,'native_captured_actions':capture_count,'perception_cache_flags':dict(cache),
        'wire':wire,
        'binding_sha256':o.sha(RUN/'binding.json'),'strict_audit_sha256':o.sha(RUN/'strict-audit-v4.json'),
        'trace_sha256':{p.name:o.sha(p) for p in paths},'time':time.time()}
    o.save(RUN/'runtime-acceptance-v4.json',result)
    return b


def execute():
    m,o=owner()
    state_path=DEPLOY/'state.json'
    def phase(name,**extra):o.save(state_path,{'phase':name,'time':time.time(),'training_started':False,**extra})
    try:
        phase('waiting_for_fixed32_collection')
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            state=o.load(RUN/'state.json')
            if state['phase']=='collection_complete_requires_source_checker':break
            failure=SERVICE/'captured-v2/failure.json'
            if failure.exists():raise RuntimeError('Collector failed; preserve all outputs and inspect')
            receipt=o.load(RUN/'process.json');o.checked(receipt)
            time.sleep(10)
        else:raise RuntimeError('Fixed32 collection wait deadline')
        phase('auditing_fixed32');binding=audit()
        phase('source_checker_and_bounded_slate_repair')
        sys.path[:0]=[str(CODE),str(CODE/'training')]
        from scripts.run_psd_round import prepare
        args=SimpleNamespace(run_dir=RUN/'episodes',benchmark=Path(binding['benchmark']),
            train_cases=Path(binding['train_cases']),private_gold=Path(binding['private_gold']),
            source_access_policy=Path(binding['source_access_policy']),snapshot=RUN/'snapshot',
            output=OUT,round_index=1,previous_round_completion=None,attempts=6,case_concurrency=2,
            source_review_concurrency=4,expected_rollouts_per_case=8,task_source_selection='longest_failed',
            repair_mode='slate',judge_model='gemini-3.1-pro-preview',teacher_device='cuda:0',defer_topk=True)
        result=asyncio.run(prepare(args));o.save(DEPLOY/'result.json',result)
        phase(result['status'],result=result)
    except BaseException as e:
        phase('held_requires_inspection',error_type=type(e).__name__)
        raise


def launch():
    m,o=owner()
    assert not CODE.exists() and not (DEPLOY/'process.json').exists() and not OUT.exists()
    assert o.load(PREVIOUS/'state.json')['phase']=='waiting_for_fixed32_collection'
    assert not (RUN/'psd-round-v3').exists(),'Never interrupt active checker or repair'
    # Replace only the idle continuation controller, not the running collector.
    prior=o.load(PREVIOUS/'process.json');o.stop(prior)
    o.save(PREVIOUS/'superseded.json',{'reason':'distinguish auxiliary HTTP retries from native policy failure',
                                    'replacement':str(DEPLOY),'time':time.time()})
    shutil.copytree(PREVIOUS/'code',CODE,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    env,checks=m.capture_environment(o);env['PYTHONPATH']=str(CODE)+':'+str(CODE/'training')
    # Bind the frozen collection plus the updated target-preparation code.
    o.save(DEPLOY/'binding.json',{'collection_binding_sha256':o.sha(RUN/'binding.json'),
        'collector_capture_required':True,'model_export_sha256':o.EXPORT_SHA,
        'wire_audit_sha256':o.sha(DEPLOY/'psd_wire_audit.py'),
        'source_sha256':{str(p.relative_to(CODE)):o.sha(p) for folder in ('src','scripts','training')
                        for p in (CODE/folder).rglob('*.py')},'credential_presence':checks,
        'limits':{'source_slots':32,'complete_repair_reruns_per_task':6,'proposals_per_task':12},
        'production_authorization':{'train_images':400,'rollouts_per_image':8,'optimization_epochs':5}})
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--execute']
    receipt=o.spawn(command,env,DEPLOY/'run.log');receipt['script_sha256']=o.sha(__file__)
    o.save(DEPLOY/'process.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch',action='store_true');modes.add_argument('--execute',action='store_true')
    args=parser.parse_args();launch() if args.launch else execute()
