"""Rebuild derived targets from unchanged completed episodes; no provider calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path('/volume/ybo/wza')
DEPLOY=ROOT/'training-artifacts/psd-raw-teacher-20260916-v18'
CODE=DEPLOY/'code'
RUNBASE=ROOT/'runs/psd-sft3084-captured-canary4x8-20260916'
OLD=RUNBASE/'psd-grounded-review-canary-v15'
OUT=RUNBASE/'psd-raw-teacher-canary-v18'
SNAPSHOT=ROOT/'training-artifacts/psd-contract-audit-20260916-v10/snapshot'


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('launch','execute','score'))
    args=parser.parse_args()
    sys.path[:0]=[str(CODE),str(CODE/'training')]
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    assert owner.load(DEPLOY/'state.json')['deployment_ready_not_live']
    if args.mode=='launch':
        OUT.mkdir(exist_ok=False)
        env={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'training'),
            'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TMPDIR':str(ROOT/'tmp'),
            'HF_HOME':str(ROOT/'cache/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],env,OUT/'prepare.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    from ifv_training.psd_materialization import materialize_bank
    paths=[OLD/'repair_candidates.jsonl',OLD/'repair_attempts.jsonl',
        RUNBASE/'psd-corrected-contract-v12/source-bank/candidates/preservation_candidates.jsonl']
    hashes={str(p):owner.sha(p) for p in paths}
    if args.mode=='execute':
        owner.save(OUT/'binding.json',{'inputs':hashes,'retained_complete_episodes':True,
            'no_agent_or_judge_calls':True,'old_unattested_logprobs_not_relabelled':True})
    else:
        assert owner.load(OUT/'binding.json')['inputs']==hashes
    owner.save(OUT/'state.json',{'phase':'exact_frozen_teacher_rescoring' if args.mode=='score' else 'rebuilding_pending_targets',
        'formal_training':False})
    result=materialize_bank(output_dir=OUT/'bank',repair_candidates=paths[0],repair_attempts=paths[1],
        preservation_candidates=paths[2],serving_profile=SNAPSHOT/'serving-profile.json',
        checkpoint_manifest=SNAPSHOT/'checkpoint-manifest.json',teacher_device='cuda:0',
        score_missing_topk=args.mode=='score')
    assert hashes=={str(p):owner.sha(p) for p in paths}
    owner.save(OUT/('score-result.json' if args.mode=='score' else 'prepare-result.json'),result)
    owner.save(OUT/'state.json',{'phase':result['status'],'formal_training':False})
    print(json.dumps({'phase':result['status']}))


if __name__=='__main__':main()
