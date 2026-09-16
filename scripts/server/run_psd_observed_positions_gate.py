"""Real corrected-slate canary on unchanged sources/localizers and SFT3 weights.

This is a new explicitly versioned experiment, not a reset of an exhausted run.
All old artifacts remain read-only. No policy training or service restart here.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path('/volume/ybo/wza')
BASE = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
CODE = DEPLOY / 'code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
OLD = RUN / 'psd-corrected-contract-v12'
OUT = RUN / 'psd-observed-positions-v14'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner():
    spec = importlib.util.spec_from_file_location('previous_corrected_gate', BASE / 'run_psd_corrected_contract_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.owner()


def validate(o):
    assert o.load(DEPLOY / 'state.json')['deployment_ready_not_live']
    for rel, expected in o.load(DEPLOY / 'code-binding.json').items():
        assert sha(CODE / rel) == expected, 'Immutable code changed'
    for path, expected in o.load(OLD / 'binding.json')['files'].items():
        assert sha(path) == expected, 'Frozen source/checker artifact changed'
    o.verify_export()
    for gpu in range(4):
        receipt = o.load(SERVICE / f'replica-{gpu}.json')
        o.checked(receipt)
        command = receipt['command']
        assert command[command.index('--mm-processor-cache-gb') + 1] == '0'


def prepare(o):
    validate(o)
    OUT.mkdir(exist_ok=False)
    files = {str(path): sha(path) for path in (OLD / 'search').rglob('*.json*') if path.is_file()}
    o.save(OUT / 'binding.json', {'old_search_files': files,
        'code_binding_sha256': sha(DEPLOY / 'code-binding.json'),
        'source_bank': str(OLD / 'source-bank'), 'source_resampling': False,
        'same_localization': True, 'new_protocol_budget': {'complete_reruns': 6, 'proposals': 12},
        'protected_epoch3_weights_unchanged': True})
    o.save(OUT / 'state.json', {'phase': 'prepared_observed_positions', 'training_started': False})


def execute(o):
    validate(o)
    binding = o.load(OUT / 'binding.json')
    assert sha(DEPLOY / 'code-binding.json') == binding['code_binding_sha256']
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    import scripts.run_psd_feedback_canary as feedback
    from ifv_training.psd_diagnostics import persist_exception
    original_run = feedback._run

    async def observed(args):
        prior = OLD / 'search/repairs' / args.output_dir.name
        old_identity = o.load(prior / 'run-inputs.json')['identity']
        for key in ('trace', 'candidate', 'audit', 'gold', 'image'):
            assert sha(getattr(args, key)) == old_identity[key]['sha256'], 'Canary case changed'
        localization = prior / 'semantic-localization.json'
        if localization.exists():
            assert sha(localization) == binding['old_search_files'][str(localization)]
            args.semantic_verification = localization
        try:
            return await original_run(args)
        except Exception as error:
            receipt = persist_exception(args.output_dir / 'private-diagnostics', error)
            o.save(OUT / (args.output_dir.name + '-failure.json'), receipt)
            raise

    feedback._run = observed
    o.save(OUT / 'state.json', {'phase': 'real_corrected_position_repairs',
                               'time': time.time(), 'training_started': False})
    args = SimpleNamespace(source=OLD / 'source-bank', output=OUT / 'search',
        snapshot=BASE / 'snapshot', attempts=6, proposal_rounds=12, case_concurrency=40,
        task_source_selection='longest_failed', repair_mode='slate',
        judge_model='gemini-3.1-pro-preview', teacher_device='cuda:0', score_missing_topk=False)
    try:
        result = asyncio.run(feedback.run(args))
        o.save(OUT / 'result.json', result)
        o.save(OUT / 'state.json', {'phase': result['status'], 'time': time.time(), 'training_started': False})
    finally:
        for path, expected in binding['old_search_files'].items():
            assert sha(path) == expected, 'Old canary artifact changed'


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'execute'))
    args = parser.parse_args()
    original, o = owner()
    if args.mode == 'prepare':
        prepare(o)
    elif args.mode == 'execute':
        execute(o)
    else:
        validate(o)
        assert o.load(OUT / 'state.json')['phase'] == 'prepared_observed_positions'
        assert not (OUT / 'process.json').exists()
        env, checks = original.capture_environment(o)
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
                   PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
        o.save(OUT / 'credential-presence.json', checks)
        receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, OUT / 'run.log')
        o.save(OUT / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)}), flush=True)


if __name__ == '__main__':
    main()
