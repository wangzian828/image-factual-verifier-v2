"""Resume the SAME fixed bank after bounded source-citation correction.

No source resampling, no rejudging valid decisions, no implicit GPU teacher load.
"""
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/volume/ybo/wza')
PREVIOUS = ROOT / 'training-artifacts/psd-epoch3-repair-20260916-v5'
DEPLOY = ROOT / 'training-artifacts/psd-epoch3-repair-20260916-v6'
CODE = DEPLOY / 'code'


def controller():
    spec = importlib.util.spec_from_file_location('validated_epoch3_controller',
        PREVIOUS / 'continue_psd_epoch3_canary.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execute():
    c = controller()
    m, o = c.owner()
    def phase(name, **extra):
        o.save(DEPLOY / 'state.json', {'phase': name, 'time': time.time(), 'training_started': False, **extra})
    try:
        phase('verifying_immutable_fixed32')
        bound = o.load(DEPLOY / 'binding.json')
        for raw, digest in bound['files'].items():
            assert o.sha(raw) == digest, 'Frozen preparation input changed'
        acceptance = o.load(c.RUN / 'runtime-acceptance-v5.json')
        assert acceptance['runtime_gate_passed'] and acceptance['collection']['episodes'] == 32
        for name, digest in acceptance['trace_sha256'].items():
            assert o.sha(c.RUN / 'episodes/traces' / name) == digest, name
        current = o.preflight()[1]
        binding = o.load(c.RUN / 'binding.json')
        for key in current:
            if key not in ('benchmark', 'source_access_policy'):
                assert current[key] == binding[key], key
        phase('source_citation_correction_then_bounded_slate_repair')
        sys.path[:0] = [str(CODE), str(CODE / 'training')]
        from scripts.run_psd_round import prepare
        args = SimpleNamespace(run_dir=c.RUN / 'episodes', benchmark=Path(binding['benchmark']),
            train_cases=Path(binding['train_cases']), private_gold=Path(binding['private_gold']),
            source_access_policy=Path(binding['source_access_policy']), snapshot=c.RUN / 'snapshot',
            output=c.OUT, round_index=1, previous_round_completion=None, attempts=6, case_concurrency=2,
            source_review_concurrency=4, expected_rollouts_per_case=8, task_source_selection='longest_failed',
            repair_mode='slate', judge_model='gemini-3.1-pro-preview', teacher_device='cuda:0', defer_topk=True)
        result = asyncio.run(prepare(args))
        o.save(DEPLOY / 'result.json', result)
        phase(result['status'], result=result)
    except BaseException as error:
        phase('held_requires_inspection', error_type=type(error).__name__)
        raise


def launch():
    c = controller()
    m, o = c.owner()
    assert not CODE.exists() and not (DEPLOY / 'process.json').exists()
    assert o.load(PREVIOUS / 'state.json')['phase'] == 'paused_source_review_requires_resolution'
    receipt = o.load(PREVIOUS / 'process.json')
    stat = Path(f'/proc/{receipt["pid"]}/stat')
    assert not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z'
    assert not (c.OUT / 'search').exists(), 'Source correction must precede repair'
    shutil.copytree(PREVIOUS / 'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copy2(DEPLOY / 'psd_source_review.py', CODE / 'training/ifv_training/psd_source_review.py')
    source = c.OUT / 'source-reviews'
    files = [c.RUN / 'binding.json', c.RUN / 'runtime-acceptance-v5.json', source / 'inputs.json',
             *source.joinpath('reviews').glob('*.json'), *source.joinpath('judge-cache').glob('*.json')]
    o.save(DEPLOY / 'binding.json', {'files': {str(p): o.sha(p) for p in files},
        'code_sha256': {str(p.relative_to(CODE)): o.sha(p) for folder in ('src', 'training', 'scripts')
                       for p in (CODE / folder).rglob('*.py')},
        'preserved_valid_source_reviews': len(list((source / 'reviews').glob('*.json'))),
        'maximum_corrective_responses_per_invalid_review': 1, 'source_resampling': False})
    env, checks = m.capture_environment(o)
    env['PYTHONPATH'] = str(CODE) + ':' + str(CODE / 'training')
    o.save(DEPLOY / 'credential-presence.json', checks)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
    receipt = o.spawn(command, env, DEPLOY / 'run.log')
    receipt['script_sha256'] = o.sha(__file__)
    o.save(DEPLOY / 'process.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    launch() if args.launch else execute()
