"""Version-correct PSD checker rerun on the SAME 32 frozen source episodes.

Does not rerun Agents, change the main experiment judge, or start optimization.
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

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
CODE = DEPLOY / 'source-review-v2-code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
OUT = RUN / 'source-reviews-transport-v2'
OVERLAYS = {'psd_source_review.py': 'training/ifv_training',
            'review_psd_sources.py': 'scripts', 'test_psd_source_review.py': 'training/tests'}


def owner():
    path = ROOT / 'training-artifacts/psd-native-repair-20260916-v9/run_psd_native_repair_v9.py'
    spec = importlib.util.spec_from_file_location('bounded_psd_v9', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.owner()


def execute(o):
    binding = o.load(DEPLOY / 'source-review-v2-binding.json')
    for path, expected in binding['files'].items():
        assert o.sha(path) == expected
    o.verify_export()
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from scripts.review_psd_sources import review_sources
    from ifv_training.psd_source_review import TRACE_PROJECTION
    assert TRACE_PROJECTION == 'transport_ids_v2'
    config = o.load(RUN / 'psd-round-v5/source-bank/prepared.json')['preparation']
    def state(phase, **extra):
        o.save(DEPLOY / 'source-review-v2-state.json', {'phase': phase,
            'time': time.time(), 'training_started': False, 'source_resampling': False, **extra})
    state('reviewing_same_32_with_transport_ids')
    try:
        result = asyncio.run(review_sources(run_dir=Path(config['rollout_dir']),
            benchmark=Path(config['benchmark']), train_cases=Path(config['case_split']),
            private_gold=Path(config['private_gold']), output=OUT,
            model='gemini-3.1-pro-preview', concurrency=4))
        assert result['selected'] == 32
        state(result['status'], counts=result['counts'], pending=result['pending'])
    except BaseException as error:
        state('held_requires_inspection', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'execute'))
    args = parser.parse_args()
    original, o = owner()
    if args.mode == 'prepare':
        assert not CODE.exists() and not OUT.exists()
        for name in OVERLAYS:
            assert (DEPLOY / name).is_file()
        shutil.copytree(DEPLOY / 'final-regression-code', CODE,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        for name, parent in OVERLAYS.items():
            shutil.copy2(DEPLOY / name, CODE / parent / name)
        files = [p for parent in ('scripts', 'training', 'src') for p in (CODE / parent).rglob('*.py')]
        files += [RUN / 'binding.json', RUN / 'episodes/run_manifest.json', RUN / 'episodes/run_results.jsonl',
                  RUN / 'psd-round-v5/source-bank/prepared.json']
        o.save(DEPLOY / 'source-review-v2-binding.json', {'files': {str(p): o.sha(p) for p in files},
            'reason': 'old checker projection omitted transport evidence IDs',
            'source_resampling': False, 'training_started': False})
    elif args.mode == 'execute':
        execute(o)
    else:
        receipt_path = DEPLOY / 'source-review-v2-process.json'
        assert CODE.is_dir() and not receipt_path.exists()
        env, checks = original.capture_environment(o)
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
                   PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
        o.save(DEPLOY / 'source-review-v2-credential-presence.json', checks)
        receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env,
                          DEPLOY / 'source-review-v2.log')
        o.save(receipt_path, receipt)
        print(json.dumps(receipt))
