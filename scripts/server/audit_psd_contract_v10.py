"""Isolated PSD audit: CPU regressions, archived failures, then three repairs.

Never starts the 400x8 collection or the optimizer. The immutable v8/v9 roots
and the epoch3 SFT export remain untouched. All outputs stay under ROOT.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
PREVIOUS = ROOT / 'training-artifacts/psd-native-repair-20260916-v9'
DEPLOY = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
OVERLAYS = {
    'psd_gemini_judge.py': 'training/ifv_training',
    'psd_source_review.py': 'training/ifv_training',
    'psd_repair_verifier.py': 'training/ifv_training',
    'psd_slate.py': 'training/ifv_training',
    'psd_slate_search.py': 'training/ifv_training',
    'run_psd_repair_driver.py': 'scripts',
    'test_psd_slate_pipeline.py': 'training/tests',
    'test_psd_review_contract_audit.py': 'training/tests',
}


def controller():
    spec = importlib.util.spec_from_file_location('contract_audit_v9', PREVIOUS / 'run_psd_native_repair_v9.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Same bounded three-case workflow, isolated paths and new immutable code.
    module.DEPLOY = DEPLOY
    module.CODE = DEPLOY / 'code'
    module.OUT = module.RUN / 'psd-contract-audit-v10'
    module.SNAPSHOT = DEPLOY / 'snapshot'
    return module


def prepare(m, o):
    assert not m.CODE.exists() and not m.OUT.exists() and not m.SNAPSHOT.exists()
    assert not (PREVIOUS / 'process.json').exists()
    # Validate transfer completion BEFORE creating an immutable code snapshot.
    for name in OVERLAYS:
        assert (DEPLOY / name).is_file(), name
    shutil.copytree(PREVIOUS / 'code', m.CODE,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, destination in OVERLAYS.items():
        shutil.copy2(DEPLOY / name, m.CODE / destination / name)
    shutil.copytree(PREVIOUS / 'snapshot', m.SNAPSHOT)
    for path in (m.CODE / 'src').rglob('*.py'):
        assert path.read_text() == (ROOT / 'image-factual-verifier-v2' / path.relative_to(m.CODE)).read_text()
    files = list(o.load(PREVIOUS / 'binding.json')['files'])
    files = [Path(p) for p in files if not Path(p).is_relative_to(PREVIOUS)]
    files += [m.SNAPSHOT / name for name in ('serving-profile.json', 'checkpoint-manifest.json')]
    files += [p for folder in ('src', 'scripts', 'training') for p in (m.CODE / folder).rglob('*.py')]
    o.save(DEPLOY / 'binding.json', {'files': {str(p): o.sha(p) for p in files},
        'native_contract_version': 'native-contract-and-complete-slate-review-v4',
        'fixed_source_rollouts_reused': True, 'formal_collection_or_training_allowed': False,
        'historical_source_review_projection': 'legacy_v1 retained, not relabeled v2'})
    o.save(DEPLOY / 'state.json', {'phase': 'prepared_requires_tests', 'training_started': False})


def test(m, o):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(ROOT / 'tmp'), PYTHONPATH=str(m.CODE) + ':' + str(m.CODE / 'training'))
    command = [sys.executable, '-m', 'pytest', 'training/tests', '-q', '-p', 'no:cacheprovider']
    result = subprocess.run(command, cwd=m.CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    report = {'passed': result.returncode == 0, 'exit_code': result.returncode,
              'output': result.stdout, 'binding_sha256': o.sha(DEPLOY / 'binding.json'), 'time': time.time()}
    o.save(DEPLOY / 'regression-tests.json', report)
    print(result.stdout)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'test', 'audit-only', 'launch', 'execute'))
    args = parser.parse_args()
    m = controller()
    _, o = m.owner()
    if args.mode == 'prepare':
        prepare(m, o)
    elif args.mode == 'test':
        test(m, o)
    elif args.mode == 'audit-only':
        m.audit_only()
    elif args.mode == 'execute':
        m.execute()
    else:
        tests = o.load(DEPLOY / 'regression-tests.json')
        assert tests['passed'] and tests['binding_sha256'] == o.sha(DEPLOY / 'binding.json')
        assert o.load(DEPLOY / 'archived-failure-verification.json')['passed']
        assert not (DEPLOY / 'process.json').exists()
        original, _ = m.owner()
        env, checks = original.capture_environment(o)
        env.update(PYTHONPATH=str(m.CODE) + ':' + str(m.CODE / 'training'),
                   PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'))
        o.save(DEPLOY / 'credential-presence.json', checks)
        receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, DEPLOY / 'run.log')
        receipt['script_sha256'] = o.sha(__file__)
        o.save(DEPLOY / 'process.json', receipt)
        print(json.dumps(receipt))
