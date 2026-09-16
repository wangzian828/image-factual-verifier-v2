"""Test an immutable collector recovery snapshot; never launch production here."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
OUT = ROOT / 'training-artifacts/psd-service-recovery-20260917-v30'
BASE = ROOT / 'training-artifacts/psd-completion-admission-20260917-v27/code'
FILES = {
    'training/ifv_training': ['psd_infrastructure_retry.py', 'psd_collection_recovery.py'],
    'training/tests': ['test_psd_infrastructure_retry.py', 'test_psd_collection_recovery.py'],
    'scripts': ['collect_psd_rollouts.py'],
    'scripts/server': ['run_psd_production_collection.py', 'probe_psd_failed_slots.py', 'psd_nan_metric_ab.py'],
    'tests': ['test_psd_failed_slots_diagnostic.py'],
}


def main():
    global OUT, BASE, FILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-cache', action='store_true')
    parser.add_argument('--storage-spool', action='store_true')
    parser.add_argument('--completed-recovery', action='store_true')
    parser.add_argument('--final-bank-gate', action='store_true')
    parser.add_argument('--slow-storage', action='store_true')
    args = parser.parse_args()
    if sum((args.storage_cache, args.storage_spool, args.completed_recovery, args.final_bank_gate, args.slow_storage)) > 1:
        parser.error('Choose one immutable snapshot variant')
    if args.storage_cache:
        OUT = ROOT / 'training-artifacts/psd-storage-reader-20260917-v31'
        BASE = ROOT / 'training-artifacts/psd-service-recovery-20260917-v30/code'
        FILES = {'training/ifv_training': ['psd_repair_storage.py'],
                 'training/tests': ['test_psd_storage_compaction.py'],
                 'scripts/server': ['compact_psd_completed_storage.py']}
    if args.storage_spool:
        OUT = ROOT / 'training-artifacts/psd-storage-spool-20260917-v32'
        BASE = ROOT / 'training-artifacts/psd-storage-reader-20260917-v31/code'
        FILES = {'training/tests': ['test_psd_storage_compaction.py'],
                 'scripts/server': ['compact_psd_completed_storage.py']}
    if args.completed_recovery:
        OUT = ROOT / 'training-artifacts/psd-completed-recovery-20260917-v33'
        BASE = ROOT / 'training-artifacts/psd-storage-spool-20260917-v32/code'
        FILES = {'training/ifv_training': ['psd_collection_recovery.py'],
                 'training/tests': ['test_psd_collection_recovery.py']}
    if args.final_bank_gate:
        OUT = ROOT / 'training-artifacts/psd-final-bank-gate-20260917-v34'
        BASE = ROOT / 'training-artifacts/psd-completed-recovery-20260917-v33/code'
        FILES = {'scripts/server': ['run_psd_dp4_resume_gate.py'],
                 'training/tests': ['test_psd_final_bank_gate.py']}
    if args.slow_storage:
        OUT = ROOT / 'training-artifacts/psd-slow-storage-20260917-v35'
        BASE = ROOT / 'training-artifacts/psd-final-bank-gate-20260917-v34/code'
        FILES = {'scripts/server': ['run_psd_production_collection.py'],
                 'training/ifv_training': ['psd_collection_recovery.py', 'psd_storage_admission.py'],
                 'training/tests': ['test_psd_collection_recovery.py', 'test_psd_storage_admission.py']}
    os.umask(0o077)
    code = OUT / 'code'
    if code.exists():
        raise RuntimeError('Do not modify an existing immutable candidate')
    for names in FILES.values():
        for name in names:
            assert (OUT/name).is_file()
    shutil.copytree(BASE, code, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for parent, names in FILES.items():
        (code/parent).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(OUT/name, code/parent/name)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (BASE/'src').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            assert sha(p) == sha(code/p.relative_to(BASE)), 'Frozen Agent changed'
    binding = {str(p.relative_to(code)): sha(p) for p in code.rglob('*') if p.is_file()}
    (OUT/'code-binding.json').write_text(json.dumps(binding, indent=2))
    env = {**os.environ, 'PYTHONPATH': str(code)+':'+str(code/'training'), 'PYTHONDONTWRITEBYTECODE': '1',
           'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(ROOT/'tmp')}
    with (OUT/'tests.log').open('x') as log:
        result = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests',
            'tests/test_psd_failed_slots_diagnostic.py', '--tb=short'], cwd=code, env=env, stdout=log, stderr=log)
    status = {'tests_returncode': result, 'frozen_agent_unchanged': True,
              'deployment_ready_not_live': result == 0, 'formal_training': False}
    (OUT/'stage-state.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
