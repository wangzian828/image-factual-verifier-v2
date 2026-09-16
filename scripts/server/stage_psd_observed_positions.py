"""Stage corrected upstream-aligned position policy without touching live code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
OLD = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10/source-review-v2-code'
CODE = DEPLOY / 'code'
OVERLAYS = {
    'psd_slate.py': 'training/ifv_training',
    'psd_slate_search.py': 'training/ifv_training',
    'run_psd_repair_driver.py': 'scripts',
    'run_psd_feedback_canary.py': 'scripts',
    'test_psd_slate.py': 'training/tests',
    'test_psd_slate_pipeline.py': 'training/tests',
    'test_psd_slate_bound_positions.py': 'training/tests',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.umask(0o077)
    if CODE.exists():
        raise RuntimeError('Immutable candidate already staged')
    for name in OVERLAYS:
        assert (DEPLOY / name).is_file()
    shutil.copytree(OLD, CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, parent in OVERLAYS.items():
        shutil.copy2(DEPLOY / name, CODE / parent / name)
    for path in (OLD / 'src').rglob('*.py'):
        assert sha(path) == sha(CODE / path.relative_to(OLD)), 'Frozen Agent changed'
    binding = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob('*.py')}
    (DEPLOY / 'code-binding.json').write_text(json.dumps(binding, indent=2))
    env = {**os.environ, 'PYTHONPATH': str(CODE) + ':' + str(CODE / 'training'),
           'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT / 'tmp')}
    with (DEPLOY / 'tests.log').open('x') as log:
        code = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests'],
                               cwd=CODE, env=env, stdout=log, stderr=log)
    result = {'tests_returncode': code, 'frozen_agent_unchanged': True, 'provider_calls': 0,
              'training_started': False, 'deployment_ready_not_live': code == 0,
              'position_policy': 'observed-decisions-not-localizer-lock-v1'}
    (DEPLOY / 'state.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
