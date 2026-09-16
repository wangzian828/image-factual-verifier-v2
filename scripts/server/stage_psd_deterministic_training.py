"""Stage a new immutable PSD trainer snapshot; no server Git or live edits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-deterministic-training-20260916-v16'
OLD = ROOT / 'training-artifacts/psd-grounded-review-20260916-v15/code'
CODE = DEPLOY / 'code'
OVERLAYS = {
    'qwen3.5-lora-r32-h20-dp4-128k.env': 'training/configs/psd',
    'qwen3.5-lora-r32-h20-sp4-128k.env': 'training/configs/psd',
    'verify_psd_training_profile.py': 'training/scripts/probe',
    'test_psd_training_profile.py': 'training/tests',
    'manifests.py': 'training/ifv_training',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.umask(0o077)
    assert not CODE.exists(), 'Immutable candidate already exists'
    for name in OVERLAYS:
        assert (DEPLOY / name).is_file()
    shutil.copytree(OLD, CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, parent in OVERLAYS.items():
        shutil.copy2(DEPLOY / name, CODE / parent / name)
    for path in (OLD / 'src').rglob('*'):
        if path.is_file():
            assert sha(path) == sha(CODE / path.relative_to(OLD)), 'Frozen Agent changed'
    binding = {str(p.relative_to(CODE)): sha(p) for p in CODE.rglob('*') if p.is_file()}
    (DEPLOY / 'code-binding.json').write_text(json.dumps(binding, indent=2))
    env = {**os.environ, 'PYTHONPATH': str(CODE) + ':' + str(CODE / 'training'),
           'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT / 'tmp'),
           'XDG_CACHE_HOME': str(ROOT / 'cache/psd-training-gate')}
    with (DEPLOY / 'tests.log').open('x') as log:
        result = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests'],
                                 cwd=CODE, env=env, stdout=log, stderr=log)
    state = {'tests_returncode': result, 'frozen_agent_unchanged': True,
             'deployment_ready_not_live': result == 0, 'formal_training': False,
             'attention_backward': 'deterministic'}
    (DEPLOY / 'state.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
