"""Build a tested immutable snapshot fixing the already-imported capture alias."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
OUT = ROOT / 'training-artifacts/psd-capture-callsite-20260916-v21'


def main():
    os.umask(0o077)
    code = OUT / 'code'
    assert not code.exists()
    shutil.copytree(ROOT / 'training-artifacts/psd-raw-teacher-resume-20260916-v20/code', code,
        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, dest in [('psd_capture_semantics.py', 'training/ifv_training'),
                       ('test_psd_raw_logprobs.py', 'training/tests')]:
        shutil.copy2(OUT / name, code / dest / name)
    hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code.rglob('*') if p.is_file()}
    (OUT / 'code-binding.json').write_text(json.dumps(hashes, indent=2))
    env = {**os.environ, 'PYTHONPATH': str(code) + ':' + str(code / 'training'),
        'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(ROOT / 'tmp')}
    with (OUT / 'tests.log').open('x') as log:
        result = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests', '--tb=short'],
            cwd=code, env=env, stdout=log, stderr=log)
    state = {'deployment_ready_not_live': result == 0, 'tests_returncode': result, 'formal_training': False}
    (OUT / 'state.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state)); return result


if __name__ == '__main__': raise SystemExit(main())
