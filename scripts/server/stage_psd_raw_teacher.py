"""Stage the raw-teacher candidate independently from the active FSDP gate."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-raw-teacher-20260916-v17'
OLD = ROOT / 'training-artifacts/psd-deterministic-training-20260916-v16/code'
CODE = DEPLOY / 'code'
GROUPS = {
    'training/ifv_training': ['psd_capture_semantics.py', 'psd_candidates.py', 'psd_repairs.py',
        'psd_repair_runtime.py', 'psd.py', 'psd_topk.py', 'psd_datums.py', 'psd_preflight.py'],
    'scripts': ['collect_psd_rollouts.py'],
    'scripts/server': ['psd_raw_logprobs.py', 'psd_raw_teacher_worker.py', 'psd_qwen_gateway.py'],
    'training/tests': ['test_psd_raw_logprobs.py', 'test_psd.py', 'test_psd_datums.py',
        'test_psd_preflight.py', 'test_psd_compact_datums.py', 'test_psd_media.py'],
}


def main():
    os.umask(0o077)
    assert not CODE.exists()
    for names in GROUPS.values():
        for name in names: assert (DEPLOY / name).is_file()
    shutil.copytree(OLD, CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for parent, names in GROUPS.items():
        for name in names: shutil.copy2(DEPLOY / name, CODE / parent / name)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (OLD / 'src').rglob('*.py'):
        assert sha(p) == sha(CODE / p.relative_to(OLD)), 'Frozen Agent changed'
    (DEPLOY / 'code-binding.json').write_text(json.dumps(
        {str(p.relative_to(CODE)): sha(p) for p in CODE.rglob('*') if p.is_file()}, indent=2))
    env = {**os.environ, 'PYTHONPATH': str(CODE) + ':' + str(CODE / 'training'),
        'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT / 'tmp'), 'CUDA_VISIBLE_DEVICES': '',
        'XDG_CACHE_HOME': str(ROOT / 'cache/psd-training-gate')}
    with (DEPLOY / 'tests.log').open('x') as log:
        code = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests', '--tb=short'],
            cwd=CODE, env=env, stdout=log, stderr=log)
    result = {'deployment_ready_not_live': code == 0, 'tests_returncode': code,
        'formal_training': False, 'teacher_logprobs': 'pre_grammar_unprocessed_v1'}
    (DEPLOY / 'state.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result)); return code


if __name__ == '__main__':
    raise SystemExit(main())
