"""Stage isolated PSD position-constraint regression; no provider or GPU update."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-position-audit-20260916-v13'
OLD = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10/source-review-v2-code'
CODE = DEPLOY / 'code'
OVERLAYS = {'psd_slate.py':'training/ifv_training', 'run_psd_repair_driver.py':'scripts',
            'test_psd_slate_bound_positions.py':'training/tests',
            'friend_generation_campaign.py':'scripts/server',
            'test_friend_generation_campaign.py':'tests'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == '__main__':
    os.umask(0o077)
    if CODE.exists():
        raise RuntimeError('Immutable candidate already staged')
    for name in OVERLAYS:
        assert (DEPLOY / name).is_file()
    shutil.copytree(OLD,CODE,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name,parent in OVERLAYS.items():
        shutil.copy2(DEPLOY/name,CODE/parent/name)
    for p in (OLD/'src').rglob('*.py'):
        assert sha(p)==sha(CODE/p.relative_to(OLD))
    binding={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')}
    (DEPLOY/'code-binding.json').write_text(json.dumps(binding,indent=2))
    env={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'training'),
         'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(ROOT/'tmp')}
    with (DEPLOY/'tests.log').open('x') as log:
        code=subprocess.call([sys.executable,'-m','pytest','-q','training/tests',
                              'tests/test_friend_generation_campaign.py'],cwd=CODE,env=env,stdout=log,stderr=log)
    result={'tests_returncode':code,'frozen_agent_unchanged':True,'provider_calls':0,
            'training_started':False,'deployment_ready_not_live':code==0}
    (DEPLOY/'state.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    raise SystemExit(code)
