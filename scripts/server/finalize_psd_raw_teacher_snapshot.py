"""Finish v18 from the rejected v17 candidate; never modify a live snapshot."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/volume/ybo/wza')
OLD=ROOT/'training-artifacts/psd-raw-teacher-20260916-v17'
DEPLOY=ROOT/'training-artifacts/psd-raw-teacher-20260916-v18'
CODE=DEPLOY/'code'


def main():
    os.umask(0o077)
    assert json.loads((OLD/'state.json').read_text())['deployment_ready_not_live'] is False
    assert not CODE.exists()
    shutil.copytree(OLD/'code', CODE, ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name,parent in [('test_psd_modality.py','training/tests'),
        ('psd_multimodal_smoke.py','training/scripts/probe'),
        ('psd_training_entrypoint_smoke.py','training/scripts/probe')]:
        shutil.copy2(DEPLOY/name,CODE/parent/name)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (DEPLOY/'code-binding.json').write_text(json.dumps({str(p.relative_to(CODE)):sha(p)
        for p in CODE.rglob('*') if p.is_file()},indent=2))
    env={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'training'),
        'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TMPDIR':str(ROOT/'tmp')}
    with (DEPLOY/'tests.log').open('x') as log:
        code=subprocess.call([sys.executable,'-m','pytest','-q','training/tests','--tb=short'],
            cwd=CODE,env=env,stdout=log,stderr=log)
    result={'deployment_ready_not_live':code==0,'tests_returncode':code,'formal_training':False,
        'teacher_logprobs':'pre_grammar_unprocessed_v1'}
    (DEPLOY/'state.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return code


if __name__=='__main__':raise SystemExit(main())
