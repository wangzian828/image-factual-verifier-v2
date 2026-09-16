"""Separate mechanical-resume and full raw-teacher integration snapshots."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/volume/ybo/wza')
DROP=ROOT/'training-artifacts/psd-fsdp-resume-bridge-20260916-v19'


def main():
    os.umask(0o077)
    for name,source in [
        ('psd-fsdp-resume-bridge-20260916-v19','psd-deterministic-training-20260916-v16'),
        ('psd-raw-teacher-resume-20260916-v20','psd-raw-teacher-20260916-v18')]:
        deploy=ROOT/'training-artifacts'/name;deploy.mkdir(exist_ok=True)
        code=deploy/'code';assert not code.exists()
        shutil.copytree(ROOT/'training-artifacts'/source/'code',code,
            ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        for file,parent in [('psd_fsdp_resume.py','training/ifv_training'),
            ('ifv_psd_topk_plugin.py','training/plugins'),('test_psd_fsdp_resume.py','training/tests')]:
            shutil.copy2(DROP/file,code/parent/file)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        (deploy/'code-binding.json').write_text(json.dumps({str(p.relative_to(code)):sha(p)
            for p in code.rglob('*') if p.is_file()},indent=2))
        env={**os.environ,'PYTHONPATH':str(code)+':'+str(code/'training'),
            'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TMPDIR':str(ROOT/'tmp')}
        with (deploy/'tests.log').open('x') as log:
            result=subprocess.call([sys.executable,'-m','pytest','-q','training/tests','--tb=short'],
                cwd=code,env=env,stdout=log,stderr=log)
        state={'deployment_ready_not_live':result==0,'tests_returncode':result,'formal_training':False,
            'purpose':'legacy_bank_resume_mechanics_only' if name.endswith('v19') else 'full_raw_teacher_candidate'}
        (deploy/'state.json').write_text(json.dumps(state,indent=2));print(name,json.dumps(state),flush=True)
        if result:return result
    return 0


if __name__=='__main__':raise SystemExit(main())
