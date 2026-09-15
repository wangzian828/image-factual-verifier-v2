"""Launch the authorized smoke -> engineering audit -> full Agent evaluation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import run_sft2056_eval
import run_sft3084_eval as controller
from benchmark_sft2056_serving import save

ARTIFACT = Path(__file__).resolve().parent
OUTPUT = controller.OUTPUT


def command(script, *args):
    subprocess.run([sys.executable, str(ARTIFACT/script), *args], check=True,
                   cwd=controller.REPO, env={**os.environ, 'PYTHONPATH': str(controller.REPO)})


def pipeline():
    try:
        save(OUTPUT/'pipeline-state.json', {'phase': 'smoke', 'time': time.time()})
        command('run_sft2056_eval.py', '--stage', 'smoke')
        command('transition_sft2056_services.py', 'guard')
        save(OUTPUT/'pipeline-state.json', {'phase': 'engineering_audit', 'time': time.time()})
        command('audit_sft2056_smoke.py', 'multiimage')
        command('audit_sft2056_smoke.py', 'audit')
        save(OUTPUT/'pipeline-state.json', {'phase': 'full_inference', 'time': time.time()})
        command('run_sft2056_eval.py', '--stage', 'full')
        save(OUTPUT/'pipeline-state.json', {'phase': 'controller_finished_check_progress', 'time': time.time()})
    except Exception as error:
        save(OUTPUT/'pipeline-state.json', {'phase': 'stopped_requires_inspection',
             'error_type': type(error).__name__, 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUTPUT/'pipeline.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                cwd=controller.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        save(OUTPUT/'pipeline-process.json', {'pid': process.pid, 'time': time.time()})
        print('PIPELINE_LAUNCHED', process.pid)
    else:
        pipeline()
