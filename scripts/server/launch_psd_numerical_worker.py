"""One drained owned GPU with eager numerical hooks, preserving library files."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
OUT = SERVICE / 'eager-numerical-worker-gpu3-v1'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'


def main():
    global OUT
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['launch', 'execute'])
    p.add_argument('--gpu', type=int, choices=range(4), default=3)
    p.add_argument('--worker', choices=('layers','boundary'), default='layers')
    args = p.parse_args()
    port=19002+args.gpu
    if args.worker=='boundary':
        OUT=SERVICE/f'numerical-boundary-gpu{args.gpu}-v1'
    elif args.gpu!=3:
        OUT=SERVICE/f'eager-numerical-worker-gpu{args.gpu}-v1'
    module='psd_boundary_worker' if args.worker=='boundary' else 'psd_nan_worker'
    worker=module+('.BoundaryWorker' if args.worker=='boundary' else '.DiagnosticWorker')
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.load(RUN / 'eager-failure-hold-v1/state.json')['completed_results_preserved']
    owner.verify_export()
    if args.mode == 'launch':
        OUT.mkdir(exist_ok=False)
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute',
                              '--gpu',str(args.gpu),'--worker',args.worker],
                              os.environ.copy(), OUT / 'run.log')
        owner.save(OUT / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)})); return
    guard = owner.load(SERVICE / 'guard.json'); owner.checked(guard)
    old = owner.load(SERVICE / f'replica-{args.gpu}.json'); env = owner.checked(old)
    assert env['CUDA_VISIBLE_DEVICES'] == str(args.gpu) and '--enforce-eager' in old['command']
    assert '--worker-cls' not in old['command']
    command = [*old['command'], '--worker-cls', worker]
    env.update(PYTHONPATH=str(DEPLOY) + ':' + env.get('PYTHONPATH', ''),
               PSD_NUMERICAL_DIAGNOSTIC_DIR=str(OUT / 'numerical'),
               VLLM_COMPUTE_NANS_IN_LOGITS='1')
    owner.save(OUT / 'before.json', {'backend': old, 'worker_source_sha256': owner.sha(DEPLOY / (module+'.py'))})
    os.kill(guard['pid'], signal.SIGSTOP)
    try:
        health = owner.http('http://127.0.0.1:19019/health')
        assert all(r['inflight'] == 0 for r in health['replicas'] if r['url'].endswith(':'+str(port)))
        for _ in range(90):
            text = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5).read().decode()
            values = [float(l.rsplit(' ', 1)[1]) for l in text.splitlines()
                      if l.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
            assert len(values) == 2
            if sum(values) == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Diagnostic GPU did not drain')
        owner.stop(old)
        receipt = owner.spawn(command, env, OUT / 'backend.log')
        owner.save(SERVICE / f'replica-{args.gpu}.json', receipt)
        owner.save(OUT / f'replica-{args.gpu}.json', receipt)
    finally:
        os.kill(guard['pid'], signal.SIGCONT)
    owner.save(OUT / 'state.json', {'phase': 'waiting_readiness', 'training_started': False})
    for _ in range(240):
        owner.checked(receipt)
        try:
            model = owner.http(f'http://127.0.0.1:{port}/v1/models')['data'][0]
            if model['root'] == old['command'][3] and model['id'] == 'ifv-psd-sft3084':
                owner.save(OUT / 'state.json', {'phase': 'ready_for_numerical_diagnosis',
                    'training_started': False, 'hooks_may_mask_timing_races': True})
                return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError('Diagnostic worker readiness failed; inspect before another restart')


if __name__ == '__main__':
    main()
