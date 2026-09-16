"""Restart only owned GPU1/3: NaN metric off on 1, unchanged on 3 (restart control).

Commands, weights, caches and generation semantics are preserved. Production
must be idle. Failure restores replaced services; no source retries occur here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'


def candidate_environment(original, gpu):
    if gpu not in (1, 3) or original.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Only the two identified owned replicas are in scope')
    if original.get('VLLM_COMPUTE_NANS_IN_LOGITS') != '1':
        raise ValueError('Original diagnostic flag no longer matches the binding')
    result = dict(original)
    if gpu == 1:
        result['VLLM_COMPUTE_NANS_IN_LOGITS'] = '0'
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    out = SERVICE / 'nan-metric-restart-ab-v1'
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    before_probe = SERVICE / 'nan-exhausted-before-v1'
    assert owner.load(before_probe / 'state.json')['phase'] == 'diagnostic_completed'
    assert owner.load(before_probe / 'summary.json')['completed'] == 16
    assert owner.load(ROOT / 'runs/psd-production400x8-20260917-v2/state.json')['phase'] == 'collection_requires_intervention'
    guard = owner.load(SERVICE / 'guard.json'); owner.checked(guard)
    old = {i: owner.load(SERVICE / f'replica-{i}.json') for i in (1, 3)}
    env = {i: owner.checked(old[i]) for i in old}
    new_env = {i: candidate_environment(env[i], i) for i in old}
    for replica in owner.http('http://127.0.0.1:19025/health')['replicas']:
        assert replica['inflight'] == 0
    owner.verify_export()
    if args.mode == 'launch':
        out.mkdir(exist_ok=False)
        owner.save(out / 'binding.json', {'before': old, 'only_environment_change':
            {'1': {'VLLM_COMPUTE_NANS_IN_LOGITS': '0'}, '3': {}},
            'same_cache_directories': True, 'restart_is_a_confound_controlled_on_gpu3': True,
            'code_sha256': owner.sha(Path(__file__)), 'diagnostic_only': True})
        process = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'],
                              os.environ.copy(), out / 'controller.log')
        owner.save(out / 'process.json', process)
        print(json.dumps({'pid': process['pid'], 'output': str(out)})); return
    binding = owner.load(out / 'binding.json')
    assert {str(i): r for i, r in old.items()} == binding['before']
    stopped = set(); candidates = {}; paused = False
    try:
        os.kill(guard['pid'], signal.SIGSTOP); paused = True
        for _ in range(90):
            busy = 0
            for gpu in old:
                raw = urllib.request.urlopen(f'http://127.0.0.1:{19002+gpu}/metrics', timeout=5).read().decode()
                values = [float(l.rsplit(' ', 1)[1]) for l in raw.splitlines()
                          if l.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
                assert len(values) == 2
                busy += sum(values)
            if not busy: break
            time.sleep(1)
        else: raise RuntimeError('Diagnostic replicas failed to drain')
        def replace(gpu):
            owner.stop(old[gpu]); stopped.add(gpu)
            receipt = owner.spawn(old[gpu]['command'], new_env[gpu], out / f'backend-{gpu}.log')
            candidates[gpu] = receipt
            owner.save(SERVICE / f'replica-{gpu}.json', receipt)
            owner.save(out / f'replica-{gpu}.json', receipt)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(replace, old))
        os.kill(guard['pid'], signal.SIGCONT); paused = False
        owner.save(out / 'state.json', {'phase': 'loading_candidates', 'formal_training': False})
        for _ in range(240):
            for receipt in candidates.values(): owner.checked(receipt)
            try:
                for gpu in candidates:
                    model = owner.http(f'http://127.0.0.1:{19002+gpu}/v1/models')['data'][0]
                    assert model['root'] == old[gpu]['command'][3] and model['max_model_len'] == 131072
                break
            except OSError: time.sleep(5)
        else: raise RuntimeError('Readiness timeout')
        owner.save(out / 'state.json', {'phase': 'ready_for_replay', 'formal_training': False,
                   'gpu0_gpu2_unchanged': True, 'source_ledgers_unchanged': True})
    except BaseException as error:
        failures = []
        if not paused: os.kill(guard['pid'], signal.SIGSTOP); paused = True
        for gpu in stopped:
            try:
                if gpu in candidates:
                    proc = Path(f'/proc/{candidates[gpu]["pid"]}/cmdline')
                    if proc.exists() and proc.read_bytes(): owner.stop(candidates[gpu])
                restored = owner.spawn(old[gpu]['command'], env[gpu], out / f'restored-{gpu}.log')
                owner.save(SERVICE / f'replica-{gpu}.json', restored)
            except Exception as exc: failures.append(type(exc).__name__)
        owner.save(out / 'state.json', {'phase': 'failed_restoration_attempted',
            'error_type': type(error).__name__, 'restoration_errors': failures})
        raise
    finally:
        if paused: os.kill(guard['pid'], signal.SIGCONT)
        owner.verify_export()


if __name__ == '__main__': main()
