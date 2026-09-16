"""Lease owned GPU2 for full-request numerical diagnostics, then restore it."""
import argparse
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request
import uuid

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
CODE = ROOT / 'training-artifacts/psd-raw-teacher-20260916-v18/code'
DROP = Path(__file__).resolve().parent


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('inspect', 'launch', 'execute'))
    p.add_argument('--output-name', default='nan-boundary-v1')
    p.add_argument('--repetitions', type=int, default=3)
    args = p.parse_args()
    assert Path(args.output_name).name == args.output_name and args.output_name.startswith('nan-')
    assert 1 <= args.repetitions <= 100
    out = SERVICE / args.output_name
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if args.mode == 'inspect':
        for i in range(4):
            receipt = owner.load(SERVICE / f'replica-{i}.json'); env = owner.checked(receipt)
            print(json.dumps({'gpu': i, 'pid': receipt['pid'], 'cache_environment': {k: env.get(k) for k in (
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'XDG_CACHE_HOME',
                'FLASHINFER_WORKSPACE_BASE', 'PYTHONPATH')}}))
        return
    if args.mode == 'launch':
        out.mkdir(exist_ok=False)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute',
            '--output-name', args.output_name, '--repetitions', str(args.repetitions)]
        receipt = owner.spawn(command, os.environ.copy(), out / 'controller.log')
        owner.save(out / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(out)})); return
    owner.verify_export()
    guard = owner.load(SERVICE / 'guard.json'); owner.checked(guard)
    backend = owner.load(SERVICE / 'replica-2.json'); original_env = owner.checked(backend)
    assert original_env['CUDA_VISIBLE_DEVICES'] == '2'
    assert backend['command'][backend['command'].index('--worker-cls') + 1] == 'scripts.server.psd_raw_teacher_worker.RawTeacherWorker'
    owner.save(out / 'binding.json', {'backend': backend, 'formal_training': False,
        'diagnostic_changes': ['layer finiteness hooks and synchronization', 'fresh per-worker JIT caches', 'request nonce'],
        'source_scope': 'reconstructed archived requests, not exact wire bytes',
        'files': {name: owner.sha(DROP/name) for name in ('psd_nan_boundary_capture.py', 'psd_diagnostic_raw_worker.py', Path(__file__).name)}})
    stopped = False; diagnostic = None; paused = False
    try:
        os.kill(guard['pid'], signal.SIGSTOP); paused = True
        for port in (19019, 19022):
            assert all(r['inflight'] == 0 for r in owner.http(f'http://127.0.0.1:{port}/health')['replicas'])
        for _ in range(90):
            metrics = urllib.request.urlopen('http://127.0.0.1:19004/metrics', timeout=5).read().decode()
            values = [float(s.rsplit(' ', 1)[1]) for s in metrics.splitlines()
                if s.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
            assert len(values) == 2
            if sum(values) == 0: break
            time.sleep(1)
        else: raise RuntimeError('Owned GPU2 did not drain')
        owner.stop(backend); stopped = True
        command = backend['command'][:]
        command[command.index('--worker-cls')+1] = 'psd_diagnostic_raw_worker.DiagnosticRawWorker'
        cache = out / 'cache'
        env = {**original_env, 'PYTHONPATH': str(DROP)+':'+str(CODE), 'PYTHONDONTWRITEBYTECODE': '1',
            'PSD_PREFILL_DIAGNOSTIC_DIR': str(out/'boundaries'), 'TMPDIR': str(ROOT/'tmp'),
            'TRITON_CACHE_DIR': str(cache/'triton'), 'TORCHINDUCTOR_CACHE_DIR': str(cache/'inductor'),
            'VLLM_CACHE_ROOT': str(cache/'vllm'), 'XDG_CACHE_HOME': str(cache),
            'FLASHINFER_WORKSPACE_BASE': str(cache/'flashinfer'), 'TORCH_EXTENSIONS_DIR': str(cache/'extensions')}
        diagnostic = owner.spawn(command, env, out/'backend.log')
        owner.save(SERVICE/'replica-2.json', diagnostic)
        owner.save(out/'diagnostic-process.json', diagnostic)
        owner.save(out/'state.json', {'phase': 'diagnostic_worker_starting', 'formal_training': False})
        # Other GPUs can keep their real idle workloads while GPU2 initializes.
        os.kill(guard['pid'], signal.SIGCONT); paused = False
        for _ in range(180):
            owner.checked(diagnostic)
            try:
                if owner.http('http://127.0.0.1:19004/v1/models')['data'][0]['id'] == 'ifv-psd-sft3084': break
            except OSError: pass
            time.sleep(5)
        else: raise RuntimeError('Diagnostic worker readiness deadline')
        results = []
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for iteration in range(args.repetitions):
            for source in sorted((SERVICE/'raw-multimage-failure-v1').glob('*/backend-body.json')):
                body = owner.load(source)
                body['cache_salt'] = 'full-request-diagnostic-'+uuid.uuid4().hex
                started = time.monotonic()
                owner.save(out/'state.json', {'phase': 'full_request', 'case': source.parent.name, 'iteration': iteration,
                    'started': time.time(), 'formal_training': False})
                request = urllib.request.Request('http://127.0.0.1:19004/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with opener.open(request, timeout=1260) as response: raw = response.read()
                with gzip.open(out/f'{iteration}-{source.parent.name}.json.gz', 'wb') as f: f.write(raw)
                payload = json.loads(raw); choice = payload['choices'][0]
                rows = (choice.get('logprobs') or {}).get('content') or []
                invalid = sum(not isinstance(x.get('logprob'), (float, int)) or not math.isfinite(x['logprob'])
                    for row in rows for x in [row, *(row.get('top_logprobs') or [])])
                assert rows and not invalid
                record = {'case': source.parent.name, 'iteration': iteration, 'seconds': time.monotonic()-started,
                    'tokens': len(rows), 'finish_reason': choice.get('finish_reason'), 'invalid_logprobs': invalid,
                    'tool_calls': len(choice['message'].get('tool_calls') or []), 'usage': payload.get('usage')}
                results.append(record); owner.save(out/'results.json', results)
                print(json.dumps(record), flush=True)
                assert choice.get('finish_reason') in ('tool_calls', 'stop'), 'Incomplete diagnostic response'
        owner.save(out/'state.json', {'phase': 'diagnostic_requests_completed', 'formal_training': False,
            'requests': len(results), 'does_not_prove_nan_fixed': True})
    except BaseException as error:
        owner.save(out/'state.json', {'phase': 'diagnostic_failed', 'error_type': type(error).__name__,
            'error': str(error), 'formal_training': False})
        raise
    finally:
        if stopped:
            if not paused: os.kill(guard['pid'], signal.SIGSTOP); paused = True
            if diagnostic is not None:
                try: owner.stop(diagnostic)
                except (ProcessLookupError, FileNotFoundError): pass
            restored = owner.spawn(backend['command'], original_env, out/'restored-backend.log')
            owner.save(SERVICE/'replica-2.json', restored)
        if paused: os.kill(guard['pid'], signal.SIGCONT)
        owner.verify_export()


if __name__ == '__main__': main()
