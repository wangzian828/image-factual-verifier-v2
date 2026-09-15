"""One-card serving-only cache ablation. Preserve all prior receipts and weights."""
from __future__ import annotations
import asyncio
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916'
OUT = SERVICE/'no-apc-backend-probe-v1'
REQUEST = ROOT/'runs/psd-real-runtime-gate-20260916-v2/failed-request-diagnostic/request.json'


def save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def process(receipt):
    old = json.loads(receipt.read_text())
    pid = old['pid']
    args = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    assert args == old['command'] and os.getpgid(pid) == pid
    env = dict(x.decode().split('=', 1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if x)
    return old, args, env


def stop(pid):
    os.killpg(pid, signal.SIGTERM)
    for _ in range(120):
        stat = Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z':
            return
        time.sleep(0.5)
    raise RuntimeError('Owned service did not stop gracefully')


def spawn(args, env, log_path):
    with log_path.open('ab') as log:
        return subprocess.Popen(args, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


async def diagnose():
    import httpx
    base = json.loads(REQUEST.read_text())
    base['model'] = 'ifv-psd-sft2056-safety'
    base.pop('thinking_token_budget')
    base['max_tokens'] = 256
    # Raw diagnostic only: no budget mask, no salt. Check first-token corruption
    # without wasting another 32k generation. This is NOT a normal Agent success.
    base.pop('vllm_xargs', None)
    base.pop('cache_salt', None)
    save(OUT/'request.json', base)
    results = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for label, port in [('no_apc_gpu2', 19004), ('apc_gpu1_control', 19003)]:
            started = time.monotonic()
            response = await client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=copy.deepcopy(base))
            (OUT/(label+'-response.json')).write_bytes(response.content)
            response.raise_for_status()
            choice = response.json()['choices'][0]
            tokens = choice.get('token_ids') or []
            result = {'variant': label, 'port': port, 'seconds': time.monotonic()-started,
                      'finish': choice['finish_reason'], 'tokens': len(tokens),
                      'zero_tokens': tokens.count(0), 'first_tokens': tokens[:20],
                      'tools': choice['message'].get('tool_calls')}
            save(OUT/(label+'-summary.json'), result)
            results.append(result)
    save(OUT/'diagnostic-summary.json', {'results': results, 'full_runtime_gate_passed': False})


def main():
    assert REQUEST.is_file()
    assert all(x['inflight'] == 0 for x in get('http://127.0.0.1:19001/health')['replicas'])
    guard, guard_args, guard_env = process(SERVICE/'guard.json')
    old, args, env = process(SERVICE/'replica-2.json')
    assert env['CUDA_VISIBLE_DEVICES'] == '2' and args[args.index('--port')+1] == '19004'
    assert '--enable-prefix-caching' in args and args[args.index('--mamba-cache-mode')+1] == 'align'
    OUT.mkdir()
    save(OUT/'backend-before.json', old)
    save(OUT/'guard-before.json', guard)
    paused = False
    try:
        stop(guard['pid'])
        paused = True
        # Guard requests can still be completing after its client process exits.
        for _ in range(60):
            with urllib.request.urlopen('http://127.0.0.1:19004/metrics', timeout=5) as response:
                metrics = response.read().decode()
            active = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
                      if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
            assert len(active) == 2
            if sum(active) == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Backend did not drain')
        stop(old['pid'])
        args[args.index('--enable-prefix-caching')] = '--no-enable-prefix-caching'
        args[args.index('--mamba-cache-mode')+1] = 'none'
        backend = spawn(args, env, OUT/'backend.log')
        save(SERVICE/'replica-2.json', {'pid': backend.pid, 'command': args, 'started_at': time.time(),
                                      'replaces': old['pid'], 'prefix_caching': False})
        save(OUT/'state.json', {'phase': 'waiting_readiness', 'pid': backend.pid})
        for _ in range(180):
            if backend.poll() is not None:
                raise RuntimeError('Ablation backend exited; inspect backend.log')
            try:
                card = get('http://127.0.0.1:19004/v1/models')['data'][0]
                assert card['id'] == 'ifv-psd-sft2056-safety' and card['max_model_len'] == 131072
                break
            except OSError:
                time.sleep(5)
        else:
            raise RuntimeError('Ablation backend readiness timed out')
        state = json.loads((SERVICE/'state.json').read_text())
        state.update(phase='single_backend_cache_ablation', gpu_verified=False,
                     source_collection_started=False,
                     backend_overrides={'19004': {'prefix_caching': False,
                                                 'mamba_cache_mode': 'none'}},
                     cache_ablation=str(OUT))
        save(SERVICE/'state.json', state)
        asyncio.run(diagnose())
        save(OUT/'state.json', {'phase': 'finished_requires_inspection', 'time': time.time()})
    except BaseException as error:
        save(OUT/'failure.json', {'type': type(error).__name__, 'message': str(error), 'time': time.time()})
        raise
    finally:
        if paused:
            guard_process = spawn(guard_args, guard_env, SERVICE/'guard.log')
            save(SERVICE/'guard.json', {'pid': guard_process.pid, 'command': guard_args,
                                       'started_at': time.time(), 'replaces': guard['pid']})
            print('GUARD_RESTORED', guard_process.pid, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    options = parser.parse_args()
    if options.launch:
        # Persist the outer controller before returning from SSH. Existing
        # receipts refuse a duplicate launch, including ambiguous prior starts.
        receipt = SERVICE/'no-apc-probe-process.json'
        assert not OUT.exists() and not receipt.exists()
        with receipt.open('x') as reserved:
            json.dump({'phase': 'launch_reserved', 'time': time.time()}, reserved)
        command = [sys.executable, '-u', str(Path(__file__).resolve())]
        child = spawn(command, os.environ.copy(), SERVICE/'no-apc-probe.log')
        save(receipt, {'pid': child.pid, 'command': command, 'time': time.time(),
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print('NO_APC_PROBE_LAUNCHED', child.pid, flush=True)
    else:
        main()
