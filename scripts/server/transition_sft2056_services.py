"""Scoped, drained service transitions; no server git or checkpoint mutation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from benchmark_sft2056_serving import ALIAS, EXPORT, ROOT, WORK, save
from benchmark_sft2056_fixed_tokens import metrics

ARTIFACT = ROOT/'training-artifacts/sft2056-evaluation-20260915'
RUN = ROOT/'inference/sft2056-epoch2-apc-20260915'
MODEL_ALIAS = 'ifv-qwen3.5-9b-sft-2056'
PYTHON = ROOT/'envs/h20-qwen35-128k/bin/python'


def drain_stop(record_path):
    import requests
    record_path.resolve().relative_to(WORK)
    record = json.loads(record_path.read_text())
    pid = record['pid']
    assert pid > 1 and os.getpgid(pid) == pid
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    command = [x.decode() for x in command if x]
    assert str(EXPORT/'model') in command and ALIAS in command and 'serve' in command
    port = int(command[command.index('--port')+1])
    for _ in range(60):
        state = metrics(requests.get(f'http://127.0.0.1:{port}/metrics', timeout=10).text)
        if all(state.get(k) == 0 for k in ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Benchmark requests are not drained; refusing to stop the service')
    save(record_path.parent/'drained-stop.json', {'pid': pid, 'port': port, 'time': time.time(), 'idle': True})
    os.killpg(pid, signal.SIGTERM)
    for _ in range(120):
        try:
            status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0]
            if status == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError('Service did not exit gracefully; inspect instead of forcing')
    return record


def apc16():
    output = WORK/'apc-s16-b32k'
    if output.exists():
        raise ValueError('APC16 already attempted')
    completed = json.loads((WORK/'fixed-token-v1/apc-s8-b32k/summary.json').read_text())
    assert len(completed['passes']) == 3 and all(p['passed'] for p in completed['passes'])
    record = drain_stop(WORK/'apc-s8-b32k/server.json')
    output.mkdir()
    command = record['command']
    command[command.index('--max-num-seqs')+1] = '16'
    short_tmp = ROOT/'tmp/ab2056-g3-16'
    short_tmp.mkdir(exist_ok=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '3',
        'HF_HOME': str(ROOT/'cache/h20-huggingface'), 'XDG_CACHE_HOME': str(output/'cache'),
        'FLASHINFER_WORKSPACE_DIR': str(output/'cache/flashinfer'), 'TMPDIR': str(short_tmp),
        'VLLM_USE_FLASHINFER_SAMPLER': '0', 'LD_LIBRARY_PATH': str(ROOT/'envs/h20-qwen35-128k/lib')}
    with (output/'server.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT/'image-factual-verifier-v2', env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(output/'server.json', {'pid': process.pid, 'command': command, 'time': time.time()})
    print('APC16_LAUNCHED', process.pid, flush=True)


def formal(seqs):
    if RUN.exists():
        raise ValueError('Formal service directory already exists')
    baseline = json.loads((WORK/'fixed-token-v1/finished.json').read_text())
    assert all(p['passed'] for r in baseline['results'] for p in r['passes'])
    if seqs == 16:
        candidate = json.loads((WORK/'fixed-token-apc16/finished.json').read_text())
        assert all(p['passed'] for r in candidate['results'] for p in r['passes'])
    guard_record = WORK/'idle-guard/process.json'
    if guard_record.exists():
        pid = json.loads(guard_record.read_text())['pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        assert str(ARTIFACT/'gpu_utilization_guard.py') in command and os.getpgid(pid) == pid
        os.killpg(pid, signal.SIGTERM)
        time.sleep(3)
    # All four GPU-specific records were created by the isolated benchmark.
    for rel in ('baseline-s8-b32k/server-r2.json', 's16-b32k/server.json',
                's16-b8k/server.json', 'apc-s16-b32k/server.json'):
        drain_stop(WORK/rel)
    env = {**os.environ, 'IFV_QWEN_MODEL': str(EXPORT/'model'), 'IFV_QWEN_MODEL_ALIAS': MODEL_ALIAS,
           'IFV_QWEN_RUN_ROOT': str(RUN), 'IFV_QWEN_MAX_NUM_SEQS': str(seqs),
           'IFV_QWEN_CACHE_ROOT': str(WORK/'apc-s16-b32k/cache'),
           'IFV_QWEN_MAX_NUM_BATCHED_TOKENS': '32768', 'IFV_QWEN_ENABLE_PREFIX_CACHING': 'true',
           'IFV_QWEN_THINKING_ENABLED': 'true', 'IFV_QWEN_TMPDIR': str(ROOT/'tmp/sft2056')}
    subprocess.run(['bash', str(ARTIFACT/'start_qwen35_base_replicas.sh'), 'start'], env=env, check=True)
    save(RUN/'acceleration.json', {'apc': True, 'mamba_cache_mode': 'align', 'max_num_seqs': seqs,
         'max_num_batched_tokens': 32768, 'gateway': 'unchanged least-inflight',
         'benchmark': str(WORK), 'formal_budget': 32768, 'thinking_budget': 8192, 'time': time.time()})


def guard(benchmark=False):
    import requests
    directory = (WORK if benchmark else RUN)/'idle-guard'
    if directory.exists():
        record = json.loads((directory/'process.json').read_text())
        command = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().decode()
        assert str(ARTIFACT/'gpu_utilization_guard.py') in command
        assert str(directory/'state.json') in command
        print('GUARD_ALREADY_RUNNING', record['pid'], flush=True)
        return
    ports = (18902, 18903, 18904) if benchmark else (8902, 8903, 8904, 8905)
    alias = ALIAS if benchmark else MODEL_ALIAS
    for port in ports:
        response = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5)
        response.raise_for_status()
        assert response.json()['data'][0]['id'] == alias
    directory.mkdir()
    command = [str(PYTHON), str(ARTIFACT/'gpu_utilization_guard.py'),
               '--state-file', str(directory/'state.json'), '--log-file', str(directory/'samples.jsonl'),
               '--poll-seconds', '5', '--low-window-seconds', '30', '--model', alias,
               '--gpu-ids', ','.join(str(i) for i in range(len(ports))),
               '--vllm-ports', ','.join(str(p) for p in ports),
               '--pulse-tokens', '2048', '--keeper-script',
               str(ROOT/'image-factual-verifier-v2/training/scripts/h20/gpu_memory_keeper.sh')]
    env = {**os.environ, 'IFV_GPU_KEEPER_GIB': '64', 'IFV_GPU_KEEPER_DUTY_CYCLE': '0.35',
           'IFV_GPU_KEEPER_RUN_ROOT': str(directory/'matrix-workers')}
    with (directory/'process.log').open('xb') as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(directory/'process.json', {'pid': process.pid, 'time': time.time(), 'command': command,
         'note': 'Stop this guard and its scoped matrix workers before any future service/training transition.'})
    print('GUARD_LAUNCHED', process.pid, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['apc16', 'formal', 'guard', 'bench-guard'])
    parser.add_argument('--max-seqs', type=int, choices=[8, 16], default=16)
    args = parser.parse_args()
    os.umask(0o077)
    if args.stage == 'apc16':
        apc16()
    elif args.stage == 'formal':
        formal(args.max_seqs)
    else:
        guard(benchmark=args.stage == 'bench-guard')
