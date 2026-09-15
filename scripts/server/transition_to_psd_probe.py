"""Explicitly drained handoff from ended epoch2 attempts to isolated PSD probes."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
OLD = ROOT/'inference/sft2056-epoch2-apc-20260915'
EVAL = ROOT/'runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915'
RUN = ROOT/'inference/psd-sft2056-safety-20260916'
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916'
ENGINE = ROOT/'envs/h20-qwen35-vllm-0181'
PYTHON = ROOT/'envs/h20-qwen35-128k/bin/python'
EXPORT = ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915'
GUARD = ROOT/'training-artifacts/sft2056-evaluation-20260915/gpu_utilization_guard.py'
ALIAS = 'ifv-psd-sft2056-safety'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def command(pid):
    return [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def validate_group(pid, required):
    assert pid > 1 and os.getpgid(pid) == pid
    args = command(pid)
    assert all(value in args for value in required), (pid, required, args)
    return args


def stop(pid):
    os.killpg(pid, signal.SIGTERM)
    for _ in range(120):
        if not alive(pid):
            return
        time.sleep(0.5)
    raise RuntimeError('Process did not stop gracefully; no forced kill attempted')


def metrics(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
        text = response.read().decode()
    counts = {}
    for line in text.splitlines():
        for key in ['vllm:num_requests_running', 'vllm:num_requests_waiting']:
            if line.startswith(key+'{'):
                counts[key] = float(line.rsplit(' ', 1)[1])
    assert len(counts) == 2
    return counts


def start(args, name, env):
    with (RUN/(name+'.log')).open('xb') as log:
        process = subprocess.Popen(args, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(RUN/(name+'.json'), {'pid': process.pid, 'command': args, 'started_at': time.time()})
    return process.pid


def main(execute):
    progress = json.loads((EVAL/'progress.json').read_text())
    assert progress['phase'] in {'inference_complete', 'retry_budget_exhausted'}
    lock = (EVAL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipeline_pid = json.loads((EVAL/'pipeline-process.json').read_text())['pid']
    assert not alive(pipeline_pid), 'Evaluation pipeline is still running'
    assert hashlib.sha256((EXPORT/'export.json').read_bytes()).hexdigest() == '55dfb77f56cb175573c5e966816c8a0a0c384385190ae5b62795963f681fbd28'
    assert json.loads((ROOT/'evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916/summary.json').read_text())['successful'] == 1524
    guard_pid = json.loads((OLD/'idle-guard/process.json').read_text())['pid']
    validate_group(guard_pid, [str(GUARD), str(OLD/'idle-guard/state.json')])
    assert not (OLD/'idle-guard/matrix-workers').exists(), 'Inspect matrix workers before handoff'
    old = []
    for gpu in range(4):
        pid = int((OLD/'pids'/f'replica-{gpu}.pid').read_text())
        args = validate_group(pid, ['serve', str(EXPORT/'model'), 'ifv-qwen3.5-9b-sft-2056'])
        assert int(args[args.index('--port')+1]) == 8902+gpu
        old.append((pid, args))
    gateway_pid = int((OLD/'pids/gateway.pid').read_text())
    validate_group(gateway_pid, ['scripts.server.qwen_replica_gateway:app', '8901'])
    for port in range(19001, 19006):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    assert not RUN.exists(), 'Handoff already attempted; inspect receipts instead of rerunning'
    if not execute:
        print(json.dumps({'preflight': True, 'old_gpu_pids': [p for p,_ in old],
                          'guard_pid': guard_pid, 'gateway_pid': gateway_pid,
                          'new_alias': ALIAS, 'new_ports': list(range(19001,19006))}), flush=True)
        return
    RUN.mkdir()
    save(RUN/'transition.json', {'phase': 'stopping_guard', 'old_gpu_pids': [p for p,_ in old],
                               'old_guard': guard_pid, 'old_gateway': gateway_pid, 'time': time.time(),
                               'evaluation_status': progress, 'export': str(EXPORT/'export.json')})
    stop(guard_pid)
    for _ in range(60):
        if all(all(v == 0 for v in metrics(port).values()) for port in range(8902,8906)):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Queues not empty after stopping guard; old GPU services left running')
    save(RUN/'drained.json', {'ports': list(range(8902,8906)), 'time': time.time()})
    stop(gateway_pid)
    for pid, _ in old:
        stop(pid)
    env = {**os.environ, 'PYTHONPATH': str(CODE), 'HF_HOME': str(ROOT/'cache/h20-huggingface'),
           'XDG_CACHE_HOME': str(ROOT/'benchmarks/sft2056-serving-ab-20260915/apc-s16-b32k/cache'),
           'FLASHINFER_WORKSPACE_DIR': str(ROOT/'benchmarks/sft2056-serving-ab-20260915/apc-s16-b32k/cache/flashinfer'),
           'LD_LIBRARY_PATH': str(ROOT/'envs/h20-qwen35-128k/lib'), 'VLLM_USE_FLASHINFER_SAMPLER': '0',
           'PSD_GATEWAY_DEADLINE_SECONDS': '1200', 'AGENT_LLM_REQUEST_TIMEOUT_SECONDS': '1230',
           'AGENT_STAGE_REQUEST_TIMEOUT_SECONDS': '1260', 'AGENT_LLM_REQUEST_MAX_RETRIES': '0'}
    for gpu, (_, old_args) in enumerate(old):
        args = list(old_args)
        args[args.index('--port')+1] = str(19002+gpu)
        args[args.index('--served-model-name')+1] = ALIAS
        args += ['--logits-processors', 'scripts.server.psd_qwen_thinking:PSDThinkingBudget']
        tmp = ROOT/f'tmp/psd916-{gpu}'
        tmp.mkdir(exist_ok=True)
        start(args, f'replica-{gpu}', {**env, 'CUDA_VISIBLE_DEVICES': str(gpu), 'TMPDIR': str(tmp)})
    gateway_env = {**env, 'QWEN_REPLICA_BACKENDS': ','.join(f'http://127.0.0.1:{p}' for p in range(19002,19006)),
                   'QWEN_REPLICA_MODEL_ID': ALIAS}
    start([str(ENGINE/'bin/python'), '-m', 'uvicorn', 'scripts.server.psd_qwen_gateway:app',
           '--app-dir', str(CODE), '--host', '127.0.0.1', '--port', '19001', '--log-level', 'warning'],
          'gateway', gateway_env)
    save(RUN/'state.json', {'phase': 'waiting_readiness', 'time': time.time(), 'gpu_verified': False})
    deadline = time.monotonic()+900
    while time.monotonic() < deadline:
        try:
            for port in range(19002,19006):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    model = json.load(response)['data'][0]
                assert model['id'] == ALIAS and model['root'] == str(EXPORT/'model')
                assert model['max_model_len'] == 131072
            break
        except (OSError, ValueError, KeyError):
            time.sleep(5)
    else:
        save(RUN/'state.json', {'phase': 'readiness_failed', 'time': time.time(), 'gpu_verified': False})
        raise RuntimeError('Inspect new service logs; do not mark gate passed')
    guard_dir = RUN/'idle-guard'
    guard_dir.mkdir()
    guard_args = [str(PYTHON), str(GUARD), '--state-file', str(guard_dir/'state.json'),
                  '--log-file', str(guard_dir/'samples.jsonl'), '--poll-seconds', '5',
                  '--low-window-seconds', '30', '--model', ALIAS, '--gpu-ids', '0,1,2,3',
                  '--vllm-ports', '19002,19003,19004,19005', '--pulse-tokens', '2048',
                  '--keeper-script', str(ROOT/'image-factual-verifier-v2/training/scripts/h20/gpu_memory_keeper.sh')]
    guard_env = {**env, 'IFV_GPU_KEEPER_GIB': '64', 'IFV_GPU_KEEPER_DUTY_CYCLE': '0.35',
                 'IFV_GPU_KEEPER_RUN_ROOT': str(guard_dir/'matrix-workers')}
    start(guard_args, 'guard', guard_env)
    save(RUN/'state.json', {'phase': 'ready_for_boundary_probes', 'time': time.time(),
                          'alias': ALIAS, 'gpu_verified': False, 'source_policy_epoch': 2,
                          'source_policy_step': 2056, 'max_model_len': 131072,
                          'formal_thinking_budget': 8192, 'source_collection_started': False})
    print('ISOLATED_SERVICES_READY_NOT_GPU_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
