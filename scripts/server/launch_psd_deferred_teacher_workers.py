"""Deploy the passed stock-worker Agent recipe to four owned, drained GPUs.

No vLLM package or sampling kernel patches; no formal collection starts here.
Failure restores all replaced backends. Old weights and results stay intact.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.request

from psd_eval_reference import deferred_teacher_command, stop_if_present

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-capture-callsite-20260916-v21/code'
SERVICE = ROOT/'inference/psd-sft3084-20260916'
OUT = SERVICE/'deferred-teacher-four-gpu-v1'
GATE = ROOT/'runs/psd-eval-style-agent-gate-20260916-v1'
PORT = 19025


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner', ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.load(GATE/'state.json')['passed']
    assert owner.load(GATE/'strict-audit.json')['passed']
    assert owner.load(SERVICE/'eval-style-agent-v1/state.json')['returncode'] == 0
    if args.mode == 'launch':
        OUT.mkdir(exist_ok=False)
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], os.environ.copy(), OUT/'run.log')
        owner.save(OUT/'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)})); return
    owner.verify_export()
    for rel, digest in owner.load(CODE.parent/'code-binding.json').items():
        assert owner.sha(CODE/rel) == digest
    prior_controller = owner.load(SERVICE/'eval-style-agent-v1/process.json')
    path = Path(f"/proc/{prior_controller['pid']}/cmdline")
    assert not path.exists() or not path.read_bytes(), 'Single-GPU controller still owns its lease'
    old = [owner.load(SERVICE/f'replica-{i}.json') for i in range(4)]
    environments = [owner.checked(receipt) for receipt in old]
    for i, env in enumerate(environments): assert env['CUDA_VISIBLE_DEVICES'] == str(i)
    guard = owner.load(SERVICE/'guard.json'); owner.checked(guard)
    previous_gateway = owner.load(SERVICE/'four-gpu-v7/gateway.json')
    gateway_env = owner.checked(previous_gateway)
    commands = [deferred_teacher_command(receipt['command'], CODE) for receipt in old]
    assert commands[2] == owner.load(SERVICE/'eval-style-agent-v1/binding.json')['candidate']
    with socket.socket() as sock: sock.bind(('127.0.0.1', PORT))
    owner.save(OUT/'binding.json', {'before': old, 'candidate_commands': commands,
        'full_agent_gate_sha256': owner.sha(GATE/'strict-audit.json'),
        'code_binding_sha256': owner.sha(CODE.parent/'code-binding.json'),
        'teacher_scoring': 'deferred_exact_tokens_multimodal_frozen_transformers',
        'sampling_top20_is_not_teacher': True, 'formal_training': False})
    stopped = [False]*4
    backends = [None]*4
    gateway = None
    paused = False

    def replace(i):
        owner.stop(old[i]); stopped[i] = True
        cache = OUT/f'cache-{i}'
        env = {**environments[i], 'PYTHONPATH': str(CODE), 'PYTHONDONTWRITEBYTECODE': '1',
            'TMPDIR': str(ROOT/'tmp'), 'XDG_CACHE_HOME': str(cache),
            'TRITON_CACHE_DIR': str(cache/'triton'), 'TORCHINDUCTOR_CACHE_DIR': str(cache/'inductor'),
            'VLLM_CACHE_ROOT': str(cache/'vllm'), 'FLASHINFER_WORKSPACE_BASE': str(cache/'flashinfer'),
            'TORCH_EXTENSIONS_DIR': str(cache/'extensions')}
        for key in list(env):
            if key.startswith('PSD_'): env.pop(key)
        backends[i] = owner.spawn(commands[i], env, OUT/f'backend-{i}.log')
        owner.save(SERVICE/f'replica-{i}.json', backends[i])
        owner.save(OUT/f'replica-{i}.json', backends[i])

    try:
        os.kill(guard['pid'], signal.SIGSTOP); paused = True
        for port in (19019, 19022):
            assert all(r['inflight'] == 0 for r in owner.http(f'http://127.0.0.1:{port}/health')['replicas'])
        for _ in range(90):
            busy = 0
            for i in range(4):
                with urllib.request.urlopen(f'http://127.0.0.1:{19002+i}/metrics', timeout=5) as response:
                    metrics = response.read().decode()
                values = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
                    if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
                assert len(values) == 2; busy += sum(values)
            if not busy: break
            time.sleep(1)
        else: raise RuntimeError('Owned inference did not drain; no replacement permitted')
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(replace, range(4)))
        os.kill(guard['pid'], signal.SIGCONT); paused = False
        owner.save(OUT/'state.json', {'phase': 'loading_stock_workers', 'formal_training': False})
        for _ in range(240):
            for receipt in backends: owner.checked(receipt)
            try:
                for i in range(4):
                    card = owner.http(f'http://127.0.0.1:{19002+i}/v1/models')['data'][0]
                    assert card['root'] == commands[i][3] and card['max_model_len'] == 131072
                break
            except OSError: time.sleep(5)
        else: raise RuntimeError('Stock worker readiness deadline exceeded')
        command = previous_gateway['command'][:]
        command[command.index('--app-dir')+1] = str(CODE)
        command[command.index('--port')+1] = str(PORT)
        for key in list(gateway_env):
            if key.startswith(('PSD_RAW_', 'PSD_POLICY_LOGPROB', 'PSD_WIRE_', 'PSD_DIAGNOSTIC_')):
                gateway_env.pop(key)
        gateway_env.update(PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
            QWEN_REPLICA_BACKENDS=','.join(f'http://127.0.0.1:{19002+i}' for i in range(4)))
        gateway = owner.spawn(command, gateway_env, OUT/'gateway.log')
        owner.save(OUT/'gateway.json', gateway)
        for _ in range(45):
            owner.checked(gateway)
            try:
                replicas = owner.http(f'http://127.0.0.1:{PORT}/health')['replicas']
                if len(replicas) == 4 and all(r['healthy'] for r in replicas): break
            except OSError: pass
            time.sleep(1)
        else: raise RuntimeError('Deferred gateway not healthy')
        owner.save(OUT/'state.json', {'phase': 'ready_for_concurrency_validation', 'formal_training': False,
            'gateway': f'http://127.0.0.1:{PORT}', 'teacher_scoring': 'deferred_frozen_forward',
            'custom_worker': False, 'full_collection_started': False})
    except BaseException as error:
        if not paused: os.kill(guard['pid'], signal.SIGSTOP); paused = True
        failures = []
        if gateway is not None:
            try: stop_if_present(owner, gateway)
            except Exception as restore_error: failures.append(type(restore_error).__name__)
        for i in range(4):
            if not stopped[i]: continue
            try:
                if backends[i] is not None: stop_if_present(owner, backends[i])
                receipt = owner.spawn(old[i]['command'], environments[i], OUT/f'restore-{i}.log')
                owner.save(SERVICE/f'replica-{i}.json', receipt)
            except Exception as restore_error: failures.append(f'gpu{i}:{type(restore_error).__name__}')
        owner.save(OUT/'state.json', {'phase': 'requires_fix', 'error_type': type(error).__name__,
            'error': str(error), 'restore_errors': failures, 'formal_training': False})
        raise
    finally:
        if paused: os.kill(guard['pid'], signal.SIGCONT)
        owner.verify_export()


if __name__ == '__main__': main()
