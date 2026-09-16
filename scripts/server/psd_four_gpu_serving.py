"""Move owned idle replicas to the protected epoch-3 policy, without restarting GPU1.

New serving identity for the 400x8 stage. The bound, already collected canary
keeps its original gateway. All writes stay under /volume/ybo/wza.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-four-gpu-20260916-v7'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
OUT = SERVICE / 'four-gpu-v7'
CAPTURE = ROOT / 'training-artifacts/psd-epoch3-capture-20260916-v2'
MODEL = ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'
BACKEND = 'ifv-psd-sft3084'
ALIAS = 'ifv-qwen3.5-9b-sft-3084'
PORT = 19019
CONCURRENCY = 40
REPLACED = (0, 2, 3)
PORTS = (19002, 19003, 19004, 19005)


def owner():
    path = ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py'
    spec = importlib.util.spec_from_file_location('protected_epoch3_owner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replica_command(original, gpu):
    command = list(original)
    required = {'--served-model-name': BACKEND, '--max-model-len': '131072',
                '--tool-call-parser': 'ifv_psd_qwen3_single', '--mamba-cache-mode': 'none',
                '--gpu-memory-utilization': '0.94', '--max-num-seqs': '16'}
    if command[3] != str(MODEL) or '--no-enable-prefix-caching' not in command:
        raise ValueError('Template is not the validated epoch3 replica')
    for key, value in required.items():
        if command[command.index(key) + 1] != value:
            raise ValueError('Unexpected serving setting: ' + key)
    if gpu not in range(4):
        raise ValueError('Only the four owned GPUs are in scope')
    command[command.index('--port') + 1] = str(PORTS[gpu])
    return command


def gateway_environment(previous):
    env = dict(previous)
    env.update(PYTHONPATH=str(CAPTURE), QWEN_REPLICA_MODEL_ID=BACKEND,
               PSD_PUBLIC_MODEL_ALIAS=ALIAS,
               QWEN_REPLICA_BACKENDS=','.join(f'http://127.0.0.1:{port}' for port in PORTS))
    # Native PSD requests themselves request and archive token IDs + top20.
    # A second full wire archive would duplicate every image/logprob payload.
    env.pop('PSD_WIRE_CAPTURE_DIR', None)
    env.pop('PSD_DIAGNOSTIC_RETURN_TOKEN_IDS', None)
    return env


def phase(o, name, **extra):
    o.save(OUT / 'state.json', {'phase': name, 'time': time.time(),
                              'training_started': False, 'full_collection_started': False, **extra})


def idle(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
        lines = response.read().decode().splitlines()
    values = [float(line.rsplit(' ', 1)[1]) for line in lines
              if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    return len(values) == 2 and sum(values) == 0


def ready(o, receipt, gpu):
    for _ in range(240):
        o.checked(receipt)
        try:
            cards = o.http(f'http://127.0.0.1:{PORTS[gpu]}/v1/models')['data']
            if len(cards) == 1 and cards[0]['id'] == BACKEND and cards[0]['root'] == str(MODEL):
                assert cards[0]['max_model_len'] == 131072
                return cards[0]
        except OSError:
            pass
        time.sleep(3)
    raise RuntimeError('Replica readiness deadline: GPU' + str(gpu))


def smoke(o):
    """40 simultaneous bounded requests; diagnostic only, never a training sample."""
    def request(index):
        payload = {'model': ALIAS, 'messages': [{'role': 'user', 'content':
                   f'Serving diagnostic {index}: count from 1 to 100 with spaces.'}],
                   'temperature': 0.7, 'seed': index, 'max_tokens': 128,
                   'chat_template_kwargs': {'enable_thinking': False},
                   'return_token_ids': True, 'logprobs': True, 'top_logprobs': 20}
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as response:
            body = json.load(response)
            replica = response.headers['x-ifv-qwen-replica']
        choice = body['choices'][0]
        assert body.get('prompt_token_ids') and choice.get('token_ids')
        assert choice.get('logprobs', {}).get('content')
        o.save(OUT / 'smoke' / f'{index:02d}.json', {'response': body, 'replica': replica,
               'elapsed_seconds': time.monotonic() - started, 'not_a_training_sample': True})
        return replica
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        replicas = list(executor.map(request, range(CONCURRENCY)))
    counts = {f'http://127.0.0.1:{port}': replicas.count(f'http://127.0.0.1:{port}') for port in PORTS}
    assert sum(counts.values()) == CONCURRENCY and all(counts.values()), counts
    o.save(OUT / 'smoke-summary.json', {'passed': True, 'submitted_concurrency': CONCURRENCY,
        'completed': len(replicas), 'requests_by_replica': counts,
        'native_prompt_completion_ids_and_top20_received': True,
        'scope': 'serving/capture plumbing only; not end-to-end PSD acceptance or speed estimate'})


def execute():
    o = owner()
    assert not OUT.exists(), 'Never overwrite a previous transition'
    o.verify_export()
    source = o.load(SERVICE / 'replica-1.json')
    source_env = o.checked(source)
    assert source_env['CUDA_VISIBLE_DEVICES'] == '1'
    commands = {gpu: replica_command(source['command'], gpu) for gpu in REPLACED}
    old = {gpu: o.load(o.OLD / f'replica-{gpu}.json') for gpu in REPLACED}
    old_env = {gpu: o.checked(receipt) for gpu, receipt in old.items()}
    for gpu in REPLACED:
        assert old_env[gpu]['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert old[gpu]['command'][old[gpu]['command'].index('--port') + 1] == str(PORTS[gpu])
        assert old[gpu]['command'][3] == str(ROOT / 'exports/h20-sft-merged4872-epoch2-step2056-20260915/model')
    guard = o.load(SERVICE / 'guard.json')
    guard_env = o.checked(guard)
    gateway = o.load(SERVICE / 'captured-v2/gateway.json')
    gateway_env = o.checked(gateway)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    OUT.mkdir()
    o.save(OUT / 'before.json', {'replicas': old, 'guard': guard, 'gpu1_untouched': source,
                              'canary_gateway_untouched': gateway})
    o.save(OUT / 'full-stage-options.json', {'cases': 400, 'rollouts_per_case': 8, 'slots': 3200,
        'collection_concurrency': CONCURRENCY, 'case_concurrency': CONCURRENCY,
        'within_case_repair': 'sequential, at most 6 replay rounds',
        'gateway': f'http://127.0.0.1:{PORT}', 'public_model': ALIAS, 'backend_model': BACKEND,
        'gpu_ids': list(range(4)), 'per_replica_max_num_seqs': 16, 'balancing': 'least_inflight',
        'temperature': 0.7, 'thinking_token_budget': 8192, 'output_budget': 32768,
        'context_length': 131072, 'images_per_request': 32, 'source_export_sha256': o.EXPORT_SHA,
        'native_capture_required': True, 'duplicate_wire_archive': False,
        'optimization_effective_batch_size': 32, 'optimization_epochs': 5,
        'gates': 'source checker, repair, native target audit, real GPU save/resume before full launch'})
    paused = False
    guard_stopped = False
    stopped = set()
    launched = {}
    new_guard = None
    new_gateway = None
    try:
        phase(o, 'draining_owned_idle_replicas')
        os.kill(guard['pid'], signal.SIGSTOP)
        paused = True
        for _ in range(120):
            if all(idle(PORTS[gpu]) for gpu in REPLACED):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Old replicas did not drain; no service stopped')
        # Resume only to terminate the guard normally; no generation is queued.
        os.kill(guard['pid'], signal.SIGCONT)
        paused = False
        o.stop(guard)
        guard_stopped = True
        for gpu in REPLACED:
            o.stop(old[gpu])
            stopped.add(gpu)
        phase(o, 'loading_three_epoch3_replicas_gpu1_preserved')
        for gpu in REPLACED:
            launched[gpu] = o.spawn(commands[gpu], {**source_env, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                   OUT / f'backend-{gpu}.log')
            o.save(OUT / f'replica-{gpu}.json', launched[gpu])
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda gpu: ready(o, launched[gpu], gpu), REPLACED))
        ready(o, source, 1)
        for gpu in REPLACED:
            o.save(SERVICE / f'replica-{gpu}.json', launched[gpu])
        guard_command = list(guard['command'])
        for option, value in {'--model': BACKEND, '--models': ','.join([BACKEND] * 4),
                              '--state-file': str(OUT / 'idle-guard/state.json'),
                              '--log-file': str(OUT / 'idle-guard/samples.jsonl')}.items():
            guard_command[guard_command.index(option) + 1] = value
        new_guard = o.spawn(guard_command, guard_env, OUT / 'guard.log')
        o.save(OUT / 'guard.json', new_guard)
        o.save(SERVICE / 'guard.json', new_guard)
        gateway_command = list(gateway['command'])
        gateway_command[gateway_command.index('--port') + 1] = str(PORT)
        new_gateway = o.spawn(gateway_command, gateway_environment(gateway_env), OUT / 'gateway.log')
        o.save(OUT / 'gateway.json', new_gateway)
        for _ in range(30):
            o.checked(new_gateway)
            try:
                health = o.http(f'http://127.0.0.1:{PORT}/health')
                if len(health['replicas']) == 4 and all(row['healthy'] for row in health['replicas']):
                    assert not health['wire_capture']['enabled'] and health['post_retries'] == 0
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Four-GPU gateway readiness deadline')
        o.save(OUT / 'health.json', health)
        phase(o, 'four_gpu_ready_serving_smoke')
        smoke(o)
        o.verify_export()
        assert o.load(SERVICE / 'replica-1.json') == source
        phase(o, 'four_gpu_ready_40_smoke_passed', gateway=f'http://127.0.0.1:{PORT}',
              configured_collection_concurrency=CONCURRENCY, configured_repair_concurrency=CONCURRENCY)
    except BaseException as error:
        phase(o, 'transition_failed_rolling_back', error_type=type(error).__name__)
        if new_gateway is not None:
            o.stop(new_gateway)
        if new_guard is not None:
            o.stop(new_guard)
        for receipt in launched.values():
            try:
                o.stop(receipt)
            except FileNotFoundError:
                pass
        for gpu in sorted(stopped):
            receipt = o.spawn(old[gpu]['command'], old_env[gpu], OUT / f'rollback-{gpu}.log')
            o.save(o.OLD / f'replica-{gpu}.json', receipt)
            o.save(OUT / f'rollback-replica-{gpu}.json', receipt)
        if guard_stopped:
            receipt = o.spawn(guard['command'], guard_env, OUT / 'rollback-guard.log')
            o.save(SERVICE / 'guard.json', receipt)
        phase(o, 'transition_failed_rolled_back', error_type=type(error).__name__)
        raise
    finally:
        if paused:
            os.kill(guard['pid'], signal.SIGCONT)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        o = owner()
        assert not OUT.exists()
        with (DEPLOY / 'controller.json').open('x') as file:
            json.dump({'reserved': True}, file)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
        receipt = o.spawn(command, os.environ.copy(), DEPLOY / 'controller.log')
        receipt['script_sha256'] = o.sha(__file__)
        o.save(DEPLOY / 'controller.json', receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
