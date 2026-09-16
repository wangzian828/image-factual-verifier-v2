"""Drain and roll only owned PSD replicas after the bounded repair audit ends.

Work around a verified vLLM multimodal sender/receiver cache miss, not a model
quality failure. No change to weights, prompts, media, budgets or sampling.
No training and no automatic rerun of an interrupted trajectory.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
AUDIT = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
OUT = SERVICE / 'mm-cache-recovery-v11'
MODEL = ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'
PORTS = (19002, 19003, 19004, 19005)


def owner():
    path = ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py'
    spec = importlib.util.spec_from_file_location('owned_psd_services', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_free_command(command):
    result = list(command)
    assert result[2] == 'serve' and result[3] == str(MODEL)
    for key, value in {'--max-model-len': '131072', '--served-model-name': 'ifv-psd-sft3084',
                       '--tool-call-parser': 'ifv_psd_qwen3_single', '--mamba-cache-mode': 'none'}.items():
        assert result[result.index(key) + 1] == value
    assert '--no-enable-prefix-caching' in result
    if '--mm-processor-cache-gb' in result:
        result[result.index('--mm-processor-cache-gb') + 1] = '0'
    else:
        result += ['--mm-processor-cache-gb', '0']
    return result


def idle(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
        values = [float(line.rsplit(' ', 1)[1]) for line in response.read().decode().splitlines()
            if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    return len(values) == 2 and sum(values) == 0


def assert_audit_finished(o):
    receipt = o.load(AUDIT / 'process.json')
    stat = Path(f'/proc/{receipt["pid"]}/stat')
    assert not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z', 'Audit still active; do not interrupt it'
    state = o.load(AUDIT / 'state.json')
    assert state['phase'] != 'native_contract_repair_running' and not state['training_started']


def wait_ready(o, receipt, port):
    for _ in range(240):
        o.checked(receipt)
        try:
            card = o.http(f'http://127.0.0.1:{port}/v1/models')['data'][0]
            if card['root'] == str(MODEL) and card['id'] == 'ifv-psd-sft3084':
                assert card['max_model_len'] == 131072
                return
        except OSError:
            pass
        time.sleep(3)
    raise RuntimeError('Owned replica readiness deadline exceeded')


def smoke(o):
    """Repeated two-image requests on EVERY backend; excluded from PSD data."""
    from PIL import Image
    images = []
    for color in ('red', 'blue'):
        stream = io.BytesIO()
        Image.new('RGB', (96, 96), color).save(stream, format='PNG')
        images.append({'type': 'image_url', 'image_url': {'url':
            'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}})
    def one(item):
        gpu, repeat = item
        payload = {'model': 'ifv-psd-sft3084', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Name the two colors briefly.'}, *images]}],
            'max_tokens': 32, 'temperature': 0.7, 'seed': repeat,
            'chat_template_kwargs': {'enable_thinking': False},
            'return_token_ids': True, 'logprobs': True, 'top_logprobs': 20}
        request = urllib.request.Request(f'http://127.0.0.1:{PORTS[gpu]}/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        choice = body['choices'][0]
        assert body['prompt_token_ids'] and choice['token_ids'] and choice['logprobs']['content']
        return {'gpu': gpu, 'repeat': repeat, 'images': 2, 'native_capture': True}
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(one, [(g, r) for r in range(4) for g in range(4)]))
    o.save(OUT / 'multimodal-smoke.json', {'passed': True, 'requests': rows,
        'scope': '16 repeated multi-image serving requests, not full Agent or training acceptance'})


def execute():
    o = owner()
    assert_audit_finished(o)
    o.verify_export()
    assert not OUT.exists(), 'Do not overwrite a previous recovery'
    old = {gpu: o.load(SERVICE / f'replica-{gpu}.json') for gpu in range(4)}
    envs = {gpu: o.checked(receipt) for gpu, receipt in old.items()}
    commands = {gpu: cache_free_command(receipt['command']) for gpu, receipt in old.items()}
    for gpu in range(4):
        assert envs[gpu]['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert commands[gpu][commands[gpu].index('--port') + 1] == str(PORTS[gpu])
    guard = o.load(SERVICE / 'guard.json')
    o.checked(guard)
    OUT.mkdir()
    o.save(OUT / 'before.json', {'replicas': old, 'guard': guard,
        'reason': '13:40:11 backend0 verified Expected a cached item AssertionError',
        'weights_and_agent_unchanged': True})
    try:
        for gpu in range(4):
            o.save(OUT / 'state.json', {'phase': 'rolling_replica', 'gpu': gpu, 'training_started': False})
            launched = None
            stopped = False
            o.checked(guard)
            os.kill(guard['pid'], signal.SIGSTOP)
            try:
                for _ in range(120):
                    health = o.http('http://127.0.0.1:19019/health')
                    assert all(row['inflight'] == 0 for row in health['replicas'])
                    if idle(PORTS[gpu]):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Owned replica did not drain')
                o.stop(old[gpu])
                stopped = True
                launched = o.spawn(commands[gpu], envs[gpu], OUT / f'backend-{gpu}.log')
                o.save(OUT / f'replica-{gpu}.json', launched)
                o.save(SERVICE / f'replica-{gpu}.json', launched)
            finally:
                os.kill(guard['pid'], signal.SIGCONT)
                if stopped and launched is None:
                    restored = o.spawn(old[gpu]['command'], envs[gpu], OUT / f'rollback-{gpu}.log')
                    o.save(SERVICE / f'replica-{gpu}.json', restored)
            try:
                wait_ready(o, launched, PORTS[gpu])
            except BaseException:
                # Preserve a working original service if the workaround cannot load.
                try:
                    o.stop(launched)
                except FileNotFoundError:
                    pass
                restored = o.spawn(old[gpu]['command'], envs[gpu], OUT / f'rollback-{gpu}.log')
                o.save(SERVICE / f'replica-{gpu}.json', restored)
                raise
        smoke(o)
        o.verify_export()
        o.save(OUT / 'state.json', {'phase': 'cache_disabled_four_replicas_smoke_passed',
            'time': time.time(), 'training_started': False, 'weights_and_agent_unchanged': True})
    except BaseException as error:
        o.save(OUT / 'state.json', {'phase': 'held_requires_inspection',
            'error_type': type(error).__name__, 'training_started': False})
        raise


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'launch', 'execute'))
    args = parser.parse_args()
    o = owner()
    if args.mode == 'plan':
        print(json.dumps({str(g): cache_free_command(o.load(SERVICE / f'replica-{g}.json')['command'])
                          for g in range(4)}))
    elif args.mode == 'execute':
        execute()
    else:
        assert_audit_finished(o)
        receipt_path = AUDIT / 'mm-recovery-process.json'
        assert not receipt_path.exists() and not OUT.exists()
        receipt = o.spawn([os.sys.executable, '-u', str(Path(__file__).resolve()), 'execute'],
                          os.environ.copy(), AUDIT / 'mm-recovery.log')
        o.save(receipt_path, receipt)
        print(json.dumps(receipt))
