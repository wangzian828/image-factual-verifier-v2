"""Fixed-token engine diagnostic, never a formal Agent evaluation.

Uses only local vLLM endpoints and the frozen multimodal workload. The 1024-token
forced continuation is diagnostic only; formal inference keeps 32768/8192.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from benchmark_sft2056_serving import ALIAS, EXPORT, PROFILES, ROOT, WORK, digest, save

OUTPUT = WORK / 'fixed-token-v1'
TOKENS = 1024
PASSES = 3


def fixed_body(original, salt):
    body = copy.deepcopy(original)
    body.update(max_tokens=TOKENS, min_tokens=TOKENS, ignore_eos=True,
                stream=True, stream_options={'include_usage': True}, cache_salt=salt)
    return body


def metrics(text):
    result = {}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            name, value = line.rsplit(' ', 1)
            name = name.split('{', 1)[0]
            result[name] = result.get(name, 0.0) + float(value)
    return result


def metric_delta(before, after):
    a, b = metrics(before), metrics(after)
    wanted = ('_sum', '_count', '_total')
    return {k: b[k] - a.get(k, 0.0) for k in b if k.endswith(wanted)}


def run():
    import requests
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = WORK / 'workload.json'
    assert digest(source) == 'be40282d75e702bb1d019652648ab896d2fdef50cc086bbd205b282d3c03fc2e'
    workload = json.loads(source.read_text())
    assert len(workload) == 16
    salt = 'ifv-epoch2-' + OUTPUT.name + '-20260915'
    identity = {'workload_sha256': digest(source), 'export_sha256': digest(EXPORT/'export.json'),
                'output_tokens_per_request': TOKENS, 'concurrency': 16,
                'cache_salt': salt, 'passes': PASSES, 'profiles': PROFILES,
                'diagnostic_only': True, 'external_tools': False,
                'formal_output_tokens': 32768, 'formal_thinking_budget': 8192,
                'note': 'Cold uses a new cache salt; later passes repeat exact inputs. Warm replay is an upper bound.'}
    if (OUTPUT/'inputs.json').exists():
        raise ValueError('Existing benchmark; inspect before an explicit new run')
    save(OUTPUT/'inputs.json', identity)

    def profile_run(profile):
        base = f'http://127.0.0.1:{profile["port"]}'
        directory = OUTPUT/profile['name']
        deadline = time.monotonic() + 600
        while True:
            try:
                response = requests.get(base+'/v1/models', timeout=10)
                response.raise_for_status()
                break
            except requests.RequestException:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(5)
        model = response.json()['data'][0]
        assert model['id'] == ALIAS and model['root'] == str(EXPORT/'model')
        initial = requests.get(base+'/metrics', timeout=10).text
        state = metrics(initial)
        assert state.get('vllm:num_requests_running', 0) == 0
        assert state.get('vllm:num_requests_waiting', 0) == 0

        def one(item):
            started = time.monotonic()
            record = {'id': item['id'], 'image_slots': item['source_images']}
            first = last = None
            output_hash = hashlib.sha256()
            try:
                with requests.post(base+'/v1/chat/completions',
                                   json=fixed_body(item['body'], salt),
                                   stream=True, timeout=(10, 600)) as response:
                    record['http_status'] = response.status_code
                    response.raise_for_status()
                    done = False
                    for line in response.iter_lines(chunk_size=1):
                        if not line.startswith(b'data: '):
                            continue
                        payload = line[6:]
                        if payload == b'[DONE]':
                            done = True
                            break
                        event = json.loads(payload)
                        if event.get('error'):
                            raise RuntimeError('SSE error event')
                        if event.get('usage'):
                            record['usage'] = event['usage']
                        for choice in event.get('choices', []):
                            delta = choice.get('delta') or {}
                            if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                                now = time.monotonic()
                                first = first or now
                                last = now
                                output_hash.update(json.dumps(delta, sort_keys=True).encode())
                            if choice.get('finish_reason'):
                                record['finish_reason'] = choice['finish_reason']
                    record['passed'] = done and record.get('usage', {}).get('completion_tokens') == TOKENS
                    if not record['passed']:
                        record['error_type'] = 'IncompleteOrWrongTokenCount'
            except Exception as error:
                record.update(passed=False, error_type=type(error).__name__)
            record.update(seconds=time.monotonic()-started, output_sha256=output_hash.hexdigest(),
                          visible_ttft_seconds=first-started if first else None,
                          visible_stream_span_seconds=last-first if first and last else None)
            return record

        summaries = []
        for index in range(PASSES):
            before = requests.get(base+'/metrics', timeout=10).text
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                records = list(pool.map(one, workload))
            wall = time.monotonic()-started
            after = requests.get(base+'/metrics', timeout=10).text
            save(directory/f'metrics-{index}.json', {'before': before, 'after': after})
            passed = all(r['passed'] for r in records)
            summary = {'pass': index, 'cache_state': 'new_salt' if index == 0 else 'exact_replay',
                       'passed': passed, 'requests': len(records),
                       'successful': sum(r['passed'] for r in records), 'wall_seconds': wall,
                       'output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in records),
                       'tokens_per_second': len(records)*TOKENS/wall if passed else None,
                       'mean_request_seconds': statistics.mean(r['seconds'] for r in records),
                       'engine_metric_delta': metric_delta(before, after), 'results': records}
            save(directory/f'pass-{index}.json', summary)
            compact = {k: v for k, v in summary.items() if k not in ('results', 'engine_metric_delta')}
            summaries.append(compact)
            save(directory/'summary.json', {'passes': summaries})
            print('PASS', profile['name'], json.dumps(compact), flush=True)
            if not passed:
                raise RuntimeError('Fixed output count validation failed; no speed claim allowed')
        return {'profile': profile['name'], 'passes': summaries}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(profile_run, PROFILES))
    save(OUTPUT/'finished.json', {'time': time.time(), 'results': results, 'formal_inference_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run-id', default='fixed-token-v1')
    parser.add_argument('--apc16-only', action='store_true')
    parser.add_argument('--passes', type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('run-id must be a simple directory name')
    OUTPUT = WORK / args.run_id
    PASSES = args.passes
    if args.apc16_only:
        PROFILES = [{'name': 'apc-s16-b32k', 'gpu': 3, 'port': 18905,
                     'seqs': 16, 'batch_tokens': 32768, 'apc': True}]
    os.umask(0o077)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUTPUT/'run.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *[a for a in sys.argv[1:] if a != '--launch']],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        save(OUTPUT/'process.json', {'pid': process.pid, 'time': time.time()})
        print('LAUNCHED', process.pid)
    else:
        run()
