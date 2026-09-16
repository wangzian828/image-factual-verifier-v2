"""Replay bound real policy bodies without changing non-streaming execution.

Read-only inference diagnostics: no tool actions, no retries, never training data.
"""
import asyncio
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
OUT = SERVICE / 'nan-nonstream-four-replica-v1'
WIRE = SERVICE / 'validated-execution-candidate-v1/wire'
FIXED = WIRE / '1789551642580005259-b2b919425c11426eb34c440879f2f248'


def body_from_ticket(ticket):
    raw = gzip.decompress((ticket / 'request.json.gz').read_bytes())
    meta = json.loads((ticket / 'request-meta.json').read_text())
    assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
    body = json.loads(raw)
    assert body['model'] == 'ifv-psd-sft3084' and body['top_logprobs'] == 20
    assert not body.get('stream') and body['max_tokens'] == 32768
    return raw, body


async def request(gpu, slot, ticket, save):
    import httpx
    raw, body = body_from_ticket(ticket)
    directory = OUT / f'gpu{gpu}-request{slot}'
    directory.mkdir(exist_ok=False)
    result = {'gpu': gpu, 'slot': slot, 'source_ticket': str(ticket),
              'request_sha256': hashlib.sha256(raw).hexdigest(),
              'changed_request_fields': [], 'not_training_target': True,
              'status': 'running', 'started': time.time()}
    save(directory / 'state.json', result)
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=1230, trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=0)) as client:
            async with client.stream('POST', f'http://127.0.0.1:{19002+gpu}/v1/chat/completions',
                    content=raw, headers={'Content-Type': 'application/json'}) as response:
                result['http_status'] = response.status_code
                chunks, size = [], 0
                async for block in response.aiter_bytes():
                    size += len(block)
                    if size > 64 * 1024 * 1024:
                        raise ValueError('Diagnostic body limit exceeded')
                    chunks.append(block)
                payload_raw = b''.join(chunks)
        with gzip.open(directory / 'response.json.gz', 'wb') as f:
            f.write(payload_raw)
        payload = json.loads(payload_raw)
        result['response_sha256'] = hashlib.sha256(payload_raw).hexdigest()
        result['status'] = 'completed' if result['http_status'] == 200 else 'http_error'
        if payload.get('error'):
            result['error'] = payload['error']
        for choice in payload.get('choices', []):
            result['finish_reason'] = choice.get('finish_reason')
            rows = (choice.get('logprobs') or {}).get('content') or []
            result['logprob_rows'] = len(rows)
            if rows:
                result['first_logprob'] = rows[0]
            result['invalid_logprobs'] = sum(
                not isinstance(item.get('logprob'), (int, float)) or not math.isfinite(item['logprob'])
                for row in rows for item in [row, *(row.get('top_logprobs') or [])])
    except Exception as error:
        result.update(status='diagnostic_exception', error_type=type(error).__name__)
    result['seconds'] = time.monotonic() - start
    save(directory / 'state.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'first_logprob'}), flush=True)
    return result


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if sys.argv[1:] == ['launch']:
        replicas = [owner.load(SERVICE / f'replica-{gpu}.json') for gpu in range(4)]
        for receipt in replicas:
            owner.checked(receipt)
        choices = []
        for ticket in sorted(WIRE.iterdir()):
            try:
                body_from_ticket(ticket); choices.append(ticket)
            except (OSError, KeyError, AssertionError, ValueError):
                continue
        assert len(choices) >= 16
        selected = [FIXED, choices[0], choices[len(choices)//2], choices[-1]]
        OUT.mkdir(exist_ok=False)
        owner.save(OUT / 'binding.json', {'backends': replicas,
            'tickets': [str(p) for p in selected], 'requests_per_gpu': 8,
            'max_concurrency_per_gpu': 4, 'retries': 0, 'never_execute_actions': True})
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'],
                              os.environ.copy(), OUT / 'run.log')
        owner.save(OUT / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)})); return
    assert sys.argv[1:] == ['execute']
    async def run():
        tickets = [Path(p) for p in owner.load(OUT / 'binding.json')['tickets']]
        results = []
        for wave in range(2):
            results.extend(await asyncio.gather(*(request(gpu, wave*4+i, ticket, owner.save)
                for gpu in range(4) for i, ticket in enumerate(tickets))))
            owner.save(OUT / 'summary.json', {'completed': len(results),
                'results': [{k:v for k,v in r.items() if k!='first_logprob'} for r in results]})
    asyncio.run(run())


if __name__ == '__main__':
    main()
