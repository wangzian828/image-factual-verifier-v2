"""Bounded archived-request streaming diagnostic; never execute returned actions."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
CODE = DEPLOY / 'code'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
ARCHIVE = ROOT / ('runs/psd-sft3084-captured-canary4x8-20260916/'
    'psd-observed-positions-v14/search/repairs/e79d4db478e45800/'
    'runtime/main-08596/slate-00-920e8224c070')


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2))
    temporary.replace(path)


def invalid_logprobs(payload):
    bad = []
    for choice in payload.get('choices', []):
        for row in (choice.get('logprobs') or {}).get('content') or []:
            for item in [row, *(row.get('top_logprobs') or [])]:
                value = item.get('logprob')
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    bad.append({'token': item.get('token'), 'kind': type(value).__name__,
                                'value': str(value)})
    return bad


async def reconstruct(archive=ARCHIVE, request_id='req-000006'):
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from src.orchestrator.llm_backend import APIBackend
    request = reconstruct_archived_request(archive, request_id)
    body = {}

    class Captured(BaseException):
        pass

    class Client:
        async def post(self, url, *, json, **kwargs):
            body.update(json)
            raise Captured()

    backend = APIBackend(provider='qwen_local', model_name='ifv-psd-sft3084',
        api_key='none', base_url='http://127.0.0.1:19004/v1', wire_api='chat_completions',
        temperature=.7, max_tokens=32768, max_retries=0)
    backend._get_shared_client = lambda: Client()
    try:
        await backend.get_response(request['input_payload'], tools=request['tool_schema'],
            tool_choice='required', generation_config=request['generation_config'],
            capture_policy_tokens=True, policy_topk=20)
    except Captured:
        pass
    assert body.pop('thinking_token_budget') == 8192
    body.update(vllm_xargs={'ifv_thinking_budget': 8192}, stream=True,
        stream_options={'include_usage': True}, cache_salt='psd-nan-diagnostic-'+uuid.uuid4().hex)
    assert body['max_tokens'] == 32768 and body['top_logprobs'] == 20
    return body


def captured_wire_body(ticket):
    """Retain original serialized message/tool ordering, unlike archive rebuild."""
    meta = json.loads((ticket / 'request-meta.json').read_text())
    raw = gzip.decompress((ticket / 'request.json.gz').read_bytes())
    if len(raw) != meta['bytes'] or hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError('Wire request receipt does not match captured bytes')
    body = json.loads(raw)
    if body.get('model') != 'ifv-psd-sft3084' or body.get('top_logprobs') != 20:
        raise ValueError('Expected the protected PSD model with native top20')
    body.update(stream=True, stream_options={'include_usage': True},
                cache_salt='psd-nan-diagnostic-'+uuid.uuid4().hex)
    return body


async def execute(out, gpu, *, archive=ARCHIVE, request_id='req-000006', wire_ticket=None):
    import httpx
    from tokenizers import Tokenizer
    body = (captured_wire_body(wire_ticket) if wire_ticket is not None else
            await reconstruct(archive, request_id))
    raw = json.dumps(body, ensure_ascii=False).encode()
    with gzip.open(out / 'request.json.gz', 'wb') as file:
        file.write(raw)
    model = ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'
    end = Tokenizer.from_file(str(model / 'tokenizer.json')).token_to_id('</think>')
    result = {'not_training_target': True, 'request_sha256': hashlib.sha256(raw).hexdigest(),
              'source_archive': str(archive) if wire_ticket is None else None,
              'source_request_id': request_id if wire_ticket is None else None,
              'source_wire_ticket': str(wire_ticket) if wire_ticket is not None else None,
              'semantic_request_sha256': hashlib.sha256(json.dumps({k:v for k,v in body.items() if k!='cache_salt'},sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
              'stream_only_diagnostic': True,
              'archive_reconstruction_not_exact_wire': wire_ticket is None,
              'changed_wire_fields': ['stream', 'stream_options', 'cache_salt'],
              'gpu': gpu, 'tokens': 0, 'zero_tokens': 0, 'think_closures': [], 'bytes': 0,
              'status': 'running', 'first_tokens': []}
    save(out / 'state.json', result)
    start = time.monotonic()
    async def consume():
        async with httpx.AsyncClient(timeout=httpx.Timeout(1230), trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=0)) as client:
            async with client.stream('POST', f'http://127.0.0.1:{19002+gpu}/v1/chat/completions',
                                     json=body) as response:
                result['http_status'] = response.status_code
                response.raise_for_status()
                with gzip.open(out / 'response.sse.gz', 'wb') as sink:
                    async for line in response.aiter_lines():
                        encoded = (line+'\n').encode()
                        result['bytes'] += len(encoded)
                        if result['bytes'] > 64 * 1024 * 1024:
                            result['status'] = 'diagnostic_byte_limit'
                            return
                        sink.write(encoded)
                        if not line.startswith('data: '):
                            continue
                        if line[6:] == '[DONE]':
                            result['status'] = 'completed'
                            return
                        payload = json.loads(line[6:])
                        if payload.get('error'):
                            result['status'] = 'server_error'
                            result['error_type'] = (payload.get('error') or {}).get('type')
                            return
                        for choice in payload.get('choices', []):
                            rows = (choice.get('logprobs') or {}).get('content') or []
                            if rows and 'first_logprob' not in result:
                                result['first_logprob'] = {k:v for k,v in rows[0].items() if k!='bytes'}
                                result['first_logprob']['top_logprobs'] = [
                                    {k:v for k,v in item.items() if k!='bytes'} for item in rows[0].get('top_logprobs',[])]
                            ids = choice.get('token_ids') or []
                            if len(result['first_tokens']) < 16:
                                result['first_tokens'].extend(ids[:16-len(result['first_tokens'])])
                            result['think_closures'].extend(result['tokens']+i for i,t in enumerate(ids) if t==end)
                            result['zero_tokens'] += ids.count(0)
                            result['tokens'] += len(ids)
                            if choice.get('finish_reason'):
                                result['finish_reason'] = choice['finish_reason']
                        bad = invalid_logprobs(payload)
                        if bad:
                            result.update(status='invalid_logprob_detected', invalid=bad[:25])
                            return
                        if result['tokens'] and result['tokens'] % 256 == 0:
                            result['seconds'] = time.monotonic()-start
                            save(out / 'state.json', result)
    try:
        await asyncio.wait_for(consume(), timeout=1200)
    except Exception as error:
        result.update(status='diagnostic_exception', error_type=type(error).__name__)
    finally:
        result['seconds'] = time.monotonic()-start
        save(out / 'state.json', result)
        print(json.dumps(result), flush=True)


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'execute'])
    parser.add_argument('--gpu', type=int, choices=range(4), default=2)
    parser.add_argument('--wire-ticket', type=Path)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    if args.wire_ticket is not None:
        args.wire_ticket = args.wire_ticket.resolve()
        args.wire_ticket.relative_to(SERVICE.resolve())
        captured_wire_body(args.wire_ticket)
    name = args.output_name or f'nan-stream-gpu{args.gpu}-v1'
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Diagnostic output must be a new direct child name')
    out = SERVICE / name
    receipt = json.loads((SERVICE / f'replica-{args.gpu}.json').read_text())
    command = [s.decode() for s in Path(f'/proc/{receipt["pid"]}/cmdline').read_bytes().split(b'\0') if s]
    assert command == receipt['command'] and 'ifv-psd-sft3084' in command
    if args.mode == 'execute':
        asyncio.run(execute(out, args.gpu, wire_ticket=args.wire_ticket))
        return
    out.mkdir(exist_ok=False)
    source = (args.wire_ticket / 'request-meta.json' if args.wire_ticket is not None else
              ARCHIVE / 'context/req-000006.json')
    save(out / 'binding.json', {'backend': receipt, 'archive': str(ARCHIVE) if args.wire_ticket is None else None,
        'wire_ticket': str(args.wire_ticket) if args.wire_ticket is not None else None,
        'requests': 1, 'retries': 0, 'never_execute_actions': True,
        'source_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute', '--gpu', str(args.gpu)]
    command += ['--output-name', name]
    if args.wire_ticket is not None:
        command += ['--wire-ticket', str(args.wire_ticket)]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'))
    with (out / 'run.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True)
    save(out / 'process.json', {'pid': child.pid, 'command': command, 'time': time.time()})
    print(json.dumps({'pid': child.pid, 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
