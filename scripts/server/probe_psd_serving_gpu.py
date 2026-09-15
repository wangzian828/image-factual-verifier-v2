"""Persist real GPU protocol probes; not a PSD training/canary attestation."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time

import httpx

ROOT = Path('/volume/ybo/wza')
RUN = ROOT/'inference/psd-sft2056-safety-20260916'
OUT = RUN/'generation-probes-v1'
MODEL = ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/model'
ALIAS = 'ifv-psd-sft2056-safety'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


async def request(client, name, port, endpoint, body):
    save(OUT/(name+'-request.json'), body)
    started = time.monotonic()
    response = await client.post(f'http://127.0.0.1:{port}/v1/{endpoint}', json=body)
    (OUT/(name+'-response.json')).write_bytes(response.content)
    response.raise_for_status()
    result = response.json()
    save(OUT/(name+'-transport.json'), {'seconds': time.monotonic()-started,
                                     'status': response.status_code,
                                     'replica': response.headers.get('x-ifv-qwen-replica')})
    return result


async def boundary(client, budget, port):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    end = tokenizer.token_to_id('</think>')
    prefix = '<|im_start|>user\nWhat is 13 + 29? Give the final answer after thinking.\n<|im_end|>\n<|im_start|>assistant\n<think>\n'
    # Diagnostic only: prevent early closure/EOS so the real sampling loop must
    # reach the boundary. Never use these controls in Agent/source collection.
    body = {'model': ALIAS, 'prompt': tokenizer.encode(prefix, add_special_tokens=False).ids,
            'max_tokens': budget+256, 'min_tokens': budget+2,
            'logit_bias': {str(end): -100}, 'temperature': 0, 'seed': 0,
            'return_token_ids': True, 'skip_special_tokens': False,
            'vllm_xargs': {'ifv_thinking_budget': budget}}
    result = await request(client, f'boundary-{budget}', port, 'completions', body)
    choice = result['choices'][0]
    tokens = choice['token_ids']
    assert tokens.index(end) == budget, (budget, tokens.index(end))
    tail = tokenizer.decode(tokens[budget+1:], skip_special_tokens=True)
    assert tail.strip(), 'Reasoning closed but no answer tokens followed'
    check = {'budget': budget, 'closure_index': tokens.index(end),
             'generated_tokens': len(tokens), 'answer_text': tail,
             'finish_reason': choice['finish_reason'], 'passed': True,
             'diagnostic_forced_length': True}
    save(OUT/f'boundary-{budget}-check.json', check)
    print(json.dumps({'probe': f'boundary-{budget}', 'passed': True,
                      'closure_index': tokens.index(end)}, ensure_ascii=False), flush=True)
    return check


async def early(client):
    result = await request(client, 'early-finish', 19001, 'chat/completions', {
        'model': ALIAS, 'messages': [{'role': 'user', 'content': 'What is 13 + 29? Answer briefly.'}],
        'max_tokens': 32768, 'thinking_token_budget': 8192, 'return_token_ids': True,
        'temperature': 0.7, 'seed': 0})
    choice = result['choices'][0]
    tokens = choice['token_ids']
    assert 0 <= tokens.index(248069) < 8192
    assert choice['finish_reason'] == 'stop' and choice['message']['content'].strip()
    check = {'passed': True, 'closure_index': tokens.index(248069),
             'finish_reason': choice['finish_reason']}
    save(OUT/'early-finish-check.json', check)
    return check


async def multi_image_tool(client):
    inputs = ROOT/'runs/psd-pilot400-preparation-20260915-v1/runtime-release/runtime_input'
    with (inputs/'cases.jsonl').open() as handle:
        rows = [json.loads(next(handle)), json.loads(next(handle))]
    parts = [{'type': 'text', 'text': 'Inspect both images and call record_images exactly once. Record the number of supplied images and a short visual description of each. Do not answer in prose.'}]
    identities = []
    for row in rows:
        path = (inputs/row['image_path']).resolve()
        path.relative_to(inputs)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['image_sha256']
        parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(data).decode()}})
        identities.append({'image_sha256': row['image_sha256'], 'bytes': len(data)})
    save(OUT/'multi-image-identities.json', identities)
    result = await request(client, 'multi-image-tool', 19001, 'chat/completions', {
        'model': ALIAS, 'messages': [{'role': 'user', 'content': parts}],
        'tools': [{'type': 'function', 'function': {'name': 'record_images',
                  'description': 'Record observations for the supplied images.',
                  'parameters': {'type': 'object', 'properties': {
                      'count': {'type': 'integer'}, 'descriptions': {'type': 'array', 'items': {'type': 'string'}}},
                      'required': ['count', 'descriptions'], 'additionalProperties': False}}}],
        'tool_choice': 'auto', 'max_tokens': 32768, 'thinking_token_budget': 8192,
        'return_token_ids': True, 'temperature': 0.7, 'seed': 0})
    choice = result['choices'][0]
    calls = choice['message']['tool_calls']
    assert choice['finish_reason'] == 'tool_calls' and len(calls) == 1
    assert calls[0]['function']['name'] == 'record_images'
    args = json.loads(calls[0]['function']['arguments'])
    assert args['count'] == 2 and len(args['descriptions']) == 2
    tokens = choice['token_ids']
    assert tokens.index(248069) <= 8192
    check = {'passed': True, 'image_count': 2, 'tool_count': 1,
             'closure_index': tokens.index(248069), 'finish_reason': choice['finish_reason']}
    save(OUT/'multi-image-tool-check.json', check)
    return check


async def main():
    assert json.loads((RUN/'state.json').read_text())['phase'] == 'ready_for_boundary_probes'
    OUT.mkdir()  # Never silently repeat a previously issued probe.
    save(OUT/'state.json', {'phase': 'running', 'started_at': time.time(),
                          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    async with httpx.AsyncClient(timeout=1200, trust_env=False) as client:
        jobs = [boundary(client, 1, 19002), boundary(client, 3, 19003),
                boundary(client, 8192, 19004), early(client), multi_image_tool(client)]
        results = await asyncio.gather(*jobs, return_exceptions=True)
    checks = [({'passed': False, 'error_type': type(r).__name__, 'error': str(r)}
               if isinstance(r, BaseException) else r) for r in results]
    save(OUT/'summary.json', {'checks': checks, 'generation_probes_passed': all(r['passed'] for r in checks),
                             'cancellation_gate_passed': False, 'psd_canary_passed': False,
                             'formal_collection_authorized_by_this_artifact': False})
    save(OUT/'state.json', {'phase': 'finished', 'time': time.time()})
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
