"""Mixed-budget multi-image GPU regression, isolated from evaluation and PSD data."""
from __future__ import annotations
import asyncio
import copy
import json
from pathlib import Path
import time
import httpx

RUN = Path('/volume/ybo/wza/inference/psd-sft2056-safety-20260916')
OUT = RUN/'cache-isolation-probes-v2'


async def main():
    assert json.loads((RUN/'cancellation-probes-v3/summary.json').read_text())['cancellation_gpu_gate_passed']
    base = json.loads((RUN/'generation-probes-v1/multi-image-tool-request.json').read_text())
    OUT.mkdir()
    results = []
    async with httpx.AsyncClient(timeout=1200, trust_env=False) as client:
        health = (await client.get('http://127.0.0.1:19001/health')).json()
        assert health['prefix_cache_policy'] == 'unique_salt_per_request'
        async def one(index):
            name = f'probe-{index:02d}'
            body = copy.deepcopy(base)
            body['thinking_token_budget'] = 64 if index % 2 else 8192
            (OUT/(name+'-request.json')).write_text(json.dumps(body))
            started = time.monotonic()
            response = await client.post('http://127.0.0.1:19001/v1/chat/completions', json=body)
            (OUT/(name+'-response.json')).write_bytes(response.content)
            response.raise_for_status()
            choice = response.json()['choices'][0]
            calls = choice['message'].get('tool_calls') or []
            tokens = choice.get('token_ids') or []
            args = json.loads(calls[0]['function']['arguments']) if len(calls) == 1 else {}
            result = {'index': index, 'budget': body['thinking_token_budget'],
                      'seconds': time.monotonic()-started, 'replica': response.headers.get('x-ifv-qwen-replica'),
                      'finish_reason': choice['finish_reason'], 'token_count': len(tokens),
                      'zero_tokens': tokens.count(0),
                      'closure_index': tokens.index(248069) if 248069 in tokens else None}
            result['passed'] = (choice['finish_reason'] == 'tool_calls' and len(calls) == 1 and
                                calls[0]['function']['name'] == 'record_images' and
                                args.get('count') == 2 and len(args.get('descriptions', [])) == 2 and
                                result['closure_index'] is not None and
                                result['closure_index'] <= body['thinking_token_budget'] and
                                tokens.count(0) < max(1, len(tokens)//2))
            (OUT/(name+'-check.json')).write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return result
        # Two batches exercise slot release/reuse and mixed closure lengths.
        for start in [0, 4]:
            batch = await asyncio.gather(*(one(i) for i in range(start, start+4)), return_exceptions=True)
            results += [({'passed': False, 'error': str(x), 'type': type(x).__name__}
                         if isinstance(x, BaseException) else x) for x in batch]
    coverage = {x.get('replica') for x in results}
    passed = all(x['passed'] for x in results) and coverage == {f'http://127.0.0.1:{p}' for p in range(19002, 19006)}
    (OUT/'summary.json').write_text(json.dumps({'passed': passed, 'checks': results,
        'real_agent_canary_passed': False, 'formal_collection_started': False}, indent=2))
    print('CACHE_ISOLATION_MULTIMODAL_GATE', passed, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
