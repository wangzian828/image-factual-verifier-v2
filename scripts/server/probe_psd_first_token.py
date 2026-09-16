"""First-token numerical diagnostic, never an Agent/PSD training rollout."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import uuid

ROOT = Path('/volume/ybo/wza/inference/psd-sft3084-20260916')


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output-name', required=True)
    args = p.parse_args()
    assert 1 <= args.repetitions <= 100
    assert Path(args.output_name).name == args.output_name and args.output_name.startswith('nan-')
    out = ROOT / args.output_name; out.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for index in range(args.repetitions):
        for source in sorted((ROOT / 'raw-multimage-failure-v1').glob('*/backend-body.json')):
            body = json.loads(source.read_text())
            # Isolate the prompt's first distribution. A future 8192-token
            # closure cannot affect this first step, but the budget plugin
            # rejects max_tokens=1, so omit only that processor in this probe.
            body['max_tokens'] = 1
            body.pop('vllm_xargs')
            body['cache_salt'] = 'first-token-diagnostic-' + uuid.uuid4().hex
            start = time.monotonic()
            request = urllib.request.Request(f'http://127.0.0.1:{19002+args.gpu}/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=120) as response:
                raw = response.read()
            with gzip.open(out / f'{index}-{source.parent.name}.json.gz', 'wb') as f: f.write(raw)
            payload = json.loads(raw); choice = payload['choices'][0]
            rows = (choice.get('logprobs') or {}).get('content') or []
            invalid = sum(not isinstance(x.get('logprob'), (int, float)) or not math.isfinite(x['logprob'])
                for row in rows for x in [row, *(row.get('top_logprobs') or [])])
            record = {'case': source.parent.name, 'iteration': index, 'gpu': args.gpu,
                'seconds': time.monotonic()-start, 'invalid_logprobs': invalid, 'rows': len(rows),
                'first_token': rows[0]['token'] if rows else None, 'usage': payload.get('usage'),
                'diagnostic_changes': ['max_tokens=1', 'omit future thinking-budget closure', 'cache nonce'],
                'formal_training': False}
            results.append(record)
            (out / 'summary.json').write_text(json.dumps({'results': results}, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
