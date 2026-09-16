"""Fixed multi-image requests, stock worker, 40 concurrent; never training data."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft3084-20260916'
OUT = SERVICE/'deferred-concurrency40-v1'


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner', ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if args.mode == 'launch':
        assert owner.load(SERVICE/'deferred-teacher-four-gpu-v1/state.json')['phase'] == 'ready_for_concurrency_validation'
        OUT.mkdir(exist_ok=False)
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], os.environ.copy(), OUT/'run.log')
        owner.save(OUT/'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)})); return
    owner.verify_export()
    sources = sorted((SERVICE/'raw-multimage-failure-v1').glob('*/backend-body.json'))
    assert len(sources) == 2
    receipts = [owner.load(SERVICE/f'replica-{gpu}.json') for gpu in range(4)]
    for gpu, receipt in enumerate(receipts):
        assert owner.checked(receipt)['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert '--worker-cls' not in receipt['command']
        assert '--enforce-eager' not in receipt['command']
    owner.save(OUT/'binding.json', {'source_hashes': {str(s): owner.sha(s) for s in sources},
        'worker_receipts': receipts, 'waves': 3, 'concurrency': 40, 'requests': 120,
        'routing': ['direct_per_gpu_10', 'direct_per_gpu_10', 'four_gpu_gateway_40'],
        'sampling_top20_not_teacher_targets': True, 'no_tools_executed': True,
        'retries': 0, 'formal_training': False})

    def request(wave, slot):
        gpu = slot // 10
        source = sources[slot % 2]
        body = owner.load(source)
        assert body['temperature'] == .7 and body['max_tokens'] == 32768
        assert body['vllm_xargs']['ifv_thinking_budget'] == 8192
        assert body['return_token_ids'] and body['logprobs'] and body['top_logprobs'] == 20
        if wave == 2:
            # Public gateway owns the translation; never inject internal xargs.
            xargs = dict(body['vllm_xargs'])
            body['thinking_token_budget'] = xargs.pop('ifv_thinking_budget')
            if xargs: body['vllm_xargs'] = xargs
            else: body.pop('vllm_xargs')
        raw_request = json.dumps(body, ensure_ascii=False).encode()
        started = time.monotonic()
        result = {'wave': wave, 'slot': slot, 'gpu': gpu if wave < 2 else None,
            'case': source.parent.name, 'request_sha256': hashlib.sha256(raw_request).hexdigest()}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            port = 19002 + gpu if wave < 2 else 19025
            req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=raw_request,
                headers={'Content-Type': 'application/json'})
            try:
                with opener.open(req, timeout=1230) as response: raw = response.read(); status = response.status
            except urllib.error.HTTPError as error: raw = error.read(); status = error.code
            with gzip.open(OUT/f'wave{wave}-slot{slot}.json.gz', 'wb') as stream: stream.write(raw)
            response = json.loads(raw)
            result['http_status'] = status
            if status != 200:
                result.update(passed=False, error=response.get('error'))
            else:
                choice = response['choices'][0]; message = choice['message']
                assert response.get('ifv_policy_logprobs') is None
                ids = choice.get('token_ids') or message.get('token_ids') or []
                rows = (choice.get('logprobs') or {}).get('content') or []
                invalid = sum(not isinstance(entry.get('logprob'), (int, float)) or not math.isfinite(entry['logprob'])
                    for row in rows for entry in [row, *(row.get('top_logprobs') or [])])
                calls = message.get('tool_calls') or []
                arguments_valid = len(calls) == 1 and isinstance(json.loads(calls[0]['function']['arguments']), dict)
                thinking = ids.index(248069) if 248069 in ids else None
                passed = bool(response.get('prompt_token_ids')) and bool(ids) and len(ids) == len(rows)
                passed &= not invalid and arguments_valid and choice.get('finish_reason') == 'tool_calls'
                passed &= thinking is not None and thinking <= 8192
                result.update(passed=bool(passed), prompt_tokens=len(response.get('prompt_token_ids') or []),
                    completion_tokens=len(ids), logprob_rows=len(rows), invalid_logprobs=invalid,
                    tool_calls=len(calls), thinking_tokens=thinking, finish_reason=choice.get('finish_reason'))
        except Exception as error:
            result.update(passed=False, error_type=type(error).__name__, error=str(error)[:500])
        result['seconds'] = time.monotonic() - started
        owner.save(OUT/f'wave{wave}-slot{slot}-result.json', result)
        return result

    results = []
    try:
        for wave in range(3):
            owner.save(OUT/'state.json', {'phase': 'probing', 'wave': wave, 'completed': len(results), 'formal_training': False})
            with ThreadPoolExecutor(max_workers=40) as pool: results.extend(pool.map(lambda slot: request(wave, slot), range(40)))
            owner.save(OUT/'summary.json', {'completed': len(results), 'passed': sum(r['passed'] for r in results),
                'results': results, 'formal_training': False})
            if not all(r['passed'] for r in results): raise RuntimeError('Concurrency gate failed; no automatic favorable resampling')
        owner.save(OUT/'state.json', {'phase': 'passed', 'completed': len(results), 'passed': len(results), 'formal_training': False})
    except BaseException as error:
        owner.save(OUT/'state.json', {'phase': 'requires_fix', 'completed': len(results),
            'error_type': type(error).__name__, 'error': str(error), 'formal_training': False})
        raise


if __name__ == '__main__': main()
