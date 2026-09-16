"""Replay fixed PSD requests on the successful evaluation serving recipe.

Never execute tools or put diagnostics into training. Original worker restored
after all three modes, including when any request fails. Other GPUs unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request

from psd_eval_reference import reference_body, reference_command, stop_if_present

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft3084-20260916'


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('launch', 'execute'))
    p.add_argument('--output-name', default='eval-reference-ab-v1')
    args = p.parse_args()
    assert args.output_name.startswith('eval-reference-') and Path(args.output_name).name == args.output_name
    out = SERVICE/args.output_name
    spec = importlib.util.spec_from_file_location('owner', ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if args.mode == 'launch':
        out.mkdir(exist_ok=False)
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute',
            '--output-name', args.output_name], os.environ.copy(), out/'controller.log')
        owner.save(out/'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(out)})); return
    owner.verify_export()
    backend = owner.load(SERVICE/'replica-2.json'); env = owner.checked(backend)
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    assert backend['command'][backend['command'].index('--worker-cls')+1] == 'scripts.server.psd_raw_teacher_worker.RawTeacherWorker'
    command = reference_command(backend['command'])
    guard = owner.load(SERVICE/'guard.json'); owner.checked(guard)
    sources = sorted((SERVICE/'raw-multimage-failure-v1').glob('*/backend-body.json'))
    assert len(sources) == 2
    owner.save(out/'binding.json', {'before': backend, 'reference_command': command,
        'old_evaluation_log_sha256': owner.sha(ROOT/'inference/sft3084-3epoch-20260915/logs/replica-0.log'),
        'source_request_hashes': {str(s): owner.sha(s) for s in sources},
        'preserved': ['same weights', 'messages and all images', 'tool schemas and choice', 'T0.7 and seed', 'out32768'],
        'diagnostic_only': True, 'training_targets': False, 'thinking_budget_enforced': False,
        'fresh_scoped_caches': True, 'alias_and_port_differ_from_original_evaluation': True})
    stopped = False; candidate = None; paused = False
    try:
        os.kill(guard['pid'], signal.SIGSTOP); paused = True
        for port in (19019, 19022):
            assert all(r['inflight'] == 0 for r in owner.http(f'http://127.0.0.1:{port}/health')['replicas'])
        for _ in range(90):
            metrics = urllib.request.urlopen('http://127.0.0.1:19004/metrics', timeout=5).read().decode()
            values = [float(s.rsplit(' ',1)[1]) for s in metrics.splitlines()
                if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(values) == 2
            if not sum(values): break
            time.sleep(1)
        else: raise RuntimeError('Owned GPU2 did not drain')
        owner.stop(backend); stopped = True
        cache = out/'cache'
        new_env = {**env, 'PYTHONDONTWRITEBYTECODE':'1', 'TMPDIR':str(ROOT/'tmp'),
            'XDG_CACHE_HOME':str(cache), 'TRITON_CACHE_DIR':str(cache/'triton'),
            'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'), 'VLLM_CACHE_ROOT':str(cache/'vllm'),
            'TORCH_EXTENSIONS_DIR':str(cache/'extensions'), 'FLASHINFER_WORKSPACE_BASE':str(cache/'flashinfer')}
        for key in list(new_env):
            if key.startswith(('PSD_', 'VLLM_DISABLE_COMPILE_CACHE')): new_env.pop(key)
        candidate = owner.spawn(command, new_env, out/'backend.log')
        owner.save(SERVICE/'replica-2.json', candidate); owner.save(out/'replica.json', candidate)
        os.kill(guard['pid'], signal.SIGCONT); paused = False
        owner.save(out/'state.json', {'phase':'loading_original_evaluation_recipe','formal_training':False})
        for _ in range(240):
            owner.checked(candidate)
            try:
                card = owner.http('http://127.0.0.1:19004/v1/models')['data'][0]
                if card['id']=='ifv-psd-sft3084' and card['root']==command[3]: break
            except OSError: pass
            time.sleep(5)
        else: raise RuntimeError('Evaluation reference did not start')
        def request(mode, index):
            source = sources[index % 2]
            body = reference_body(owner.load(source), mode)
            raw_request = json.dumps(body, ensure_ascii=False).encode()
            started = time.monotonic()
            result = {'mode':mode,'index':index,'case':source.parent.name,
                'request_sha256':hashlib.sha256(raw_request).hexdigest(),'formal_training':False}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                req = urllib.request.Request('http://127.0.0.1:19004/v1/chat/completions', data=raw_request,
                    headers={'Content-Type':'application/json'})
                try:
                    with opener.open(req, timeout=1230) as response: raw = response.read(); code = response.status
                except urllib.error.HTTPError as e: raw = e.read(); code = e.code
                with gzip.open(out/f'{mode}-{index}.json.gz','wb') as f: f.write(raw)
                response = json.loads(raw); result['http_status'] = code
                if code != 200:
                    result['error'] = response.get('error'); result['passed'] = False
                else:
                    choice = response['choices'][0]; message = choice['message']
                    rows = (choice.get('logprobs') or {}).get('content') or []
                    invalid = sum(not isinstance(r.get('logprob'),(int,float)) or not math.isfinite(r['logprob'])
                        for row in rows for r in [row, *(row.get('top_logprobs') or [])])
                    result.update(finish_reason=choice.get('finish_reason'),usage=response.get('usage'),
                        prompt_ids=len(response.get('prompt_token_ids') or []), completion_ids=len(choice.get('token_ids') or message.get('token_ids') or []),
                        logprob_rows=len(rows),invalid_logprobs=invalid,tool_calls=len(message.get('tool_calls') or []),
                        passed=choice.get('finish_reason') in ('stop','tool_calls') and not invalid)
                    if mode != 'plain': result['passed'] &= bool(result['prompt_ids'] and result['completion_ids'])
                    if mode == 'top20': result['passed'] &= bool(rows)
            except Exception as error:
                result.update(passed=False,error_type=type(error).__name__)
            result['seconds'] = time.monotonic()-started
            owner.save(out/f'{mode}-{index}-result.json', result)
            print(json.dumps(result),flush=True)
            return result
        results = []
        for mode in ('plain','token_ids','top20'):
            owner.save(out/'state.json', {'phase':'replaying','mode':mode,'concurrency':4,'formal_training':False})
            with ThreadPoolExecutor(max_workers=4) as pool:
                results.extend(pool.map(lambda i: request(mode,i),range(4)))
            owner.save(out/'summary.json', {'results':results,'never_training_targets':True})
        owner.save(out/'state.json', {'phase':'reference_comparison_completed','requests':len(results),
            'passed':sum(r['passed'] for r in results),'formal_training':False,'full_agent_gate':False})
    except BaseException as error:
        owner.save(out/'state.json', {'phase':'reference_comparison_failed','error_type':type(error).__name__,
            'error':str(error),'formal_training':False}); raise
    finally:
        try:
            if stopped:
                if not paused: os.kill(guard['pid'], signal.SIGSTOP); paused=True
                if candidate is not None: stop_if_present(owner,candidate)
                restored = owner.spawn(backend['command'],env,out/'restored-backend.log')
                owner.save(SERVICE/'replica-2.json',restored)
        finally:
            if paused: os.kill(guard['pid'],signal.SIGCONT)
            owner.verify_export()


if __name__ == '__main__': main()
