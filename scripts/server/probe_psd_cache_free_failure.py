"""Bounded raw-output diagnosis, not Agent resampling or a PSD target."""
from __future__ import annotations
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-adjustments-20260915/code'
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
CANARY = ROOT/'runs/psd-slate-canary4x8-no-perception-cache-20260916'
OUT = SERVICE/'cache-free-failure-diagnostic-v1'
VALIDATOR = ROOT/'training-artifacts/psd-serving-safety-20260916/psd_collection_validator_v2.py'
ARCHIVE = CANARY/'episodes/traces/runtime/main-05485--d4337e0a--r001/230642-3be047'
REQUEST_ID = 'req-000015'
CONCURRENT = False


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not OUT.exists(), 'Preserve previous diagnostic; inspect instead of relaunching'
    receipt = json.loads((CANARY/'process.json').read_text())
    proc = Path(f'/proc/{receipt["pid"]}/cmdline')
    assert not proc.exists() or not proc.read_bytes(), 'Do not overlap the grouped collector'
    backend = json.loads((SERVICE/'replica-2.json').read_text())
    args = [x.decode() for x in Path(f'/proc/{backend["pid"]}/cmdline').read_bytes().split(b'\0') if x]
    assert args == backend['command']
    assert '--no-enable-prefix-caching' in args and args[args.index('--mamba-cache-mode')+1] == 'none'
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    spec = importlib.util.spec_from_file_location('canary_slot_validator_v2', VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = json.loads((CANARY/'binding.json').read_text())
    rows = [json.loads(line) for line in (CANARY/'episodes/run_results.jsonl').read_text().splitlines()]
    manifest = json.loads((CANARY/'episodes/run_manifest.json').read_text())
    integrity = module.verify_collection(rows, case_ids=binding['case_ids'], manifest=manifest, expected_rollouts=8)
    calls = Counter()
    failures = []
    for row in rows:
        path = CANARY/'episodes'/row['trace_path']
        trace = json.loads(path.read_text())
        for step in trace.get('state', {}).get('all_steps', []):
            if step.get('tool_name') in {'perceive_scene', 'ocr_with_position'}:
                calls['perception_ocr_calls'] += 1
                calls['cache_hits'] += step.get('metadata', {}).get('cache_hit') is True
        if trace.get('error'):
            failures.append({'episode_id': row['episode_id'], 'error': trace['error'],
                             'trace': str(path), 'sha256': sha(path)})
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from src.orchestrator.llm_backend import APIBackend
    archived = reconstruct_archived_request(ARCHIVE, REQUEST_ID)
    config = archived['generation_config']
    context = json.loads((ARCHIVE/'context'/f'{REQUEST_ID}.json').read_text())
    assert context['status'] == 'error' and 'finish_reason=length' in context['error']
    body = {'model': 'ifv-psd-sft2056-safety', 'messages': archived['input_payload'],
            'tools': APIBackend._openai_tool_schemas(archived['tool_schema']),
            'tool_choice': 'auto', 'parallel_tool_calls': False,
            'max_tokens': context['max_output_tokens'],
            'chat_template_kwargs': {'enable_thinking': config['enable_thinking']},
            'return_token_ids': True}
    for key in ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty', 'seed']:
        if key in config:
            body[key] = config[key]
    assert body['max_tokens'] == 32768 and config['thinking_token_budget'] == 8192
    assert body['temperature'] == .7 and context['image_count'] == 15
    OUT.mkdir()
    save(OUT/'collection-integrity.json', {**integrity, 'validator_sha256': sha(VALIDATOR),
        'native_results_sha256': sha(CANARY/'episodes/run_results.jsonl'),
        'native_manifest_sha256': sha(CANARY/'episodes/run_manifest.json'),
        'raw_history_rewritten': False, 'psd_gate_passed': False})
    save(OUT/'canary-audit.json', {'slots': len(rows), 'errors': len(failures), 'calls': dict(calls),
        'failures': failures, 'psd_gate_passed': False, 'full_collection_started': False})
    save(OUT/'request.json', body)
    save(OUT/'binding.json', {'source_archive': str(ARCHIVE), 'request_id': REQUEST_ID,
        'context_sha256': sha(ARCHIVE/'context'/f'{REQUEST_ID}.json'), 'image_count': context['image_count'],
        'exact_original_wire_attested': False, 'purpose': 'reconstructed context with budget on/off',
        'tool_choice': 'auto', 'tool_choice_matches_original_wire': False,
        'limitation': 'Cannot rule out required/structured-output failures; use captured-wire probes',
        'not_training_or_evaluation': True, 'backend_pid': backend['pid'], 'backend_command': args})


def probe_plan(concurrent):
    if concurrent:
        return [(f'budget-{mode}-{index}', mode == 'on')
                for index in range(4) for mode in ('on', 'off')]
    return [('budget-on', True), ('budget-off-control', False)]


async def execute():
    import httpx
    base = json.loads((OUT/'request.json').read_text())
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(1230), trust_env=False) as client:
        # Fixed 2 serial or 8 concurrent requests; no retry, no Agent tools.
        async def one(label, enabled):
            body = {**base, 'cache_salt': 'ifv-raw-diagnostic-'+uuid.uuid4().hex}
            if enabled:
                body['vllm_xargs'] = {'ifv_thinking_budget': 8192}
            save(OUT/(label+'-request.json'), body)
            start = time.monotonic()
            try:
                response = await client.post('http://127.0.0.1:19004/v1/chat/completions', json=body)
                (OUT/(label+'-response.json')).write_bytes(response.content)
                response.raise_for_status()
                payload = response.json()
                choice = payload['choices'][0]
                ids = choice.get('token_ids') or []
                message = choice['message']
                result = {'label': label, 'seconds': time.monotonic()-start,
                    'finish_reason': choice['finish_reason'], 'tokens': len(ids),
                    'most_common_tokens': Counter(ids).most_common(6), 'zero_tokens': ids.count(0),
                    'endthink_indices': [i for i,t in enumerate(ids) if t == 248069][:10],
                    'reasoning_chars': len(message.get('reasoning_content') or message.get('reasoning') or ''),
                    'content_chars': len(message.get('content') or ''),
                    'tool_calls': len(message.get('tool_calls') or []), 'usage': payload.get('usage')}
            except Exception as error:
                result = {'label': label, 'seconds': time.monotonic()-start,
                          'error_type': type(error).__name__, 'response_saved': (OUT/(label+'-response.json')).exists()}
            save(OUT/(label+'-summary.json'), result)
            print(json.dumps(result), flush=True)
            return result
        if CONCURRENT:
            results = await asyncio.gather(*(one(label, enabled) for label, enabled in probe_plan(True)))
        else:
            for label, enabled in probe_plan(False):
                results.append(await one(label, enabled))
    save(OUT/'summary.json', {'results': results, 'psd_gate_passed': False,
        'full_collection_started': False, 'concurrency': 8 if CONCURRENT else 1, 'time': time.time()})


def main():
    global OUT, CONCURRENT
    os.umask(0o077)
    if '--concurrent' in sys.argv:
        CONCURRENT = True
        OUT = SERVICE/'cache-free-concurrency-diagnostic-v1'
    if '--launch' in sys.argv:
        prepare()
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
        if CONCURRENT:
            command.append('--concurrent')
        with (OUT/'run.log').open('x') as log:
            child = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(OUT/'process.json', {'pid': child.pid, 'command': command, 'script_sha256': sha(Path(__file__)),
                                'started_at': time.time()})
        state = json.loads((SERVICE/'state.json').read_text())
        state.update(phase='cache_free_canary_failed_raw_diagnostic', active_diagnostic=str(OUT),
                     gpu_verified=False, source_collection_started=False)
        save(SERVICE/'state.json', state)
        print(json.dumps({'launched': True, 'pid': child.pid, 'requests': len(probe_plan(CONCURRENT))}))
    elif '--execute' in sys.argv:
        asyncio.run(execute())
    else:
        raise ValueError('Use --launch or --execute')


if __name__ == '__main__':
    main()
