"""Bounded archived-context budget/APC diagnosis, never a PSD training target."""
from __future__ import annotations
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916'
OUT = SERVICE/'archived-context-budget-ab-v2'
REQUEST = ROOT/'runs/psd-real-runtime-gate-20260916-v2/failed-request-diagnostic/request.json'


def helpers():
    spec = importlib.util.spec_from_file_location('owned_service_process_helpers', CODE/'probe_psd_no_apc_backend_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def diagnose(f):
    import httpx
    import jsonschema
    base = json.loads(REQUEST.read_text())
    base['model'] = 'ifv-psd-sft2056-safety'
    assert base.pop('thinking_token_budget') == 8192
    base['max_tokens'] = 32768
    base['vllm_xargs'] = {'ifv_thinking_budget': 8192}
    base['return_token_ids'] = True
    schemas = {row['function']['name']: row['function']['parameters'] for row in base['tools']}
    results = []
    async with httpx.AsyncClient(timeout=1230, trust_env=False) as client:
        async def one(port, round_index):
            body = copy.deepcopy(base)
            body['cache_salt'] = 'ifv-psd-ab-'+uuid.uuid4().hex
            label = f'port-{port}-round-{round_index}'
            f.save(OUT/(label+'-request.json'), body)
            start = time.monotonic()
            response = await client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=body)
            (OUT/(label+'-response.json')).write_bytes(response.content)
            response.raise_for_status()
            choice = response.json()['choices'][0]
            ids = choice.get('token_ids') or []
            calls = choice['message'].get('tool_calls') or []
            errors = []
            if len(calls) != 1:
                errors.append('Expected exactly one native tool call')
            else:
                function = calls[0]['function']
                try:
                    jsonschema.validate(json.loads(function['arguments']), schemas[function['name']])
                except (ValueError, KeyError, jsonschema.ValidationError) as error:
                    errors.append(type(error).__name__+': '+str(error)[:400])
            closure = ids.index(248069) if 248069 in ids else None
            result = {'port': port, 'round': round_index, 'seconds': time.monotonic()-start,
                      'tokens': len(ids), 'zero_tokens': ids.count(0), 'finish': choice['finish_reason'],
                      'closure_index': closure, 'tool_schema_errors': errors,
                      'passed': choice['finish_reason'] == 'tool_calls' and not errors
                                and closure is not None and closure <= 8192
                                and ids.count(0) < max(1, len(ids)//2)}
            f.save(OUT/(label+'-summary.json'), result)
            return result
        for round_index in [1, 2]:
            batch = await asyncio.gather(*(one(p, round_index) for p in [19004, 19003]), return_exceptions=True)
            results.extend({'passed': False, 'error_type': type(r).__name__} if isinstance(r, BaseException) else r for r in batch)
    f.save(OUT/'summary.json', {'results': results, 'real_agent_gate_passed': False, 'source_collection_started': False})


def main():
    import httpx  # Fail before stopping the guard if the environment is wrong.
    import jsonschema
    f = helpers()
    if '--launch' in sys.argv:
        receipt = SERVICE/'archived-context-budget-ab-v2-process.json'
        assert not OUT.exists() and not receipt.exists()
        with receipt.open('x') as reserved:
            json.dump({'phase': 'launch_reserved'}, reserved)
        command = [sys.executable, '-u', str(Path(__file__).resolve())]
        child = f.spawn(command, os.environ.copy(), SERVICE/'archived-context-budget-ab-v2.log')
        f.save(receipt, {'pid': child.pid, 'command': command, 'time': time.time()})
        print('BUDGET_AB_LAUNCHED', child.pid, flush=True)
        return
    assert all(x['inflight'] == 0 for x in f.get('http://127.0.0.1:19001/health')['replicas'])
    guard, args, env = f.process(SERVICE/'guard.json')
    OUT.mkdir()
    f.save(OUT/'guard-before.json', guard)
    state = json.loads((SERVICE/'state.json').read_text())
    state.update(phase='archived_context_budget_ab_running', active_diagnostic=str(OUT),
                 gpu_verified=False, source_collection_started=False)
    f.save(SERVICE/'state.json', state)
    f.stop(guard['pid'])
    try:
        asyncio.run(diagnose(f))
    finally:
        child = f.spawn(args, env, SERVICE/'guard.log')
        f.save(SERVICE/'guard.json', {'pid': child.pid, 'command': args, 'started_at': time.time(), 'replaces': guard['pid']})
        state['phase'] = 'archived_context_budget_ab_requires_inspection'
        f.save(SERVICE/'state.json', state)
        print('GUARD_RESTORED', child.pid, flush=True)


if __name__ == '__main__':
    main()
