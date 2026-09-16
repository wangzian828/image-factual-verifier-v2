"""User-authorized tail repair: cancelled Batch items, exhausted 503s, GIF failure.

Preserves immutable old ownership, requests, receipts and successful results.
The additive plan transfers only explicitly cancelled items, never active jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/volume/ybo/wza')
WORK = ROOT / 'evaluation/sft3084-gemini31pro-v3-judge-20260915'
CONTROL = WORK / 'tail-repair-20260916-v1'
FROZEN = ROOT / 'training-artifacts/submit_completed_agent_judges_20260915.py'
GIF_CASE = 'route-aware-hrc-final-3000-20260816-input:2311:web_r005-web_crawled_refuted-web-backlog-00001'
GIF_SHA = '72aed244f121f6599e0a863674d77e2d72ec800109bf5935a9c37f5d7c2f0078'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_batch_item(row, valid_output, schema):
    if row.get('status') == 'completed' and valid_output(row.get('judge_output'), schema):
        return 'retain'
    if (row.get('status') == 'error' and row.get('error_type') == 'BatchItemError'
            and 'cancelled because the batch was cancelled' in str(row.get('error', '')).lower()):
        return 'transfer'
    raise ValueError('Unresolved Batch item; refusing automatic duplicate request')


def validate_history(directory, *, base_attempts, limit, bound_hashes):
    for filename, sha in bound_hashes.items():
        if digest(directory / filename) != sha:
            raise ValueError('Frozen request/attempt changed')
    attempts = sorted(p for p in directory.glob('attempt-*.json')
                      if not p.name.endswith('.response.json'))
    numbers = [json.loads(p.read_text())['attempt'] for p in attempts]
    if numbers != list(range(1, len(numbers) + 1)) or not base_attempts <= len(numbers) <= limit:
        raise ValueError('Attempt sequence or bounded budget changed')
    for path in attempts:
        row = json.loads(path.read_text())
        if path.with_suffix('.response.json').exists():
            continue  # Runner recovers the archived response without generating again.
        if row.get('state') != 'provider_rejected' or row.get('http_code') not in {408,429,500,502,503,504}:
            raise ValueError('Unreconciled transport or non-retryable attempt')


def prepare(runner, journal, ownership, module):
    bindings = {'journal_sha256': digest(runner.f.JOURNAL),
                'ownership_sha256': digest(WORK / 'transport-switch.json'),
                'source_hashes': runner.record['sources'],
                'runner_sha256': digest(Path(module.__file__))}
    checkpoint_hashes, transfer, retry = {}, [], []
    for job in journal:
        path = runner.collect._checkpoint_path(runner.f.OUTPUT / 'collector-state', job)
        if not path.exists():
            raise ValueError('Wait for all terminal Batch checkpoints before transfer')
        rows = runner.f.rows(path)
        if len(rows) != len(job['case_ids']) or {r['case_id'] for r in rows} != set(job['case_ids']):
            raise ValueError('Batch checkpoint identity mismatch')
        checkpoint_hashes[str(path)] = digest(path)
        for row in rows:
            if classify_batch_item(row, module.valid_output, runner.schema) == 'transfer':
                directory = runner.directory(job['source_name'], row['case_id'])
                if directory.exists() and list(directory.iterdir()):
                    raise ValueError('Cancelled Batch case unexpectedly already attempted by realtime')
                transfer.append({'source': job['source_name'], 'case': row['case_id'],
                                 'base_attempts': 0, 'limit': 6, 'bound_hashes': {}})
    for name, members in ownership['sources'].items():
        for case in members['realtime']:
            directory = runner.directory(name, case)
            if (directory / 'result.json').exists():
                continue
            status = json.loads((directory / 'status.json').read_text())
            request = json.loads((directory / 'request.json').read_text())
            if case == GIF_CASE:
                if request['image_sha256'] != GIF_SHA or request['image_mime'] != 'image/gif':
                    raise ValueError('GIF identity mismatch')
                continue
            attempts = sorted(p for p in directory.glob('attempt-*.json')
                              if not p.name.endswith('.response.json'))
            if (len(attempts) != 6 or status.get('state') != 'provider_rejected'
                    or list(directory.glob('attempt-*.response.json'))):
                raise ValueError('Only six explicit 503 rejections may receive this extension')
            for number, path in enumerate(attempts, 1):
                receipt = json.loads(path.read_text())
                if receipt.get('attempt') != number or receipt.get('state') != 'provider_rejected' or receipt.get('http_code') != 503:
                    raise ValueError('Unexpected exhausted attempt')
            hashes = {p.name: digest(p) for p in [directory / 'request.json', *attempts]}
            retry.append({'source': name, 'case': case, 'base_attempts': 6,
                          'limit': 8, 'bound_hashes': hashes})
    if len(retry) != 24:
        raise ValueError('Expected exactly the 24 diagnosed 503 cases')
    return {'schema': 'ifv-judge-tail-repair-v1', 'created_at': time.time(), 'bindings': bindings,
            'checkpoint_hashes': checkpoint_hashes, 'retry_503': retry, 'transferred_batch': transfer,
            'terminal_input_failures': {name: [GIF_CASE] for name in ownership['sources']},
            'reported_denominator': 1527, 'authorization': '2026-09-16: user requested judge repair, cancelled stalled Batch, and GIF counted as failure'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','run','merge'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use one to four workers')
    os.umask(0o077)
    import fcntl
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'submit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(ROOT / 'image-factual-verifier-v2'))
    module = load(Path(__file__).with_name('run_realtime_agent_judges.py'), 'tail_runner')
    f = load(FROZEN, 'tail_frozen')
    record = f.prepare()
    candidates = {name: {row['case_id']: row for row in f.rows(WORK / (name + '.jsonl.gz'))} for name in f.SOURCES}
    journal = f.rows(f.JOURNAL)
    ownership = json.loads((WORK / 'transport-switch.json').read_text())
    if digest(f.JOURNAL) != ownership['journal_sha256'] or module.partition(candidates, journal, record) != ownership['sources']:
        raise ValueError('Original ownership changed')
    module.check_intents([json.loads(p.read_text()) for p in (WORK/'create-intents').glob('*.json')], journal)
    from google import genai
    from google.genai import types
    from dotenv import dotenv_values
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT as prompt, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA as schema
    if (hashlib.sha256(prompt.encode()).hexdigest() != record['prompt_sha256'] or
            hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest() != record['response_schema_sha256']):
        raise ValueError('Judge protocol changed')
    client = genai.Client(api_key=dotenv_values(ROOT/'private/runtime.env')['GEMINI_API_KEY'],
        http_options=types.HttpOptions(timeout=300000, retry_options=types.HttpRetryOptions(attempts=1)))
    runner = module.Runner(f, client, types, record, candidates, ownership, schema, prompt)
    plan_path = CONTROL / 'plan.json'
    if args.stage == 'prepare':
        plan = prepare(runner, journal, ownership, module)
        with plan_path.open('x') as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        print(json.dumps({'retry_503':len(plan['retry_503']), 'transferred_batch':len(plan['transferred_batch']),
                          'gif_terminal_failures':2, 'plan':str(plan_path)}), flush=True)
        client.close()
        return
    plan = json.loads(plan_path.read_text())
    expected = {'journal_sha256':digest(f.JOURNAL), 'ownership_sha256':digest(WORK/'transport-switch.json'),
                'source_hashes':record['sources'], 'runner_sha256':digest(Path(module.__file__))}
    if plan['bindings'] != expected:
        raise ValueError('Repair binding changed')
    for path, sha in plan['checkpoint_hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError('Batch checkpoint changed')
    effective = copy.deepcopy(ownership)
    for item in plan['transferred_batch']:
        members = effective['sources'][item['source']]
        members['batch'].remove(item['case'])
        members['realtime'].append(item['case'])
    jobs = plan['retry_503'] + plan['transferred_batch']
    runner.ownership = effective
    runner.attempt_limits = {(x['source'],x['case']):x['limit'] for x in jobs}
    runner.terminal_failed_cases = plan['terminal_input_failures']
    for name, cases in runner.terminal_failed_cases.items():
        for case in cases:
            f.save(runner.directory(name,case)/'terminal-input-failure.json',
                   {'state':'terminal_input_failure','reason':'GIF counted as failure per user; do not submit',
                    'image_sha256':GIF_SHA,'reported_denominator':1527})
    if args.stage == 'merge':
        print(json.dumps(runner.merge()), flush=True)
        client.close()
        return
    pending_jobs = []
    for item in jobs:
        directory = runner.directory(item['source'],item['case'])
        validate_history(directory, base_attempts=item['base_attempts'], limit=item['limit'], bound_hashes=item['bound_hashes'])
        if (directory/'result.json').exists():
            continue
        attempts = [p for p in directory.glob('attempt-*.json') if not p.name.endswith('.response.json')]
        if len(attempts) < item['limit'] or list(directory.glob('attempt-*.response.json')):
            pending_jobs.append(item)
    runner.merge()  # Validate retained results before dispatch.
    receipt = {'pid':os.getpid(),'started_at':time.time(),'script_sha256':digest(Path(__file__)),
               'runner_sha256':digest(Path(module.__file__)),'plan_sha256':digest(plan_path),
               'workers':args.workers,'selected_cases':len(pending_jobs)}
    f.save(CONTROL/('process-'+str(os.getpid())+'.json'),receipt)
    queue = iter(pending_jobs); pending={}; failures=0; completed=0; exhausted=False; last_merge=0; held=False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while pending or not exhausted:
            while len(pending)<args.workers and not exhausted and failures<3:
                try:item=next(queue)
                except StopIteration:exhausted=True;break
                pending[executor.submit(runner.run_one,item['source'],item['case'])]=item
            if failures>=3: held=True;exhausted=True
            done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                item=pending.pop(future)
                try:outcome=future.result()
                except Exception as exc:
                    outcome='internal_error'
                    f.save(CONTROL/('error-'+hashlib.sha256((item['source']+item['case']).encode()).hexdigest()[:20]+'.json'),
                           {'error_type':type(exc).__name__,'time':time.time()})
                if outcome=='completed':completed+=1;failures=0
                else:failures+=1
                print(json.dumps({'source':item['source'],'case_id':item['case'],'state':outcome}),flush=True)
            f.save(CONTROL/'progress.json',{'phase':'running','time':time.time(),'in_flight':len(pending),
                'completed_this_run':completed,'consecutive_errors':failures,'selected_cases':len(pending_jobs)})
            if time.time()-last_merge>60:runner.merge();last_merge=time.time()
    summaries=runner.merge()
    f.save(CONTROL/'progress.json',{'phase':'held_consecutive_errors' if held else 'pass_finished',
        'time':time.time(),'in_flight':0,'completed_this_run':completed,'sources':summaries})
    client.close()


if __name__ == '__main__':
    main()
