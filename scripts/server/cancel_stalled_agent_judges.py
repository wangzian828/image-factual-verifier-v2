"""Cancel only this experiment's unfinished Batch jobs; retain all evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/volume/ybo/wza')
WORK = ROOT / 'evaluation/sft3084-gemini31pro-v3-judge-20260915'
CONTROL = WORK / 'batch-cancellation-20260916'
JOURNAL = WORK / 'submitted_shards.jsonl'
ACTIVE = {'JOB_STATE_RUNNING', 'JOB_STATE_PENDING', 'JOB_STATE_QUEUED'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def job_key(name):
    return hashlib.sha256(name.encode()).hexdigest()[:20]


def state_name(job):
    return str(getattr(job.state, 'name', job.state))


def select_targets(journal, states):
    names = [row['batch_name'] for row in journal]
    if len(names) != len(set(names)) or set(names) != set(states):
        raise ValueError('Batch identity mismatch')
    return [row for row in journal if states[row['batch_name']] in ACTIVE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['cancel', 'status'], required=True)
    args = parser.parse_args()
    os.umask(0o077)
    import fcntl
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'control.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ownership = json.loads((WORK / 'transport-switch.json').read_text())
    if digest(JOURNAL) != ownership['journal_sha256']:
        raise ValueError('Journal changed')
    journal = [json.loads(line) for line in JOURNAL.read_text().split('\n') if line.strip()]
    sys.path.insert(0, str(ROOT / 'training-artifacts/judge-sdk-20260915'))
    from dotenv import dotenv_values
    from google import genai
    from google.genai import types
    key = dotenv_values(ROOT / 'private/runtime.env')['GEMINI_API_KEY']

    def client():
        return genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))

    def inspect(row):
        with client() as api:
            job = api.batches.get(name=row['batch_name'])
        snapshot = CONTROL / 'snapshots' / (job_key(row['batch_name']) + '-' + str(time.time_ns()) + '.json')
        save_new(snapshot, job.model_dump(mode='json', exclude_none=True))
        return row['batch_name'], {'state': state_name(job), 'snapshot': str(snapshot),
                                  'snapshot_sha256': digest(snapshot), 'time': time.time()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        before = dict(pool.map(inspect, journal))
    plan_path = CONTROL / 'plan.json'
    if args.stage == 'cancel':
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['journal_sha256'] != digest(JOURNAL):
                raise ValueError('Cancellation plan journal changed')
        else:
            targets = select_targets(journal, {name: row['state'] for name, row in before.items()})
            plan = {'time': time.time(), 'authorization': 'User requested clearing stalled Batch jobs on 2026-09-16',
                    'journal_sha256': digest(JOURNAL), 'targets': targets, 'before': before}
            save_new(plan_path, plan)

        def cancel(row):
            name = row['batch_name']
            if before[name]['state'] not in ACTIVE:
                return {'batch_key': job_key(name), 'action': 'already_not_active', 'state': before[name]['state']}
            intent = CONTROL / 'intents' / (job_key(name) + '.json')
            if intent.exists():
                return {'batch_key': job_key(name), 'action': 'prior_cancel_intent_preserved'}
            save_new(intent, {'batch_name': name, 'time': time.time(), 'before': before[name]})
            try:
                with client() as api:
                    api.batches.cancel(name=name)
                result = {'batch_key': job_key(name), 'action': 'cancel_acknowledged', 'time': time.time()}
            except Exception as exc:
                result = {'batch_key': job_key(name), 'action': 'cancel_unconfirmed',
                          'error_type': type(exc).__name__, 'http_code': getattr(exc, 'code', None), 'time': time.time()}
            save_new(CONTROL / 'receipts' / (job_key(name) + '.json'), result)
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(cancel, plan['targets']):
                print(json.dumps(result), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        after = dict(pool.map(inspect, journal))
    summary = {'time': time.time(), 'journal_sha256': digest(JOURNAL), 'jobs': after}
    save_new(CONTROL / ('status-' + str(time.time_ns()) + '.json'), summary)
    from collections import Counter
    print(json.dumps({'states': dict(Counter(row['state'] for row in after.values())),
                      'control': str(CONTROL)}), flush=True)


if __name__ == '__main__':
    main()
