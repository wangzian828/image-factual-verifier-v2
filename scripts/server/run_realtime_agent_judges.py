"""Switch unsubmitted frozen v3 judges to ordinary generateContent, without overlap.

Server-only inputs and outputs. Accepted Batch requests retain exclusive ownership.
Normal responses are durably archived before validation; restarts recover them.
No protocol, image, candidate-material, model or judge-quality changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


def audit_text(candidate, prompt):
    return prompt + '\n\nAUDIT INPUT:\n' + json.dumps({
        'private_gold': candidate['gold'],
        'candidate_material': {
            'mode': 'agent_trace', 'candidate_answer': candidate['candidate_answer'],
            'supporting_trace_material': candidate['candidate_output'],
        },
    }, ensure_ascii=False, indent=2)


def partition(candidates, journal, record):
    owned = {name: set() for name in candidates}
    displays = set()
    batches = set()
    for row in journal:
        name = row['source_name']
        if name not in owned or row['source_sha256'] != record['sources'][name]['sha256']:
            raise ValueError('Unknown source or changed Batch source hash')
        ids = row['case_ids']
        if len(ids) != len(set(ids)) or len(ids) != row['sample_count']:
            raise ValueError('Invalid Batch case membership')
        if owned[name].intersection(ids) or not set(ids).issubset(candidates[name]):
            raise ValueError('Duplicate or unknown Batch case')
        if row['display_name'] in displays or row['batch_name'] in batches:
            raise ValueError('Duplicate Batch job')
        displays.add(row['display_name'])
        batches.add(row['batch_name'])
        owned[name].update(ids)
    return {name: {'batch': sorted(owned[name]),
                   'realtime': [case for case in values if case not in owned[name]]}
            for name, values in candidates.items()}


def check_intents(intents, journal):
    accepted = {row['display_name'] for row in journal}
    for intent in intents:
        if intent['display_name'] not in accepted and intent.get('state') != 'server_rejected_429':
            raise ValueError('Ambiguous Batch create: reconcile before realtime dispatch')


def valid_output(value, schema):
    if not isinstance(value, dict) or set(value) != set(schema['required']):
        return False
    for name, definition in schema['properties'].items():
        item = value[name]
        if definition['type'] == 'string':
            if not isinstance(item, str) or ('enum' in definition and item not in definition['enum']):
                return False
        elif definition['type'] == 'array':
            if not isinstance(item, list) or len(item) > definition.get('maxItems', float('inf')):
                return False
            if any(not isinstance(entry, str) for entry in item):
                return False
        else:
            raise ValueError('Unhandled schema type')
    return True


def retry_delay(attempt, code, *, limit=6):
    # Only explicit transient provider rejections; ambiguous transport errors pause.
    if code not in {408, 429, 500, 502, 503, 504} or attempt >= limit:
        return None
    return min(300, 15 * 2 ** (attempt - 1)) + random.uniform(0, 5)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_unattempted(jobs, directory_for):
    """Resume untouched cases without resetting receipts or replaying old failures."""
    pending, skipped = [], Counter()
    for name, case in jobs:
        directory = directory_for(name, case)
        if (directory / 'result.json').exists():
            skipped['completed'] += 1
            continue
        attempts = sorted(p for p in directory.glob('attempt-*.json')
                          if not p.name.endswith('.response.json'))
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if list(directory.glob('attempt-*.response.json')):
            raise ValueError('Recover archived response before unattempted-only resume')
        if attempts:
            if (any(json.loads(p.read_text()).get('state') != 'provider_rejected' for p in attempts)
                    or status.get('state') != 'provider_rejected'):
                raise ValueError('Reconcile nonterminal or ambiguous attempts before resume')
            skipped['previously_rejected'] += 1
        elif str(status.get('state', '')).startswith('deferred_'):
            skipped['deferred'] += 1
        elif status:
            raise ValueError('Unexpected status without attempt receipts')
        else:
            pending.append((name, case))
    return pending, dict(skipped)


def final_phase(stage, held_for_errors):
    if stage.startswith('held'):
        return stage
    return 'held_consecutive_errors' if held_for_errors else 'realtime_pass_finished'


class Runner:
    def __init__(self, frozen, client, types, record, candidates, ownership, schema, prompt,
                 *, attempt_limits=None, terminal_failed_cases=None):
        self.f, self.client, self.types = frozen, client, types
        self.record, self.candidates, self.ownership = record, candidates, ownership
        self.schema, self.prompt = schema, prompt
        self.attempt_limits = attempt_limits or {}
        if any(not 1 <= limit <= 8 for limit in self.attempt_limits.values()):
            raise ValueError('Retry extension must remain bounded by eight total attempts')
        self.terminal_failed_cases = terminal_failed_cases or {}
        self.root = frozen.ROOT / 'runs/eval/sft3084-gemini31pro-hybrid-judge-v3-low32k-20260915'
        self.cases = {row['case_id']: row for row in frozen.rows(frozen.CASES)}
        self.indexes = {case: index for index, case in enumerate(self.cases, 1)}
        self.collect = load_module(frozen.REPO / 'scripts/collect_gemini_inline_agent_audits.py', 'frozen_collector')
        self.config = types.GenerateContentConfig(
            max_output_tokens=32768,
            thinking_config=types.ThinkingConfig(thinking_level='low', include_thoughts=True),
            response_mime_type='application/json', response_json_schema=schema)

    def directory(self, name, case):
        token = hashlib.sha256(case.encode()).hexdigest()[:20]
        return self.root / 'realtime' / name / f'c{self.indexes[case]:04d}-{token}'

    def normalize(self, name, case, response, request_sha, raw_path):
        choices = response.candidates or []
        if len(choices) != 1 or str(getattr(choices[0].finish_reason, 'name', choices[0].finish_reason)) != 'STOP':
            raise ValueError('Judge did not finish normally')
        result = self.collect._audit_record(
            source_name=name, candidate=self.candidates[name][case],
            item=SimpleNamespace(metadata={'case_id': case, 'key': f'c{self.indexes[case]:04d}'},
                                 response=response, error=None),
            judge_model=self.f.MODEL, thinking_level='low')
        if not valid_output(result.get('judge_output'), self.schema):
            raise ValueError('Judge JSON does not match frozen schema')
        result.update(transport='realtime_generate_content', max_output_tokens=32768,
                      request_sha256=request_sha, raw_response=str(raw_path),
                      source_sha256=self.record['sources'][name]['sha256'])
        result['private_gold_category'] = self.collect._category(result)
        return result

    def run_one(self, name, case):
        if case not in self.ownership['sources'][name]['realtime']:
            raise ValueError('Attempt to steal Batch-owned case')
        if case in self.terminal_failed_cases.get(name, set()):
            return 'terminal_input_failure'
        limit = self.attempt_limits.get((name, case), 6)
        directory = self.directory(name, case)
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / 'result.json'
        if result_path.exists():
            return 'completed'
        text = audit_text(self.candidates[name][case], self.prompt)
        raw, mime = self.f.wire_image(self.f.CASES.parent / self.cases[case]['image_path'])
        image_sha = hashlib.sha256(raw).hexdigest()
        binding = {'source_name': name, 'case_id': case,
                   'source_sha256': self.record['sources'][name]['sha256'],
                   'image_sha256': image_sha, 'image_mime': mime,
                   'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'model': self.f.MODEL, 'thinking_level': 'low',
                   'include_thoughts': True, 'max_output_tokens': 32768,
                   'response_schema_sha256': self.record['response_schema_sha256']}
        request_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        request_path = directory / 'request.json'
        if request_path.exists() and json.loads(request_path.read_text()) != binding:
            raise ValueError('Changed request identity')
        self.f.save(request_path, binding)
        attempts = sorted(path for path in directory.glob('attempt-*.json')
                          if not path.name.endswith('.response.json'))
        for path in attempts:
            receipt = json.loads(path.read_text())
            raw_path = path.with_suffix('.response.json')
            if raw_path.exists():
                response = self.types.GenerateContentResponse.model_validate(json.loads(raw_path.read_text()))
                try:
                    result = self.normalize(name, case, response, request_sha, raw_path)
                except ValueError:
                    return 'invalid_response'
                self.f.save(result_path, result)
                return 'completed'
            if receipt['state'] == 'in_flight':
                self.f.save(directory / 'status.json', {'state': 'ambiguous_transport', 'time': time.time()})
                return 'ambiguous_transport'
            if receipt['state'] not in {'provider_rejected'}:
                return receipt['state']
        # Respect conservative image-doc limit. Do not resize or omit large images.
        estimated = len(text.encode()) + 4 * ((len(raw) + 2) // 3) + 8192
        if estimated >= 20_000_000:
            registry = self.f.WORK / 'large-image-files.json'
            files = json.loads(registry.read_text()) if registry.exists() else {}
            if image_sha not in files:
                self.f.save(directory / 'status.json', {
                    'state': 'deferred_file_quota', 'request_bytes_estimate': estimated,
                    'image_bytes': len(raw), 'time': time.time()})
                return 'deferred_file_quota'
            remote = self.client.files.get(name=files[image_sha]['name'])
            if str(getattr(remote.state, 'name', remote.state)) != 'ACTIVE':
                return 'deferred_file_not_active'
            image = self.types.Part.from_uri(file_uri=files[image_sha]['uri'], mime_type=mime)
        else:
            image = self.types.Part.from_bytes(data=raw, mime_type=mime)
        contents = [self.types.Content(role='user', parts=[image, self.types.Part.from_text(text=text)])]
        for attempt in range(len(attempts) + 1, limit + 1):
            if attempts:
                previous = json.loads(attempts[-1].read_text())
                if retry_delay(previous['attempt'], previous.get('http_code'), limit=limit) is None:
                    return 'retry_exhausted'
            receipt_path = directory / f'attempt-{attempt:02d}.json'
            receipt = {'state': 'in_flight', 'attempt': attempt,
                       'request_sha256': request_sha, 'started_at': time.time()}
            self.f.save(receipt_path, receipt)
            self.f.save(directory / 'status.json', receipt)
            try:
                response = self.client.models.generate_content(model=self.f.MODEL, contents=contents, config=self.config)
            except Exception as error:
                code = getattr(error, 'code', None)
                try:
                    code = int(code) if code is not None else None
                except (TypeError, ValueError):
                    code = None
                receipt.update(state='provider_rejected' if code else 'ambiguous_transport',
                               http_code=code, error_type=type(error).__name__, ended_at=time.time())
                self.f.save(receipt_path, receipt)
                delay = retry_delay(attempt, code, limit=limit)
                self.f.save(directory / 'status.json', {**receipt, 'retry_in_seconds': delay})
                if delay is None:
                    return receipt['state']
                time.sleep(delay)
                continue
            raw_path = receipt_path.with_suffix('.response.json')
            self.f.save(raw_path, response.model_dump(mode='json', exclude_none=True))
            try:
                result = self.normalize(name, case, response, request_sha, raw_path)
            except ValueError:
                receipt.update(state='invalid_response', ended_at=time.time())
                self.f.save(receipt_path, receipt)
                self.f.save(directory / 'status.json', receipt)
                return 'invalid_response'
            self.f.save(result_path, result)
            receipt.update(state='completed', ended_at=time.time())
            self.f.save(receipt_path, receipt)
            self.f.save(directory / 'status.json', receipt)
            return 'completed'
        return 'retry_exhausted'

    def merge(self):
        if self.f.digest(self.f.JOURNAL) != self.ownership['journal_sha256']:
            raise ValueError('Batch journal changed after transport switch')
        outputs = {name: {} for name in self.candidates}
        for job in self.f.rows(self.f.JOURNAL):
            path = self.collect._checkpoint_path(self.f.OUTPUT / 'collector-state', job)
            if not path.exists():
                continue
            batch_rows = self.f.rows(path)
            ids = [row['case_id'] for row in batch_rows]
            if len(ids) != len(set(ids)) or set(ids) != set(job['case_ids']):
                raise ValueError('Batch checkpoint identity mismatch')
            for row in batch_rows:
                if row.get('status') == 'completed' and valid_output(row.get('judge_output'), self.schema):
                    outputs[job['source_name']][row['case_id']] = {**row, 'transport': 'batch'}
        summaries = {}
        for name, members in self.ownership['sources'].items():
            statuses = Counter()
            for case in members['realtime']:
                directory = self.directory(name, case)
                result_path = directory / 'result.json'
                if case in self.terminal_failed_cases.get(name, set()):
                    if result_path.exists():
                        raise ValueError('Cannot relabel an existing judge result as input failure')
                    statuses['terminal_input_failure'] += 1
                    continue
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                    if (row['case_id'] != case or row['source_name'] != name or
                            row['source_sha256'] != self.record['sources'][name]['sha256'] or
                            not valid_output(row.get('judge_output'), self.schema)):
                        raise ValueError('Invalid realtime checkpoint')
                    if case in outputs[name]:
                        raise ValueError('Duplicate Batch/realtime result')
                    outputs[name][case] = row
                    statuses['completed'] += 1
                elif (directory / 'status.json').exists():
                    statuses[json.loads((directory / 'status.json').read_text())['state']] += 1
                else:
                    statuses['pending'] += 1
            ordered = [outputs[name][case] for case in self.candidates[name] if case in outputs[name]]
            categories = Counter(self.collect._category(row) or 'uncategorized' for row in ordered)
            strict = categories['correct_point_with_strong_evidence']
            complete = len(ordered) == len(self.candidates[name])
            failed = statuses['terminal_input_failure']
            finished = len(ordered) + failed == len(self.candidates[name])
            summary = {'source_name': name, 'complete': complete, 'valid_judges': len(ordered),
                       'finished': finished, 'terminal_failed_judges': failed,
                       'batch_reserved': len(members['batch']),
                       'batch_valid': len(ordered) - statuses['completed'],
                       'realtime_statuses': dict(statuses), 'available_cases': 1526,
                       'reported_denominator': 1527, 'judge_model': self.f.MODEL,
                       'thinking_level': 'low', 'max_output_tokens': 32768,
                       'private_gold_categories': dict(categories),
                       'strict_evidence_sufficient_count': strict,
                       'sesr_reported_percent': 100 * strict / 1527 if finished else None,
                       'time': time.time()}
            self.collect._write_gzip_jsonl(self.root / name / 'audit-results.jsonl.gz', ordered)
            self.f.save(self.root / name / 'summary.json', summary)
            summaries[name] = summary
        self.f.save(self.root / 'summary.json', {'sources': summaries, 'time': time.time()})
        return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['run', 'merge'], default='run')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--unattempted-only', action='store_true',
                        help='After diagnosis, continue untouched cases; preserve all prior attempts')
    parser.add_argument('--frozen-script', type=Path, default=Path('/volume/ybo/wza/training-artifacts/submit_completed_agent_judges_20260915.py'))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Initial realtime rollout allows 1-4 workers')
    import os
    import fcntl
    os.umask(0o077)
    f = load_module(args.frozen_script, 'frozen_submitter')
    lock = (f.WORK / 'submit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = f.prepare()
    candidates = {name: {row['case_id']: row for row in f.rows(f.WORK / (name + '.jsonl.gz'))}
                  for name in f.SOURCES}
    journal = f.rows(f.JOURNAL)
    sources = partition(candidates, journal, record)
    check_intents([json.loads(p.read_text()) for p in (f.WORK / 'create-intents').glob('*.json')], journal)
    marker = f.WORK / 'transport-switch.json'
    ownership = {'schema': 'ifv-batch-realtime-exclusive-v1', 'sources': sources,
                 'journal_sha256': f.digest(f.JOURNAL), 'source_hashes': record['sources']}
    if marker.exists():
        if json.loads(marker.read_text()) != ownership:
            raise ValueError('Ownership changed; refusing dispatch')
    elif args.stage == 'merge':
        raise ValueError('Missing transport ownership receipt')
    else:
        f.save(marker, ownership)
    from google import genai
    from google.genai import types
    from dotenv import dotenv_values
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    assert hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest() == record['prompt_sha256']
    assert hashlib.sha256(json.dumps(PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest() == record['response_schema_sha256']
    client = genai.Client(api_key=dotenv_values(f.ROOT / 'private/runtime.env')['GEMINI_API_KEY'],
                         http_options=types.HttpOptions(timeout=300000,
                             retry_options=types.HttpRetryOptions(attempts=1)))
    runner = Runner(f, client, types, record, candidates, ownership,
                    PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, PRIVATE_GOLD_JUDGE_PROMPT)
    if args.stage == 'merge':
        print(json.dumps(runner.merge()), flush=True)
        return 0
    f.save(runner.root / 'process.json', {'pid': os.getpid(), 'started_at': time.time(),
                                        'workers': args.workers, 'script_sha256': f.digest(Path(__file__)),
                                        'unattempted_only': args.unattempted_only})
    jobs = []
    for index in range(max(len(v['realtime']) for v in sources.values())):
        for name, members in sources.items():
            if index < len(members['realtime']):
                jobs.append((name, members['realtime'][index]))
    pilot_file = runner.root / 'pilot-acceptance.json'
    pilot = not pilot_file.exists()
    if args.unattempted_only:
        if pilot:
            raise ValueError('Unattempted-only resume requires accepted pilot')
        jobs, skipped = select_unattempted(jobs, runner.directory)
        f.save(runner.root / ('resume-selection-' + str(time.time_ns()) + '.json'), {
            'time': time.time(), 'mode': 'unattempted_only', 'workers': args.workers,
            'selected_cases': jobs, 'skipped': skipped,
            'journal_sha256': ownership['journal_sha256'],
            'script_sha256': f.digest(Path(__file__))})
    queue = iter(jobs[:4] if pilot else jobs)
    pending = {}
    failures = 0
    stage = 'pilot' if pilot else 'realtime_running'
    completed_this_run = 0
    last_merge = 0
    held_for_errors = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < args.workers and not exhausted and failures < 3:
                try:
                    name, case = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(runner.run_one, name, case)] = (name, case)
            if failures >= 3:
                held_for_errors = True
                exhausted = True
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                name, case = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = 'internal_error'
                    f.save(runner.directory(name, case) / 'status.json', {
                        'state': outcome, 'error_type': type(error).__name__, 'time': time.time()})
                print(json.dumps({'source': name, 'case_id': case, 'state': outcome, 'time': time.time()}), flush=True)
                if outcome == 'completed':
                    failures = 0
                    completed_this_run += 1
                elif not outcome.startswith('deferred_'):
                    failures += 1
            f.save(runner.root / 'progress.json', {'phase': stage, 'time': time.time(),
                'in_flight': len(pending), 'completed_this_run': completed_this_run,
                'consecutive_errors': failures, 'workers': args.workers})
            if time.time() - last_merge > 60:
                runner.merge()
                last_merge = time.time()
            if pilot and exhausted and not pending:
                if all((runner.directory(n, c) / 'result.json').exists() for n, c in jobs[:4]):
                    f.save(pilot_file, {'cases': jobs[:4], 'time': time.time(),
                                       'journal_sha256': ownership['journal_sha256'],
                                       'model': f.MODEL, 'thinking_level': 'low', 'max_output_tokens': 32768})
                    pilot = False
                    queue, exhausted = iter(jobs[4:]), False
                    stage = 'realtime_running'
                    print('PILOT_ACCEPTED_FULL_DISPATCH', flush=True)
                else:
                    stage = 'held_pilot_failed'
                    break
    summaries = runner.merge()
    f.save(runner.root / 'progress.json', {'phase': final_phase(stage, held_for_errors),
        'time': time.time(), 'in_flight': 0, 'sources': summaries})
    client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
