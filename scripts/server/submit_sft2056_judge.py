"""Independent, ID-joined epoch2 Batch submission with durable no-replay receipts.

Reuse only the frozen image encoding/helper functions, never the old two-source
submit stage. Explicit quota rejection may resume later; ambiguous creates halt.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
REPO = ROOT/'image-factual-verifier-v2'
WORK = ROOT/'evaluation/sft2056-v3-judge-20260916'
OUTPUT = ROOT/'runs/eval/sft2056-batch-judge-v3-low32k-20260916'
HELPER = ROOT/'training-artifacts/submit_completed_agent_judges_20260915.py'
SDK = ROOT/'training-artifacts/judge-sdk-20260915'
NAME = 'qwen35-9b-sft2056-agent'
JOURNAL = WORK/'submitted_shards.jsonl'


def save_durable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def join_cases(cases, candidates):
    by_id = {r['case_id']: (i, r) for i, r in enumerate(cases, 1)}
    if len(by_id) != len(cases):
        raise ValueError('Duplicate benchmark case')
    ids = [r['case_id'] for r in candidates]
    if len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise ValueError('Duplicate or unknown candidate case')
    return [(by_id[r['case_id']][0], by_id[r['case_id']][1], r) for r in candidates]


def check_intent(prior, fingerprint, now):
    if not prior:
        return 1
    if prior['fingerprint'] != fingerprint:
        raise ValueError('Changed request binding')
    if prior['state'] != 'server_rejected_429':
        raise ValueError('Ambiguous create; reconcile remote job before retry')
    if prior['attempt'] >= 6:
        raise ValueError('Quota retry budget exhausted')
    if now - prior['time'] < 3600:
        raise ValueError('Quota cooldown has not elapsed')
    return prior['attempt'] + 1


def load():
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == 'f32e364375e383f4a209bb95c7dff8cdffd12cc1b4059d9bdf32c5b72f576da4'
    spec = importlib.util.spec_from_file_location('epoch2_frozen_image_helpers', HELPER)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    sys.path[:0] = [str(SDK), str(REPO)]
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    record = json.loads((WORK/'prepared.json').read_text())
    assert record['count'] == 1524 and record['formal_denominator'] == 1527
    assert record['source_name'] == NAME
    assert record['source_sha256'] == f.digest(WORK/(NAME+'.jsonl.gz')) == '0bffefb2f4f9b6d9ecb8fcce4d8f0b64eb2cfd3cdb3fa2864cec032dae1a577c'
    assert record['cases_sha256'] == f.digest(f.CASES)
    assert record['gold_sha256'] == f.digest(f.GOLD)
    assert record['prompt_sha256'] == hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest()
    assert record['response_schema_sha256'] == hashlib.sha256(json.dumps(PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest()
    assert record['model'] == 'gemini-3.7-flash' and record['thinking_level'] == 'low'
    assert record['max_output_tokens'] == 32768
    candidates = f.rows(WORK/(NAME+'.jsonl.gz'))
    assert len(candidates) == record['count']
    joined = join_cases(f.rows(f.CASES), candidates)
    return f, record, joined, PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA


def plan(f, joined, prompt):
    shards, current, size, deferred = [], [], 0, []
    for index, case, candidate in joined:
        text = prompt + '\n\nAUDIT INPUT:\n' + json.dumps({
            'private_gold': candidate['gold'], 'candidate_material': {'mode': 'agent_trace',
            'candidate_answer': candidate['candidate_answer'],
            'supporting_trace_material': candidate['candidate_output']}}, ensure_ascii=False, indent=2)
        path = f.CASES.parent/case['image_path']
        raw, mime = f.wire_image(path)
        estimate = len(text.encode()) + 4*((len(raw)+2)//3) + 8192
        item = {'index': index, 'case_id': case['case_id'], 'text': text,
                'image_path': str(path), 'image_sha256': hashlib.sha256(raw).hexdigest(),
                'mime': mime, 'estimated_bytes': estimate}
        if estimate >= 15_000_000:
            # Shared File API quota is already known to be constrained. Do not
            # mutate anyone else's files or silently resize/drop this image.
            deferred.append({k: v for k, v in item.items() if k != 'text'})
            continue
        if current and (size + estimate > 15_000_000 or len(current) >= 20):
            shards.append(current)
            current, size = [], 0
        current.append(item)
        size += estimate
    if current:
        shards.append(current)
    assert sum(map(len, shards)) + len(deferred) == 1524
    f.save(WORK/'plan.json', {'shards': len(shards), 'inline_cases': sum(map(len, shards)),
           'deferred': deferred, 'case_join': 'exact_case_id', 'total_candidates': 1524})
    return shards, deferred


def submit(f, record, shards, deferred, schema):
    from google import genai
    from google.genai import types
    from dotenv import dotenv_values
    client = genai.Client(api_key=dotenv_values(ROOT/'private/runtime.env')['GEMINI_API_KEY'],
        http_options=types.HttpOptions(timeout=120000, retry_options=types.HttpRetryOptions(attempts=1)))
    config = types.GenerateContentConfig(max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_level='low', include_thoughts=True),
        response_mime_type='application/json', response_json_schema=schema)
    prior = f.rows(JOURNAL) if JOURNAL.exists() else []
    accepted = {row['shard_index']: row for row in prior}
    assert len(accepted) == len(prior)
    for offset, shard in enumerate(shards, 1):
        ids = [item['case_id'] for item in shard]
        fingerprint = hashlib.sha256(json.dumps({'record': record, 'shard': shard}, sort_keys=True).encode()).hexdigest()
        if offset in accepted:
            assert accepted[offset]['case_ids'] == ids
            assert accepted[offset]['fingerprint'] == fingerprint
            continue
        display = f'ifv-{NAME}-v3-low32k-20260916-s{offset:03d}'
        intent = WORK/'create-intents'/(display+'.json')
        previous = json.loads(intent.read_text()) if intent.exists() else None
        attempt = check_intent(previous, fingerprint, time.time())
        requests = []
        for item in shard:
            raw, mime = f.wire_image(Path(item['image_path']))
            assert hashlib.sha256(raw).hexdigest() == item['image_sha256']
            requests.append(types.InlinedRequest(contents=[types.Content(role='user', parts=[
                types.Part.from_bytes(data=raw, mime_type=mime), types.Part.from_text(text=item['text'])])],
                metadata={'key': f'c{item["index"]:04d}', 'case_id': item['case_id'], 'source_name': NAME}, config=config))
        receipt = {'display_name': display, 'fingerprint': fingerprint, 'case_ids': ids,
                   'source_sha256': record['source_sha256'], 'attempt': attempt, 'time': time.time()}
        save_durable(intent, {**receipt, 'state': 'create_inflight'})
        try:
            batch = client.batches.create(model=record['model'], src=requests, config={'display_name': display})
        except Exception as error:
            code = getattr(error, 'code', None)
            save_durable(intent, {**receipt, 'state': 'server_rejected_429' if code == 429 else 'ambiguous_or_rejected',
                           'error_type': type(error).__name__, 'http_status': code, 'time': time.time()})
            f.save(WORK/'submission-progress.json', {'phase': 'waiting_batch_quota' if code == 429 else 'held_for_inspection',
                'submitted_jobs': len(accepted), 'submitted_cases': sum(r['sample_count'] for r in accepted.values()),
                'deferred_cases': len(deferred), 'time': time.time(), 'http_status': code, 'attempt': attempt})
            if code == 429:
                return
            raise RuntimeError('Batch create held; inspect receipt without blind replay') from None
        assert batch.name
        value = {**receipt, 'source_name': NAME, 'shard_index': offset, 'shard_count': len(shards),
                 'sample_count': len(ids), 'case_indexes': [item['index'] for item in shard],
                 'batch_name': batch.name, 'state': getattr(batch.state, 'name', str(batch.state)),
                 'create_time': str(batch.create_time)}
        with JOURNAL.open('a') as stream:
            stream.write(json.dumps(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        accepted[offset] = value
        save_durable(intent, {**receipt, 'state': 'accepted', 'batch_name': batch.name})
        f.save(WORK/'submission-progress.json', {'phase': 'submitting', 'submitted_jobs': len(accepted),
            'submitted_cases': sum(r['sample_count'] for r in accepted.values()), 'time': time.time()})
    f.save(WORK/'submission-progress.json', {'phase': 'partial_deferred_image' if deferred else 'all_submitted',
        'submitted_jobs': len(accepted), 'submitted_cases': sum(r['sample_count'] for r in accepted.values()),
        'deferred_cases': len(deferred), 'time': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['plan', 'submit', 'collect'], required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.launch:
        receipt = WORK/(args.stage+'-process.json')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            proc = Path(f'/proc/{old["pid"]}/cmdline')
            assert not proc.exists() or not proc.read_bytes(), 'Existing controller still alive'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', args.stage]
        with (WORK/(args.stage+'.log')).open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', dir=WORK, delete=False) as handle:
            json.dump({'pid': child.pid, 'command': command, 'time': time.time()}, handle)
        Path(handle.name).replace(receipt)
        print(args.stage, 'LAUNCHED', child.pid, flush=True)
        return
    import fcntl
    lock = (WORK/(args.stage+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    f, record, joined, prompt, schema = load()
    if args.stage == 'collect':
        assert JOURNAL.exists()
        from dotenv import dotenv_values
        env = {**os.environ, 'GEMINI_API_KEY': dotenv_values(ROOT/'private/runtime.env')['GEMINI_API_KEY'],
               'PYTHONPATH': str(SDK)+os.pathsep+str(REPO), 'PYTHONNOUSERSITE': '1'}
        command = [sys.executable, str(REPO/'scripts/collect_gemini_inline_agent_audits.py'),
            '--journal', str(JOURNAL), '--source', NAME+'='+str(WORK/(NAME+'.jsonl.gz')),
            '--output-dir', str(OUTPUT), '--expected-cases', '1524', '--reported-denominator', '1527',
            '--judge-model', record['model'], '--thinking-level', 'low', '--workers', '2', '--wait', '--poll-seconds', '120']
        return subprocess.call(command, cwd=REPO, env=env)
    shards, deferred = plan(f, joined, prompt)
    if args.stage == 'submit':
        submit(f, record, shards, deferred, schema)
    print((WORK/'plan.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
