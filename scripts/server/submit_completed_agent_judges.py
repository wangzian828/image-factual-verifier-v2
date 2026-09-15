"""Frozen 2026-09-15 v3 judges: epoch-3 Qwen and Gemini 3.1 Pro Agent.

All inputs/results stay under the authorized server root. Inline Batch avoids
the shared File API quota. Ambiguous creates halt instead of duplicating jobs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
REPO = ROOT / 'image-factual-verifier-v2'
WORK = ROOT / 'evaluation/sft3084-gemini31pro-v3-judge-20260915'
OUTPUT = ROOT / 'runs/eval/sft3084-gemini31pro-batch-judge-v3-low32k-20260915'
CASES = ROOT / 'evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl'
GOLD = ROOT / 'data/factcheck-test-1527-filtered-frozen-20260909/evaluator_private/private-gold-v1/private-gold.jsonl'
SFT = ROOT / 'runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915'
PRO = ROOT / 'runs/eval/gemini31pro-agent-formal1527-20260913'
SOURCES = {'qwen35-9b-sft3084-agent':'ifv-qwen3.5-9b-sft-3084',
           'gemini31pro-agent':'gemini-3.1-pro-preview'}
JOURNAL = WORK / 'submitted_shards.jsonl'
MODEL = 'gemini-3.7-flash'
SDK = ROOT / 'training-artifacts/judge-sdk-20260915'
sys.path.insert(0, str(SDK))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def wire_image(path):
    path = path.resolve()
    path.relative_to(ROOT)
    raw = path.read_bytes()
    for header, mime in [(b'\xff\xd8\xff','image/jpeg'), (b'\x89PNG','image/png'),
                         (b'GIF87a','image/gif'), (b'GIF89a','image/gif')]:
        if raw.startswith(header):
            return raw, mime
    if raw.startswith(b'RIFF') and raw[8:12] == b'WEBP':
        return raw, 'image/webp'
    if raw[4:12] == b'ftypavif':
        from PIL import Image
        buffer = io.BytesIO()
        with Image.open(io.BytesIO(raw)) as image:
            image.save(buffer, format='PNG')
        return buffer.getvalue(), 'image/png'
    raise ValueError('Unsupported image format: ' + str(path))


def selected():
    sft = json.loads((SFT / 'selected-traces.json').read_text())
    assert json.loads((SFT / 'progress.json').read_text())['phase'] == 'inference_complete'
    pro = {}
    for name in ['full-r2-c4','resume-smoke-r3-c4-20260914',
                 'resume-attempt-01-c4-20260914','resume-attempt-02-c4-20260914']:
        directory = PRO / name
        assert (directory / 'summary.json').is_file()
        for row in rows(directory / 'run_results.jsonl'):
            if row.get('status') != 'success':
                continue
            case = row['case_id']
            if case in pro:
                raise ValueError('Successful Pro case resampled')
            trace = (directory / row['trace_path']).resolve()
            trace.relative_to(directory)
            pro[case] = {'path':str(trace), 'sha256':digest(trace), 'verdict':row['verdict']}
    merged = rows(PRO / 'resume-control-20260914/merged-success-results.jsonl')
    assert len(merged) == len(pro) == 1526
    assert {r['case_id']:r['verdict'] for r in merged} == {k:v['verdict'] for k,v in pro.items()}
    return dict(zip(SOURCES, [sft, pro]))


def prepare():
    sys.path.insert(0, str(REPO))
    from src.eval.agent_private_gold import build_agent_private_gold_candidate, agent_candidate_answer
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    from scripts.audit_direct_qa_baseline import _private_gold
    assert digest(CASES) == 'c6568c302147893f7a648ea4e5cccd29e4c11dae831e1344399330ef75d7cf32'
    assert digest(GOLD) == '49a5785522b982dc4f97b28270a4b5bf5f4da0d7247f8dc7fd30cdf0de5ea671'
    assert digest(REPO / 'src/eval/agent_private_gold.py') == 'a7072d30cfaef2af5add48648cf376cbe316e93deeab33878873fd347900117e'
    assert digest(REPO / 'src/eval/private_gold_judge_contract.py') == '09c97955e292786b0a2259a653c0d198b86eeff33d3984e6e451dd4fcdc3577a'
    cases = rows(CASES)
    gold = {r['case_id']:r for r in rows(GOLD)}
    expected = [r['case_id'] for r in cases]
    assert len(expected) == len(set(expected)) == 1526
    report = WORK / 'prepared.json'
    if report.exists():
        record = json.loads(report.read_text())
        for name, details in record['sources'].items():
            assert digest(WORK / (name + '.jsonl.gz')) == details['sha256']
        return record
    source_sets = selected()
    record = {'count_per_source':1526, 'formal_denominator':1527,
              'model':MODEL, 'thinking_level':'low', 'max_output_tokens':32768,
              'candidate_schema':'ifv-agent-private-gold-candidate-v3',
              'prompt_sha256':hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest(),
              'response_schema_sha256':hashlib.sha256(json.dumps(PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,sort_keys=True).encode()).hexdigest(),
              'cases_sha256':digest(CASES), 'gold_sha256':digest(GOLD), 'sources':{}}
    for name, paths in source_sets.items():
        assert set(paths) == set(expected)
        target = WORK / (name + '.jsonl.gz')
        if target.exists():
            raise ValueError('Incomplete preparation exists; inspect before overwriting')
        total_actions = 0
        with gzip.open(target, 'wt', encoding='utf-8') as stream:
            for case in expected:
                path = Path(paths[case]['path']).resolve()
                path.relative_to(SFT if name.startswith('qwen') else PRO)
                assert digest(path) == paths[case]['sha256']
                trace = json.loads(path.read_text())
                assert not trace.get('error') and trace.get('verdict') in {'real','fake'} and trace.get('fact_check_report')
                candidate = build_agent_private_gold_candidate(trace)
                answer = agent_candidate_answer(candidate)
                assert candidate['projection_schema_version'] == record['candidate_schema']
                assert answer['verdict'] == trace['verdict']
                total_actions += len(candidate['raw_observations'])
                value = {'case_id':case, 'source_status':'success', 'source_model':SOURCES[name],
                         'source_trace_path':str(path), 'source_trace_sha256':digest(path),
                         'gold':_private_gold(gold[case]), 'candidate_output':candidate, 'candidate_answer':answer}
                assert value['gold']['auditable']
                stream.write(json.dumps(value, ensure_ascii=False, separators=(',',':')) + '\n')
        save(WORK / (name + '-selected-traces.json'), paths)
        record['sources'][name] = {'sha256':digest(target), 'count':1526, 'raw_actions':total_actions}
        print('PREPARED',name,json.dumps(record['sources'][name]),flush=True)
    save(report, record)
    return record


def plan(record):
    sys.path.insert(0, str(REPO))
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT
    cases = rows(CASES)
    result = {}
    for name in SOURCES:
        candidates = rows(WORK / (name + '.jsonl.gz'))
        shards, current, size = [], [], 0
        for index, (case, candidate) in enumerate(zip(cases,candidates),1):
            assert case['case_id'] == candidate['case_id']
            text = PRIVATE_GOLD_JUDGE_PROMPT + '\n\nAUDIT INPUT:\n' + json.dumps({
                'private_gold':candidate['gold'], 'candidate_material':{'mode':'agent_trace',
                'candidate_answer':candidate['candidate_answer'],
                'supporting_trace_material':candidate['candidate_output']}},ensure_ascii=False,indent=2)
            raw, mime = wire_image(CASES.parent / case['image_path'])
            estimate = len(text.encode()) + 4*((len(raw)+2)//3) + 8192
            use_file = estimate >= 15_000_000
            if use_file:
                estimate = len(text.encode()) + 16384
                if estimate >= 15_000_000:
                    raise ValueError('Text exceeds inline limit; no silent material truncation')
            if current and (use_file or size + estimate > 15_000_000 or len(current)>=20):
                shards.append(current)
                current, size = [], 0
            current.append({'index':index,'case_id':case['case_id'],'text':text,
                            'image_path':str(CASES.parent / case['image_path']),
                            'image_sha256':hashlib.sha256(raw).hexdigest(),'mime':mime,'bytes':estimate,
                            'use_file':use_file})
            size += estimate
            if use_file:
                shards.append(current)
                current, size = [], 0
        if current:
            shards.append(current)
        result[name] = shards
    return result


def submit(record, plans):
    from google import genai
    from google.genai import types
    from dotenv import dotenv_values
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    key = dotenv_values(ROOT / 'private/runtime.env')['GEMINI_API_KEY']
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120000,
                         retry_options=types.HttpRetryOptions(attempts=1)))
    files_path = WORK / 'large-image-files.json'
    files = json.loads(files_path.read_text()) if files_path.exists() else {}
    deferred_files = set()
    for shards in plans.values():
        for shard in shards:
            for item in shard:
                if not item['use_file']:
                    continue
                sha = item['image_sha256']
                if sha in deferred_files:
                    continue
                if sha not in files:
                    raw, mime = wire_image(Path(item['image_path']))
                    try:
                        uploaded = client.files.upload(file=io.BytesIO(raw),config={
                            'mime_type':mime,'display_name':'ifv-judge-large-image-20260915-'+sha[:16]})
                    except Exception as error:
                        if getattr(error,'code',None) != 429:
                            raise
                        deferred_files.add(sha)
                        save(WORK / 'deferred-large-image.json',{'reason':'File API quota 429',
                            'sha256':sha,'case_id':item['case_id'],'time':time.time()})
                        print('DEFERRED_LARGE_IMAGE_QUOTA',item['case_id'],flush=True)
                        continue
                    files[sha] = {'name':uploaded.name,'uri':uploaded.uri,'mime_type':mime,
                                  'sha256':sha,'bytes':len(raw)}
                    save(files_path,files)
                remote = client.files.get(name=files[sha]['name'])
                if getattr(remote.state,'name',str(remote.state)) != 'ACTIVE':
                    raise RuntimeError('Large image file not ACTIVE; inspect before submitting')
    prior = rows(JOURNAL) if JOURNAL.exists() else []
    existing = {(r['source_name'],r['shard_index']):r for r in prior}
    assert len(existing) == len(prior)
    config = types.GenerateContentConfig(max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_level='low',include_thoughts=True),
        response_mime_type='application/json',response_json_schema=PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA)
    for offset in range(max(map(len,plans.values()))):
        for name, shards in plans.items():
            if offset >= len(shards):
                continue
            shard = shards[offset]
            if any(item['use_file'] and item['image_sha256'] not in files for item in shard):
                continue
            ids = [r['case_id'] for r in shard]
            if (name,offset+1) in existing:
                old = existing[name,offset+1]
                assert old['case_ids'] == ids and old['source_sha256'] == record['sources'][name]['sha256']
                continue
            display = f'ifv-{name}-v3-low32k-20260915-s{offset+1:03d}'
            intent = WORK / 'create-intents' / (display + '.json')
            if intent.exists():
                previous = json.loads(intent.read_text())
                if previous.get('state') != 'server_rejected_429':
                    raise RuntimeError('Ambiguous previous create; reconcile remote display name before any retry')
                assert previous['case_ids']==ids and previous['source_sha256']==record['sources'][name]['sha256']
            requests = []
            for item in shard:
                raw, mime = wire_image(Path(item['image_path']))
                assert hashlib.sha256(raw).hexdigest() == item['image_sha256']
                image = types.Part.from_uri(file_uri=files[item['image_sha256']]['uri'],mime_type=mime) if item['use_file'] else types.Part.from_bytes(data=raw,mime_type=mime)
                requests.append(types.InlinedRequest(contents=[types.Content(role='user',parts=[
                    image,types.Part.from_text(text=item['text'])])],
                    metadata={'key':f'c{item["index"]:04d}','case_id':item['case_id'],'source_name':name},config=config))
            intent_record = {'display_name':display,'case_ids':ids,'source_sha256':record['sources'][name]['sha256'],'time':time.time()}
            for attempt in range(30):
                save(intent, {**intent_record,'state':'create_inflight','attempt':attempt+1})
                try:
                    batch = client.batches.create(model=MODEL,src=requests,config={'display_name':display})
                    break
                except Exception as error:
                    if getattr(error,'code',None) != 429:
                        raise
                    save(intent,{**intent_record,'state':'server_rejected_429','attempt':attempt+1,'time':time.time()})
                    counts = Counter()
                    for row in existing.values():
                        counts[row['source_name']] += row['sample_count']
                    save(WORK/'submission-progress.json',{'state':'waiting_for_batch_quota','time':time.time(),
                        'submitted_cases':dict(counts),'submitted_jobs':len(existing),
                        'waiting_display_name':display,'judge_model':MODEL})
                    if attempt == 29:
                        raise
                    print('QUOTA_BACKOFF',display,attempt+1,flush=True)
                    time.sleep(min(300,60*(attempt+1)))
            value = {'source_name':name,'source_sha256':record['sources'][name]['sha256'],
                     'shard_index':offset+1,'shard_count':len(shards),'sample_count':len(ids),
                     'case_ids':ids,'case_indexes':[r['index'] for r in shard],
                     'batch_name':batch.name,'display_name':display,
                     'state':getattr(batch.state,'name',str(batch.state)), 'create_time':str(batch.create_time)}
            with JOURNAL.open('a') as stream:
                stream.write(json.dumps(value)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            existing[name,offset+1] = value
            counts = Counter()
            for row in existing.values():
                counts[row['source_name']] += row['sample_count']
            save(WORK / 'submission-progress.json', {'state':'submitting','time':time.time(),
                 'submitted_cases':dict(counts),'submitted_jobs':len(existing),'judge_model':MODEL})
            print('SUBMITTED',name,offset+1,len(ids),batch.name,flush=True)
    counts = Counter()
    for row in existing.values():
        counts[row['source_name']] += row['sample_count']
    save(WORK / 'submission-progress.json',{'state':'all_submitted' if all(counts[name]==1526 for name in SOURCES) else 'partial_submitted',
        'time':time.time(),'submitted_cases':dict(counts),'submitted_jobs':len(existing),'judge_model':MODEL,
        'remaining_cases':{name:1526-counts[name] for name in SOURCES}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','submit','collect'],required=True)
    args = parser.parse_args()
    os.umask(0o077)
    WORK.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (WORK / (args.stage + '.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record = prepare()
    if args.stage == 'collect':
        from dotenv import dotenv_values
        env = {**os.environ,'GEMINI_API_KEY':dotenv_values(ROOT / 'private/runtime.env')['GEMINI_API_KEY'],
               'PYTHONPATH':str(SDK)+os.pathsep+str(REPO),'PYTHONNOUSERSITE':'1'}
        command = [sys.executable,str(REPO / 'scripts/collect_gemini_inline_agent_audits.py'),
            '--journal',str(JOURNAL),'--output-dir',str(OUTPUT),'--wait','--poll-seconds','120',
            '--workers','4','--judge-model',MODEL,'--thinking-level','low',
            '--expected-cases','1526','--reported-denominator','1527']
        for name in SOURCES:
            command += ['--source',name+'='+str(WORK / (name+'.jsonl.gz'))]
        return subprocess.call(command,cwd=REPO,env=env)
    plans = plan(record)
    save(WORK / 'plan.json', {name:{'shards':len(shards),'samples':sum(map(len,shards)),
        'max_estimated_shard_bytes':max(sum(r['bytes'] for r in shard) for shard in shards)} for name,shards in plans.items()})
    print('PLAN', (WORK / 'plan.json').read_text(),flush=True)
    if args.stage == 'submit':
        submit(record,plans)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
