"""Warm exact PSD source-review responses while immutable source slots finish.

This never declares the source collection complete, routes repair candidates,
or starts training. The normal full-bank gates remain mandatory. Provider
responses (including malformed decisions) are reused by their existing exact
request hash, so prefetch cannot reroll a judge until it gives a favorable label.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'training')]
from ifv_training.io import load_json, load_jsonl, sha256_file
from ifv_training.psd_candidates import _load_train_case_allowlist, _trace_case_id
from ifv_training.psd_collection_recovery import numerical_failure_evidence
from ifv_training.psd_gemini_judge import _atomic_json
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_search import search_lock
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_source_review import VERSION, TRACE_PROJECTION, PROMPT, SCHEMA, judge_source, validate_source_review
from src.eval.rollout import rollout_specs

PREFETCH_VERSION = 'ifv-psd-source-review-prefetch-v1'
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
TRANSPORT_ERROR_NAMES = {'GeminiInteractionsResponseError', 'TimeoutError', 'ReadTimeout',
    'ConnectTimeout', 'WriteTimeout', 'PoolTimeout', 'ConnectError', 'ReadError',
    'WriteError', 'RemoteProtocolError', 'LocalProtocolError'}


def review_attempts(record):
    if not record:
        return []
    history = record.get('attempt_history')
    if history is None:
        # Charge the earlier prefetch attempt instead of resetting its budget.
        return [{k: v for k, v in record.items() if k in {
            'status', 'error_type', 'status_code', 'retry_attempts', 'elapsed_seconds'}}]
    if (not isinstance(history, list) or not history
            or any(not isinstance(row, dict) or not row.get('status') for row in history)):
        raise ValueError('Malformed persisted source-review attempt history')
    return history


def retryable_record(record):
    if record.get('status') == 'in_progress':
        return True  # Interrupted work stays charged; exact cached replies are reused.
    if record.get('status') != 'pending_error':
        return False  # Never retry pass/fail/unresolved for a preferred decision.
    if record.get('retryable_transport') is not None:
        return record['retryable_transport'] is True
    return (record.get('error_type') in TRANSPORT_ERROR_NAMES
            or (record.get('error_type') == 'GeminiInteractionsHTTPError'
                and record.get('status_code') in RETRYABLE_HTTP))


def retryable_error(error):
    from src.integrations.gemini import GeminiInteractionsHTTPError, GeminiInteractionsResponseError
    import httpx
    return (isinstance(error, GeminiInteractionsHTTPError) and error.status_code in RETRYABLE_HTTP
            or isinstance(error, (GeminiInteractionsResponseError, httpx.TransportError, TimeoutError))
            or type(error) is ValueError and str(error) == 'PSD judge interaction did not complete')


def load_scope(*, run_dir, benchmark, train_cases, private_gold, model):
    run_dir = Path(run_dir).resolve()
    binding_path = run_dir.parent/'binding.json'
    binding = load_json(binding_path)
    inputs = {'benchmark': Path(benchmark).resolve(), 'train_cases': Path(train_cases).resolve(),
              'private_gold': Path(private_gold).resolve()}
    if (binding.get('formal_source_collection') is not True or binding.get('rollouts_per_case') != 8
            or binding.get('temperature') != .7 or not model.strip()):
        raise ValueError('Prefetch requires the attested grouped source collector')
    for name, path in inputs.items():
        if str(path) != binding[name] or sha256_file(path) != binding['files'][str(path)]:
            raise ValueError('Prefetch input differs from frozen collection binding')
    policy = Path(binding['source_access_policy']).resolve()
    if sha256_file(policy) != binding['files'][str(policy)]:
        raise ValueError('Source access policy changed')
    rows, gold_rows = load_jsonl(inputs['benchmark']), load_jsonl(inputs['private_gold'])
    public, gold = {r['case_id']: r for r in rows}, {r['case_id']: r for r in gold_rows}
    allowed = _load_train_case_allowlist(inputs['train_cases'])
    if (not rows or len(public) != len(rows) or len(gold) != len(gold_rows)
            or list(public) != binding['case_ids'] or binding['slots'] != len(rows)*8
            or any(allowed.get(c) != 'train' or c not in gold for c in public)):
        raise ValueError('Prefetch training membership/slot coverage invalid')
    manifest = load_json(run_dir/'run_manifest.json')
    agent = manifest['agent']
    if (manifest['benchmark'].get('training_prohibited') or manifest['git_commit'] != binding['policy_revision']
            or agent.get('rollouts_per_case') != 8 or agent.get('base_sampling_seed') != binding['seed']
            or manifest['benchmark'].get('sha256') != sha256_file(inputs['benchmark'])):
        raise ValueError('Prefetch source policy or sampling configuration changed')
    snapshot = Path(binding['snapshot']).resolve()
    if snapshot != run_dir.parent/'snapshot': raise ValueError('Wrong frozen source snapshot')
    profile = load_json(snapshot/'serving-profile.json')
    if profile['profile_id'] != agent['model']: raise ValueError('Frozen policy model differs')
    specs = rollout_specs(rows, [SimpleNamespace(case_id=c) for c in public], rollouts_per_case=8,
        base_sampling_seed=binding['seed'], policy_revision=binding['policy_revision'], model=agent['model'],
        episode_namespace=agent.get('episode_namespace'))
    paths = [binding_path, *inputs.values(), policy, snapshot/'serving-profile.json', snapshot/'checkpoint-manifest.json']
    identity = {'version': PREFETCH_VERSION, 'review_version': VERSION, 'trace_projection': TRACE_PROJECTION,
        'model': model, 'prompt_sha256': _sha(PROMPT), 'schema_sha256': _sha(SCHEMA),
        'generation': {'thinking_level': 'high', 'max_output_tokens': 8192}, 'run_dir': str(run_dir),
        'policy': {k: agent.get(k) for k in ('model', 'provider', 'base_url', 'timeout_seconds', 'episode_namespace')},
        'files': {str(p): sha256_file(p) for p in paths}}
    return {'identity': identity, 'public': public, 'gold': gold, 'benchmark': inputs['benchmark'],
            'run_dir': run_dir, 'expected': {r['episode_id']: r for r in specs}}


def bind_output(output, scope, cache_source=None):
    output = Path(output).resolve()
    marker = output/'inputs.json'
    source_name = str(Path(cache_source).resolve()) if cache_source is not None else None
    if marker.exists():
        payload = load_bound(marker, identity=scope['identity'])
        if payload.get('cache_source') != source_name: raise ValueError('Prefetch cache source changed')
    else:
        if (output/'records').exists() or (output/'judge-cache').exists():
            raise ValueError('Unbound prefetch data already exists')
        save_bound(marker, identity=scope['identity'], payload={'training_only': True,
            'admission_complete': False, 'cache_source': source_name})


def validate_prefetch_cache(output, **kwargs):
    scope = load_scope(**kwargs)
    output = Path(output).resolve()
    saved = load_json(output/'inputs.json')['identity']
    if saved.get('run_dir') != scope['identity']['run_dir']:
        original_run = Path(saved['run_dir']).resolve()
        current = scope['run_dir'].parent; seen = set()
        while current != original_run.parent:
            if current in seen: raise ValueError('Recovery lineage cycle')
            seen.add(current)
            previous = load_json(current/'binding.json').get('reuse_run')
            if not previous: raise ValueError('Prefetch cache is not from a recovery ancestor')
            current = Path(previous).resolve()
            if current.parent != scope['run_dir'].parent.parent: raise ValueError('Recovery lineage escaped run root')
        original = load_scope(**{**kwargs, 'run_dir': original_run})
        for key in ('model', 'prompt_sha256', 'schema_sha256', 'trace_projection', 'generation', 'policy'):
            if original['identity'][key] != scope['identity'][key]: raise ValueError('Recovered review policy differs')
        if (load_json(original_run.parent/'binding.json')['policy_revision'] !=
                load_json(scope['run_dir'].parent/'binding.json')['policy_revision']
                or sha256_file(original_run.parent/'snapshot/checkpoint-manifest.json') !=
                sha256_file(scope['run_dir'].parent/'snapshot/checkpoint-manifest.json')):
            raise ValueError('Recovered source checkpoint differs')
        scope = original
    payload = load_bound(output/'inputs.json', identity=scope['identity'])
    if payload.get('cache_source'):
        if Path(payload['cache_source']).resolve() == output: raise ValueError('Prefetch cache cycle')
        return validate_prefetch_cache(payload['cache_source'], **{**kwargs, 'run_dir': scope['run_dir']})
    return output/'judge-cache'


def ready_slots(scope):
    ready = []
    for episode, spec in scope['expected'].items():
        slot = scope['run_dir']/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
        marker = slot/'retry-state.json'
        if not marker.exists(): continue
        header = load_json(marker)
        state = load_bound(marker, identity=header['identity'])
        name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in episode)+'.json'
        if (state.get('attempts') and state['attempts'][-1]['status'] == 'completed'
                and (slot/'result.json').is_file() and (scope['run_dir']/'traces'/name).is_file()):
            ready.append(episode)
    return ready


def completed_source(scope, episode):
    spec = scope['expected'][episode]
    slot = scope['run_dir']/'psd-infrastructure-attempts'/hashlib.sha256(episode.encode()).hexdigest()
    # The compactor can atomically change cache representation, not contents.
    with search_lock(slot):
        header = load_json(slot/'retry-state.json'); identity = header['identity']; inputs = identity['inputs']
        state = load_bound(slot/'retry-state.json', identity=identity)
        policy = scope['identity']['policy']; case = spec['case_id']
        if (not state.get('attempts') or state['attempts'][-1]['status'] != 'completed'
                or any(inputs.get(k) != spec[k] for k in ('case_id', 'episode_id', 'sampling_seed'))
                or any(inputs.get(k) != policy[k] for k in ('model', 'provider', 'base_url'))
                or inputs.get('temperature') != .7 or inputs.get('timeout') != policy['timeout_seconds']):
            raise ValueError('Source slot is unfinished or differs from frozen sampling identity')
        name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in episode)+'.json'
        path = scope['run_dir']/'traces'/name
        digest = sha256_file(path); trace = load_json(path)
        if (trace != load_bound(slot/'result.json', identity=identity) or sha256_file(path) != digest
                or _trace_case_id(trace) != case or trace.get('image_id') != episode):
            raise ValueError('Source canonical trace differs from durable completed result')
        if numerical_failure_evidence(trace, inputs=inputs, source_slot=slot) is not None:
            raise ValueError('Unresolved numerical failure cannot enter source review')
    image = (scope['benchmark'].parent/scope['public'][case]['image_path']).resolve()
    image_hash = sha256_file(image)
    if (image_hash != inputs['image_sha256'] or image_hash != scope['public'][case]['image_sha256']
            or trace.get('state', {}).get('runtime_case', {}).get('image_sha256') != image_hash):
        raise ValueError('Source task image binding changed')
    binding = {'prefetch': _sha(scope['identity']), 'episode_id': episode,
        'trace_sha256': digest, 'gold_sha256': _sha(scope['gold'][case]), 'image_sha256': image_hash}
    return trace, image, path, binding


async def prefetch(*, run_dir, benchmark, train_cases, private_gold, output, model,
                   concurrency=16, follow=False, limit=None, poll_seconds=30, client=None, cache_source=None,
                   retry_errors=False, max_review_attempts=4, retry_delay_seconds=60):
    if type(concurrency) is not int or not 1 <= concurrency <= 64: raise ValueError('Invalid concurrency')
    if limit is not None and (type(limit) is not int or limit < 1): raise ValueError('Invalid prefetch limit')
    if poll_seconds <= 0: raise ValueError('Polling interval must be positive')
    if type(max_review_attempts) is not int or not 1 <= max_review_attempts <= 10:
        raise ValueError('Invalid bounded source-review retry budget')
    if not 0 <= retry_delay_seconds <= 3600:
        raise ValueError('Invalid source-review retry delay')
    scope = load_scope(run_dir=run_dir, benchmark=benchmark, train_cases=train_cases, private_gold=private_gold, model=model)
    output = Path(output).resolve()
    # Separate lock directory keeps the private reviewer independent of source writing.
    with search_lock(output/'process-lock'):
        cache_dir = output/'judge-cache'
        if cache_source is not None:
            cache_dir = validate_prefetch_cache(cache_source, run_dir=run_dir, benchmark=benchmark,
                train_cases=train_cases, private_gold=private_gold, model=model)
            with search_lock(Path(cache_source)/'process-lock'): pass
        bind_output(output, scope, cache_source)
        records = {}
        for path in (output/'records').glob('*.json'):
            raw = load_json(path); value = load_bound(path, identity=raw['identity'])
            episode = raw['identity']['episode_id']
            if raw['identity']['prefetch'] != _sha(scope['identity']) or episode not in scope['expected']:
                raise ValueError('Existing prefetch record belongs to another scope')
            records[episode] = value
        def can_retry(record):
            return retry_errors and retryable_record(record) and len(review_attempts(record)) < max_review_attempts

        queue = asyncio.Queue(); scheduled = {e for e,r in records.items() if not can_retry(r)}
        active = set(); submitted = 0
        started = time.time(); available = 0; stop_scan = False
        cooldown_until = 0.0

        def progress(phase):
            counts = Counter(r['status'] for r in records.values())
            value = {'phase': phase, 'time': time.time(), 'started_at': started,
                'expected': len(scope['expected']), 'available': available, 'attempted': len(records),
                'active': len(active), 'queued': queue.qsize(), 'concurrency': concurrency,
                'counts': dict(counts), 'reviewed': sum(counts[s] for s in ('pass', 'fail', 'unresolved')),
                'training_started': False, 'full_collection_admitted': False, 'model': model}
            value.update(retry_errors=retry_errors, max_review_attempts=max_review_attempts,
                retrying=sum(e in active and len(review_attempts(records.get(e))) > 1 for e in active),
                exhausted_transport=sum(retryable_record(r) and len(review_attempts(r)) >= max_review_attempts
                    for r in records.values()), cooldown_until=cooldown_until)
            _atomic_json(output/'progress.json', value)
            return value

        async def worker():
            nonlocal cooldown_until
            while True:
                episode = await queue.get()
                if episode is None: queue.task_done(); return
                active.add(episode); progress('prefetching')
                record_path = output/'records'/(hashlib.sha256(episode.encode()).hexdigest()+'.json')
                binding = {'prefetch': _sha(scope['identity']), 'episode_id': episode}
                previous = records.get(episode)
                source_validated = False
                try:
                    # Only the short local lock acquisition is retried here, never a paid decision.
                    for lock_attempt in range(20):
                        try:
                            trace, image, trace_path, binding = await asyncio.to_thread(completed_source, scope, episode)
                            break
                        except BlockingIOError:
                            if lock_attempt == 19: raise
                            await asyncio.sleep(.25)
                    if previous is None and cache_source is not None:
                        ancestor_path = Path(cache_source)/'records'/record_path.name
                        if ancestor_path.exists():
                            header = load_json(ancestor_path)
                            old = load_bound(ancestor_path, identity=header['identity'])
                            if (retryable_record(old) and all(header['identity'].get(k) == binding[k]
                                    for k in ('episode_id','trace_sha256','gold_sha256','image_sha256'))):
                                previous = {**old, 'inherited_attempt_record': str(ancestor_path),
                                            'inherited_attempt_record_sha256': sha256_file(ancestor_path)}
                    source_validated = True
                    history = review_attempts(previous)
                    if retry_errors and previous and not can_retry(previous):
                        result = previous
                    else:
                        while True:
                            deadline = max(cooldown_until, (previous or {}).get('retry_after', 0))
                            while retry_errors and time.time() < deadline:
                                await asyncio.sleep(min(5, deadline-time.time()))
                                deadline = max(deadline, cooldown_until)
                            attempt_started = time.monotonic()
                            entry = {'status':'in_progress', 'started_at':time.time()}
                            history = [*history, entry]
                            inherited = {k:v for k,v in (previous or {}).items() if k.startswith('inherited_attempt_record')}
                            records[episode] = {'status':'in_progress', 'attempt_history':history, **inherited}
                            save_bound(record_path, identity=binding, payload=records[episode])
                            progress('prefetching')
                            try:
                                artifact = await judge_source(client, trace,
                                    gold=scope['gold'][scope['expected'][episode]['case_id']],
                                    image_path=image, model=model, cache_dir=cache_dir)
                                status = validate_source_review(artifact, trace=trace,
                                    gold=scope['gold'][scope['expected'][episode]['case_id']])
                                if await asyncio.to_thread(sha256_file, trace_path) != binding['trace_sha256']:
                                    raise ValueError('Source changed while being reviewed')
                                result = {'status':status, 'artifact':artifact}
                            except Exception as error:
                                result = {'status':'pending_error', 'error_type':type(error).__name__,
                                          'retryable_transport':retryable_error(error)}
                                for key in ('status_code','retry_attempts'):
                                    if type(getattr(error,key,None)) is int: result[key] = getattr(error,key)
                            elapsed = time.monotonic()-attempt_started
                            history[-1] = {**entry, **{k:v for k,v in result.items() if k != 'artifact'},
                                           'finished_at':time.time(), 'elapsed_seconds':elapsed}
                            result.update(attempt_history=history, elapsed_seconds=elapsed, **inherited)
                            if can_retry(result):
                                delay = retry_delay_seconds * 2**(len(history)-1)
                                # Deterministic jitter avoids all 16 retries firing together.
                                jitter = retry_delay_seconds * (int(record_path.stem[:4],16) % 21) / 100
                                result['retry_after'] = time.time()+min(900,delay+jitter)
                                if result.get('status_code') == 429:
                                    cooldown_until = max(cooldown_until, result['retry_after'])
                            save_bound(record_path, identity=binding, payload=result)
                            records[episode] = result
                            progress('prefetching')
                            if not can_retry(result): break
                            previous = result
                except Exception as error:
                    if source_validated:
                        raise  # Persistence/control faults must stop, not masquerade as API errors.
                    # No raw exception text: provider errors can contain secrets.
                    result = {'status': 'pending_error', 'error_type': type(error).__name__,
                              'retryable_transport': False, 'attempt_history': review_attempts(previous) or [
                                  {'status':'pending_error','error_type':type(error).__name__,'phase':'source_validation'}]}
                    for key in ('status_code', 'retry_attempts'):
                        if type(getattr(error, key, None)) is int: result[key] = getattr(error, key)
                save_bound(record_path, identity=binding, payload=result)
                records[episode] = result; active.discard(episode); queue.task_done(); progress('prefetching')

        async def scan_and_drain():
            nonlocal available, submitted, stop_scan
            while True:
                # Fail before admitting more calls if a frozen input changed.
                for p, digest in scope['identity']['files'].items():
                    if await asyncio.to_thread(sha256_file, Path(p)) != digest:
                        raise ValueError('Bound prefetch input changed during collection')
                ready = await asyncio.to_thread(ready_slots, scope); available = len(ready)
                for episode in ready:
                    if episode not in scheduled and (limit is None or submitted < limit):
                        scheduled.add(episode); queue.put_nowait(episode); submitted += 1
                progress('prefetching' if active or not queue.empty() else 'waiting_for_completed_sources')
                if not follow or available == len(scope['expected']) or (limit is not None and submitted >= limit): break
                state_path = scope['run_dir'].parent/'state.json'
                if state_path.exists() and load_json(state_path).get('phase') == 'collection_requires_intervention':
                    stop_scan = True; break
                await asyncio.sleep(poll_seconds)
            await queue.join()

        async def collect():
            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            scanner = asyncio.create_task(scan_and_drain())
            try:
                done, _ = await asyncio.wait([scanner, *workers], return_when=asyncio.FIRST_COMPLETED)
                # A failed persistence worker must stop the run, not strand queue.join forever.
                for task in done: task.result()
                if scanner not in done: raise RuntimeError('Prefetch worker exited unexpectedly')
            except BaseException:
                progress('interrupted_requires_inspection')
                raise
            finally:
                for task in [scanner, *workers]: task.cancel()
                await asyncio.gather(scanner, *workers, return_exceptions=True)
            return progress('source_intervention_required' if stop_scan else
                'prefetch_complete_requires_full_collection_gate' if available == len(scope['expected']) else 'prefetch_pass_finished')

        if client is not None: return await collect()
        from src.integrations.gemini import GeminiInteractionsClient, GeminiRequestGate
        # Explicit gate, not the generic client's default of four. Other experiments are untouched.
        async with GeminiInteractionsClient(timeout=240, max_retries=2,
                request_gate=GeminiRequestGate(concurrency)) as live_client:
            client = live_client
            return await collect()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'benchmark', 'train-cases', 'private-gold', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', default='gemini-3.1-pro-preview')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--cache-source', type=Path,
        help='Reuse exact response cache from an idle, validated recovery ancestor')
    parser.add_argument('--retry-errors', action='store_true', help='Automatically retry transient errors, never valid decisions')
    parser.add_argument('--max-review-attempts', type=int, default=4)
    parser.add_argument('--retry-delay-seconds', type=float, default=60)
    print(json.dumps(asyncio.run(prefetch(**vars(parser.parse_args())))), flush=True)


if __name__ == '__main__': main()
