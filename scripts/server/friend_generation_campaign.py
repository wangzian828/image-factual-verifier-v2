"""Isolated model lanes and append-only retry waves for the authorized friend job.

Imports the already audited, immutable Linux runner; does not modify its prompt,
tools or timeout/503 policy. No original result is overwritten. This orchestrator
does not render slides or perform quality evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path('/volume/ybo/wza/external-projects/wanglinhaotrain-20260915')
PROJECT = ROOT / 'project'
SOURCE = PROJECT / 'src_linux_fallback'
BASE = PROJECT / 'outputs/gemini-3.6-flash-medium-linux-full-20260915'
BASE_HASH = 'e25fd65d089a0c17310b7be229d72491bb4911e4b4115f25b2dc0ac08f102ef1'
MODELS = {'flash': 'gemini-3.6-flash', 'pro': 'gemini-3.1-pro-preview'}
STOP = False


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen campaign identity changed: ' + str(path))
    else:
        atomic(path, value)


def load_statuses(root, allowed, expected_hash):
    values = {}
    for path in root.glob('sample-*/status.json'):
        item = read(path)
        key = path.parent.name
        if key not in allowed or item.get('sample_id') != key:
            raise ValueError('Foreign or mismatched sample')
        if item.get('config_hash') != expected_hash:
            raise ValueError('Foreign configuration in campaign output')
        if item.get('state') not in {'succeeded', 'failed', 'running', 'retry_wait'}:
            raise ValueError('Unknown sample state')
        values[key] = {**item, 'status_path': str(path), 'status_sha256': sha(path)}
    return values


def aggregate(ids, waves):
    """First structurally successful attempt wins; never select by quality."""
    selected, failures, unattempted, active = {}, [], [], []
    for key in ids:
        attempts = [w[key] for w in waves if key in w]
        winners = [v for v in attempts if v['state'] == 'succeeded']
        if winners:
            selected[key] = {k: winners[0][k] for k in ('status_path', 'status_sha256')}
        elif any(v['state'] in {'running', 'retry_wait'} for v in attempts):
            active.append(key)
        elif attempts:
            failures.append(key)
        else:
            unattempted.append(key)
    return {'success': len(selected), 'failed': len(failures), 'unattempted': len(unattempted),
            'active': len(active), 'selected': selected, 'failed_ids': failures,
            'unattempted_ids': unattempted, 'active_ids': active}


def wire_checks(rows, model, source_sha, expected_thinking):
    requests = [r for r in rows if r.get('event') == 'request']
    responses = [r for r in rows if r.get('event') == 'response']
    return {
        'exact_model': bool(requests) and all(r.get('model') == model for r in requests),
        'source_image': any(i.get('sha256') == source_sha for r in requests for i in r.get('images', [])),
        'tool_results': any(r.get('function_responses', 0) > 0 for r in requests),
        'provider_completed': any(r.get('generation_complete') for r in responses),
        'output_budget': bool(requests) and all(r.get('max_output_tokens') == (
            65535 if model == MODELS['pro'] else 65536) for r in requests),
        'thinking': bool(requests) and all((r.get('thinking') or {}) == expected_thinking for r in requests),
    }


def pro_model_route(path):
    """Installed CLI selects customtools; the requested experiment uses plain Pro.

    Rewrite ONLY that exact model segment. The broker still authenticates and
    applies its exact-model/method allowlist. No prompt, tools or body is changed.
    """
    parsed = urlsplit(path)
    prefix = '/v1beta/models/gemini-3.1-pro-preview-customtools:'
    if parsed.path.startswith(prefix) and not parsed.netloc and not parsed.scheme:
        return path.replace(prefix, '/v1beta/models/gemini-3.1-pro-preview:', 1)
    return path


def configuration(lane, campaign, wave, expected_thinking):
    original = read(PROJECT / ('configs/gemini-3.1-pro-high.json' if lane == 'pro'
                              else 'configs/gemini-36-flash-full-20260915.json'))
    release = read(ROOT / 'runtime/agy-release.json')
    config = {**original, 'run_id': f'friend-{lane}-20260916-wave-{wave:02d}',
        'run_root': str((campaign / f'wave-{wave:02d}').relative_to(PROJECT)),
        'cli_binary': '../runtime/bin/agy', 'cli_version': release['version'],
        'cli_sha256': release['binary_sha256'], 'concurrency': 1,
        'resume_compatible_config_hashes': [], 'pricing_usd_per_million_tokens': None,
        'pricing_note': 'Usage retained; no inferred prices.',
        'campaign_controller_sha256': sha(Path(__file__)),
        'immutable_runner_sha256': {p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
        'expected_wire_thinking': expected_thinking,
        'campaign_protocol': {'authorization': '2026-09-16: rerun failed friend cases and also run Pro.',
            'model_lanes': 2, 'per_model_concurrency': 1,
            'original_outputs_read_only': True, 'generation_only': True,
            'retry_wave': wave, 'max_additional_waves_before_diagnosis': 3,
            'retry_selection': 'all unsuccessful cases; retain first structural success',
            'pro_route_adapter': 'CLI customtools alias -> requested plain gemini-3.1-pro-preview; body unchanged',
            'pro_high_wire': 'Installed CLI high maps to thinkingBudget=-1 (dynamic/default), not a numeric high cap.',
            'note': 'Three exhausted waves require diagnosis, not an assertion of completion.'}}
    target = campaign / f'config-wave-{wave:02d}.json'
    freeze(target, config)
    return target


def child_one(args):
    import run_antigravity_batch as runner
    config = read(args.config)
    manifest = read(PROJECT / config['manifest'])
    prompt = (PROJECT / config['prompt']).read_text()
    digest = runner.config_hash(config, prompt, manifest)
    sample = next(s for s in manifest['samples'] if s['sample_id'] == args.sample)
    runner.run_one(sample, PROJECT / config['run_root'], prompt, config, digest, {digest})


def main():
    global STOP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=MODELS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--thinking-json', type=json.loads)
    parser.add_argument('--one', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--sample')
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    if args.one:
        child_one(args)
        return 0
    if not args.lane or not isinstance(args.thinking_json, dict):
        parser.error('lane and observed CLI thinking configuration are required')
    import fcntl
    import run_antigravity_batch as runner
    import gemini_broker
    from linux_support import PYTHON
    os.umask(0o077)
    campaign = PROJECT / f'outputs/friend-{args.lane}-campaign-20260916'
    campaign.mkdir(parents=True, exist_ok=True)
    lock = (campaign / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stopping(*_):
        global STOP
        STOP = True  # Drain current sample; do not orphan its sandbox or broker.
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    identity = {'controller_sha256': sha(Path(__file__)), 'model': MODELS[args.lane],
        'lane': args.lane, 'thinking': args.thinking_json,
        'source_sha256': {p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}}
    freeze(campaign / 'identity.json', identity)
    def state(status, **kwargs):
        atomic(campaign / 'state.json', {'time':time.time(), 'pid':os.getpid(),
            'state':status, 'model':MODELS[args.lane], 'concurrency':1, **kwargs})
    # Flash uses the SAME lock as its old controller: cannot race original work.
    flash_lock = None
    if args.lane == 'flash' and args.execute:
        flash_lock = (ROOT / 'state/flash-controller.lock').open('a')
        while not STOP:
            try:
                fcntl.flock(flash_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                state('waiting_original_flash_drain', retry_failed=True)
                time.sleep(5)
        if STOP:
            state('stopped_before_dispatch')
            return 2
    # Only this isolated parent extends the broker's exact-model allowlist.
    # Neither the running Flash broker nor its immutable source is changed.
    gemini_broker.AUTHORIZED_MODELS = gemini_broker.AUTHORIZED_MODELS | {MODELS[args.lane]}
    broker = gemini_broker.Broker(MODELS[args.lane])
    broker.log = campaign / 'wire.jsonl'
    if args.lane == 'pro':
        handler = broker.server.RequestHandlerClass
        original_relay = handler.relay
        def pro_relay(request):
            route = pro_model_route(request.path)
            if route != request.path:
                broker.record({'event':'model_alias_adapter', 'requested_cli_model':'gemini-3.1-pro-preview-customtools',
                               'forwarded_api_model':MODELS['pro'], 'body_unchanged':True})
                request.path = route
            return original_relay(request)
        handler.relay = pro_relay
    env = {k:v for k,v in os.environ.items() if k in {'PATH','LANG','LC_ALL','HOME'}}
    env.update(broker.start(), PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(SOURCE))
    os.environ.update({k:v for k,v in env.items() if k in {'GEMINI_API_KEY','GOOGLE_GEMINI_BASE_URL'}})
    try:
        config_path = configuration(args.lane, campaign, 0, args.thinking_json)
        config, manifest, prompt, report = runner.preflight(config_path)
        atomic(campaign / 'preflight.json', report)
        if not report['ok']:
            state('held_preflight')
            return 2
        samples = {s['sample_id']:s for s in manifest['samples']}
        ids = list(samples)
        freeze(campaign / 'input-binding.json', {'samples':[(k,samples[k]['sha256']) for k in ids],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        if not args.execute:
            state('preflight_passed_no_generation')
            return 0
        waves = []
        if args.lane == 'flash':
            base = load_statuses(BASE, ids, BASE_HASH)
            if set(base) != set(ids) or any(s['state'] in {'running','retry_wait'} for s in base.values()):
                state('held_original_incomplete', original=len(base))
                return 2
            freeze(campaign / 'original-binding.json', {k:v['status_sha256'] for k,v in base.items()})
            waves.append(base)
        # Pro's first wave + 3 retries; Flash has 3 retry waves after the original.
        for wave in range(4 if args.lane == 'pro' else 3):
            path = configuration(args.lane, campaign, wave, args.thinking_json)
            cfg = read(path)
            digest = runner.config_hash(cfg, prompt, manifest)
            out = PROJECT / cfg['run_root']
            current = load_statuses(out, ids, digest)
            if any(v['state'] in {'running','retry_wait'} for v in current.values()):
                raise RuntimeError('Interrupted attempt needs explicit recovery; not silently resampled')
            waves.append(current)
            for key in ids:
                summary = aggregate(ids, waves)
                atomic(campaign / 'aggregate.json', summary)
                if STOP:
                    state('stopped_after_drain', wave=wave)
                    return 2
                if key in summary['selected'] or key in current:
                    continue
                gate_path = campaign / 'pro-canary.json'
                is_pilot = args.lane == 'pro' and not gate_path.exists()
                start_bytes = broker.log.stat().st_size if broker.log.exists() else 0
                state('canary_running' if is_pilot else 'running', wave=wave, sample_id=key,
                      success=summary['success'], failed=summary['failed'], unattempted=summary['unattempted'])
                command = [PYTHON, str(Path(__file__).resolve()), '--one', '--config', str(path), '--sample',key]
                with (campaign / 'runner.log').open('ab') as log:
                    process = subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                                               stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        time.sleep(5)
                current.update(load_statuses(out, ids, digest))
                if process.returncode != 0 or key not in current:
                    raise RuntimeError('Runner did not close attempt; inspect before resuming')
                if is_pilot:
                    raw = broker.log.read_bytes()[start_bytes:]
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    checks = wire_checks(rows,MODELS[args.lane],samples[key]['sha256'],args.thinking_json)
                    checks['structural_success'] = current[key]['state'] == 'succeeded'
                    receipt = {'passed':all(checks.values()),'checks':checks,'sample_id':key,
                        'status_sha256':current[key]['status_sha256'], 'wire_sha256':hashlib.sha256(raw).hexdigest()}
                    if not receipt['passed']:
                        atomic(campaign / 'pro-canary-held.json',receipt)
                        state('held_pro_canary',checks=checks)
                        return 2
                    freeze(gate_path,receipt)
            summary = aggregate(ids,waves)
            atomic(campaign / 'aggregate.json',summary)
            if summary['success'] == len(ids):
                state('completed_all_structurally_valid',success=len(ids))
                return 0
        state('held_exhausted_waves_needs_diagnosis',success=summary['success'],failed=summary['failed'])
        return 2
    except Exception as error:
        state('held_controller_exception',error_type=type(error).__name__)
        raise
    finally:
        broker.close()


if __name__ == '__main__':
    raise SystemExit(main())
