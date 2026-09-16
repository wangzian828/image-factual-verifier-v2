"""One bounded review-only recovery; never resample a completed Agent episode."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
CODE = DEPLOY / 'code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'
CASE = RUN / 'search/repairs/7996ba9752652f37'
CONTROL = RUN / 'review-recovery-v1'


def load_review_inputs(case):
    from ifv_training.io import load_json, sha256_file
    from ifv_training.psd_repair import _sha
    from ifv_training.psd_repair_storage import load_bound
    saved = load_json(case / 'run-inputs.json')
    config = saved['identity']
    load_bound(case / 'run-inputs.json', identity=config)
    for value in config.values():
        if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
            if sha256_file(Path(value['path'])) != value['sha256']:
                raise ValueError('Original review input changed')
    saved = load_json(case / 'slate-state.json')
    if saved['identity']['inputs'] != _sha(config):
        raise ValueError('Search inputs differ from original driver')
    state = load_bound(case / 'slate-state.json', identity=saved['identity'])
    pending = state.get('pending_proposal')
    if state['status'] != 'repairing' or not pending or pending['round_index'] != len(state['rounds']):
        raise ValueError('No pending reviewed-round recovery')
    directory = case / 'slate-rounds' / f"{pending['round_index']:02d}"
    identity = {'inputs': _sha(config), 'hints': {k: h['text'] for k, h in pending['hints'].items()}}
    continuation = load_bound(directory / 'continuation.json', identity=identity)
    if continuation['teacher_complete'] is not True:
        raise ValueError('Cannot recover a review of an incomplete episode')
    episode = load_json(directory / 'episode.json')
    if _sha(episode) != _sha(continuation['teacher_episode_trace']):
        raise ValueError('Cached episode differs from completed continuation')
    return config, directory, continuation


def owner():
    spec = importlib.util.spec_from_file_location('observed_gate', DEPLOY / 'run_psd_observed_positions_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.owner()


async def execute(o):
    from ifv_training.io import load_json, write_json
    from ifv_training.psd_slate import review_slate
    from ifv_training.psd_repair_search import search_lock
    from ifv_training.psd_diagnostics import persist_exception
    from src.integrations.gemini import GeminiInteractionsClient
    with search_lock(CASE.parent / ('.' + CASE.name + '-writer')):
        config, directory, continuation = load_review_inputs(CASE)
        if (directory / 'review.json').exists():
            raise ValueError('Completed review already exists; do not resample')
        before = {str(p): o.sha(p) for p in [CASE / 'run-inputs.json', CASE / 'slate-state.json',
                                           directory / 'continuation.json', directory / 'episode.json']}
        o.save(CONTROL / 'binding.json', {'files': before, 'model': config['judge_model'],
            'agent_resampling': False, 'transport_timeout_seconds': 900, 'max_transport_retries': 0})
        o.save(CONTROL / 'state.json', {'phase': 'review_only_running', 'time': time.time()})
        try:
            # Only the transport deadline changes: same model/material/prompt/schema,
            # high reasoning, max output 8192, and original completed-result cache.
            async with GeminiInteractionsClient(timeout=900, max_retries=0) as client:
                result = await review_slate(client, source=load_json(Path(config['trace']['path'])),
                    episode=continuation['teacher_episode_trace'], gold=load_json(Path(config['gold']['path'])),
                    image_path=Path(config['image']['path']), model=config['judge_model'],
                    cache_dir=CASE / 'judge-cache', targets=continuation['local_targets'])
            write_json(directory / 'review.json', result)
            o.save(CONTROL / 'state.json', {'phase': 'review_complete', 'decision': result['decision']['status'],
                'time': time.time(), 'review_path': str(directory / 'review.json'),
                'agent_resampling': False, 'training_started': False})
        except Exception as error:
            receipt = persist_exception(CONTROL / 'private-diagnostics', error)
            o.save(CONTROL / 'failure.json', receipt)
            o.save(CONTROL / 'state.json', {'phase': 'held_requires_diagnosis',
                                           'error_type': type(error).__name__, 'time': time.time()})
            raise
        finally:
            for raw, expected in before.items():
                assert o.sha(raw) == expected, 'Recovery mutated original continuation/state'


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    original, o = owner()
    for path, digest in o.load(DEPLOY / 'code-binding.json').items():
        assert o.sha(CODE / path) == digest, 'Immutable code changed'
    if args.mode == 'execute':
        asyncio.run(execute(o))
        return
    assert not Path(f"/proc/{o.load(RUN / 'process.json')['pid']}").exists(), 'Original driver still active'
    load_review_inputs(CASE)
    CONTROL.mkdir(exist_ok=False)
    env, checks = original.capture_environment(o)
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
               PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
    o.save(CONTROL / 'credential-presence.json', checks)
    receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, CONTROL / 'run.log')
    o.save(CONTROL / 'process.json', receipt)
    print(json.dumps({'pid': receipt['pid'], 'control': str(CONTROL)}), flush=True)


if __name__ == '__main__':
    main()
