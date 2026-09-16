"""One versioned review calibration on unchanged real episodes, not production."""
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
OUT = RUN / 'hint-review-calibration-v1'
CASES = ('7996ba9752652f37', '82897bdeb31b6ade', 'e79d4db478e45800')
CANDIDATE = DEPLOY / 'psd_slate_review_candidate_v3.py'


def read_completed(case):
    from ifv_training.io import load_json, sha256_file
    from ifv_training.psd_repair import _sha
    from ifv_training.psd_repair_storage import load_bound
    config = load_json(case / 'run-inputs.json')['identity']
    load_bound(case / 'run-inputs.json', identity=config)
    for value in config.values():
        if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
            assert sha256_file(Path(value['path'])) == value['sha256']
    bound = load_json(case / 'slate-state.json')
    assert bound['identity']['inputs'] == _sha(config)
    state = load_bound(case / 'slate-state.json', identity=bound['identity'])
    row = state['rounds'][0]
    for path, digest in row['files'].items():
        assert sha256_file(Path(path)) == digest
    directory = case / 'slate-rounds/00'
    continuation = load_bound(directory / 'continuation.json', identity={
        'inputs': _sha(config), 'hints': row['hints']})
    assert continuation['teacher_complete']
    assert _sha(continuation['teacher_episode_trace']) == _sha(load_json(directory / 'episode.json'))
    before = {**row['files'], str(case / 'run-inputs.json'): sha256_file(case / 'run-inputs.json'),
              str(case / 'slate-state.json'): sha256_file(case / 'slate-state.json')}
    return config, continuation, before


async def execute(owner):
    from src.integrations.gemini import GeminiInteractionsClient
    from ifv_training.psd_diagnostics import persist_exception
    spec = importlib.util.spec_from_file_location('ifv_training.psd_slate_review_candidate', CANDIDATE)
    candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
    binding = owner.load(OUT / 'binding.json')
    assert owner.sha(CANDIDATE) == binding['candidate_sha256']
    owner.save(OUT / 'state.json', {'phase': 'calibrating_new_review_protocol', 'training_started': False})
    async def one(key):
        config, continuation, before = read_completed(RUN / 'search/repairs' / key)
        owner.save(OUT / (key + '-binding.json'), {'files': before, 'model': config['judge_model'],
            'old_review_not_overwritten': True, 'agent_resampled': False})
        try:
            async with GeminiInteractionsClient(timeout=900, max_retries=0) as client:
                review = await candidate.review_slate(client,
                    source=owner.load(config['trace']['path']), episode=continuation['teacher_episode_trace'],
                    gold=owner.load(config['gold']['path']), image_path=Path(config['image']['path']),
                    model=config['judge_model'], cache_dir=OUT / key / 'judge-cache',
                    targets=continuation['local_targets'])
            owner.save(OUT / (key + '-review.json'), review)
            return {'case': key, 'status': review['decision']['status']}
        except Exception as error:
            failure = persist_exception(OUT / key / 'private-diagnostics', error)
            owner.save(OUT / (key + '-failure.json'), failure)
            return {'case': key, 'error_type': type(error).__name__}
        finally:
            for path, digest in before.items():
                assert owner.sha(path) == digest
    results = await asyncio.gather(*(one(key) for key in CASES))
    owner.save(OUT / 'results.json', {'results': results, 'production_review_replaced': False})
    owner.save(OUT / 'state.json', {'phase': 'completed_requires_semantic_inspection', 'training_started': False})


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['launch', 'execute'])
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('observed', DEPLOY / 'run_psd_observed_positions_gate.py')
    observed = importlib.util.module_from_spec(spec); spec.loader.exec_module(observed)
    original, owner = observed.owner()
    if args.mode == 'execute':
        asyncio.run(execute(owner)); return
    assert owner.load(RUN / 'eager-failure-hold-v1/state.json')['completed_results_preserved']
    for key in CASES:
        read_completed(RUN / 'search/repairs' / key)
    OUT.mkdir(exist_ok=False)
    owner.save(OUT / 'binding.json', {'candidate_sha256': owner.sha(CANDIDATE),
        'purpose': 'protocol calibration, not resampling valid judgments for favorable results',
        'cases': list(CASES), 'round_index': 0, 'time': time.time()})
    env, checks = original.capture_environment(owner)
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
               PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
    owner.save(OUT / 'credential-presence.json', checks)
    receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, OUT / 'run.log')
    owner.save(OUT / 'process.json', receipt)
    print(json.dumps({'pid': receipt['pid'], 'output': str(OUT)}))


if __name__ == '__main__':
    main()
