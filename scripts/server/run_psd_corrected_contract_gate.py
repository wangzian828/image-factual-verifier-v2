"""New checker/material protocol, same frozen 32 sources, bounded repairs only.

Derived rewards/audits go into a separate view: never rewrite the old source
bank or its hash-bound postprocess files. Raw trace hardlinks are read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10'
CODE = DEPLOY / 'source-review-v2-code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
OUT = RUN / 'psd-corrected-contract-v12'
REVIEWS = RUN / 'source-reviews-transport-v2'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'


def owner():
    spec = importlib.util.spec_from_file_location('source_protocol_owner', DEPLOY / 'review_psd_contract_sources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.owner()


def prepare(o):
    assert not OUT.exists()
    assert o.load(DEPLOY / 'source-review-v2-state.json')['phase'] == 'source_reviews_complete'
    assert o.load(REVIEWS / 'summary.json')['pending'] == 0
    assert o.load(SERVICE / 'mm-cache-recovery-v11/multimodal-smoke.json')['passed']
    o.verify_export()
    for path, expected in o.load(DEPLOY / 'source-review-v2-binding.json')['files'].items():
        assert o.sha(path) == expected
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from scripts.postprocess_psd_training import postprocess
    from ifv_training.psd_round import verify_psd_round_rollout
    from ifv_training.psd_candidates import build_psd_candidate_package
    from ifv_training.psd_repair_storage import load_bound
    from ifv_training.psd_source_review import TRACE_PROJECTION
    for path in (REVIEWS / 'reviews').glob('*.json'):
        saved = o.load(path)
        assert load_bound(path, identity=saved['identity'])['trace_projection'] == TRACE_PROJECTION
    config = o.load(RUN / 'psd-round-v5/source-bank/prepared.json')['preparation']
    original = Path(config['rollout_dir'])
    before = {str(p): o.sha(p) for p in original.rglob('*.json*') if p.is_file()}
    view = OUT / 'source-episodes'
    view.mkdir(parents=True)
    for name in ('run_manifest.json', 'run_results.jsonl'):
        shutil.copy2(original / name, view / name)
    shutil.copytree(original / 'traces', view / 'traces', copy_function=os.link)
    # No old derived file is copied/hardlinked, so writers cannot truncate it.
    result = postprocess(run_dir=view, train_cases=Path(config['case_split']),
        private_gold=Path(config['private_gold']), source_access_policy=Path(config['source_access_policy']),
        source_reviews=REVIEWS)
    assert result['episodes'] == 32 and result['source_review_pending'] == 0
    gate = OUT / 'rollout-gate.json'
    # Attest where the SOURCE really ran (19018); do not relabel history 19019.
    report = verify_psd_round_rollout(round_index=1, run_dir=view,
        train_cases_path=Path(config['case_split']),
        serving_profile_path=RUN / 'snapshot/serving-profile.json',
        round_start_checkpoint_manifest_path=RUN / 'snapshot/checkpoint-manifest.json', output=gate)
    assert report['passed']
    bank = OUT / 'source-bank'
    candidates = build_psd_candidate_package(run_dir=view, train_cases_path=Path(config['case_split']),
        rollout_gate_path=gate, output_dir=bank / 'candidates')
    o.save(bank / 'prepared.json', {'schema_version': 'ifv-psd-round-bank-v1', 'round_index': 1,
        'training_only': True, 'preparation': {**config, 'rollout_dir': str(view)}})
    for raw, expected in before.items():
        assert o.sha(raw) == expected, 'Original frozen source artifact changed'
    files = [*before, str(REVIEWS / 'summary.json'), str(gate), str(bank / 'prepared.json'),
             *[str(p) for p in (bank / 'candidates').glob('*') if p.is_file()]]
    files += list(o.load(DEPLOY / 'source-review-v2-binding.json')['files'])
    o.save(OUT / 'binding.json', {'files': {p: o.sha(p) for p in files},
        'source_resampling': False, 'old_derived_artifacts_unchanged': True,
        'source_gateway': 19018, 'repair_gateway': 19019, 'same_epoch3_weights': True})
    o.save(OUT / 'state.json', {'phase': 'prepared_corrected_checker_bank',
        'counts': candidates['counts'], 'training_started': False})
    print(json.dumps({'counts': candidates['counts'], 'training_started': False}))


def execute(o):
    for path, expected in o.load(OUT / 'binding.json')['files'].items():
        assert o.sha(path) == expected
    for gpu in range(4):
        receipt = o.load(SERVICE / f'replica-{gpu}.json')
        o.checked(receipt)
        command = receipt['command']
        assert command[command.index('--mm-processor-cache-gb') + 1] == '0'
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    import scripts.run_psd_feedback_canary as feedback
    from ifv_training.psd_diagnostics import persist_exception
    original_run = feedback._run
    async def observed(args):
        try:
            return await original_run(args)
        except Exception as error:
            receipt = persist_exception(args.output_dir / 'private-diagnostics', error)
            o.save(OUT / (args.output_dir.name + '-failure.json'), receipt)
            raise
    feedback._run = observed
    def phase(name):
        o.save(OUT / 'state.json', {'phase': name, 'time': time.time(), 'training_started': False})
    phase('bounded_corrected_contract_repairs')
    try:
        args = SimpleNamespace(source=OUT / 'source-bank', output=OUT / 'search',
            snapshot=DEPLOY / 'snapshot', attempts=6, proposal_rounds=12, case_concurrency=40,
            task_source_selection='longest_failed', repair_mode='slate',
            judge_model='gemini-3.1-pro-preview', teacher_device='cuda:0', score_missing_topk=False)
        result = asyncio.run(feedback.run(args))
        o.save(OUT / 'result.json', result)
        phase(result['status'])
    except BaseException as error:
        phase('held_requires_inspection_' + type(error).__name__)
        raise


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'execute'))
    args = parser.parse_args()
    original, o = owner()
    if args.mode == 'prepare':
        prepare(o)
    elif args.mode == 'execute':
        execute(o)
    else:
        assert o.load(OUT / 'state.json')['phase'] == 'prepared_corrected_checker_bank'
        assert not (OUT / 'process.json').exists()
        env, checks = original.capture_environment(o)
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
                   PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
        o.save(OUT / 'credential-presence.json', checks)
        receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, OUT / 'run.log')
        o.save(OUT / 'process.json', receipt)
        print(json.dumps(receipt))
