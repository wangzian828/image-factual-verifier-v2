"""Build an explicitly versioned engineering bank from audited real episodes.

No new Agent/judge calls, no search-budget reset, no mutation of earlier reviews.
This bank is for end-to-end training validation, not the formal 400x8 bank.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-grounded-review-20260916-v15'
CODE = DEPLOY / 'code'
PRIOR = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
RUNBASE = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
OLD = RUNBASE / 'psd-observed-positions-v14'
OUT = RUNBASE / 'psd-grounded-review-canary-v15'
KEY = '82897bdeb31b6ade'
REVIEW = OLD / 'hint-review-calibration-v1' / (KEY + '-review.json')
PRESERVATION = RUNBASE / 'psd-corrected-contract-v12/source-bank/candidates/preservation_candidates.jsonl'


def execute(owner):
    from ifv_training.io import load_json, write_jsonl, sha256_file
    from ifv_training.psd_repair import PSDModelRoles
    from ifv_training.psd_source_review import source_review_reference
    from ifv_training.psd_slate import assemble_slate_attempts
    from ifv_training.psd_media import media_from_archive, validate_media
    from ifv_training.psd_materialization import materialize_bank
    from src.orchestrator.source_access import SourceAccessPolicy
    spec = importlib.util.spec_from_file_location('calibration', PRIOR / 'calibrate_psd_hint_review.py')
    calibration = importlib.util.module_from_spec(spec); spec.loader.exec_module(calibration)
    config, continuation, before = calibration.read_completed(OLD / 'search/repairs' / KEY)
    before[str(REVIEW)] = sha256_file(REVIEW)
    before[str(PRESERVATION)] = sha256_file(PRESERVATION)
    owner.save(OUT / 'binding.json', {'files': before, 'review_policy': 'grounded-procedural-advice-v3',
        'agent_resampled': False, 'old_review_replaced': False, 'search_budgets_reset': False,
        'formal_collection': False, 'purpose': 'real multimodal optimizer and resume validation'})
    owner.save(OUT / 'state.json', {'phase': 'strict_audit_and_multimodal_target_build', 'training_started': False})
    try:
        source, seed, gold, audit = [load_json(Path(config[k]['path'])) for k in ('trace','candidate','gold','audit')]
        review, profile = load_json(REVIEW), load_json(Path(config['policy_serving_profile']['path']))
        assert review['schema_version'] == 'ifv-psd-slate-review-v3' and review['decision']['status'] == 'pass'
        roles = PSDModelRoles(hint_constructor_provider=config['hint_constructor_provider'],
            hint_constructor_model=config['hint_constructor_model'],
            frozen_self_teacher_provider=config['policy_provider'], frozen_self_teacher_model=config['policy_model'],
            round_start_checkpoint=config['round_start_checkpoint'],
            round_start_checkpoint_manifest_sha256=config['round_start_checkpoint_manifest']['sha256'],
            trainable_student_provider=config['policy_provider'], trainable_student_model=config['policy_model'],
            trainable_student_initial_checkpoint=config['round_start_checkpoint'])
        targets = continuation['local_targets']
        for target in targets:
            if 248056 in target['teacher_prompt_ids']:
                target['psd_media'] = media_from_archive(target, processor_path=profile['engine_model_path'],
                    output_dir=OUT / 'media', prompt_ids=target['teacher_prompt_ids'])
                validate_media(target['psd_media'], target['student_prompt_ids'])
        candidates, attempts = assemble_slate_attempts(seed=seed, source=source,
            source_hash=config['trace']['sha256'], episode=continuation['teacher_episode_trace'],
            targets=targets, review=review, gold=gold, source_task_review=source_review_reference(seed['source']),
            source_audit=audit if audit.get('source_trace_canonical_sha256') else None,
            source_policy=SourceAccessPolicy.load(Path(config['source_access_policy']['path'])), roles=roles)
        write_jsonl(OUT / 'repair_candidates.jsonl', candidates)
        write_jsonl(OUT / 'repair_attempts.jsonl', attempts)
        assert attempts and all(r['accepted'] for r in attempts), 'Real repaired episode failed strict verification'
        result = materialize_bank(output_dir=OUT / 'bank', repair_candidates=OUT / 'repair_candidates.jsonl',
            repair_attempts=OUT / 'repair_attempts.jsonl', preservation_candidates=PRESERVATION,
            serving_profile=Path(config['policy_serving_profile']['path']),
            checkpoint_manifest=Path(config['round_start_checkpoint_manifest']['path']),
            score_missing_topk=False)
        owner.save(OUT / 'result.json', result)
        owner.save(OUT / 'state.json', {'phase': result['status'], 'training_started': False})
    finally:
        for path, digest in before.items():
            assert sha256_file(Path(path)) == digest, 'Historical artifact mutated'


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    parser.add_argument('--resume-after-fix', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.load(DEPLOY / 'state.json')['deployment_ready_not_live']
    for rel, digest in owner.load(DEPLOY / 'code-binding.json').items():
        assert owner.sha(CODE / rel) == digest
    if args.mode == 'launch':
        if args.resume_after_fix:
            assert owner.load(OUT/'state.json')['phase'] == 'held_requires_diagnosis'
            assert not (OUT/'result.json').exists()
            previous = owner.load(OUT/'process.json')
            assert not Path(f'/proc/{previous["pid"]}').exists()
            for path, digest in owner.load(OUT/'binding.json')['files'].items():
                assert owner.sha(path) == digest
        else:
            OUT.mkdir(exist_ok=False)
        env = {**os.environ, 'PYTHONPATH': str(CODE)+':'+str(CODE/'training'),
               'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT/'tmp'),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        suffix = 'retry-01' if args.resume_after_fix else 'run'
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, OUT/(suffix+'.log'))
        owner.save(OUT/('retry-01.json' if args.resume_after_fix else 'process.json'), receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)})); return
    try:
        execute(owner)
    except Exception as error:
        from ifv_training.psd_diagnostics import persist_exception
        failure = persist_exception(OUT / 'private-diagnostics', error)
        owner.save(OUT / 'failure.json', failure)
        owner.save(OUT / 'state.json', {'phase':'held_requires_diagnosis', 'error_type':type(error).__name__})
        raise


if __name__ == '__main__':
    main()
