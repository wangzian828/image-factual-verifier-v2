"""Launch the authorized 400x8 collection; immutable Agent and durable retries.

The first forty are real source slots, not a disposable canary. This controller
only adds scheduling, bounded storage admission and compact in-memory results.
It never supplies private gold to the policy or changes generation settings.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-infrastructure-retry-20260916-v23/code'
RUN = ROOT/'runs/psd-production400x8-20260916-v1'
PREP = ROOT/'runs/psd-pilot400-preparation-20260915-v1'
SERVICE = ROOT/'inference/psd-sft3084-20260916'
PRIOR = ROOT/'runs/psd-sft3084-captured-canary4x8-20260916'
GATEWAY = 'http://127.0.0.1:19025'
GIB = 1024**3
# An admission ceiling, NOT an assertion about the user's unknown GPFS quota.
STORAGE_CEILING = 128 * GIB
REUSE_RUN = None
NUMERICAL_RECOVERY = None


def owner():
    spec = importlib.util.spec_from_file_location('production_owner',
        ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_result(result):
    """Canonical trace was persisted already; retain exact result-record fields."""
    value = dict(result)
    value['state'] = {'stage_timings': result.get('state', {}).get('stage_timings', {})}
    return value


def capture_stats(trace):
    count = external_ok = 0
    for step in trace.get('state', {}).get('all_steps', []):
        meta = step.get('metadata', {})
        external_ok += int(meta.get('tool_success') is True and step.get('action_type') == 'tool_call')
        if (step.get('stage') not in ('unified_react', 'unified_judgment')
                or meta.get('deterministic_segment_boundary')
                or step.get('action_type') not in ('tool_call', 'output')):
            continue
        cap = meta.get('policy_token_capture', {})
        ids, probabilities = cap.get('completion_token_ids', []), cap.get('completion_logprobs', [])
        if (cap.get('status') != 'complete' or not cap.get('prompt_token_ids')
                or not ids or len(ids) != len(probabilities)
                or any(not math.isfinite(x) for x in probabilities)):
            raise ValueError('Native source capture incomplete or nonfinite')
        count += 1
    return {'captures': count, 'successful_tool_calls': external_ok}


def checked_inputs(o):
    o.verify_export()
    if o.load(CODE.parent/'stage-state.json')['tests_returncode'] != 0:
        raise ValueError('Retry snapshot not validated')
    for relative, digest in o.load(CODE.parent/'code-binding.json').items():
        if o.sha(CODE/relative) != digest:
            raise ValueError('Immutable retry snapshot changed')
    old = o.load(PRIOR/'binding.json')
    for relative, digest in old['source_sha256'].items():
        if o.sha(CODE/relative) != digest:
            raise ValueError('Frozen Agent source changed')
    for i in range(4):
        o.checked(o.load(SERVICE/f'replica-{i}.json'))
    o.checked(o.load(SERVICE/'guard.json'))
    health = o.http(GATEWAY+'/health')
    if (len(health['replicas']) != 4 or not all(r['healthy'] for r in health['replicas'])
            or health['post_retries'] != 0 or health['wire_capture']['enabled']):
        raise ValueError('Serving or bounded-storage contract differs')
    source = o.load(PREP/'prepared.json')['payload']
    rows = [json.loads(line) for line in Path(source['benchmark']).read_text().splitlines() if line.strip()]
    splits = [json.loads(line) for line in Path(source['train_cases']).read_text().splitlines() if line.strip()]
    ids = [row['case_id'] for row in rows]
    if (len(rows) != 400 or len(set(ids)) != 400 or len(splits) != 400
            or {r['case_id'] for r in splits if r.get('split') == 'train'} != set(ids)):
        raise ValueError('Expected the complete frozen 400 training cases')
    for row in rows:
        image = (Path(source['benchmark']).parent/row['image_path']).resolve()
        image.relative_to(ROOT)
        if o.sha(image) != row['image_sha256']:
            raise ValueError('Training image mismatch')
    return old, source, ids


def launch(o):
    old, source, ids = checked_inputs(o)
    if REUSE_RUN is not None:
        prior = o.load(REUSE_RUN/'binding.json')
        for key in ('benchmark', 'train_cases', 'private_gold', 'source_access_policy'):
            if prior[key] != source[key] or prior['files'][source[key]] != o.sha(Path(source[key])):
                raise ValueError('Recovered run has different training inputs')
        process = o.load(REUSE_RUN/'process.json')
        cmdline = Path(f'/proc/{process["pid"]}/cmdline')
        if cmdline.exists() and cmdline.read_bytes():
            raise ValueError('Original collector still alive; drain before recovery')
    if any(r['inflight'] for r in o.http(GATEWAY+'/health')['replicas']):
        raise ValueError('Another inference client is active; inspect before launch')
    RUN.mkdir(exist_ok=False)
    snapshot = RUN/'snapshot'
    snapshot.mkdir()
    for name in ('checkpoint-manifest.json', 'sft-provenance.json'):
        shutil.copy2(Path(old['snapshot'])/name, snapshot/name)
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from ifv_training.checkpoints import build_serving_profile
    build_serving_profile(output_path=snapshot/'serving-profile.json',
        profile_id=old['policy_model_name'], model_path=str(o.EXPORT.parent/'model'),
        engine='vllm', port=19025, tensor_parallel_size=1, dtype='bfloat16',
        context_length=131072, tool_call_parser='ifv_psd_qwen3_single',
        reasoning_parser='qwen3', thinking_enabled=True,
        checkpoint_manifest_path=snapshot/'checkpoint-manifest.json')
    binding = {key: source[key] for key in ('benchmark', 'train_cases', 'private_gold', 'source_access_policy')}
    binding.update(case_ids=ids, snapshot=str(snapshot), policy_revision=old['policy_revision'],
        code=str(CODE), slots=3200, concurrency=40, rollouts_per_case=8,
        first_batch=40, temperature=.7, seed=0, formal_source_collection=True,
        storage_ceiling_bytes=STORAGE_CEILING, personal_quota_known=False,
        controller_sha256=o.sha(Path(__file__)),
        reuse_run=str(REUSE_RUN) if REUSE_RUN else None,
        numerical_recovery=o.load(NUMERICAL_RECOVERY) if NUMERICAL_RECOVERY else None,
        files={str(Path(source[key])): o.sha(Path(source[key])) for key in
            ('benchmark', 'train_cases', 'private_gold', 'source_access_policy')})
    o.save(RUN/'binding.json', binding)
    h = o.helper(); h.CODE = CODE; h.GATEWAY = GATEWAY
    env, checks = h.environment()
    env.update(IFV_CAPTURE_POLICY_TOKENS='1', IFV_POLICY_TOPK='20', PYTHONDONTWRITEBYTECODE='1')
    o.save(RUN/'credential-presence.json', checks)
    o.save(RUN/'state.json', {'phase': 'launching_formal_collection', 'slots': 3200, 'optimizer_steps': 0})
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute',
               '--run-name', RUN.name, '--code-directory', str(CODE)]
    if REUSE_RUN is not None:
        command += ['--reuse-run', str(REUSE_RUN)]
    if NUMERICAL_RECOVERY is not None:
        command += ['--numerical-recovery', str(NUMERICAL_RECOVERY)]
    receipt = o.spawn(command, env, RUN/'controller.log')
    o.save(RUN/'process.json', receipt)
    print(json.dumps({'pid': receipt['pid'], 'run': str(RUN), 'slots': 3200, 'concurrency': 40}))


def execute(o):
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from scripts import collect_psd_rollouts as collector
    from src.eval import run_cases
    from ifv_training.psd_repair_search import search_lock
    from scripts.audit_real_trace import audit_trace, SourceAccessPolicy, _json_summary
    binding = o.load(RUN/'binding.json')
    if o.sha(Path(__file__)) != binding['controller_sha256']:
        raise ValueError('Controller changed since launch')
    for path, digest in binding['files'].items():
        if o.sha(Path(path)) != digest:
            raise ValueError('Bound training input changed')
    started = time.time()
    progress = {'selected': 3200, 'completed': 0, 'canonical': 0, 'normal_errors': 0,
                'pending_infrastructure': 0, 'concurrency': 40, 'started_at': started}
    from ifv_training.psd_storage_admission import StorageAdmission
    storage_admission = StorageAdmission(run=RUN, ceiling=STORAGE_CEILING, save=o.save)
    reused_results = {}

    class ProductionWorkflow(collector.PSDWorkflow):
        def _new_batch_child(self, config):
            return ProductionWorkflow(config)

        async def run_single(self, image_path, image_id='', *, runtime_case=None):
            if image_id in reused_results:
                # Import already verified the complete immutable cache and
                # canonical trace. Reuse its compact record without rewriting
                # multi-GB histories or generating a new model request.
                result = reused_results[image_id]
            else:
                await storage_admission()
                result = await super().run_single(image_path, image_id, runtime_case=runtime_case)
            progress['completed'] += 1
            pending = bool(result.get('psd_infrastructure_pending'))
            progress['pending_infrastructure'] += int(pending)
            progress['canonical'] += int(not pending)
            progress['normal_errors'] += int(not pending and bool(result.get('error')))
            progress['elapsed_seconds'] = time.time()-started
            o.save(RUN/'collection-progress.json', progress)
            # Avoid retaining all 3200 multimodal top20 histories in RAM.
            # Full native traces and attempt ledgers remain unchanged on disk.
            return compact_result(result)

        async def run_batch(self, image_paths, image_ids=None, runtime_cases=None,
                            sampling_seeds=None, concurrency=40):
            arrays = dict(image_paths=image_paths, image_ids=image_ids,
                runtime_cases=runtime_cases, sampling_seeds=sampling_seeds)
            if any(value is None or len(value) != 3200 for value in arrays.values()) or concurrency != 40:
                raise ValueError('Formal sampling identity or concurrency mismatch')
            if binding.get('reuse_run'):
                from ifv_training.psd_collection_recovery import import_prior_slots
                imported = import_prior_slots(source=Path(binding['reuse_run'])/'episodes',
                    destination=RUN/'episodes', episode_ids=image_ids, seeds=sampling_seeds,
                    numerical_recovery=binding.get('numerical_recovery'))
                o.save(RUN/'slot-recovery.json', imported)
                reused_results.update({r['episode_id']: r['cached_result'] for r in imported['records']
                                       if r['action'] == 'reuse_exact_outcome'})
            first = await super().run_batch(**{k: v[:40] for k, v in arrays.items()}, concurrency=40)
            paths = []
            for episode in image_ids[:40]:
                name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in episode)+'.json'
                path = Path(self.config.output_dir)/name
                if not path.is_file():
                    raise RuntimeError('First40 has unresolved infrastructure slots; inspect before extending')
                paths.append(path)
            healthy, tool_calls, audits, captures = 0, 0, [], []
            policy = SourceAccessPolicy.load(Path(binding['source_access_policy']))
            for path in paths:
                trace = o.load(path)
                stats = capture_stats(trace)
                captures.append({'episode': trace['image_id'], **stats})
                if trace.get('termination') == 'success' and not trace.get('error'):
                    healthy += 1
                    tool_calls += stats['successful_tool_calls']
                    audits.append(audit_trace(path, source_access_policy=policy))
            audit = _json_summary(audits, strict_scheduler=True)
            passed = healthy > 0 and tool_calls > 0 and audit['passed']
            o.save(RUN/'first40-gate.json', {'passed': passed, 'slots': 40, 'complete_reports': healthy,
                'captures': captures, 'audit': audit, 'valid_sources_retained': True,
                'selection_by_answer': False, 'teacher_top20': 'requires_frozen_raw_rescoring'})
            if not passed:
                raise RuntimeError('Online first40 engineering gate requires diagnosis')
            o.save(RUN/'state.json', {'phase': 'collecting_remaining_3160', 'slots': 3200,
                'first40_passed': True, 'optimizer_steps': 0, 'time': time.time()})
            rest = await super().run_batch(**{k: v[40:] for k, v in arrays.items()}, concurrency=40)
            return first+rest

    collector.PSDWorkflow = ProductionWorkflow
    run_cases._git_commit = lambda: binding['policy_revision']
    sys.argv = ['psd-formal400x8', '--train-cases', binding['train_cases'],
        '--benchmark', binding['benchmark'], '--source-access-policy', binding['source_access_policy'],
        '--profile', 'student-qwen3.5-local', '--output-dir', str(RUN/'episodes'),
        '--concurrency', '40', '--rollouts-per-case', '8', '--base-sampling-seed', '0', '--timeout', '3000']
    with search_lock(RUN/'controller-lock'):
        try:
            o.save(RUN/'state.json', {'phase': 'formal_first40_online_acceptance', 'slots': 3200,
                'concurrency': 40, 'optimizer_steps': 0, 'time': time.time()})
            collector.main()
            from ifv_training.psd_collection import verify_collection
            from ifv_training.io import load_jsonl
            verified = verify_collection(load_jsonl(RUN/'episodes/run_results.jsonl'),
                case_ids=binding['case_ids'], manifest=o.load(RUN/'episodes/run_manifest.json'), expected_rollouts=8)
            o.save(RUN/'collection-verification.json', verified)
            o.save(RUN/'state.json', {'phase': 'collection_complete_requires_source_checker',
                'slots': 3200, 'optimizer_steps': 0, 'time': time.time()})
        except BaseException as error:
            o.save(RUN/'failure.json', {'type': type(error).__name__, 'time': time.time(),
                'preserve_completed_slots': True, 'requires_diagnosis_not_blind_relaunch': True})
            o.save(RUN/'state.json', {'phase': 'collection_requires_intervention',
                'optimizer_steps': 0, 'time': time.time()})
            raise
        finally:
            o.verify_export()


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    parser.add_argument('--run-name', default=RUN.name)
    parser.add_argument('--code-directory', type=Path, default=CODE)
    parser.add_argument('--reuse-run', type=Path)
    parser.add_argument('--numerical-recovery', type=Path)
    arguments = parser.parse_args()
    if Path(arguments.run_name).name != arguments.run_name or not arguments.run_name.startswith('psd-production400x8-'):
        parser.error('Invalid formal run name')
    RUN = ROOT/'runs'/arguments.run_name
    CODE = arguments.code_directory.resolve()
    CODE.relative_to(ROOT/'training-artifacts')
    if CODE.name != 'code':
        parser.error('Use an immutable code snapshot')
    if arguments.reuse_run:
        REUSE_RUN = arguments.reuse_run.resolve()
        REUSE_RUN.relative_to(ROOT/'runs')
        if REUSE_RUN == RUN:
            parser.error('Recovery must preserve the original run')
    if arguments.numerical_recovery:
        if REUSE_RUN is None:
            parser.error('A numerical recovery gate requires --reuse-run')
        NUMERICAL_RECOVERY = arguments.numerical_recovery.resolve()
        NUMERICAL_RECOVERY.relative_to(SERVICE)
    {'launch': launch, 'execute': execute}[arguments.mode](owner())
