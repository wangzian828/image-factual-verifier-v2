"""Fork only broken repair construction; reuse fixed sources/checkers verbatim.

The new snapshot explicitly uses four identical epoch3 replicas at port19019.
No source sampling, GPU optimization, or modification to previous attempts.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/volume/ybo/wza')
PREVIOUS = ROOT / 'training-artifacts/psd-epoch3-diagnostic-20260916-v8'
DEPLOY = ROOT / 'training-artifacts/psd-native-repair-20260916-v9'
CODE = DEPLOY / 'code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'
SOURCE = RUN / 'psd-round-v5/source-bank'
OUT = RUN / 'psd-native-repair-v9'
SNAPSHOT = DEPLOY / 'snapshot'


def owner():
    spec = importlib.util.spec_from_file_location('v8_diagnostic_owner', PREVIOUS / 'diagnostic_resume_psd_epoch3.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value.controller().owner()


async def verify_archived_failures(o):
    """Exact re-execution of local validation only; never admit these old targets."""
    import httpx
    from ifv_training.psd_repair import HintProposal
    from ifv_training.psd_slate import capture_target
    from ifv_training.psd_repair_runtime import QwenContinuationAdapter
    from src.orchestrator.stage_runner import StageStep
    from src.orchestrator.react_runtime import compile_react_judgment_basis
    results = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async def tokenize(body):
            response = await client.post('http://127.0.0.1:19019/tokenize', json=body)
            response.raise_for_status()
            return response.json()['tokens']
        for path in sorted(PREVIOUS.glob('*-failure.json')):
            receipt = o.load(path)
            with gzip.open(receipt['path'], 'rb') as file:
                body = file.read()
            import hashlib
            assert hashlib.sha256(body).hexdigest() == receipt['sha256']
            failure = json.loads(body)
            contexts = {frame['function']: frame['locals'] for frame in failure['selected_model_context']}
            if 'capture_target' in contexts:
                values = contexts['capture_target']
                steps = [StageStep(**row) for row in values['steps']]
                runtime_path = Path(receipt['path']).parent.parent
                # The captured context_request_id is bound to the most recent
                # runtime whose request exists and has this exact prompt capture.
                matches = []
                for context in runtime_path.glob('runtime/*/slate-*/context/' + values['request_id'] + '.json'):
                    from src.orchestrator.runtime_events import reconstruct_archived_request
                    request = reconstruct_archived_request(str(context.parent.parent), values['request_id'])
                    if request == values['request']:
                        matches.append(context.parent.parent)
                # values.request has restored live order/text; compare the
                # stable context ID using the step's archived input instead.
                if not matches:
                    from ifv_training.psd_slate import hydrate_live_snapshot
                    from src.orchestrator.stage_runner import StageRunner
                    for context in runtime_path.glob('runtime/*/slate-*/context/' + values['request_id'] + '.json'):
                        request = reconstruct_archived_request(str(context.parent.parent), values['request_id'])
                        try:
                            hydrate_live_snapshot(StageRunner._snapshot_input_payload(values['messages']), request['input_payload'])
                        except ValueError:
                            continue
                        matches.append(context.parent.parent)
                assert len(matches) == 1, 'Diagnostic runtime binding is ambiguous'
                target = await capture_target(position=values['position'], hint=HintProposal(**values['hint']),
                    steps=steps, unhinted_prefix=values['unhinted_prefix'], runtime_store=SimpleNamespace(root=matches[0]),
                    system_instruction=values['system_instruction'], tokenize=tokenize, model='ifv-qwen3.5-9b-sft-3084')
                assert target['teacher_prompt_ids'] == values['capture']['prompt_token_ids']
                results.append({'case': path.stem, 'exact_teacher_tokens': len(target['teacher_prompt_ids']),
                                'student_hint_removed': True, 'passed': True, 'old_target_not_admitted': True})
            else:
                values = contexts['run_hinted_episode']
                steps = [StageStep(**row) for row in values['teacher_steps']]
                state = SimpleNamespace(objective='', action_count=0, stop_reason='', finish_rationale='')
                before = compile_react_judgment_basis(state, steps)
                after = QwenContinuationAdapter._judgment_basis(state, steps)
                assert not before['observation_ids'] and after['observation_ids']
                decisions = [row['metadata'].get('policy_action', {}) for row in values['judgment_steps']]
                cited = [row.get('verdict_observation_ids', []) for row in decisions]
                assert any(ids and set(ids) <= set(after['observation_ids']) for ids in cited)
                results.append({'case': path.stem, 'before_allowed_ids': 0, 'after_allowed_ids': len(after['observation_ids']),
                                'previously_rejected_citations_are_observed': True, 'passed': True,
                                'old_target_not_admitted': True})
    assert len(results) == 3 and all(row['passed'] for row in results)
    o.save(DEPLOY / 'archived-failure-verification.json', {'passed': True, 'cases': results,
           'no_generation_or_external_api': True, 'training_started': False})


async def execute_async(o):
    from ifv_training.psd_diagnostics import persist_exception
    import ifv_training.psd_gemini_judge as judge
    import scripts.run_psd_feedback_canary as feedback
    from ifv_training.psd_repair import _sha
    original_request = judge._request
    reused = []
    async def cached_request(client, packet, *, prompt, schema, model, images=(), cache_dir=None):
        identity = {'version': judge.VERSION, 'model': model, 'prompt_sha256': _sha(prompt),
                    'schema_sha256': _sha(schema), 'packet_sha256': _sha(packet), 'images_sha256': _sha(images)}
        if cache_dir:
            cache = Path(cache_dir) / (_sha(identity) + '.json')
            if not cache.exists():
                for donor in sorted((RUN / 'psd-round-v5/search/repairs').glob('*/judge-cache/' + cache.name)):
                    saved = o.load(donor)
                    assert saved['identity'] == identity and saved['response_sha256'] == _sha(saved['response'])
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(donor, cache)
                    reused.append({'source': str(donor), 'destination': str(cache), 'sha256': o.sha(cache)})
                    o.save(DEPLOY / 'reused-judge-responses.json', {'responses': reused})
                    break
        return await original_request(client, packet, prompt=prompt, schema=schema, model=model,
                                      images=images, cache_dir=cache_dir)
    judge._request = cached_request
    original_run = feedback._run
    async def observed(args):
        try:
            return await original_run(args)
        except Exception as error:
            receipt = persist_exception(args.output_dir / 'private-diagnostics', error)
            o.save(DEPLOY / (args.output_dir.name + '-failure.json'), receipt)
            raise
    feedback._run = observed
    await verify_archived_failures(o)
    args = feedback._build_parser().parse_args(['--source', str(SOURCE), '--snapshot', str(SNAPSHOT),
        '--output', str(OUT), '--attempts', '6', '--proposal-rounds', '12', '--case-concurrency', '40',
        '--task-source-selection', 'longest_failed', '--repair-mode', 'slate',
        '--judge-model', 'gemini-3.1-pro-preview', '--teacher-device', 'cuda:0'])
    o.save(DEPLOY / 'state.json', {'phase': 'native_contract_repair_running', 'time': time.time(),
           'case_concurrency': 40, 'eligible_tasks': 3, 'four_gpu_gateway': 19019, 'training_started': False})
    result = await feedback.run(args)
    o.save(DEPLOY / 'result.json', result)
    o.save(DEPLOY / 'state.json', {'phase': result['status'], 'time': time.time(), 'training_started': False,
                                 'search': str(OUT / 'progress.json')})


def execute():
    m, o = owner()
    try:
        for path, digest in o.load(DEPLOY / 'binding.json')['files'].items():
            assert o.sha(path) == digest, path
        o.verify_export()
        health = o.http('http://127.0.0.1:19019/health')
        assert len(health['replicas']) == 4 and all(row['healthy'] for row in health['replicas'])
        sys.path[:0] = [str(CODE), str(CODE / 'training')]
        asyncio.run(execute_async(o))
    except BaseException as error:
        o.save(DEPLOY / 'state.json', {'phase': 'held_requires_inspection', 'error_type': type(error).__name__,
                                     'training_started': False, 'time': time.time()})
        raise


def prepare():
    m, o = owner()
    assert not CODE.exists() and not OUT.exists() and not SNAPSHOT.exists()
    assert o.load(PREVIOUS / 'state.json')['phase'] == 'paused_search_requires_resume'
    process = o.load(PREVIOUS / 'process.json')
    stat = Path(f'/proc/{process["pid"]}/stat')
    assert not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z'
    shutil.copytree(PREVIOUS / 'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, destination in {
        'psd_repair_runtime.py': 'training/ifv_training', 'psd_slate.py': 'training/ifv_training',
        'run_psd_repair_driver.py': 'scripts', 'test_psd_native_repair_contract.py': 'training/tests',
        'test_psd_diagnostics.py': 'training/tests'}.items():
        shutil.copy2(DEPLOY / name, CODE / destination / name)
    for path in (CODE / 'src').rglob('*.py'):
        assert path.read_text() == (ROOT / 'image-factual-verifier-v2' / path.relative_to(CODE)).read_text()
    SNAPSHOT.mkdir()
    shutil.copy2(RUN / 'snapshot/checkpoint-manifest.json', SNAPSHOT / 'checkpoint-manifest.json')
    profile = o.load(RUN / 'snapshot/serving-profile.json')
    assert profile['base_url'] == 'http://127.0.0.1:19018/v1'
    profile['base_url'] = 'http://127.0.0.1:19019/v1'
    o.save(SNAPSHOT / 'serving-profile.json', profile)
    files = [RUN / 'binding.json', RUN / 'runtime-acceptance-v5.json', SOURCE / 'prepared.json',
             SOURCE / 'candidates/repair_candidates.jsonl', SOURCE / 'candidates/preservation_candidates.jsonl',
             SNAPSHOT / 'serving-profile.json', SNAPSHOT / 'checkpoint-manifest.json',
             *[path for folder in ('src', 'scripts', 'training') for path in (CODE / folder).rglob('*.py')]]
    o.save(DEPLOY / 'binding.json', {'files': {str(path): o.sha(path) for path in files},
        'source_resampling': False, 'old_repairs_retained': str(RUN / 'psd-round-v5/search'),
        'native_contract_version': 'native-base-prompts-and-observation-locator-v3',
        'source_checker_rerun': False, 'new_snapshot_endpoint_only': True})
    o.save(DEPLOY / 'state.json', {'phase': 'prepared_requires_tests', 'training_started': False})


def launch():
    m, o = owner()
    assert o.load(DEPLOY / 'state.json')['phase'] == 'prepared_requires_tests'
    assert not (DEPLOY / 'process.json').exists()
    env, checks = m.capture_environment(o)
    env['PYTHONPATH'] = str(CODE) + ':' + str(CODE / 'training')
    o.save(DEPLOY / 'credential-presence.json', checks)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
    receipt = o.spawn(command, env, DEPLOY / 'run.log')
    receipt['script_sha256'] = o.sha(__file__)
    o.save(DEPLOY / 'process.json', receipt)
    print(json.dumps(receipt))


def audit_only():
    """Read archived failures and call tokenize only; no rollout or training."""
    _, o = owner()
    for path, digest in o.load(DEPLOY / 'binding.json')['files'].items():
        assert o.sha(path) == digest, path
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    asyncio.run(verify_archived_failures(o))
    print(json.dumps(o.load(DEPLOY / 'archived-failure-verification.json')))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'launch', 'execute', 'audit-only'):
        modes.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else launch() if args.launch else audit_only() if args.audit_only else execute()
