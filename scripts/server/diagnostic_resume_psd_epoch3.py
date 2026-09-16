"""One bounded diagnostic resume of the same pending slate proposals.

Only adds private failure observations in a new immutable code deployment.
Never changes the frozen Agent or samples the original 32 source slots again.
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
PREVIOUS = ROOT / 'training-artifacts/psd-epoch3-repair-20260916-v6'
DEPLOY = ROOT / 'training-artifacts/psd-epoch3-diagnostic-20260916-v8'
CODE = DEPLOY / 'code'


def controller():
    path = PREVIOUS / 'resume_psd_epoch3_canary.py'
    spec = importlib.util.spec_from_file_location('previous_resume', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result.controller()


def execute():
    c = controller()
    m, o = c.owner()
    def phase(name, **extra):
        o.save(DEPLOY / 'state.json', {'phase': name, 'time': time.time(),
                                     'training_started': False, **extra})
    try:
        binding = o.load(DEPLOY / 'binding.json')
        for raw, digest in binding['files'].items():
            assert o.sha(raw) == digest, 'Diagnostic input changed'
        original = o.load(c.RUN / 'binding.json')
        current = o.preflight()[1]
        assert all(current[key] == original[key] for key in current
                   if key not in ('benchmark', 'source_access_policy'))
        sys.path[:0] = [str(CODE), str(CODE / 'training')]
        from ifv_training.psd_diagnostics import persist_exception
        import scripts.run_psd_feedback_canary as feedback
        original_run = feedback._run
        async def observed_run(args):
            try:
                return await original_run(args)
            except Exception as error:
                receipt = persist_exception(args.output_dir / 'private-diagnostics', error)
                o.save(DEPLOY / (args.output_dir.name + '-failure.json'), receipt)
                raise
        feedback._run = observed_run
        from scripts.run_psd_round import prepare
        args = SimpleNamespace(run_dir=c.RUN / 'episodes', benchmark=Path(original['benchmark']),
            train_cases=Path(original['train_cases']), private_gold=Path(original['private_gold']),
            source_access_policy=Path(original['source_access_policy']), snapshot=c.RUN / 'snapshot',
            output=c.OUT, round_index=1, previous_round_completion=None, attempts=6, case_concurrency=40,
            source_review_concurrency=4, expected_rollouts_per_case=8, task_source_selection='longest_failed',
            repair_mode='slate', judge_model='gemini-3.1-pro-preview', teacher_device='cuda:0', defer_topk=True)
        phase('diagnostic_resume_three_pending_tasks', configured_concurrency=40,
              eligible_tasks=3, serving='original_bound_canary_gateway_19018')
        result = asyncio.run(prepare(args))
        o.save(DEPLOY / 'result.json', result)
        phase(result['status'], result=result)
    except BaseException as error:
        phase('held_requires_inspection', error_type=type(error).__name__)
        raise


def launch():
    c = controller()
    m, o = c.owner()
    assert not CODE.exists() and not (DEPLOY / 'process.json').exists()
    assert o.load(PREVIOUS / 'state.json')['phase'] == 'paused_search_requires_resume'
    receipt = o.load(PREVIOUS / 'process.json')
    stat = Path(f'/proc/{receipt["pid"]}/stat')
    assert not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z'
    progress = o.load(c.OUT / 'search/progress.json')
    assert progress['status'] == 'paused_search_requires_resume' and len(progress['cases']) == 3
    assert all(row['result']['status'] == 'paused_case_exception' for row in progress['cases'])
    shutil.copytree(PREVIOUS / 'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copy2(DEPLOY / 'psd_diagnostics.py', CODE / 'training/ifv_training/psd_diagnostics.py')
    files = [c.RUN / 'binding.json', c.RUN / 'runtime-acceptance-v5.json',
             c.OUT / 'search/inputs.json', c.OUT / 'search/task-source-selection.json']
    o.save(DEPLOY / 'before-progress.json', progress)
    o.save(DEPLOY / 'binding.json', {'files': {str(path): o.sha(path) for path in files},
        'code_sha256': {str(path.relative_to(CODE)): o.sha(path) for parent in ('src', 'scripts', 'training')
                       for path in (CODE / parent).rglob('*.py')},
        'source_resampling': False, 'diagnostic_resume_invocations': 1,
        'policy_prompt_sampling_and_acceptance_changed': False})
    env, checks = m.capture_environment(o)
    env['PYTHONPATH'] = str(CODE) + ':' + str(CODE / 'training')
    o.save(DEPLOY / 'credential-presence.json', checks)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
    receipt = o.spawn(command, env, DEPLOY / 'run.log')
    receipt['script_sha256'] = o.sha(__file__)
    o.save(DEPLOY / 'process.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    launch() if args.launch else execute()
