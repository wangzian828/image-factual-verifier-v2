"""Resume one reviewed v14 case, preserving its cached episode and search budget."""
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
CONTROL = RUN / 'reviewed-case-continuation-v1'


def restore_args(config):
    values = {k: Path(v['path']) if isinstance(v, dict) and set(v) == {'path', 'sha256'} else v
              for k, v in config.items() if k != 'continuation_policy_version'}
    values.update(output_dir=Path(values['output_dir']), resume=True, skip_auto_judge=False, generation_retries=1)
    return argparse.Namespace(**values)


def recovery():
    spec = importlib.util.spec_from_file_location('cached_review_recovery', DEPLOY / 'recover_psd_cached_review.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def execute(o, helper):
    from scripts.run_psd_repair_driver import _run
    from ifv_training.psd_diagnostics import persist_exception
    config, directory, continuation = helper.load_review_inputs(CASE)
    assert (directory / 'review.json').is_file()
    args = restore_args(config)
    before = {str(p): o.sha(p) for p in [CASE / 'run-inputs.json', directory / 'continuation.json',
                                       directory / 'episode.json', directory / 'review.json']}
    o.save(CONTROL / 'binding.json', {'files': before, 'complete_rerun_budget': args.repair_attempts,
        'proposal_budget': args.proposal_rounds, 'completed_episode_resampling': False})
    o.save(CONTROL / 'state.json', {'phase': 'continuing_original_case_budget', 'time': time.time()})
    try:
        result = await _run(args)
        o.save(CONTROL / 'result.json', result)
        o.save(CONTROL / 'state.json', {'phase': result['status'], 'time': time.time(),
                                       'training_started': False})
    except Exception as error:
        o.save(CONTROL / 'failure.json', persist_exception(CONTROL / 'private-diagnostics', error))
        o.save(CONTROL / 'state.json', {'phase': 'held_requires_diagnosis', 'time': time.time(),
                                       'error_type': type(error).__name__})
        raise
    finally:
        for raw, expected in before.items():
            assert o.sha(raw) == expected, 'Completed artifact changed on resume'


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    helper = recovery()
    original, o = helper.owner()
    for path, digest in o.load(DEPLOY / 'code-binding.json').items():
        assert o.sha(CODE / path) == digest, 'Immutable code changed'
    assert o.load(helper.CONTROL / 'state.json')['phase'] == 'review_complete'
    if args.mode == 'execute':
        asyncio.run(execute(o, helper))
        return
    for p in [RUN / 'process.json', helper.CONTROL / 'process.json']:
        assert not Path(f"/proc/{o.load(p)['pid']}").exists(), 'Previous controller is still alive'
    helper.load_review_inputs(CASE)
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
