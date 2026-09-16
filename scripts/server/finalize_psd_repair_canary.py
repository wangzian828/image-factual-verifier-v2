"""Aggregate completed canary searches; never authorize another repair attempt."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
CODE = DEPLOY / 'code'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'
CONTROL = RUN / 'completed-canary-materialization-v1'
CASES = ('7996ba9752652f37', '82897bdeb31b6ade', 'e79d4db478e45800')


def completed_inputs(run):
    from ifv_training.io import load_json, sha256_file
    from ifv_training.psd_slate_search import audit_slate_search
    recovery = load_json(run / 'execution-recovery-v1/result.json')['results']
    if len(recovery) != len(CASES) or {r['case'] for r in recovery} != set(CASES):
        raise ValueError('Recovery has not completed all fixed cases')
    if any('error_type' in row for row in recovery):
        raise ValueError('Engineering failures must be diagnosed before aggregation')
    files, accepted = {}, 0
    for key in CASES:
        case = run / 'search/repairs' / key
        if not audit_slate_search(case)['passed']:
            raise ValueError('Search is pending or paused; no new generations allowed here')
        manifest = load_json(case / 'manifest.json')
        accepted += manifest['accepted_count']
        for name in ('run-inputs.json', 'slate-state.json', 'manifest.json',
                     'repair_candidates.jsonl', 'repair_attempts.jsonl'):
            path = case / name
            files[str(path)] = sha256_file(path)
    if not accepted:
        raise ValueError('No verified repair target; preservation-only is not a full PSD gate')
    return {'files': files, 'accepted_targets': accepted, 'new_agent_generations_allowed': False}


def main():
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('observed', DEPLOY / 'run_psd_observed_positions_gate.py')
    observed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observed)
    original, owner = observed.owner()
    observed.validate(owner)
    binding = completed_inputs(RUN)
    if args.mode == 'launch':
        receipt = owner.load(RUN / 'execution-recovery-v1/process.json')
        stat = Path(f'/proc/{receipt["pid"]}/stat')
        assert not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z'
        CONTROL.mkdir(exist_ok=False)
        owner.save(CONTROL / 'binding.json', binding)
        env, checks = original.capture_environment(owner)
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
                   PYTHONPATH=str(CODE) + ':' + str(CODE / 'training'))
        owner.save(CONTROL / 'credential-presence.json', checks)
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'],
                              env, CONTROL / 'run.log')
        owner.save(CONTROL / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(CONTROL)}))
        return
    assert owner.load(CONTROL / 'binding.json') == binding
    owner.save(CONTROL / 'state.json', {'phase': 'aggregating_cached_searches', 'training_started': False})
    try:
        observed.execute(owner)
        owner.save(CONTROL / 'state.json', {
            'phase': owner.load(RUN / 'result.json')['status'], 'training_started': False})
    finally:
        for path, digest in binding['files'].items():
            assert owner.sha(path) == digest, 'Completed repair search was modified'


if __name__ == '__main__':
    main()
