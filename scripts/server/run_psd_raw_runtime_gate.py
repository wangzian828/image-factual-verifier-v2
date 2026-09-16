"""Two fixed complete Agent episodes to test raw capture; never training data."""
import argparse
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
CODE = ROOT / 'training-artifacts/psd-raw-teacher-resume-20260916-v20/code'
RUN = ROOT / 'runs/psd-raw-runtime-gate-20260916-v1'
OLD = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916'


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('launch', 'execute'))
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    binding = owner.load(OLD / 'binding.json')
    if args.mode == 'launch':
        RUN.mkdir(exist_ok=False)
        h = owner.helper(); h.CODE = CODE; h.GATEWAY = 'http://127.0.0.1:19022'
        env, checks = h.environment()
        env.update(IFV_CAPTURE_POLICY_TOKENS='1', IFV_POLICY_TOPK='20', PYTHONDONTWRITEBYTECODE='1')
        owner.save(RUN / 'credential-presence.json', checks)
        owner.save(RUN / 'binding.json', {'case_ids': binding['case_ids'][:2], 'formal_training': False,
            'purpose': 'fixed raw-probability runtime verification only; never add to training bank',
            'source_binding_sha256': owner.sha(OLD / 'binding.json'),
            'code_binding_sha256': owner.sha(CODE.parent / 'code-binding.json')})
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, RUN / 'run.log')
        owner.save(RUN / 'process.json', receipt); print(json.dumps({'pid': receipt['pid'], 'run': str(RUN)})); return
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from ifv_training.psd_capture_semantics import install_capture_semantics, RAW_POLICY_LOGPROBS
    install_capture_semantics()
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.eval import run_cases
    sys.argv = ['raw-runtime-gate', '--benchmark', binding['benchmark'],
        '--source-access-policy', binding['source_access_policy'], '--profile', 'student-qwen3.5-local',
        '--output-dir', str(RUN / 'episodes'), '--concurrency', '2', '--rollouts-per-case', '1',
        '--base-sampling-seed', '0', '--timeout', '3000']
    for case in binding['case_ids'][:2]: sys.argv += ['--case-id', case]
    run_cases.VerificationWorkflow = PSDWorkflow
    run_cases._git_commit = lambda: ''  # No server Git; immutable source binding above.
    owner.save(RUN / 'state.json', {'phase': 'full_agent_runtime_validation', 'formal_training': False})
    try:
        summary = asyncio.run(run_cases._run_cases(run_cases._parse_args()))
        from scripts.audit_real_trace import audit_trace, SourceAccessPolicy, _json_summary
        paths = sorted((RUN / 'episodes/traces').glob('*.json'))
        assert len(paths) == 2
        audit = _json_summary([audit_trace(path, source_access_policy=SourceAccessPolicy.load(
            Path(binding['source_access_policy']))) for path in paths], strict_scheduler=True)
        owner.save(RUN / 'strict-audit.json', audit)
        assert audit['passed']
        captured = 0
        for path in paths:
            trace = owner.load(path)
            assert not trace.get('error') and trace.get('termination') == 'success'
            for step in trace['state']['all_steps']:
                metadata = step.get('metadata', {})
                if step.get('stage') not in ('unified_react', 'unified_judgment') or metadata.get('deterministic_segment_boundary'):
                    continue
                if step.get('action_type') not in ('tool_call', 'output'): continue
                cap = metadata.get('policy_token_capture', {})
                assert cap.get('status') == 'complete'
                assert cap.get('teacher_logprob_semantics') == RAW_POLICY_LOGPROBS
                assert cap.get('prompt_token_ids') and cap.get('completion_token_ids')
                assert len(cap['completion_token_ids']) == len(cap['completion_logprobs'])
                assert all(math.isfinite(x) for x in cap['completion_logprobs'])
                captured += 1
        assert captured >= 2
        owner.save(RUN / 'state.json', {'phase': 'passed_full_agent_raw_capture', 'captured_actions': captured,
            'summary': summary, 'formal_training': False})
    except BaseException as error:
        owner.save(RUN / 'state.json', {'phase': 'requires_fix', 'error_type': type(error).__name__, 'formal_training': False})
        raise


if __name__ == '__main__': main()
