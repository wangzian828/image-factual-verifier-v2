"""Stage immutable PSD retries and prove real full-Agent recovery with one fault.

No SFT weights/services are changed. Injected diagnostics are NOT source data.
All remote writes remain within /volume/ybo/wza; no server Git commands.
"""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/volume/ybo/wza')
DROP = ROOT/'training-artifacts/psd-infrastructure-retry-20260916-v23'
CODE = DROP/'code'
PRIOR = ROOT/'training-artifacts/psd-capture-callsite-20260916-v21/code'
SERVICE = ROOT/'inference/psd-sft3084-20260916'
RUN = ROOT/'runs/psd-infrastructure-retry-gate-20260916-v1'
FILES = {
    'collect_psd_rollouts.py': 'scripts',
    'psd_infrastructure_retry.py': 'training/ifv_training',
    'psd_collection.py': 'training/ifv_training',
    'psd_repair_storage.py': 'training/ifv_training',
    'psd_slate_search.py': 'training/ifv_training',
    'test_psd_infrastructure_retry.py': 'training/tests',
    'test_psd_slate_pipeline.py': 'training/tests',
}


def owner():
    spec = importlib.util.spec_from_file_location('owner', ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def stage(o):
    assert not CODE.exists()
    shutil.copytree(PRIOR, CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, dest in FILES.items():
        shutil.copy2(DROP/name, CODE/dest/name)
    for path in (CODE/'src').rglob('*.py'):
        assert o.sha(path) == o.sha(PRIOR/path.relative_to(CODE))
    o.save(DROP/'code-binding.json', {str(p.relative_to(CODE)): o.sha(p) for p in CODE.rglob('*') if p.is_file()})
    env = {**os.environ, 'PYTHONPATH': str(CODE)+':'+str(CODE/'training'),
        'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(ROOT/'tmp')}
    with (DROP/'tests.log').open('x') as log:
        result = subprocess.call([sys.executable, '-m', 'pytest', '-q', 'training/tests', '--tb=short'],
            cwd=CODE, env=env, stdout=log, stderr=log)
    o.save(DROP/'stage-state.json', {'tests_returncode': result, 'frozen_agent_unchanged': True,
        'formal_training': False})
    if result: raise RuntimeError('Retry snapshot regression failed')


def launch(o):
    assert o.load(DROP/'stage-state.json')['tests_returncode'] == 0
    o.verify_export()
    for name, digest in o.load(DROP/'code-binding.json').items():
        assert o.sha(CODE/name) == digest
    for i in range(4): o.checked(o.load(SERVICE/f'replica-{i}.json'))
    o.checked(o.load(SERVICE/'guard.json'))
    health = o.http('http://127.0.0.1:19025/health')
    assert all(r['healthy'] and r['inflight'] == 0 for r in health['replicas'])
    RUN.mkdir(exist_ok=False)
    h = o.helper(); h.CODE = CODE; h.GATEWAY = 'http://127.0.0.1:19025'
    env, checks = h.environment()
    env.update(IFV_CAPTURE_POLICY_TOKENS='1', IFV_POLICY_TOPK='20', PYTHONDONTWRITEBYTECODE='1')
    o.save(RUN/'credential-presence.json', checks)
    receipt = o.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'], env, RUN/'run.log')
    o.save(RUN/'process.json', receipt)
    print(json.dumps({'pid': receipt['pid'], 'run': str(RUN), 'formal_training': False}))


def execute(o):
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from ifv_training.psd_capture_semantics import install_capture_semantics
    from scripts.collect_psd_rollouts import PSDWorkflow
    from src.eval import run_cases
    from src.orchestrator.llm_backend import APIBackend
    from src.orchestrator.runtime_events import current_case_runtime_store
    from src.redaction import sanitize_for_persistence
    from scripts.audit_real_trace import audit_trace, SourceAccessPolicy, _json_summary
    install_capture_semantics()
    binding = o.load(ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/binding.json')
    cases = binding['case_ids'][:2]
    injected, calls = [], {}
    original = APIBackend.get_response

    async def diagnostic(self, *args, **kwargs):
        response = await original(self, *args, **kwargs)
        runtime = current_case_runtime_store()
        if runtime is not None and runtime.case_id == cases[0] and 'attempt-001' in runtime.root.parts:
            key = str(runtime.root)
            calls[key] = calls.get(key, 0) + 1
            if calls[key] == 2:
                original_artifact = runtime.artifacts.put_text(
                    json.dumps(sanitize_for_persistence(response.raw), ensure_ascii=False),
                    media_type='application/json', suffix='.json', metadata={'kind': 'before_diagnostic_fault'})
                response = copy.deepcopy(response)
                response.raw['choices'][0]['logprobs']['content'][0]['logprob'] = float('nan')
                runtime.append_event('psd_diagnostic_fault_injected', {'original_response': original_artifact,
                    'type': 'nonfinite_selected_logprob', 'not_a_natural_failure': True})
                injected.append({'case': runtime.case_id, 'runtime': key, 'request': 2})
                o.save(RUN/'fault-injection.json', {'events': injected, 'never_training_data': True})
        return response

    APIBackend.get_response = diagnostic
    run_cases.VerificationWorkflow = PSDWorkflow
    run_cases._git_commit = lambda: binding['policy_revision']
    sys.argv = ['psd-retry-gate', '--benchmark', binding['benchmark'], '--source-access-policy',
        binding['source_access_policy'], '--profile', 'student-qwen3.5-local', '--output-dir',
        str(RUN/'episodes'), '--concurrency', '2', '--rollouts-per-case', '1',
        '--base-sampling-seed', '0', '--timeout', '3000']
    for case in cases: sys.argv += ['--case-id', case]
    o.save(RUN/'state.json', {'phase': 'full_agent_retry_validation', 'formal_training': False,
        'cases': cases, 'fault_injection': 'one selected logprob on second request of first case'})
    try:
        summary = asyncio.run(run_cases._run_cases(run_cases._parse_args()))
        traces = sorted((RUN/'episodes/traces').glob('*.json'))
        audit = _json_summary([audit_trace(p, source_access_policy=SourceAccessPolicy.load(
            Path(binding['source_access_policy']))) for p in traces], strict_scheduler=True)
        o.save(RUN/'strict-audit.json', audit)
        states = [o.load(p) for p in (RUN/'episodes/psd-infrastructure-attempts').glob('*/retry-state.json')]
        attempts = [{'identity': s['identity']['inputs'], 'attempts': s['payload']['attempts']} for s in states]
        o.save(RUN/'recovery-audit.json', {'slots': attempts, 'summary': summary})
        recovered = next(s for s in attempts if s['identity']['case_id'] == cases[0])
        passed = (len(injected) == 1 and len(traces) == 2 and audit['passed']
            and summary['num_errors'] == 0 and recovered['attempts'][0]['status'] == 'infrastructure_failed'
            and recovered['attempts'][-1]['status'] == 'completed')
        o.save(RUN/'state.json', {'phase': 'full_agent_retry_gate_passed' if passed else 'requires_fix',
            'passed': passed, 'slots': 2, 'attempts': sum(len(s['attempts']) for s in attempts),
            'injected_failures': len(injected), 'formal_training': False, 'never_training_data': True})
        if not passed: raise RuntimeError('Real Agent infrastructure recovery gate failed')
    except BaseException as error:
        o.save(RUN/'failure.json', {'type': type(error).__name__, 'formal_training': False})
        raise
    finally:
        o.verify_export()


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'launch', 'execute'])
    args = parser.parse_args()
    {'stage': stage, 'launch': launch, 'execute': execute}[args.mode](owner())
