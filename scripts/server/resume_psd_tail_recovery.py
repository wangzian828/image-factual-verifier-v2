"""Switch at a completed prepare pass to bounded PSD tail recovery.

The immutable deployment adds two narrow corrections: evidence-only repair for
cached slate-review citation contracts, and transport classification for direct
httpx failures. It never interrupts an active prepare pass or resets a case,
proposal, rerun, or infrastructure budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path('/volume/ybo/wza')
OLD = ROOT / 'training-artifacts/psd-source-gate-20260917-v59'
DEPLOY = ROOT / 'training-artifacts/psd-tail-recovery-20260918-v60'
CODE = DEPLOY / 'code'
PREVIOUS = ROOT / 'runs/psd-formal-prepare-controller-20260917-checker-feedback-v2'
CONTROL = ROOT / 'runs/psd-formal-prepare-controller-20260918-tail-recovery-v3'
ROUND = ROOT / 'runs/psd-production-round1-20260917-v1'
SEARCH = ROUND / 'search-gemini37-flash-high'
OVERLAYS = {
    'psd_slate.py': 'training/ifv_training',
    'psd_infrastructure_retry.py': 'training/ifv_training',
    'test_psd_slate_pipeline.py': 'training/tests',
    'test_psd_infrastructure_retry.py': 'training/tests',
    'resume_psd_tail_recovery.py': 'scripts/server',
}
TRANSPORT_ERRORS = {
    'ConnectError': 'model_transport_failure',
    'ReadError': 'model_transport_failure',
    'RemoteProtocolError': 'model_transport_failure',
    'WriteError': 'model_transport_failure',
    'ConnectTimeout': 'model_request_timeout',
    'PoolTimeout': 'model_request_timeout',
    'ReadTimeout': 'model_request_timeout',
    'WriteTimeout': 'model_request_timeout',
}


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def base():
    path = CODE / 'scripts/server/deploy_psd_checker_feedback.py'
    spec = importlib.util.spec_from_file_location('psd_tail_recovery_base', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.CODE, result.DEPLOY, result.CONTROL = CODE, DEPLOY, CONTROL
    return result


def stage():
    if CODE.exists():
        raise RuntimeError('immutable snapshot already exists')
    if not (OLD / 'code').is_dir():
        raise RuntimeError('validated source-gate snapshot is missing')
    shutil.copytree(OLD / 'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, parent in OVERLAYS.items():
        source = DEPLOY / name
        if not source.is_file():
            raise RuntimeError('missing overlay: ' + name)
        shutil.copy2(source, CODE / parent / name)
    tests = [
        'test_psd_slate_feedback.py', 'test_psd_slate_pipeline.py', 'test_psd_slate.py',
        'test_psd_slate_bound_positions.py', 'test_psd_candidate_binding.py',
        'test_psd_repair_storage.py', 'test_psd_model_route.py',
        'test_psd_infrastructure_retry.py',
    ]
    with (DEPLOY / 'tests.log').open('x') as log:
        result = subprocess.call([sys.executable, '-m', 'pytest', '-q',
            *[str(CODE / 'training/tests' / name) for name in tests]],
            cwd=CODE, env=base().environment(), stdout=log, stderr=log)
    save(DEPLOY / 'stage-state.json', {
        'deployment_ready_not_live': result == 0,
        'tests_returncode': result,
        'source_snapshot': str(OLD / 'code'),
        'overlays': {name: sha(DEPLOY / name) for name in OVERLAYS},
        'semantic_review_resampling': False,
        'infrastructure_budget_reset': False,
        'time': time.time(),
    })
    if result:
        raise RuntimeError('tail recovery regression failed')
    print(json.dumps({'tests_returncode': 0}), flush=True)


def start():
    state = load(DEPLOY / 'stage-state.json')
    if state.get('deployment_ready_not_live') is not True:
        raise RuntimeError('unvalidated stage')
    if (DEPLOY / 'handoff-process.json').exists():
        raise RuntimeError('handoff already started')
    launcher = base()
    receipt = launcher.owner().spawn([
        sys.executable, '-u', str(CODE / 'scripts/server/resume_psd_tail_recovery.py'),
        'wait-boundary'], launcher.environment(), DEPLOY / 'handoff.log')
    save(DEPLOY / 'handoff-process.json', receipt)
    print(json.dumps({'handoff_pid': receipt['pid']}), flush=True)


def alive(pid):
    stat = Path('/proc', str(pid), 'stat')
    return stat.exists() and stat.read_text().split(') ', 1)[1][0] != 'Z'


def migrate_transport_ledgers():
    """Reclassify only persisted, exact transport exceptions in place."""
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from ifv_training.psd_repair_storage import load_bound, save_bound
    rows = []
    pattern = 'repairs/*/slate-rounds/*/infrastructure-attempts/retry-state.json'
    for marker in sorted(SEARCH.glob(pattern)):
        raw = load(marker)
        identity = raw.get('identity')
        if not isinstance(identity, dict):
            raise RuntimeError('retry ledger identity is invalid')
        state = load_bound(marker, identity=identity)
        attempts = state.get('attempts')
        if not isinstance(attempts, list) or not attempts:
            continue
        latest = attempts[-1]
        error_type = latest.get('error_type')
        if latest.get('status') != 'nonretryable_error' or error_type not in TRANSPORT_ERRORS:
            continue
        if (marker.parent / 'result.json').exists():
            raise RuntimeError('transport-failed ledger already has a result')
        before = sha(marker)
        backup = DEPLOY / 'ledger-snapshots' / (before + '.json')
        if backup.exists():
            if sha(backup) != before:
                raise RuntimeError('transport ledger backup changed')
        else:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(marker, backup)
            if sha(backup) != before:
                raise RuntimeError('transport ledger backup failed')
        latest.update(status='infrastructure_failed', reason=TRANSPORT_ERRORS[error_type],
            legacy_status='nonretryable_error',
            migration_version='ifv-psd-transport-ledger-recovery-v1')
        save_bound(marker, identity=identity, payload=state)
        rows.append({'path': str(marker.relative_to(ROOT)), 'before_sha256': before,
            'backup': str(backup.relative_to(ROOT)), 'after_sha256': sha(marker),
            'attempt_index': latest.get('index'), 'error_type': error_type,
            'reason': latest['reason']})
    report = {'schema_version': 'ifv-psd-transport-ledger-recovery-v1',
        'migrated': len(rows), 'attempt_budget_reset': False,
        'answer_or_checker_conditioned': False, 'ledgers': rows, 'time': time.time()}
    save(DEPLOY / 'ledger-migration.json', report)
    return report


def wait_boundary():
    launcher = base()
    owner = launcher.owner()
    while not (PREVIOUS / 'process.json').exists():
        upstream = OLD / 'handoff-state.json'
        if upstream.exists() and load(upstream).get('phase') == 'no_handoff_needed':
            save(DEPLOY / 'handoff-state.json', {
                'phase': 'no_handoff_needed', 'upstream': load(upstream), 'time': time.time()})
            return
        save(DEPLOY / 'handoff-state.json', {'phase': 'waiting_upstream_controller', 'time': time.time()})
        time.sleep(5)
    previous = load(PREVIOUS / 'process.json')
    expected = [sys.executable, '-u', str(OLD / 'code/scripts/server/resume_psd_source_gate.py'), 'worker']
    if previous.get('command') != expected:
        raise RuntimeError('unexpected previous owner')
    while True:
        state = load(PREVIOUS / 'state.json')
        if state.get('phase') in {'requires_frozen_teacher_topk', 'ready_for_training'}:
            save(DEPLOY / 'handoff-state.json', {
                'phase': 'no_handoff_needed', 'previous_state': state, 'time': time.time()})
            return
        if state.get('phase') == 'formal_prepare_result':
            break
        if not alive(previous['pid']):
            raise RuntimeError('prepare died outside a verified boundary')
        save(DEPLOY / 'handoff-state.json', {
            'phase': 'waiting_current_pass_no_interruption', 'time': time.time()})
        time.sleep(5)
    frozen = False
    try:
        if alive(previous['pid']):
            owner.checked(previous)
            os.kill(previous['pid'], signal.SIGSTOP)
            frozen = True
        current = load(PREVIOUS / 'state.json')
        if current != state:
            raise RuntimeError('boundary changed before suspension')
        group = []
        for entry in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(entry.name)
                if os.getpgid(pid) != previous['pid'] or not alive(pid):
                    continue
                command = [part.decode() for part in (entry / 'cmdline').read_bytes().split(b'\0') if part]
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            tracker = (len(command) == 3 and command[:2] == [sys.executable, '-c'] and
                re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\(\d+\)', command[2]))
            if command != expected and not tracker:
                raise RuntimeError('active child remains; refusing handoff')
            group.append(pid)
        if CONTROL.exists():
            raise RuntimeError('new controller already exists')
        if group:
            os.killpg(previous['pid'], signal.SIGTERM)
            if frozen:
                os.kill(previous['pid'], signal.SIGCONT)
                frozen = False
            for _ in range(40):
                if not any(alive(pid) for pid in group):
                    break
                time.sleep(.25)
            else:
                raise RuntimeError('old owner still alive')
        migration = migrate_transport_ledgers()
        CONTROL.mkdir(exist_ok=False)
        receipt = owner.spawn([sys.executable, '-u',
            str(CODE / 'scripts/server/resume_psd_tail_recovery.py'), 'worker'],
            launcher.environment(), CONTROL / 'controller.log')
        save(CONTROL / 'process.json', receipt)
        save(DEPLOY / 'handoff-state.json', {'phase': 'switched_at_completed_pass',
            'previous_state': state, 'new_controller': str(CONTROL),
            'new_pid': receipt['pid'], 'interrupted_rollouts': 0,
            'migrated_transport_ledgers': migration['migrated'], 'time': time.time()})
        save(PREVIOUS / 'tail-recovery-handoff.json', {
            'new_controller': str(CONTROL), 'time': time.time()})
    finally:
        if frozen and alive(previous['pid']):
            os.kill(previous['pid'], signal.SIGCONT)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage', 'start', 'wait-boundary', 'worker'])
    mode = parser.parse_args().mode
    if mode == 'stage':
        stage()
    elif mode == 'start':
        start()
    elif mode == 'wait-boundary':
        wait_boundary()
    else:
        base().launcher().worker()
