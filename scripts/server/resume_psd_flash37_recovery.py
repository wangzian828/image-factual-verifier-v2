"""Hand off the PSD repair controller to direct Gemini 3.7 Flash/high.

This deployment waits for the current v60 controller to reach its next
completed-pass boundary. It preserves all completed responses and persisted
budgets, changes only uncached PSD external requests, and starts at eight
external requests in flight.
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
import sys
import time


ROOT = Path('/volume/ybo/wza')
OLD = ROOT / 'training-artifacts/psd-tail-recovery-20260918-v60'
DEPLOY = ROOT / 'training-artifacts/psd-flash37-recovery-20260918-v61'
CODE = DEPLOY / 'code'
PREVIOUS = ROOT / 'runs/psd-formal-prepare-controller-20260918-tail-recovery-v3'
CONTROL = ROOT / 'runs/psd-formal-prepare-controller-20260918-flash37-recovery-v4'
ROUND = ROOT / 'runs/psd-production-round1-20260917-v1'
SEARCH = ROUND / 'search-gemini37-flash-high'
ROUTE = DEPLOY / 'external-model-route.json'
OVERLAYS = {
    'psd_model_route.py': 'training/ifv_training',
    'run_psd_lightweight_resume.py': 'training/scripts/h20',
    'test_psd_model_route.py': 'training/tests',
    'resume_psd_flash37_recovery.py': 'scripts/server',
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
    spec = importlib.util.spec_from_file_location('psd_flash37_base', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    # The copied base launcher keeps its behavior, but all mutable deployment
    # paths and the route must resolve to this immutable v61 snapshot.
    result.OLD = DEPLOY
    result.DEPLOY = DEPLOY
    result.CODE = CODE
    result.CONTROL = CONTROL
    return result


def stage():
    if CODE.exists():
        raise RuntimeError('immutable snapshot already exists')
    if not (OLD / 'code').is_dir():
        raise RuntimeError('v60 code snapshot is missing')
    shutil.copytree(OLD / 'code', CODE, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name, parent in OVERLAYS.items():
        source = DEPLOY / name
        if not source.is_file():
            raise RuntimeError('missing overlay: ' + name)
        shutil.copy2(source, CODE / parent / name)
    tests = [
        'test_psd_repair_storage.py', 'test_psd_model_route.py',
        'test_psd_slate.py', 'test_psd_slate_pipeline.py',
        'test_psd_slate_bound_positions.py', 'test_psd_candidate_binding.py',
        'test_psd_infrastructure_retry.py',
    ]
    with (DEPLOY / 'tests.log').open('x') as log:
        result = __import__('subprocess').call([
            sys.executable, '-m', 'pytest', '-q',
            *[str(CODE / 'training/tests' / name) for name in tests]],
            cwd=CODE, env=base().environment(), stdout=log, stderr=log)
    save(DEPLOY / 'stage-state.json', {
        'deployment_ready_not_live': result == 0,
        'tests_returncode': result,
        'source_snapshot': str(OLD / 'code'),
        'external_model': 'gemini-3.7-flash',
        'thinking_level': 'high',
        'external_concurrency': 8,
        'source_and_completed_results_preserved': True,
        'overlays': {name: sha(DEPLOY / name) for name in OVERLAYS},
        'time': time.time(),
    })
    if result:
        raise RuntimeError('flash37 recovery regression failed')
    print(json.dumps({'tests_returncode': 0}), flush=True)


def alive(pid):
    stat = Path('/proc', str(pid), 'stat')
    return stat.exists() and stat.read_text().split(') ', 1)[1][0] != 'Z'


def migrate_transport_ledgers():
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    from ifv_training.psd_repair_storage import load_bound, save_bound
    rows = []
    for marker in sorted(SEARCH.glob('repairs/*/slate-rounds/*/infrastructure-attempts/retry-state.json')):
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
        latest.update(status='infrastructure_failed', reason=TRANSPORT_ERRORS[error_type],
                      legacy_status='nonretryable_error',
                      migration_version='ifv-psd-transport-ledger-recovery-v2')
        save_bound(marker, identity=identity, payload=state)
        rows.append({'path': str(marker.relative_to(ROOT)), 'before_sha256': before,
                     'after_sha256': sha(marker), 'error_type': error_type})
    report = {'schema_version': 'ifv-psd-transport-ledger-recovery-v2',
              'migrated': len(rows), 'attempt_budget_reset': False,
              'answer_or_checker_conditioned': False, 'ledgers': rows, 'time': time.time()}
    save(DEPLOY / 'ledger-migration.json', report)
    return report


def start():
    state = load(DEPLOY / 'stage-state.json')
    if state.get('deployment_ready_not_live') is not True:
        raise RuntimeError('unvalidated stage')
    if (DEPLOY / 'handoff-process.json').exists():
        raise RuntimeError('handoff already started')
    receipt = base().owner().spawn([
        sys.executable, '-u', str(CODE / 'scripts/server/resume_psd_flash37_recovery.py'),
        'wait-boundary'], base().environment(), DEPLOY / 'handoff.log')
    save(DEPLOY / 'handoff-process.json', receipt)
    print(json.dumps({'handoff_pid': receipt['pid']}), flush=True)


def wait_boundary():
    manager = base().owner()
    previous = load(PREVIOUS / 'process.json')
    expected_parent = [sys.executable, '-u', str(OLD / 'code/scripts/server/resume_psd_tail_recovery.py'), 'worker']
    if previous.get('command') != expected_parent:
        raise RuntimeError('unexpected v60 owner')
    while True:
        state = load(PREVIOUS / 'state.json')
        if state.get('phase') in {'requires_frozen_teacher_topk', 'ready_for_training'}:
            save(DEPLOY / 'handoff-state.json', {'phase': 'no_handoff_needed',
                'previous_state': state, 'time': time.time()})
            return
        if state.get('phase') == 'formal_prepare_result':
            break
        if not alive(previous['pid']):
            raise RuntimeError('v60 owner died outside a verified boundary')
        save(DEPLOY / 'handoff-state.json', {'phase': 'waiting_current_pass_no_interruption',
            'time': time.time()})
        time.sleep(5)
    manager.checked(previous)
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
        prepare = (len(command) >= 4 and command[:4] == [sys.executable, '-u',
                   str(OLD / 'code/scripts/run_psd_round.py'), 'prepare'] and str(ROUND) in command)
        if command != expected_parent and not tracker and not prepare:
            raise RuntimeError('unexpected active member in v60 group')
        group.append({'pid': pid, 'command': command})
    if CONTROL.exists():
        raise RuntimeError('v61 controller already exists')
    progress = load(SEARCH / 'progress.json')
    save(DEPLOY / 'handoff.json', {'previous': previous, 'group': group,
        'previous_state': state, 'completed_cases': len(progress.get('cases', [])),
        'reason': 'user_requested_gemini37_direct_route',
        'external_model': 'gemini-3.7-flash', 'external_concurrency': 8,
        'source_and_completed_results_preserved': True, 'time': time.time()})
    os.killpg(previous['pid'], signal.SIGTERM)
    for _ in range(80):
        if not any(alive(row['pid']) for row in group):
            break
        time.sleep(.25)
    else:
        raise RuntimeError('v60 group did not terminate')
    save(DEPLOY / 'controlled-interruptions.json', {
        'schema_version': 'ifv-psd-controlled-interruptions-v1',
        'round': str(ROUND), 'terminated_prepare_pids': [row['pid'] for row in group],
        'reason': 'explicit_external_model_switch_to_gemini37', 'time': time.time()})
    migration = migrate_transport_ledgers()
    CONTROL.mkdir(exist_ok=False)
    receipt = manager.spawn([
        sys.executable, '-u', str(CODE / 'scripts/server/resume_psd_flash37_recovery.py'),
        'worker'], base().environment(), CONTROL / 'controller.log')
    save(CONTROL / 'process.json', receipt)
    save(CONTROL / 'state.json', {'phase': 'launched', 'external_model': 'gemini-3.7-flash',
        'thinking_level': 'high', 'external_concurrency': 8,
        'migrated_transport_ledgers': migration['migrated'], 'time': time.time()})
    save(PREVIOUS / 'flash37-handoff.json', {'new_controller': str(CONTROL), 'time': time.time()})


def hard_switch():
    """Stop exactly the current v60 prepare group and continue on v61."""
    manager = base().owner()
    previous = load(PREVIOUS / 'process.json')
    expected_parent = [sys.executable, '-u', str(OLD / 'code/scripts/server/resume_psd_tail_recovery.py'), 'worker']
    if previous.get('command') != expected_parent:
        raise RuntimeError('unexpected v60 owner')
    manager.checked(previous)
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
        prepare = (len(command) >= 4 and command[:4] == [sys.executable, '-u',
                   str(OLD / 'code/scripts/run_psd_round.py'), 'prepare'] and str(ROUND) in command)
        if command != expected_parent and not tracker and not prepare:
            raise RuntimeError('unexpected active member in v60 group')
        group.append({'pid': pid, 'command': command})
    if not group:
        raise RuntimeError('v60 process group is empty')
    progress = load(SEARCH / 'progress.json')
    save(DEPLOY / 'hard-switch.json', {'previous': previous, 'group': group,
        'previous_state': load(PREVIOUS / 'state.json'),
        'completed_cases': len(progress.get('cases', [])),
        'reason': 'user_requested_hard_switch_to_gemini37',
        'external_model': 'gemini-3.7-flash', 'external_concurrency': 8,
        'source_and_completed_results_preserved': True, 'time': time.time()})
    os.killpg(previous['pid'], signal.SIGTERM)
    for _ in range(80):
        if not any(alive(row['pid']) for row in group):
            break
        time.sleep(.25)
    else:
        os.killpg(previous['pid'], signal.SIGKILL)
        for _ in range(20):
            if not any(alive(row['pid']) for row in group):
                break
            time.sleep(.25)
        else:
            raise RuntimeError('v60 group did not terminate after hard stop')
    save(DEPLOY / 'controlled-interruptions.json', {
        'schema_version': 'ifv-psd-controlled-interruptions-v1',
        'round': str(ROUND), 'terminated_prepare_pids': [row['pid'] for row in group],
        'reason': 'explicit_external_model_hard_switch_to_gemini37', 'time': time.time()})
    launch = base().launcher()
    reconciled = launch.reconcile_controlled_interruptions()
    migration = migrate_transport_ledgers()
    if CONTROL.exists():
        raise RuntimeError('v61 controller already exists')
    CONTROL.mkdir(exist_ok=False)
    receipt = manager.spawn([
        sys.executable, '-u', str(CODE / 'scripts/server/resume_psd_flash37_recovery.py'),
        'worker'], base().environment(), CONTROL / 'controller.log')
    save(CONTROL / 'process.json', receipt)
    save(CONTROL / 'state.json', {'phase': 'launched', 'external_model': 'gemini-3.7-flash',
        'thinking_level': 'high', 'external_concurrency': 8,
        'reconciled_interrupted_attempts': reconciled['reconciled'],
        'migrated_transport_ledgers': migration['migrated'], 'time': time.time()})
    save(PREVIOUS / 'flash37-hard-switch.json', {'new_controller': str(CONTROL),
        'reconciled': reconciled['reconciled'], 'time': time.time()})
    print(json.dumps({'controller_pid': receipt['pid'],
        'reconciled_interrupted_attempts': reconciled['reconciled']}), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage', 'start', 'wait-boundary', 'hard-switch', 'worker'])
    mode = parser.parse_args().mode
    if mode == 'stage':
        stage()
    elif mode == 'start':
        start()
    elif mode == 'wait-boundary':
        wait_boundary()
    elif mode == 'hard-switch':
        hard_switch()
    else:
        base().launcher().worker()
