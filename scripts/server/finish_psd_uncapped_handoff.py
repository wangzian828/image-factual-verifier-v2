"""Finish the already authorized storage-policy change after the v4 drain."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
OUT = ROOT/'training-artifacts/psd-uncapped-storage-20260917-v38'
CODE = OUT/'code'
OLD = ROOT/'runs/psd-production400x8-20260917-v4'
NEW = ROOT/'runs/psd-production400x8-20260917-v5'


def main():
    spec = importlib.util.spec_from_file_location('owner',
        ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
    def status(phase, **extra):
        o.save(OUT/'storage-policy-handoff.json', {'phase': phase, 'time': time.time(), **extra})
        print(json.dumps({'phase': phase, **extra}), flush=True)
    try:
        if NEW.exists(): raise ValueError('Destination already exists; inspect instead of duplicate launch')
        if o.load(OUT/'stage-state.json')['tests_returncode'] != 0: raise ValueError('Stage tests failed')
        status('waiting_for_inflight_episodes_to_finish')
        deadline = time.monotonic()+3700
        while not (OUT/'source-drain.json').exists():
            o.checked(o.load(OUT/'drain-process.json'))
            if time.monotonic() > deadline: raise TimeoutError('Owned drain did not complete')
            time.sleep(5)
        proof = o.load(OUT/'source-drain.json')
        if not proof['all_attempts_completed'] or not proof['no_active_episode_interrupted']:
            raise ValueError('Invalid drain proof')
        old_pid = o.load(OLD/'process.json')['pid']
        for _ in range(60):
            cmdline = Path(f'/proc/{old_pid}/cmdline')
            if not cmdline.exists() or not cmdline.read_bytes(): break
            time.sleep(1)
        else: raise ValueError('Original collector still alive')
        status('waiting_for_compactor_idle_window', completed=proof['completed'])
        compactor = o.load(OLD/'storage-compaction-process.json')
        deadline = time.monotonic()+900
        while True:
            o.checked(compactor)
            wait = Path(f'/proc/{compactor["pid"]}/wchan').read_text().strip()
            if wait == 'hrtimer_nanosleep':
                o.stop(compactor)
                break
            if time.monotonic() > deadline: raise TimeoutError('Compactor did not finish its sweep')
            time.sleep(2)
        status('launching_uncapped_recovery', completed=proof['completed'])
        deadline = time.monotonic()+120
        while any(r['inflight'] for r in o.http('http://127.0.0.1:19025/health')['replicas']):
            if time.monotonic() > deadline: raise TimeoutError('Gateway did not drain')
            time.sleep(1)
        subprocess.run([sys.executable, '-u', str(OUT/'run_psd_production_collection.py'),
            'launch', '--run-name', NEW.name, '--code-directory', str(CODE), '--reuse-run', str(OLD)],
            check=True, timeout=180)
        status('validating_and_reusing_completed_sources', completed=proof['completed'])
        deadline = time.monotonic()+5400
        while True:
            o.checked(o.load(NEW/'process.json'))
            state = o.load(NEW/'state.json')
            if state['phase'] == 'collection_requires_intervention': raise ValueError('Recovery requires diagnosis')
            if state['phase'] == 'collecting_remaining_3160': break
            if time.monotonic() > deadline: raise TimeoutError('Recovery validation exceeded maintenance window')
            time.sleep(10)
        recovery = o.load(NEW/'slot-recovery.json')
        if recovery['reused'] != proof['completed'] or recovery['explicit_budget_extensions'] != 0:
            raise ValueError('Recovery did not preserve every completed outcome and original budget')
        if o.load(NEW/'binding.json')['storage_ceiling_bytes'] is not None:
            raise ValueError('Arbitrary run cap unexpectedly remains')
        if (NEW/'storage-compaction-process.json').exists(): raise ValueError('Compactor already launched')
        env = {**os.environ, 'PYTHONPATH': str(CODE)+':'+str(CODE/'training'),
               'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT/'tmp')}
        compactor = o.spawn([sys.executable, '-u', str(CODE/'scripts/server/compact_psd_completed_storage.py'),
            '--run', str(NEW), '--reader-snapshot', str(CODE), '--follow'], env,
            NEW/'storage-compaction-v38-controller.log')
        o.save(NEW/'storage-compaction-process.json', compactor)
        status('uncapped_collection_resumed', reused=recovery['reused'], run=str(NEW), ceiling_bytes=None)
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            o.checked(o.load(NEW/'process.json'))
            path = NEW/'storage.json'
            space = o.load(path) if path.exists() else {}
            progress = o.load(NEW/'collection-progress.json')
            if (space.get('measurement_status') == 'complete' and space.get('admission_open')
                    and 'ceiling_bytes' in space and space['ceiling_bytes'] is None
                    and progress['completed'] > recovery['reused']):
                o.checked(compactor); o.verify_export()
                status('verified_new_sampling_without_run_cap', reused=recovery['reused'],
                       completed=progress['completed'], ceiling_bytes=None, originals_preserved=True)
                return
            time.sleep(10)
        raise TimeoutError('No verified new uncapped sampling before maintenance deadline')
    except BaseException as error:
        status('requires_diagnosis', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    os.umask(0o077)
    main()
