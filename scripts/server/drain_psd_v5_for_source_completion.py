"""One-off owned v5 drain: hold admission's du child, never an Agent request.

The old controller has no drain signal. Its existing measurement failure loop
waits without spending slots. Suspend only its exact du subprocess until all
already-admitted episodes finish, then stop the verified collector. Do not use
this utility as a normal scheduler or against any other run.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/volume/ybo/wza')
RUN = ROOT/'runs/psd-production400x8-20260917-v5'
OUT = ROOT/'training-artifacts/psd-complete-source-20260917-v41'


def main():
    spec = importlib.util.spec_from_file_location('owner',
        ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
    receipt = o.load(RUN/'process.json'); pid = receipt['pid']
    expected = ROOT/'training-artifacts/psd-uncapped-storage-20260917-v38/run_psd_production_collection.py'
    if str(expected) not in receipt['command'] or RUN.name not in receipt['command']:
        raise ValueError('Not the explicitly owned v5 collector')
    o.checked(receipt)
    if o.load(OUT/'stage-state.json')['tests_returncode'] != 0:
        raise ValueError('Replacement is not validated')
    if (OUT/'source-drain.json').exists():
        raise ValueError('Drain already completed; do not repeat')
    held, frozen, started, next_check = {}, False, time.monotonic(), 0
    try:
        while time.monotonic()-started < 3600:
            o.checked(receipt)
            # subprocess.run may be launched by an asyncio worker thread.
            for children in Path(f'/proc/{pid}/task').glob('*/children'):
                try: child_ids = children.read_text().split()
                except FileNotFoundError: continue
                for value in child_ids:
                    child = int(value)
                    try:
                        args = [x.decode() for x in Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0') if x]
                        if args[:4] != ['du', '-s', '-B1', '-c'] or str(RUN) not in args[4:]: continue
                        for directory in args[4:]: Path(directory).resolve().relative_to(ROOT/'runs')
                        if os.getpgid(child) != pid: raise ValueError('du process group changed')
                        os.kill(child, signal.SIGSTOP)
                        held[child] = args
                    except ProcessLookupError: continue
                    except FileNotFoundError: continue
            if time.monotonic() >= next_check:
                states = [o.load(p)['payload']['attempts'][-1]['status'] for p in
                    (RUN/'episodes/psd-infrastructure-attempts').glob('*/retry-state.json')]
                progress = o.load(RUN/'collection-progress.json')
                summary = {'completed': states.count('completed'), 'running': states.count('running'),
                    'other': len(states)-states.count('completed')-states.count('running'),
                    'held_measurements': len(held), 'time': time.time(), 'maintenance': 'adopt_user_authorized_complete_source_retries'}
                o.save(OUT/'drain-progress.json', summary)
                print(json.dumps(summary), flush=True)
                next_check = time.monotonic()+10
                if held and states and all(s == 'completed' for s in states) and progress['completed'] == len(states):
                    o.checked(receipt); os.kill(pid, signal.SIGSTOP); frozen = True
                    markers = list((RUN/'episodes/psd-infrastructure-attempts').glob('*/retry-state.json'))
                    if (len(markers) != len(states)
                        or any(o.load(p)['payload']['attempts'][-1]['status'] != 'completed'
                               or not (p.parent/'result.json').is_file() for p in markers)
                        or len(list((RUN/'episodes/traces').glob('*.json'))) != len(states)
                        or o.load(RUN/'collection-progress.json')['completed'] != len(states)):
                        os.kill(pid, signal.SIGCONT); frozen = False
                        continue
                    o.save(OUT/'source-drain.json', {**summary, 'source_run': str(RUN),
                        'receipt': receipt, 'all_attempts_completed': True, 'no_active_episode_interrupted': True})
                    os.killpg(pid, signal.SIGTERM)
                    os.killpg(pid, signal.SIGCONT); frozen = False
                    for _ in range(120):
                        path = Path(f'/proc/{pid}/stat')
                        if not path.exists() or path.read_text().split(') ', 1)[1][0] == 'Z':
                            print(json.dumps({'drained': True, 'completed': len(states)}), flush=True)
                            return
                        time.sleep(.5)
                    raise RuntimeError('Verified drained collector did not exit')
            time.sleep(.25)
        raise RuntimeError('Drain timed out; resume held measurement and inspect')
    finally:
        if frozen:
            o.checked(receipt); os.kill(pid, signal.SIGCONT)
        for child, args in held.items():
            try:
                actual = [x.decode() for x in Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0') if x]
                if actual == args and os.getpgid(child) == pid: os.kill(child, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError): pass


if __name__ == '__main__':
    os.umask(0o077)
    main()
