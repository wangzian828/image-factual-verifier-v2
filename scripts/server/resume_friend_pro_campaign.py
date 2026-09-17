"""Resume the explicitly authorized, drained friend Pro campaign unchanged."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/volume/ybo/wza/external-projects/wanglinhaotrain-20260915')
CAMPAIGN = ROOT / 'project/outputs/friend-pro-campaign-20260916'
RECEIPT = ROOT / 'state/pro-campaign-20260916-process.json'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    os.umask(0o077)
    with (ROOT / 'state/pro-campaign-resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = load(RECEIPT)
        if Path('/proc', str(old['pid'])).exists():
            raise RuntimeError('Previous campaign owner is still present')
        state, summary, identity = [load(CAMPAIGN / name) for name in
            ('state.json', 'aggregate.json', 'identity.json')]
        if state.get('state') != 'stopped_after_drain' or summary.get('active') != 0:
            raise RuntimeError('Campaign must be drained before resuming')
        script = ROOT / 'friend_generation_campaign_v1.py'
        if hashlib.sha256(script.read_bytes()).hexdigest() != identity['controller_sha256']:
            raise RuntimeError('Frozen friend controller changed')
        command = old['command']
        if (command[2] != str(script) or command[3:6] != ['--lane', 'pro', '--execute']
                or json.loads(command[-1]) != identity['thinking']):
            raise RuntimeError('Frozen friend command differs')
        stamp = str(time.time_ns())
        save(ROOT / 'state' / ('pro-campaign-previous-' + stamp + '.json'), old)
        log_path = ROOT / 'state' / ('pro-campaign-resumed-' + stamp + '.log')
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, cwd=ROOT / 'project',
                env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': process.pid, 'command': command, 'lane': 'pro',
            'time': time.time(), 'log': str(log_path), 'resumed_from_pid': old['pid']}
        save(RECEIPT, receipt)
        print(json.dumps({'pid': process.pid, 'model': identity['model'],
            'previous_success': summary['success'], 'failed_to_retry': summary['failed'],
            'unattempted': summary['unattempted'], 'concurrency': 1}))


if __name__ == '__main__':
    main()
