"""Resume the explicitly authorized, drained friend Pro campaign unchanged."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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


def json_value(value):
    """Compare JSON-representable identities using their persisted types."""
    return json.loads(json.dumps(value))


def worker():
    original = ROOT / 'friend_generation_campaign_v1.py'
    spec = importlib.util.spec_from_file_location('friend_frozen_campaign', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = module.freeze
    module.freeze = lambda path, value: frozen(path, json_value(value))
    sys.argv = [str(original), *sys.argv[2:]]
    return module.main()


def main():
    os.umask(0o077)
    with (ROOT / 'state/pro-campaign-resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = load(RECEIPT)
        if Path('/proc', str(old['pid'])).exists():
            raise RuntimeError('Previous campaign owner is still present')
        state, summary, identity = [load(CAMPAIGN / name) for name in
            ('state.json', 'aggregate.json', 'identity.json')]
        tuple_resume_error = False
        if state.get('state') == 'held_controller_exception' and old.get('log'):
            log = Path(old['log']).read_text()
            tuple_resume_error = ('Frozen campaign identity changed: ' +
                str(CAMPAIGN / 'input-binding.json')) in log
        if (state.get('state') != 'stopped_after_drain' and not tuple_resume_error
                or summary.get('active') != 0):
            raise RuntimeError('Campaign must be drained before resuming')
        script = ROOT / 'friend_generation_campaign_v1.py'
        if hashlib.sha256(script.read_bytes()).hexdigest() != identity['controller_sha256']:
            raise RuntimeError('Frozen friend controller changed')
        command = old.get('original_command', old['command'])
        if (command[2] != str(script) or command[3:6] != ['--lane', 'pro', '--execute']
                or json.loads(command[-1]) != identity['thinking']):
            raise RuntimeError('Frozen friend command differs')
        config = load(CAMPAIGN / 'config-wave-00.json')
        manifest = load(ROOT / 'project' / config['manifest'])
        prompt = (ROOT / 'project' / config['prompt']).read_text()
        expected = {'samples': [[row['sample_id'], row['sha256']]
            for row in manifest['samples']],
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        if load(CAMPAIGN / 'input-binding.json') != expected:
            raise RuntimeError('Actual frozen sample or prompt content changed')
        stamp = str(time.time_ns())
        save(ROOT / 'state' / ('pro-campaign-previous-' + stamp + '.json'), old)
        log_path = ROOT / 'state' / ('pro-campaign-resumed-' + stamp + '.log')
        resumed_command = [command[0], '-u', str(Path(__file__).resolve()), '--worker', *command[3:]]
        with log_path.open('xb') as log:
            process = subprocess.Popen(resumed_command, cwd=ROOT / 'project',
                env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': process.pid, 'command': resumed_command,
            'original_command': command, 'lane': 'pro',
            'time': time.time(), 'log': str(log_path), 'resumed_from_pid': old['pid']}
        save(RECEIPT, receipt)
        print(json.dumps({'pid': process.pid, 'model': identity['model'],
            'previous_success': summary['success'], 'failed_to_retry': summary['failed'],
            'unattempted': summary['unattempted'], 'concurrency': 1}))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        raise SystemExit(worker())
    main()
