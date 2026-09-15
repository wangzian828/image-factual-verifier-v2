"""Preserve cross-image cache evidence and gracefully stop only the bound canary."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/volume/ybo/wza')
RUN = ROOT/'runs/psd-slate-canary4x8-20260916'
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    out = RUN/'cache-contamination-hold'
    assert not out.exists(), 'Existing hold must be inspected, not repeated'
    receipt = load(RUN/'process.json')
    command = [x.decode() for x in Path(f'/proc/{receipt["pid"]}/cmdline').read_bytes().split(b'\0') if x]
    assert command == receipt['command']
    assert digest(Path(command[2])) == receipt['script_sha256']
    env = dict(x.decode().split('=', 1) for x in Path(f'/proc/{receipt["pid"]}/environ').read_bytes().split(b'\0') if x)
    public = [json.loads(line) for line in Path(load(RUN/'binding.json')['benchmark']).read_text().splitlines()]
    by_sha = {r['image_sha256']: r['case_id'] for r in public}
    records, groups = [], {}
    for path in sorted((RUN/'episodes/traces').glob('*.json')):
        trace = load(path)
        image = Path(trace['image_path']).resolve()
        image.relative_to(RUN)
        sha = digest(image)
        assert sha in by_sha
        for step in trace.get('state', {}).get('all_steps', []):
            if step.get('tool_name') != 'perceive_scene' or not step.get('tool_result'):
                continue
            result_sha = hashlib.sha256(step['tool_result'].encode()).hexdigest()
            groups.setdefault(result_sha, set()).add(sha)
            records.append({'trace': str(path), 'trace_sha256': digest(path), 'case_id': by_sha[sha],
                'image_sha256': sha, 'result_sha256': result_sha,
                'cache_hit': step.get('metadata', {}).get('cache_hit'),
                'tool_args': step.get('tool_args'), 'has_error': bool(trace.get('error'))})
    collisions = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    assert collisions and any(r['cache_hit'] for r in records)
    out.mkdir()
    save(out/'evidence.json', {'time': time.time(), 'passed': False, 'records': records,
        'cross_image_result_groups': collisions,
        'cache_environment': {name: env.get(name) for name in
            ['TOOL_CACHE_ENABLED', 'PERCEPTION_CACHE_ENABLED', 'TOOL_CACHE_NAMESPACE', 'TOOL_CACHE_DIR']},
        'explanation': 'Distinct images received identical cached perception; not admissible PSD source data.',
        'history_scope': 'Other experiments not yet audited; no automatic historical reruns authorized.'})
    save(out/'controller-before.json', receipt)
    save(out/'state-before.json', load(RUN/'state.json'))
    os.kill(receipt['pid'], signal.SIGINT)
    ended = False
    for _ in range(80):
        proc = Path(f'/proc/{receipt["pid"]}/cmdline')
        if not proc.exists() or not proc.read_bytes():
            ended = True
            break
        time.sleep(.25)
    save(out/'stop.json', {'time': time.time(), 'pid': receipt['pid'], 'signal': 'SIGINT', 'exited': ended})
    if not ended:
        raise RuntimeError('Canary did not drain after SIGINT; inspect before escalation')
    save(RUN/'state.json', {'phase': 'held_cross_image_perception_cache', 'time': time.time(),
        'completed_traces_retained': len(list((RUN/'episodes/traces').glob('*.json'))),
        'source_bank_admissible': False, 'psd_training_started': False,
        'full_collection_started': False, 'evidence': str(out/'evidence.json')})
    state = load(SERVICE/'state.json')
    state.update(phase='held_cross_image_perception_cache', gpu_verified=False,
        source_collection_started=False, cache_contamination_evidence=str(out/'evidence.json'),
        next_phase='disable separate perception cache in isolated launcher; verify image binding before a new diagnostic')
    save(SERVICE/'state.json', state)
    print(json.dumps({'stopped': True, 'retained_perception_records': len(records),
        'cross_image_groups': len(collisions), 'psd_training_started': False}), flush=True)


if __name__ == '__main__':
    main()
