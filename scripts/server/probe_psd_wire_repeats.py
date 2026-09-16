"""Finite real-wire history test on the one instrumented eager replica."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-observed-positions-20260916-v14'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
OUT = SERVICE / 'nan-exact-wire-instrumented-repeats-v1'
WIRE = SERVICE / 'validated-execution-candidate-v1/wire'
FIXED = WIRE / '1789551642580005259-b2b919425c11426eb34c440879f2f248'


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('probe', DEPLOY / 'probe_psd_nan_stream_v4.py')
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    if sys.argv[1:] == ['launch']:
        assert json.loads((SERVICE / 'eager-numerical-worker-gpu3-v1/state.json').read_text())['phase'] == 'ready_for_numerical_diagnosis'
        replica = json.loads((SERVICE / 'replica-3.json').read_text())
        actual = [s.decode() for s in Path(f'/proc/{replica["pid"]}/cmdline').read_bytes().split(b'\0') if s]
        assert actual == replica['command'] and 'psd_nan_worker.DiagnosticWorker' in actual
        # Only policy requests with top20, full input images and tools retained.
        choices = []
        for p in sorted(WIRE.iterdir()):
            if (p / 'response-meta.json').exists():
                try:
                    probe.captured_wire_body(p)
                    choices.append(p)
                except ValueError:
                    continue
        assert len(choices) >= 16
        selected = [choices[0], choices[len(choices)//2], choices[-1], FIXED]
        OUT.mkdir(exist_ok=False)
        probe.save(OUT / 'binding.json', {'backend': replica, 'wire_tickets': [str(p) for p in selected],
            'concurrency_waves': [2, 10, 10, 2, 10, 10], 'not_training_target': True, 'retries': 0})
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute']
        with (OUT / 'run.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, start_new_session=True)
        probe.save(OUT / 'process.json', {'pid': child.pid, 'command': cmd, 'time': time.time()})
        print(json.dumps({'pid': child.pid, 'output': str(OUT)})); return
    assert sys.argv[1:] == ['execute']
    async def run():
        binding = json.loads((OUT / 'binding.json').read_text())
        results = []
        for wave, count in enumerate(binding['concurrency_waves']):
            jobs = []
            for index in range(count):
                ticket = FIXED if count == 2 else Path(binding['wire_tickets'][index % 4])
                directory = OUT / f'wave{wave}-req{index}'
                directory.mkdir(exist_ok=False); jobs.append((directory, ticket))
            await asyncio.gather(*(probe.execute(d, 3, wire_ticket=t) for d, t in jobs))
            results += [json.loads((d / 'state.json').read_text()) for d, _ in jobs]
            probe.save(OUT / 'summary.json', {'completed': len(results), 'results': results, 'not_training_target': True})
    asyncio.run(run())


if __name__ == '__main__':
    main()
