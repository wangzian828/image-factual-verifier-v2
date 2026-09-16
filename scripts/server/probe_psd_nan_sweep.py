"""Eight fixed, diagnostic-only requests across two existing owned replicas."""
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
OUT = ROOT / 'inference/psd-sft3084-20260916/nan-stream-sweep-v1'


def main():
    global OUT
    os.umask(0o077)
    ab = '--eager-ab' in sys.argv
    argv = [value for value in sys.argv[1:] if value != '--eager-ab']
    if ab:
        OUT = ROOT / 'inference/psd-sft3084-20260916/nan-stream-eager-ab-v1'
    spec = importlib.util.spec_from_file_location('nan_stream_probe', DEPLOY / 'probe_psd_nan_stream.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if argv == ['launch']:
        if ab:
            for gpu in (2, 3):
                receipt = json.loads((m.SERVICE / f'replica-{gpu}.json').read_text())
                assert ('--enforce-eager' in receipt['command']) == (gpu == 3)
        OUT.mkdir(exist_ok=False)
        m.save(OUT / 'binding.json', {'diagnostic_only': True, 'requests': 32 if ab else 8, 'concurrency': 4,
            'gpu_indices': [2, 3], 'archive': str(m.ARCHIVE), 'retries': 0})
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute']
        if ab:
            command.append('--eager-ab')
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'))
        with (OUT / 'run.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        m.save(OUT / 'process.json', {'pid': child.pid, 'command': command, 'time': time.time()})
        print(json.dumps({'pid': child.pid, 'output': str(OUT)}))
        return
    assert argv == ['execute']
    sys.path[:0] = [str(m.CODE), str(m.CODE / 'training')]
    async def run():
        results = []
        for wave in range(8 if ab else 2):
            paths = [(OUT / f'wave{wave}-gpu{gpu}-copy{copy}', gpu) for gpu in [2, 3] for copy in range(2)]
            for path, gpu in paths:
                path.mkdir(exist_ok=False)
            await asyncio.gather(*(m.execute(path, gpu) for path, gpu in paths))
            results += [json.loads((path / 'state.json').read_text()) for path, _ in paths]
            m.save(OUT / 'summary.json', {'diagnostic_only': True, 'completed': len(results), 'results': results})
    asyncio.run(run())


if __name__ == '__main__':
    main()
