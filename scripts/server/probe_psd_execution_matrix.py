"""Bounded four-mode diagnostic. Full requests, no actions and no training targets."""
import argparse
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


def validate_mode(gpu, command):
    eager = '--enforce-eager' in command
    config = json.loads(command[command.index('--compilation-config')+1]) if '--compilation-config' in command else None
    expected = {0: None, 1: {'mode': 3, 'cudagraph_mode': 'NONE'},
                2: {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY'}, 3: None}
    assert eager == (gpu == 3) and config == expected[gpu]
    assert command[command.index('--port')+1] == str(19002+gpu)
    assert '--no-enable-prefix-caching' in command
    assert command[command.index('--mamba-cache-mode')+1] == 'none'


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch','execute'])
    parser.add_argument('--part', type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    out = SERVICE / f'nan-execution-matrix-part{args.part}-v1'
    spec = importlib.util.spec_from_file_location('nan_stream_probe', DEPLOY / 'probe_psd_nan_stream_v2.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if args.mode == 'launch':
        replicas = {}
        for gpu in range(4):
            receipt = json.loads((SERVICE / f'replica-{gpu}.json').read_text())
            actual = [s.decode() for s in Path(f'/proc/{receipt["pid"]}/cmdline').read_bytes().split(b'\0') if s]
            assert actual == receipt['command']
            validate_mode(gpu, actual)
            replicas[str(gpu)] = receipt
        out.mkdir(exist_ok=False)
        m.save(out/'binding.json', {'diagnostic_only':True,'requests_per_gpu':50,'concurrency_per_gpu':2,
            'archive':str(m.ARCHIVE),'replicas':replicas,'retries':0,'part':args.part,
            'not_a_strict_factorial': 'GPU0 uses FULL_AND_PIECEWISE; GPU2 uses FULL_DECODE_ONLY because mode0 disables piecewise',
            'fixed_request_except_unique_cache_salt':True})
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'execute','--part',str(args.part)]
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
        with (out/'run.log').open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=env,start_new_session=True)
        m.save(out/'process.json',{'pid':child.pid,'command':command,'time':time.time()})
        print(json.dumps({'pid':child.pid,'output':str(out)}))
        return
    sys.path[:0] = [str(m.CODE),str(m.CODE/'training')]
    async def run():
        results = []
        for wave in range(25):
            binding = json.loads((out/'binding.json').read_text())
            for gpu in range(4):
                assert json.loads((SERVICE/f'replica-{gpu}.json').read_text()) == binding['replicas'][str(gpu)]
            paths = [(out/f'wave{wave}-gpu{gpu}-copy{copy}',gpu) for gpu in range(4) for copy in range(2)]
            for p,_ in paths:
                p.mkdir(exist_ok=False)
            await asyncio.gather(*(m.execute(p,g) for p,g in paths))
            results += [json.loads((p/'state.json').read_text()) for p,_ in paths]
            m.save(out/'summary.json',{'diagnostic_only':True,'completed':len(results),'results':results})
    asyncio.run(run())


if __name__ == '__main__':
    main()
