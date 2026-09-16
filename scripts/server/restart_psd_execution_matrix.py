"""One finite rolling restart for diagnostic part two; not a scheduler."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft3084-20260916'
DEPLOY = ROOT/'training-artifacts/psd-observed-positions-20260916-v14'
OUT = SERVICE/'nan-execution-matrix-restarts-v1'
OWNER = ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py'


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['launch','execute'])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner',OWNER)
    o = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o)
    binding = o.load(SERVICE/'nan-execution-matrix-part1-v1/binding.json')
    summary = o.load(SERVICE/'nan-execution-matrix-part1-v1/summary.json')
    assert summary['completed']==200 and len(summary['results'])==200
    probe = o.load(SERVICE/'nan-execution-matrix-part1-v1/process.json')
    stat = Path(f'/proc/{probe["pid"]}/stat')
    assert not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z'
    o.verify_export()
    if args.mode=='launch':
        for gpu in range(4):
            assert o.load(SERVICE/f'replica-{gpu}.json')==binding['replicas'][str(gpu)]
        OUT.mkdir(exist_ok=False)
        o.save(OUT/'binding.json',{'before':binding['replicas'],'same_commands':True,
            'compilation_cache_unchanged':True,'finite_rolling_restart':True})
        receipt = o.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        o.save(OUT/'process.json',receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}))
        return
    guard = o.load(SERVICE/'guard.json')
    o.checked(guard)
    for gpu in range(4):
        old = o.load(SERVICE/f'replica-{gpu}.json')
        assert old==binding['replicas'][str(gpu)]
        env = o.checked(old)
        assert env['CUDA_VISIBLE_DEVICES']==str(gpu)
        assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19019/health')['replicas'])
        port = 19002+gpu
        new = None
        os.kill(guard['pid'],signal.SIGSTOP)
        try:
            for _ in range(90):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=5) as response:
                    numbers = [float(l.rsplit(' ',1)[1]) for l in response.read().decode().splitlines()
                        if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
                assert len(numbers)==2
                if sum(numbers)==0:break
                time.sleep(1)
            else:raise RuntimeError('Diagnostic backend did not drain')
            o.stop(old)
            new = o.spawn(old['command'],env,OUT/f'backend-{gpu}.log')
            o.save(SERVICE/f'replica-{gpu}.json',new)
            o.save(OUT/f'replica-{gpu}.json',new)
            o.save(OUT/'state.json',{'phase':'waiting_readiness','gpu':gpu,'training_started':False})
        finally:
            os.kill(guard['pid'],signal.SIGCONT)
        for _ in range(240):
            o.checked(new)
            try:
                model = o.http(f'http://127.0.0.1:{port}/v1/models')['data'][0]
                if model['id']=='ifv-psd-sft3084' and model['root']==old['command'][3]:
                    assert model['max_model_len']==131072
                    break
            except OSError:
                pass
            time.sleep(2)
        else:raise RuntimeError('Restart diagnostic readiness deadline; no retry loop')
    o.save(OUT/'state.json',{'phase':'four_restarts_ready','training_started':False,'time':time.time()})


if __name__=='__main__':
    main()
