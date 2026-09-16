"""Finite mixed-length request history stress; never execute diagnostic actions."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT/'training-artifacts/psd-observed-positions-20260916-v14'
SERVICE = ROOT/'inference/psd-sft3084-20260916'
OUT = SERVICE/'nan-mixed-history-v1'
CASE = ROOT/'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14/search/repairs/7996ba9752652f37'
LONG = CASE/'runtime/route-aware-hrc-stage2-1-622974dc62/slate-01-f7bb2495bcca'


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('nan_probe',DEPLOY/'probe_psd_nan_stream_v3.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    plan = [(m.ARCHIVE,'req-000006'),(LONG,'req-000001'),(LONG,'req-000025'),
            (m.ARCHIVE,'req-000001'),(LONG,'req-000015')]
    if sys.argv[1:]==['launch']:
        assert json.loads((SERVICE/'nan-execution-matrix-part2-v1/summary.json').read_text())['completed']==200
        receipts = {}
        for gpu in range(4):
            receipts[str(gpu)] = json.loads((SERVICE/f'replica-{gpu}.json').read_text())
            actual = [s.decode() for s in Path(f'/proc/{receipts[str(gpu)]["pid"]}/cmdline').read_bytes().split(b'\0') if s]
            assert actual==receipts[str(gpu)]['command']
        OUT.mkdir(exist_ok=False)
        m.save(OUT/'binding.json',{'diagnostic_only':True,'replicas':receipts,'retries':0,
            'waves':2,'concurrency':40,'requests_per_gpu_per_wave':10,
            'request_plan':[[str(p),r] for p,r in plan],
            'baseline_after_each_wave_per_gpu':2,'source_actions_not_executed':True})
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'execute']
        env = os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
        with (OUT/'run.log').open('x') as log:
            child = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        m.save(OUT/'process.json',{'pid':child.pid,'command':cmd,'time':time.time()})
        print(json.dumps({'pid':child.pid,'output':str(OUT)}))
        return
    assert sys.argv[1:]==['execute']
    sys.path[:0] = [str(m.CODE),str(m.CODE/'training')]
    async def run():
        results=[]
        for wave in range(2):
            jobs=[]
            for gpu in range(4):
                for n in range(10):
                    archive,request_id = plan[n%len(plan)]
                    out=OUT/f'mixed{wave}-gpu{gpu}-req{n}';out.mkdir(exist_ok=False)
                    jobs.append((out,gpu,archive,request_id))
            await asyncio.gather(*(m.execute(p,g,archive=a,request_id=r) for p,g,a,r in jobs))
            results += [json.loads((p/'state.json').read_text()) for p,_,_,_ in jobs]
            controls=[(OUT/f'after{wave}-gpu{g}-copy{c}',g) for g in range(4) for c in range(2)]
            for p,_ in controls:p.mkdir(exist_ok=False)
            await asyncio.gather(*(m.execute(p,g) for p,g in controls))
            results += [json.loads((p/'state.json').read_text()) for p,_ in controls]
            m.save(OUT/'summary.json',{'diagnostic_only':True,'completed':len(results),'results':results})
    asyncio.run(run())


if __name__=='__main__':
    main()
