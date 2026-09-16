"""Same exact-wire corpus on sync/Triton, async/Triton and async/FlashInfer."""
import asyncio
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT/'inference/psd-sft3084-20260916'
DEPLOY = ROOT/'training-artifacts/psd-observed-positions-20260916-v14'
OUT = SERVICE/'nan-scheduler-gdn-corpus-v1'
WIRE = SERVICE/'validated-execution-candidate-v1/wire'
FIXED = WIRE/'1789551642580005259-b2b919425c11426eb34c440879f2f248'


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('probe', DEPLOY/'probe_psd_nan_stream_v4.py')
    probe = importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    spec = importlib.util.spec_from_file_location('owner', ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if sys.argv[1:]==['launch']:
        assert owner.load(SERVICE/'nan-scheduler-gdn-ablation-v1/state.json')['phase']=='ready_for_ablation_not_production'
        backends={gpu:owner.load(SERVICE/f'replica-{gpu}.json') for gpu in (0,1,2)}
        for receipt in backends.values():owner.checked(receipt)
        choices=[]
        for ticket in sorted(WIRE.iterdir()):
            try:
                body=probe.captured_wire_body(ticket)
                if body['max_tokens']==32768:choices.append(ticket)
            except (ValueError,KeyError,OSError):continue
        assert len(choices)>=24
        tickets=[FIXED if n%4==0 else choices[n*3%len(choices)] for n in range(24)]
        OUT.mkdir(exist_ok=False)
        owner.save(OUT/'binding.json',{'backends':backends,'tickets':[str(p) for p in tickets],
            'concurrency_per_replica':8,'requests_per_replica':24,'retries':0,
            'source_or_repair_sampling':False,'weights_or_generation_budget_changed':False})
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    assert sys.argv[1:]==['execute']
    async def run():
        tickets=[Path(p) for p in owner.load(OUT/'binding.json')['tickets']]
        results=[]
        for wave in range(3):
            jobs=[]
            for gpu in (0,1,2):
                for index,ticket in enumerate(tickets[wave*8:wave*8+8]):
                    d=OUT/f'gpu{gpu}-request{wave*8+index}';d.mkdir(exist_ok=False)
                    jobs.append((d,gpu,ticket))
            await asyncio.gather(*(probe.execute(d,gpu,wire_ticket=ticket) for d,gpu,ticket in jobs))
            results.extend(owner.load(d/'state.json') for d,_,_ in jobs)
            counts=Counter((r['gpu'],r['status'],r.get('finish_reason')) for r in results)
            owner.save(OUT/'summary.json',{'completed':len(results),
                'counts':[{'gpu':k[0],'status':k[1],'finish_reason':k[2],'count':v} for k,v in counts.items()],
                'results':[{k:v for k,v in r.items() if k!='first_logprob'} for r in results]})
    asyncio.run(run())


if __name__=='__main__':main()
