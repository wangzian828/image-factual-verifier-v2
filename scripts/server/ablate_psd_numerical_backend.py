"""Bounded two-replica numerical isolation; no changes to weights or Agent."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
OUT = SERVICE / 'nan-scheduler-gdn-ablation-v1'
PROBE = SERVICE / 'nan-nonstream-four-replica-v1'


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if sys.argv[1:] == ['launch']:
        OUT.mkdir(exist_ok=False)
        owner.verify_export()
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute'],
                              os.environ.copy(), OUT / 'run.log')
        owner.save(OUT/'process.json', receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    assert sys.argv[1:] == ['execute']
    # Stop only this bounded diagnostic controller. Its two slow requests are
    # cancelled, not relabelled model failures; completed results are retained.
    probe = owner.load(PROBE/'process.json')
    states = {str(p):owner.load(p) for p in PROBE.glob('gpu*/state.json')}
    owner.save(OUT/'stopped-diagnostic.json', {'process':probe,'states':states,
        'reason':'first-token nonfinite reproduced; isolate execution backend',
        'source_or_repair_generation_cancelled':False})
    if Path(f'/proc/{probe["pid"]}').exists():
        assert Path(probe['command'][2]).name == 'probe_psd_nonstream.py'
        owner.stop(probe)
    guard = owner.load(SERVICE/'guard.json');owner.checked(guard)
    originals = {gpu:owner.load(SERVICE/f'replica-{gpu}.json') for gpu in (0,2)}
    environments = {gpu:owner.checked(receipt) for gpu,receipt in originals.items()}
    owner.save(OUT/'before.json', {'replicas':originals,'changed_sampling':False,'changed_model':False})
    owner.save(OUT/'state.json', {'phase':'draining_owned_diagnostic_replicas','training_started':False})
    os.kill(guard['pid'],signal.SIGSTOP)
    try:
        assert all(v['inflight']==0 for v in owner.http('http://127.0.0.1:19019/health')['replicas'])
        for gpu in originals:
            for _ in range(90):
                metrics=urllib.request.urlopen(f'http://127.0.0.1:{19002+gpu}/metrics',timeout=5).read().decode()
                values=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines()
                    if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
                assert len(values)==2
                if sum(values)==0:break
                time.sleep(1)
            else:raise RuntimeError('Replica failed to drain after diagnostic cancellation')
            old=originals[gpu];command=list(old['command'])
            assert '--enforce-eager' in command and '--worker-cls' not in command
            assert environments[gpu]['CUDA_VISIBLE_DEVICES']==str(gpu)
            if gpu==0:
                assert '--no-async-scheduling' not in command
                command += ['--no-async-scheduling']
            else:
                offset=command.index('--gdn-prefill-backend')+1
                assert command[offset]=='triton';command[offset]='flashinfer'
                cuda=ROOT/'envs/h20-qwen35-128k'
                assert (cuda/'targets/x86_64-linux/include/cuda/ptx').is_file()
                environments[gpu].update(FLASHINFER_WORKSPACE_BASE=str(ROOT/'cache/psd-flashinfer'),
                    CUDA_HOME=str(cuda),CUDACXX=str(cuda/'bin/nvcc'),MAX_JOBS='8',
                    CPLUS_INCLUDE_PATH=str(cuda/'targets/x86_64-linux/include'),
                    LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib')+':'+str(cuda/'targets/x86_64-linux/lib/stubs'),
                    LD_LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib')+':'+environments[gpu].get('LD_LIBRARY_PATH',''),
                    PATH=str(cuda/'bin')+':'+environments[gpu]['PATH'])
            owner.stop(old)
            receipt=owner.spawn(command,environments[gpu],OUT/f'backend-{gpu}.log')
            owner.save(SERVICE/f'replica-{gpu}.json',receipt)
            owner.save(OUT/f'replica-{gpu}.json',receipt)
    finally:
        os.kill(guard['pid'],signal.SIGCONT)
    owner.save(OUT/'state.json',{'phase':'waiting_readiness','training_started':False})
    pending={0,2}
    for _ in range(360):
        for gpu in list(pending):
            receipt=owner.load(OUT/f'replica-{gpu}.json');owner.checked(receipt)
            try:
                card=owner.http(f'http://127.0.0.1:{19002+gpu}/v1/models')['data'][0]
                if card['root']==receipt['command'][3] and card['id']=='ifv-psd-sft3084':pending.remove(gpu)
            except OSError:pass
        if not pending:
            owner.save(OUT/'state.json',{'phase':'ready_for_ablation_not_production','training_started':False});return
        time.sleep(2)
    raise RuntimeError('Ablation readiness deadline; do not retry blindly')


if __name__=='__main__':main()
