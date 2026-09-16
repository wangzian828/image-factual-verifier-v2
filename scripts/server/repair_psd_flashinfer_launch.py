"""Recover only the failed diagnostic JIT worker with workspace-bound caches."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/volume/ybo/wza')
SERVICE=ROOT/'inference/psd-sft3084-20260916'
OLD=SERVICE/'nan-scheduler-gdn-ablation-v1'
OUT=SERVICE/'nan-flashinfer-toolchain-recovery-v1'


def main():
    os.umask(0o077)
    spec=importlib.util.spec_from_file_location('owner',ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    if sys.argv[1:]==['launch']:
        OUT.mkdir(exist_ok=False)
        receipt=owner.spawn([sys.executable,'-u',str(Path(__file__).resolve()),'execute'],os.environ.copy(),OUT/'run.log')
        owner.save(OUT/'process.json',receipt);print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}));return
    assert sys.argv[1:]==['execute']
    old=owner.load(SERVICE/'replica-2.json');env=owner.checked(old)
    assert owner.load(OLD/'flashinfer-build-failure.json')['backend']==old
    assert old==owner.load(OLD/'replica-2.json') and env['CUDA_VISIBLE_DEVICES']=='2'
    assert not (SERVICE/'nan-scheduler-gdn-corpus-v1').exists()
    guard=owner.load(SERVICE/'guard.json');owner.checked(guard)
    owner.save(OUT/'before.json',{'backend':old,'sigterm_wait_already_exceeded_seconds':60,
        'no_agent_or_training_on_this_replica':True,'outside_workspace_untouched':True})
    os.kill(guard['pid'],signal.SIGSTOP)
    try:
        assert all(r['inflight']==0 for r in owner.http('http://127.0.0.1:19019/health')['replicas'])
        # The preceding 60-second graceful stop failed during JIT startup.
        # Exact command + owned process group + diagnostic-only binding above.
        owner.checked(old);os.killpg(old['pid'],signal.SIGKILL)
        for _ in range(60):
            live=[]
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    fields=(p/'stat').read_text().split(') ',1)[1].split()
                    if int(fields[2])==old['pid'] and fields[0]!='Z':live.append(p.name)
                except (OSError,ProcessLookupError):pass
            if not live:break
            time.sleep(.5)
        else:raise RuntimeError('Failed diagnostic process group did not exit')
        cuda=ROOT/'envs/h20-qwen35-128k'
        assert (cuda/'bin/nvcc').is_file() and (cuda/'targets/x86_64-linux/include/cuda/ptx').is_file()
        env.update(FLASHINFER_WORKSPACE_BASE=str(ROOT/'cache/psd-flashinfer'),
            CUDA_HOME=str(cuda),CUDACXX=str(cuda/'bin/nvcc'),
            CPLUS_INCLUDE_PATH=str(cuda/'targets/x86_64-linux/include'),
            LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib')+':'+str(cuda/'targets/x86_64-linux/lib/stubs'),
            LD_LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib')+':'+env.get('LD_LIBRARY_PATH',''),
            MAX_JOBS='8',
            PATH=str(cuda/'bin')+':'+env['PATH'])
        receipt=owner.spawn(old['command'],env,OUT/'backend.log')
        owner.save(SERVICE/'replica-2.json',receipt);owner.save(OUT/'replica-2.json',receipt)
    finally:os.kill(guard['pid'],signal.SIGCONT)
    owner.save(OUT/'state.json',{'phase':'waiting_scoped_jit_readiness','training_started':False})
    for _ in range(360):
        owner.checked(receipt)
        try:
            model=owner.http('http://127.0.0.1:19004/v1/models')['data'][0]
            if model['root']==receipt['command'][3]:
                log=(OUT/'backend.log').read_text()
                assert 'Using FlashInfer GDN prefill kernel' in log
                assert 'compilation terminated' not in log and 'Falling back to Triton' not in log
                owner.save(OUT/'state.json',{'phase':'ready_scoped_flashinfer_candidate','training_started':False})
                owner.save(OLD/'state.json',{'phase':'ready_for_ablation_not_production',
                    'training_started':False,'flashinfer_recovery':str(OUT)});return
        except OSError:pass
        time.sleep(2)
    raise RuntimeError('Scoped FlashInfer readiness failed')


if __name__=='__main__':main()
