"""One-card eager execution diagnostic, preserving weights and original receipt."""
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
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
OUT = SERVICE / 'eager-diagnostic-gpu3-v1'
OWNER = ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'


def eager_command(command, *, gpu=3, decode_only=False):
    result = list(command)
    assert result[2] == 'serve'
    assert result[3] == str(ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model')
    assert gpu in (2,3)
    for flag, value in {'--port': str(19002+gpu), '--served-model-name': 'ifv-psd-sft3084',
                        '--max-model-len': '131072', '--mamba-cache-mode': 'none',
                        '--mm-processor-cache-gb': '0'}.items():
        assert result[result.index(flag)+1] == value
    assert '--enforce-eager' not in result
    if decode_only:
        assert '--compilation-config' not in result
        return [*result, '--compilation-config', json.dumps({'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'})]
    return [*result, '--enforce-eager']


def execute(o, gpu=3, decode_only=False):
    for path in [RUN/'process.json', RUN/'reviewed-case-continuation-v1/process.json']:
        receipt = o.load(path)
        stat = Path(f'/proc/{receipt["pid"]}/stat')
        assert not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z', 'Repair still active'
    o.verify_export()
    old = o.load(SERVICE / f'replica-{gpu}.json')
    env = o.checked(old)
    assert env['CUDA_VISIBLE_DEVICES'] == str(gpu)
    command = eager_command(old['command'], gpu=gpu, decode_only=decode_only)
    receipt_path = SERVICE / f'replica-{gpu}.json'
    port = 19002+gpu
    guard = o.load(SERVICE / 'guard.json')
    o.checked(guard)
    o.save(OUT/'before.json', {'backend': old, 'guard': guard, 'changes': command[len(old['command']):],
                            'purpose': 'diagnostic_only_not_a_claim_of_fix'})
    launched = None
    stopped = False
    os.kill(guard['pid'], signal.SIGSTOP)
    try:
        for _ in range(90):
            assert all(r['inflight']==0 for r in o.http('http://127.0.0.1:19019/health')['replicas'])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
                values = [float(l.rsplit(' ',1)[1]) for l in response.read().decode().splitlines()
                    if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            assert len(values)==2
            if sum(values)==0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Owned diagnostic GPU did not drain')
        o.stop(old)
        stopped = True
        launched = o.spawn(command, env, OUT/'backend.log')
        o.save(OUT/f'replica-{gpu}.json', launched)
        o.save(receipt_path, launched)
        o.save(OUT/'state.json', {'phase': 'waiting_readiness', 'training_started': False})
    finally:
        os.kill(guard['pid'], signal.SIGCONT)
        if stopped and launched is None:
            o.save(receipt_path, o.spawn(old['command'], env, OUT/'rollback.log'))
    try:
        for _ in range(240):
            o.checked(launched)
            try:
                model = o.http(f'http://127.0.0.1:{port}/v1/models')['data'][0]
                if model['id']=='ifv-psd-sft3084' and model['root']==command[3]:
                    assert model['max_model_len']==131072
                    o.save(OUT/'state.json', {'phase': 'ready_for_ab_diagnostic', 'time': time.time(),
                        'training_started': False, 'other_three_replicas_unchanged': True})
                    return
            except OSError:
                pass
            time.sleep(2)
        raise RuntimeError('Eager replica readiness deadline')
    except BaseException:
        try:
            o.stop(launched)
        except FileNotFoundError:
            pass
        o.save(receipt_path, o.spawn(old['command'], env, OUT/'rollback.log'))
        o.save(OUT/'state.json', {'phase':'rolled_back_requires_diagnosis'})
        raise


def main():
    global OUT
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location('owned_psd_services', OWNER)
    o = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch','execute'])
    parser.add_argument('--gpu',type=int,choices=[2,3],default=3)
    parser.add_argument('--decode-only',action='store_true')
    args = parser.parse_args()
    if args.decode_only:
        OUT = SERVICE / f'no-compile-decode-graph-gpu{args.gpu}-v1'
    else:
        OUT = SERVICE / f'eager-diagnostic-gpu{args.gpu}-v1'
    if args.mode=='execute':
        execute(o,args.gpu,args.decode_only)
    else:
        eager_command(o.load(SERVICE/f'replica-{args.gpu}.json')['command'],gpu=args.gpu,decode_only=args.decode_only)
        OUT.mkdir(exist_ok=False)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'execute','--gpu',str(args.gpu)]
        if args.decode_only:
            command.append('--decode-only')
        receipt = o.spawn(command,
                          os.environ.copy(), OUT/'run.log')
        o.save(OUT/'process.json',receipt)
        print(json.dumps({'pid':receipt['pid'],'output':str(OUT)}))


if __name__=='__main__':
    main()
