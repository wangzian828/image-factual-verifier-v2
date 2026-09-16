"""Lease the four owned replicas for native FSDP2 PSD checkpoint/resume proof.

Uses the complete 199-target real engineering bank, not the formal 400x8 run.
Restores every stopped replica on failure or success. No protected model writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
DEPLOY = ROOT / 'training-artifacts/psd-deterministic-training-20260916-v16'
CODE = DEPLOY / 'code'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-grounded-review-canary-v15'
SNAPSHOT = ROOT / 'training-artifacts/psd-contract-audit-20260916-v10/snapshot'
MODEL = ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/model'


def main():
    global DEPLOY, CODE
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    parser.add_argument('--output-name', default='dp4-resume-gate-v1')
    parser.add_argument('--deployment', default=DEPLOY.name)
    parser.add_argument('--resume-only-from', type=Path)
    args = parser.parse_args()
    assert Path(args.deployment).name == args.deployment and args.deployment.startswith('psd-')
    DEPLOY = ROOT / 'training-artifacts' / args.deployment
    CODE = DEPLOY / 'code'
    if args.resume_only_from:
        args.resume_only_from = args.resume_only_from.resolve()
        args.resume_only_from.relative_to(ROOT / 'checkpoints')
        assert (args.resume_only_from / 'psd-resume-binding.json').is_file()
    assert Path(args.output_name).name == args.output_name and args.output_name.startswith('dp4-resume-gate-')
    out = RUN / args.output_name
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.load(DEPLOY / 'state.json')['deployment_ready_not_live']
    if args.mode == 'launch':
        assert owner.load(RUN / 'state.json')['phase'] == 'ready_for_trainer'
        out.mkdir(exist_ok=False)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'execute', '--output-name', args.output_name,
                   '--deployment', args.deployment]
        if args.resume_only_from: command += ['--resume-only-from', str(args.resume_only_from)]
        receipt = owner.spawn(command, os.environ.copy(), out / 'run.log')
        owner.save(out / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(out)})); return
    owner.verify_export()
    bindings = owner.load(DEPLOY / 'code-binding.json')
    for name, expected in bindings.items():
        assert owner.sha(CODE / name) == expected, f'Immutable code changed: {name}'
    backends = [owner.load(SERVICE / f'replica-{i}.json') for i in range(4)]
    environments = [owner.checked(b) for b in backends]
    assert [e['CUDA_VISIBLE_DEVICES'] for e in environments] == ['0', '1', '2', '3']
    guard = owner.load(SERVICE / 'guard.json'); owner.checked(guard)
    owner.save(out / 'binding.json', {'backends': backends, 'code_binding_sha256': owner.sha(DEPLOY / 'code-binding.json'),
        'datums_sha256': owner.sha(RUN / 'bank/datums/datums.jsonl'), 'formal_training': False})
    stopped = []
    def stop_one(i):
        owner.stop(backends[i])
        stopped.append(i)
    try:
        os.kill(guard['pid'], signal.SIGSTOP)
        assert all(r['inflight'] == 0 for r in owner.http('http://127.0.0.1:19019/health')['replicas'])
        for attempt in range(90):
            busy = []
            for i in range(4):
                with urllib.request.urlopen(f'http://127.0.0.1:{19002+i}/metrics', timeout=5) as response:
                    metrics = response.read().decode()
                values = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
                          if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
                assert len(values) == 2
                busy.append(sum(values))
            if not any(busy): break
            time.sleep(1)
        else: raise RuntimeError('Replicas are still running requests; did not take GPUs')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stop_one, range(4)))
        owner.save(out / 'state.json', {'phase': 'native_fsdp2_baseline', 'formal_training': False})
        cuda = ROOT / 'envs/h20-qwen35-128k'
        env = {**os.environ, 'PATH': str(cuda / 'bin') + ':' + os.environ['PATH'], 'CUDA_HOME': str(cuda),
            'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'IFV_ALLOWED_GPU_IDS': '0,1,2,3',
            'PYTHONPATH': str(CODE) + ':' + str(CODE / 'training'), 'PYTHONDONTWRITEBYTECODE': '1',
            'IFV_TRAINING_DATA_ROOT': str(ROOT), 'IFV_MODEL_ID': str(MODEL),
            'IFV_PSD_PROFILE_MODE': 'resume_probe', 'IFV_PSD_SERVING_PROFILE': str(SNAPSHOT / 'serving-profile.json'),
            'IFV_PSD_ROUND_START_MANIFEST': str(SNAPSHOT / 'checkpoint-manifest.json'),
            'FLASH_ATTENTION_DETERMINISTIC': '1', 'OMP_NUM_THREADS': '4', 'MAX_JOBS': '8',
            'TMPDIR': str(ROOT / 'tmp'), 'HF_HOME': str(ROOT / 'cache/huggingface'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TORCH_HOME': str(ROOT / 'cache/torch'),
            'TRITON_CACHE_DIR': str(ROOT / 'cache/psd-gpu-probe/triton'),
            'TORCH_EXTENSIONS_DIR': str(ROOT / 'cache/psd-gpu-probe/extensions'),
            'FLASHINFER_WORKSPACE_BASE': str(ROOT / 'cache/psd-flashinfer'),
            'XDG_CACHE_HOME': str(ROOT / 'cache/psd-gpu-probe'), 'WANDB_DISABLED': 'true', 'MASTER_PORT': '29546'}
        base_id = f'psd-{args.output_name}-baseline'
        resume_id = f'psd-{args.output_name}-resumed'
        def train(experiment, checkpoint=None):
            command = ['bash', str(CODE / 'training/scripts/train/run_psd_topk.sh'),
                str(CODE / 'training/configs/models/qwen3.5-9b.env'),
                str(CODE / 'training/configs/psd/qwen3.5-lora-r32-h20-dp4-128k.env'),
                str(RUN / 'bank/datums/datums.jsonl'), experiment]
            if checkpoint is not None: command.append(str(checkpoint))
            with (out / f'{experiment}.log').open('x') as log:
                p = subprocess.Popen(command, env=env, cwd=CODE, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, start_new_session=True)
                owner.save(out / 'training-process.json', {'pid': p.pid, 'command': command})
                code = p.wait()
            if code: raise RuntimeError(f'Native training failed with {code}; see {experiment}.log')
            return ROOT / 'checkpoints' / experiment
        def checkpoint(root, step):
            found = list(root.glob(f'**/checkpoint-{step}'))
            assert len(found) == 1, (root, step, found)
            return found[0]
        baseline = args.resume_only_from or train(base_id)
        ckpt1, ckpt2 = checkpoint(baseline, 1), checkpoint(baseline, 2)
        owner.save(out / 'state.json', {'phase': 'native_fsdp2_resume', 'checkpoint': str(ckpt1), 'formal_training': False})
        resumed = train(resume_id, ckpt1)
        resumed2 = checkpoint(resumed, 2)
        for label, source in [('baseline', ckpt2), ('resumed', resumed2)]:
            command = [str(cuda / 'bin/python'), str(CODE / 'training/scripts/train/export_fsdp2_lora_checkpoint.py'),
                '--checkpoint', str(source), '--base-model', str(MODEL), '--output', str(out / f'{label}-adapter'),
                '--rank', '32', '--alpha', '32', '--dropout', '0']
            with (out / f'{label}-export.log').open('x') as log:
                subprocess.run(command, env=env, cwd=CODE, check=True, stdout=log, stderr=log)
        # CPU comparison does not alter either checkpoint.
        from safetensors.torch import load_file
        import torch
        a = load_file(str(out / 'baseline-adapter/adapter_model.safetensors'))
        b = load_file(str(out / 'resumed-adapter/adapter_model.safetensors'))
        assert set(a) == set(b)
        delta = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
        exact = all(torch.equal(a[k], b[k]) for k in a)
        report = {'passed': exact, 'adapter_bitwise_equal': exact, 'max_parameter_difference': delta,
            'baseline_checkpoint': str(ckpt2), 'resume_checkpoint': str(resumed2), 'formal_training': False}
        owner.save(out / 'result.json', report)
        if not exact: raise RuntimeError('Native FSDP resume is not bitwise equal; investigate before formal training')
        owner.save(out / 'state.json', {'phase': 'native_fsdp2_resume_passed', **report})
    except Exception as error:
        owner.save(out / 'state.json', {'phase': 'failed_requires_fix', 'error': str(error), 'formal_training': False})
        raise
    finally:
        restore_errors = []
        try:
            for i in sorted(stopped):
                try:
                    receipt = owner.spawn(backends[i]['command'], environments[i], out / f'restored-backend-{i}.log')
                    owner.save(SERVICE / f'replica-{i}.json', receipt)
                except Exception as error:
                    restore_errors.append({'gpu': i, 'error_type': type(error).__name__})
        finally:
            os.kill(guard['pid'], signal.SIGCONT)
        if restore_errors:
            owner.save(out / 'restore-errors.json', restore_errors)
            raise RuntimeError('An owned backend needs recovery; see restore-errors.json')
        owner.verify_export()


if __name__ == '__main__':
    main()
