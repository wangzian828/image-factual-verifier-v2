"""Create an isolated, hash-bound raw-probability gateway; no backend restart."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
CODE = ROOT / 'training-artifacts/psd-raw-teacher-resume-20260916-v20/code'
WORKER_CODE = ROOT / 'training-artifacts/psd-raw-teacher-20260916-v18/code'


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-ids', default='0,1,2,3')
    p.add_argument('--port', type=int, default=19022)
    p.add_argument('--output-name', required=True)
    args = p.parse_args()
    ids = [int(x) for x in args.gpu_ids.split(',')]
    assert ids and len(set(ids)) == len(ids) and set(ids) <= {0, 1, 2, 3}
    assert 19020 <= args.port <= 19029
    assert Path(args.output_name).name == args.output_name and args.output_name.startswith('raw-')
    out = SERVICE / args.output_name
    assert not out.exists()
    spec = importlib.util.spec_from_file_location('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    owner.verify_export()
    for deployment in (CODE.parent, WORKER_CODE.parent):
        assert owner.load(deployment / 'state.json')['deployment_ready_not_live']
        for rel, digest in owner.load(deployment / 'code-binding.json').items():
            assert owner.sha(deployment / 'code' / rel) == digest
    receipts = [SERVICE / f'replica-{i}.json' for i in ids]
    urls = [f'http://127.0.0.1:{19002+i}' for i in ids]
    files = {str(WER): owner.sha(WER) for WER in [
        WORKER_CODE / 'scripts/server/psd_raw_logprobs.py',
        WORKER_CODE / 'scripts/server/psd_raw_teacher_worker.py']}
    sys.path.insert(0, str(CODE))
    from scripts.server.psd_raw_logprobs import validate_raw_worker_receipts, SEMANTICS
    validate_raw_worker_receipts([str(x) for x in receipts], urls, files)
    for receipt in receipts:
        process = owner.load(receipt)
        env = owner.checked(process)
        assert env['PYTHONPATH'].split(':')[0] == str(WORKER_CODE)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    previous = owner.load(SERVICE / 'four-gpu-v7/gateway.json')
    env = owner.checked(previous)
    command = previous['command'][:]
    command[command.index('--app-dir') + 1] = str(CODE)
    command[command.index('--port') + 1] = str(args.port)
    for key in ('PSD_WIRE_CAPTURE_DIR', 'PSD_DIAGNOSTIC_RETURN_TOKEN_IDS'):
        env.pop(key, None)
    env.update(PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
        QWEN_REPLICA_BACKENDS=','.join(urls), PSD_POLICY_LOGPROB_SEMANTICS=SEMANTICS,
        PSD_RAW_WORKER_RECEIPTS=json.dumps([str(x) for x in receipts]),
        PSD_RAW_WORKER_FILES=json.dumps(files))
    out.mkdir()
    owner.save(out / 'binding.json', {'files': files, 'receipts': [str(x) for x in receipts],
        'gateway_code_sha256': owner.sha(CODE.parent / 'code-binding.json'),
        'raw_semantics': SEMANTICS, 'port': args.port, 'duplicate_wire_archive': False})
    receipt = owner.spawn(command, env, out / 'gateway.log')
    owner.save(out / 'gateway.json', receipt)
    for _ in range(30):
        owner.checked(receipt)
        try:
            health = owner.http(f'http://127.0.0.1:{args.port}/health')
            assert len(health['replicas']) == len(ids)
            if all(r['healthy'] for r in health['replicas']):
                owner.save(out / 'state.json', {'phase': 'ready_for_runtime_validation', 'port': args.port,
                    'gpu_ids': ids, 'formal_collection_started': False})
                print(json.dumps({'pid': receipt['pid'], 'port': args.port, 'gpu_ids': ids})); return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError('Raw gateway did not become ready; retained private logs')


if __name__ == '__main__':
    main()
