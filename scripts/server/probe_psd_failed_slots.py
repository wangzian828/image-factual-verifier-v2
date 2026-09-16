"""Replay archived exhausted source requests on each owned replica, for diagnosis only.

No actions are executed, source ledgers are never edited, and outputs cannot be
used as PSD targets. Uses the existing bounded streaming diagnostic.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
SERVICE = ROOT / 'inference/psd-sft3084-20260916'
SOURCE = ROOT / 'runs/psd-production400x8-20260917-v2'
CODE = ROOT / 'training-artifacts/psd-numerical-recovery-20260917-v25/code'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exhausted_requests(source):
    selected = []
    for ledger in sorted((source / 'episodes/psd-infrastructure-attempts').glob('*/retry-state.json')):
        state = json.loads(ledger.read_text())
        attempts = state['payload']['attempts']
        if len(attempts) != state['identity']['max_attempts'] or any(
                a['status'] != 'infrastructure_failed' for a in attempts):
            continue
        last = attempts[-1]
        if last.get('reason') != 'model_http_400_nonfinite_serialization':
            raise ValueError('Unexpected exhausted failure: diagnose separately')
        directory = Path(last['directory']).resolve()
        directory.relative_to(source.resolve())
        matches = []
        for context in directory.glob('traces/runtime/*/*/context/req-*.json'):
            record = json.loads(context.read_text())
            error = str(record.get('error'))
            if record.get('status') == 'error' and (
                    error == 'PolicyInfrastructureFailure: model_http_400_nonfinite_serialization'
                    or 'Out of range float values are not JSON compliant: nan' in error):
                matches.append((context.parent.parent, record['request_id']))
        if len(matches) != 1:
            raise ValueError('Need one native numerical error receipt per exhausted slot')
        archive, request_id = matches[0]
        selected.append({'episode_id': state['identity']['inputs']['episode_id'],
                         'archive': str(archive), 'request_id': request_id, 'ledger': str(ledger)})
    if not selected:
        raise ValueError('No exhausted numerical source requests')
    return selected


async def replay(out, selected, owner, repetitions):
    sys.path[:0] = [str(CODE), str(CODE / 'training')]
    probe = load_module('bounded_stream_probe', Path(__file__).with_name('probe_psd_nan_stream.py'))
    summaries = []
    # At most two diagnostics per GPU. Keep complete generation budgets; abort
    # immediately on invalid probabilities rather than waiting for token0 loops.
    limits = [asyncio.Semaphore(2) for _ in range(4)]
    async def one(index, source, gpu, repetition):
        async with limits[gpu]:
            path = out / f'repeat{repetition}-source{index}-gpu{gpu}'
            path.mkdir(exist_ok=False)
            await probe.execute(path, gpu, archive=Path(source['archive']), request_id=source['request_id'])
            result = owner.load(path / 'state.json')
            compact = {k: result.get(k) for k in ('gpu', 'status', 'tokens', 'zero_tokens', 'seconds', 'finish_reason', 'think_closures')}
            compact.update(source_index=index, repetition=repetition)
            summaries.append(compact)
            owner.save(out / 'summary.json', {'diagnostic_only': True, 'completed': len(summaries), 'results': summaries})
    for repetition in range(repetitions):
        await asyncio.gather(*(one(i, source, gpu, repetition)
                              for i, source in enumerate(selected) for gpu in range(4)))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'execute'))
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--repetitions', type=int, default=1)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or not args.output_name.startswith('nan-exhausted-'):
        raise ValueError('Require a new direct child diagnostic directory')
    if not 1 <= args.repetitions <= 4:
        raise ValueError('Bounded diagnostic: repetitions must be 1..4')
    owner = load_module('owner', ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    out = SERVICE / args.output_name
    owner.verify_export()
    selected = exhausted_requests(SOURCE)
    receipts = [owner.load(SERVICE / f'replica-{gpu}.json') for gpu in range(4)]
    environments = [owner.checked(receipt) for receipt in receipts]
    owner.checked(owner.load(SERVICE / 'guard.json'))
    if args.mode == 'launch':
        if any(r['inflight'] for r in owner.http('http://127.0.0.1:19025/health')['replicas']):
            raise RuntimeError('Do not diagnose through active production traffic')
        out.mkdir(exist_ok=False)
        owner.save(out / 'binding.json', {'diagnostic_only': True, 'sources': selected,
            'backend_receipts': receipts, 'repetitions': args.repetitions,
            'native_context_not_exact_wire': True, 'never_execute_actions': True,
            'source_ledgers_sha256': {s['ledger']: owner.sha(Path(s['ledger'])) for s in selected},
            'nan_metric_switch': [e.get('VLLM_COMPUTE_NANS_IN_LOGITS', '0') for e in environments],
            'code_sha256': {p.name: owner.sha(p) for p in (Path(__file__), Path(__file__).with_name('probe_psd_nan_stream.py'))}})
        receipt = owner.spawn([sys.executable, '-u', str(Path(__file__).resolve()), 'execute',
            '--output-name', args.output_name, '--repetitions', str(args.repetitions)],
            {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(ROOT / 'tmp')}, out / 'controller.log')
        owner.save(out / 'process.json', receipt)
        print(json.dumps({'pid': receipt['pid'], 'output': str(out)}))
        return
    binding = owner.load(out / 'binding.json')
    for source, digest in binding['source_ledgers_sha256'].items():
        assert owner.sha(Path(source)) == digest
    asyncio.run(replay(out, selected, owner, args.repetitions))
    owner.save(out / 'state.json', {'phase': 'diagnostic_completed', 'formal_training': False})


if __name__ == '__main__':
    main()
