"""Report bounded packing throughput, with explicit warmup windows and SP accounting."""
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ['IFV_H20_RUN_ROOT'])
reports = []
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run / 'result.json').exists():
        continue
    result = json.loads((run / 'result.json').read_text())
    report = {'run': run.name, **result}
    if result['exit_code']:
        reports.append(report)
        continue
    provenance = json.loads((run / 'provenance.json').read_text())
    rows = [json.loads(line) for line in (run / 'output/logging.jsonl').read_text().split('\n') if line.strip()]
    steps = [r for r in rows if 'loss' in r and 'train_speed(s/it)' in r]
    assert steps and all(math.isfinite(float(r[k])) for r in steps for k in ('loss', 'grad_norm'))
    sp = provenance['sequence_parallel_size']
    def step(r):
        return int(r['global_step/max_steps'].split('/')[0])
    windows = []
    for n in (3, 6):
        if len(steps) <= n:
            continue
        last, anchor = steps[-1], steps[-n-1]
        elapsed = step(last) * float(last['train_speed(s/it)']) - step(anchor) * float(anchor['train_speed(s/it)'])
        tokens = (int(last['num_input_tokens_seen']) - int(anchor['num_input_tokens_seen'])) / sp
        windows.append({'steps': [step(anchor)+1, step(last)], 'seconds': elapsed, 'effective_tokens': tokens,
                        'tokens_per_second': tokens / elapsed, 'seconds_per_step': elapsed/n,
                        'projected_62673659_token_epoch_minutes': 62673659 / (tokens/elapsed) / 60})
    peaks = {}
    with (run / 'gpu.csv').open() as f:
        for row in csv.reader(f):
            if len(row) >= 6:
                key = row[1].strip()
                peaks[key] = max(peaks.get(key, 0), float(row[2]))
    report.update({'sequence_parallel_size': sp, 'finite_loss_and_gradients': True,
                   'windows': windows, 'sampled_peak_MiB_by_gpu': peaks,
                   'note': 'Warmup/shape variation remains; extrapolation excludes startup/eval/save. Same pool, not identical episodes per measured window or equal optimization batches.'})
    reports.append(report)
(root / 'packing-comparison-summary.json').write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
