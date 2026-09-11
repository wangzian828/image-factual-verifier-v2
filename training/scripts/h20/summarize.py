"""Summarize completed H20 experiments without double-counting SP tokens."""
import csv
import json
import os
from pathlib import Path

root = Path(os.environ['IFV_H20_RUN_ROOT'])
reports = []
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run / 'result.json').exists():
        continue
    result = json.loads((run / 'result.json').read_text())
    if result['exit_code'] != 0 or not (run / 'provenance.json').exists():
        continue
    provenance = json.loads((run / 'provenance.json').read_text())
    logs = [json.loads(line) for line in (run / 'output/logging.jsonl').read_text().splitlines() if line.strip()]
    steps = [row for row in logs if 'loss' in row and 'train_speed(s/it)' in row]
    if not steps:
        continue
    sp = provenance['sequence_parallel_size']
    def step(row):
        return int(row['global_step/max_steps'].split('/')[0])
    last = steps[-1]
    command = json.loads((run / 'command.json').read_text())
    start_step = 0
    if '--resume_from_checkpoint' in command:
        checkpoint = Path(command[command.index('--resume_from_checkpoint') + 1])
        start_step = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
    # Use the final three steps where possible; report the interval explicitly.
    anchor = steps[-4] if len(steps) > 3 else steps[0]
    seconds = (step(last) - start_step) * last['train_speed(s/it)'] - (step(anchor) - start_step) * anchor['train_speed(s/it)']
    tokens = (last['num_input_tokens_seen'] - anchor['num_input_tokens_seen']) / sp
    peaks = {}
    for row in csv.reader((run / 'gpu.csv').open()):
        if len(row) >= 6:
            gpu = row[1].strip()
            peaks[gpu] = max(peaks.get(gpu, 0), float(row[2]))
    reports.append({
        'run': run.name, 'provenance': provenance, 'exit_code': result['exit_code'],
        'completed_step': step(last), 'measured_steps': [step(anchor) + 1, step(last)],
        'interval_seconds': seconds, 'effective_input_tokens': tokens,
        'effective_tokens_per_second': tokens / seconds if seconds > 0 else None,
        'seconds_per_step': seconds / (step(last) - step(anchor)) if step(last) > step(anchor) else None,
        'resumed_from_step': start_step,
        'gpu_peak_MiB': peaks, 'losses': [r['loss'] for r in steps],
        'grad_norms': [r.get('grad_norm') for r in steps],
        'note': 'Short-run interval; shape warmup may remain. Synthetic and real runs are not quality comparisons.'
    })
(root / 'benchmark-summary.json').write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
