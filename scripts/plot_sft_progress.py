"""Plot a read-only snapshot of ms-swift logging.jsonl (requires matplotlib)."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def read_snapshot(data):
    records = {}
    ignored_tail = False
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                ignored_tail = True
                break
            raise
        if 'loss' not in row:
            continue
        step = int(str(row.get('global_step/max_steps', row.get('step'))).split('/')[0])
        if step in records:
            raise ValueError(f'Duplicate step {step}; plot resumed runs separately')
        records[step] = row
    if not records:
        raise ValueError('No training loss records')
    return sorted(records.items()), ignored_tail


def moving_average(values, window):
    result = []
    for end in range(1, len(values) + 1):
        chunk = values[max(0, end - window):end]
        result.append(sum(chunk) / len(chunk) if all(map(math.isfinite, chunk)) else float('nan'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--title', default='Qwen3.5-9B | Phase-1 full SFT')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    data = args.log.read_bytes()
    rows, ignored_tail = read_snapshot(data)
    steps = [step for step, _ in rows]
    metrics = {key: [float(row.get(key, float('nan'))) for _, row in rows]
               for key in ('loss', 'grad_norm', 'token_acc', 'learning_rate')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    panels = [('loss', 'Training loss'), ('grad_norm', 'Gradient norm'),
              ('token_acc', 'Training token accuracy'), ('learning_rate', 'Learning rate')]

    def draw(ax, key, label):
        values = metrics[key]
        ax.plot(steps, values, color='#8caac9', alpha=.55, linewidth=.7, label='Per step')
        if key != 'learning_rate':
            ax.plot(steps, moving_average(values, args.window), color='#164b80', linewidth=2,
                    label=f'Trailing mean (up to {args.window} logged steps)')
        ax.set(xlabel='Optimizer step', ylabel=label)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)

    fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
    draw(ax, 'loss', 'Training loss')
    ax.set_title(f'{args.title}\nSnapshot through step {steps[-1]}')
    for suffix in ('png', 'svg'):
        fig.savefig(args.output_dir / f'loss.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, (key, label) in zip(axes.flat, panels):
        draw(ax, key, label)
    fig.suptitle(f'{args.title} — through step {steps[-1]}')
    fig.savefig(args.output_dir / 'training-overview.png', dpi=160)
    plt.close(fig)
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'source': str(args.log.resolve()), 'snapshot_sha256': hashlib.sha256(data).hexdigest(),
              'rows': len(rows), 'first_step': steps[0], 'last_step': steps[-1],
              'ignored_incomplete_tail': ignored_tail, 'window': args.window,
              'last_record': rows[-1][1], 'metrics': {}}
    for key, values in metrics.items():
        finite = [v for v in values if math.isfinite(v)]
        trailing = moving_average(values, args.window)[-1]
        report['metrics'][key] = {
            'nonfinite_count': len(values) - len(finite),
            'min': min(finite) if finite else None, 'max': max(finite) if finite else None,
            'trailing_mean': trailing if math.isfinite(trailing) else None}
    (args.output_dir / 'plot-summary.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
