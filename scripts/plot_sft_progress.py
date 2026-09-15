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
    if window < 1:
        raise ValueError('window must be positive')
    result = []
    for end in range(1, len(values) + 1):
        chunk = values[max(0, end - window):end]
        result.append(sum(chunk) / len(chunk) if all(map(math.isfinite, chunk)) else float('nan'))
    return result


def loss_segments(rows, steps_per_epoch, window):
    """Summarize equal-size trailing windows and disjoint epoch intervals."""
    if steps_per_epoch < 1 or window < 1:
        raise ValueError('epoch length and window must be positive')
    def describe(part):
        losses = [float(row['loss']) for _, row in part]
        return {'rows': len(part), 'first_step': part[0][0] if part else None,
                'last_step': part[-1][0] if part else None,
                'mean': sum(losses) / len(losses) if losses and all(map(math.isfinite, losses)) else None}
    epochs = []
    for number in range(1, math.ceil(rows[-1][0] / steps_per_epoch) + 1):
        part = [(step, row) for step, row in rows
                if (number - 1) * steps_per_epoch < step <= number * steps_per_epoch]
        epochs.append({'epoch': number, 'complete': len(part) == steps_per_epoch, **describe(part)})
    return {'first_window': describe(rows[:window]), 'last_window': describe(rows[-window:]),
            'previous_window': describe(rows[-2 * window:-window]), 'epochs': epochs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--title', default='Qwen3.5-9B | Phase-1 full SFT')
    parser.add_argument('--steps-per-epoch', type=int, help='Mark verified epoch boundaries')
    parser.add_argument('--zoom-last', type=int, default=0, help='Add a recent-step loss subplot')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    if args.zoom_last < 0 or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error('epoch length must be positive; zoom length cannot be negative')
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

    def draw(ax, key, label, *, start=None):
        values = metrics[key]
        indices = [i for i, step in enumerate(steps) if start is None or step >= start]
        plot_steps = [steps[i] for i in indices]
        ax.plot(plot_steps, [values[i] for i in indices], color='#8caac9', alpha=.55,
                linewidth=.7, label='Per step')
        if key != 'learning_rate':
            smooth = moving_average(values, args.window)
            ax.plot(plot_steps, [smooth[i] for i in indices], color='#164b80', linewidth=2,
                    label=f'Trailing mean (up to {args.window} logged steps)')
        if args.steps_per_epoch:
            for boundary in range(args.steps_per_epoch, steps[-1] + 1, args.steps_per_epoch):
                if boundary >= plot_steps[0]:
                    ax.axvline(boundary, color='#9ca3af', linestyle='--', linewidth=.9)
                    ax.text(boundary + 8, .97, f'Epoch {boundary // args.steps_per_epoch} end',
                            transform=ax.get_xaxis_transform(), va='top', fontsize=8, color='#4b5563')
        ax.set(xlabel='Optimizer step', ylabel=label)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)

    panels_count = 2 if args.zoom_last else 1
    fig, loss_axes = plt.subplots(panels_count, 1, figsize=(11, 7.2 if args.zoom_last else 4.8),
                                 squeeze=False, layout='constrained')
    ax = loss_axes[0, 0]
    draw(ax, 'loss', 'Training loss (logged)')
    ax.set_ylim(bottom=0)
    fig.suptitle(f'{args.title}\nSnapshot through step {steps[-1]}', fontsize=13)
    if args.zoom_last:
        start = max(steps[0], steps[-1] - args.zoom_last + 1)
        draw(loss_axes[1, 0], 'loss', 'Training loss (zoom)', start=start)
        loss_axes[1, 0].set_title(f'Recent {steps[-1] - start + 1} steps — independent y scale', fontsize=10)
    fig.supxlabel('Training loss only; not validation loss. Shown values are not extrapolated.', fontsize=9)
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
    if args.steps_per_epoch:
        report['loss_segments'] = loss_segments(rows, args.steps_per_epoch, args.window)
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
