import json

import pytest

from scripts.plot_sft_progress import read_snapshot, moving_average, loss_segments


def test_loss_snapshot_ignores_only_incomplete_tail_and_does_not_merge_duplicates():
    first = json.dumps({'loss': .4, 'global_step/max_steps': '1/6'}).encode() + b'\n'
    rows, ignored = read_snapshot(first + b'{"loss":')
    assert ignored and rows[0][0] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        read_snapshot(first + first)
    with pytest.raises(json.JSONDecodeError):
        read_snapshot(first + b'bad line\n')


def test_trailing_mean_and_epoch_partition_are_not_centered_or_overlapping():
    rows = [(i, {'loss': float(i)}) for i in range(1, 8)]
    assert moving_average([1., 2., 3., 4.], 2) == [1., 1.5, 2.5, 3.5]
    stats = loss_segments(rows, 3, 2)
    assert stats['last_window']['mean'] == 6.5
    assert stats['previous_window']['mean'] == 4.5
    assert [r['mean'] for r in stats['epochs']] == [2., 5., 7.]
    assert [r['complete'] for r in stats['epochs']] == [True, True, False]
    with pytest.raises(ValueError):
        moving_average([1.], 0)
