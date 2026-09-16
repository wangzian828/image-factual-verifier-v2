import asyncio
import subprocess
from pathlib import Path

import pytest

from ifv_training.psd_storage_admission import StorageAdmission, measure_run_storage


def test_measure_allows_slow_gpfs_scan(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, '123\t/run\n', '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('ifv_training.psd_storage_admission.shutil.disk_usage',
                        lambda path: type('Usage', (), {'free': 32*1024**3})())
    assert measure_run_storage(Path('/run')) == (123, 32*1024**3)
    assert calls[0]['timeout'] == 180 and calls[0]['check'] is True


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired(['du'], 180),
                                  subprocess.CalledProcessError(1, ['du']),
                                  OSError('transient metadata failure'), ValueError('bad du output')])
def test_measurement_failure_waits_without_skipping_or_admitting_slot(error):
    rows, sleeps, calls = [], [], []
    def measure(path):
        calls.append(path)
        if len(calls) == 1: raise error
        return 10, 32*1024**3
    async def sleep(seconds):
        assert rows[-1]['admission_open'] is False
        sleeps.append(seconds)
    gate = StorageAdmission(run=Path('/run'), ceiling=100, measure=measure,
                            save=lambda p, v: rows.append(dict(v)), sleep=sleep)
    asyncio.run(gate())
    assert sleeps == [10] and len(calls) == 2
    assert rows[0]['measurement_error'] == type(error).__name__
    assert rows[-1]['admission_open'] and gate.state['measurement_status'] == 'complete'


@pytest.mark.parametrize('blocked', [(100, 32*1024**3), (10, 15*1024**3)])
def test_real_space_hold_keeps_limits_and_waits(blocked):
    values = iter([blocked, (99, 16*1024**3)])
    rows, sleeps = [], []
    async def sleep(seconds): sleeps.append(seconds)
    gate = StorageAdmission(run=Path('/run'), ceiling=100, measure=lambda p: next(values),
                            save=lambda p, v: rows.append(dict(v)), sleep=sleep)
    asyncio.run(gate())
    assert sleeps == [60] and not rows[0]['admission_open'] and rows[-1]['admission_open']
    assert all(r['ceiling_bytes'] == 100 for r in rows)


def test_parallel_admission_shares_one_fresh_measurement():
    calls = []
    def measure(path):
        calls.append(path)
        return 1, 32*1024**3
    async def execute():
        gate = StorageAdmission(run=Path('/run'), ceiling=100, measure=measure,
                                save=lambda *args: None)
        await asyncio.gather(*(gate() for _ in range(40)))
    asyncio.run(execute())
    assert len(calls) == 1


def test_cancellation_during_space_hold_is_not_swallowed():
    async def sleep(seconds): raise asyncio.CancelledError()
    gate = StorageAdmission(run=Path('/run'), ceiling=100,
                            measure=lambda p: (100, 32*1024**3),
                            save=lambda *args: None, sleep=sleep)
    with pytest.raises(asyncio.CancelledError): asyncio.run(gate())
