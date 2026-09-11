import asyncio
import pytest

from ifv_training.psd_case_pool import completed_cases


def test_slow_first_case_does_not_block_fast_cases_or_refilling():
    async def scenario():
        release_slow = asyncio.Event()
        started, finished = [], []
        active, peak = 0, 0
        async def worker(item):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.append(item)
            if item == "slow":
                await release_slow.wait()
            else:
                await asyncio.sleep(0)
            active -= 1
            return item
        async for _, item, result, error in completed_cases(["slow", "fast1", "fast2"], worker, concurrency=2):
            finished.append(item)
            assert error is None and result == item
            if item == "fast2":
                release_slow.set()
        assert finished == ["fast1", "fast2", "slow"]
        assert set(started) == {"slow", "fast1", "fast2"} and peak == 2
    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_case_failure_does_not_cancel_other_cases_or_expose_exception_text():
    async def scenario():
        async def worker(item):
            if item == 1:
                raise RuntimeError("SECRET API KEY")
            return item * 2
        rows = [row async for row in completed_cases([1, 2, 3], worker, concurrency=2)]
        assert sorted((item, result, error) for _, item, result, error in rows) == [
            (1, None, "RuntimeError"), (2, 4, None), (3, 6, None)]
        assert "SECRET" not in str(rows)
    asyncio.run(scenario())


def test_stopping_consumer_cancels_inflight_workers():
    async def scenario():
        entered = asyncio.Event()
        canceled = asyncio.Event()
        async def worker(item):
            if item == 0:
                await entered.wait()
                return "done"
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
        stream = completed_cases([0, 1], worker, concurrency=2)
        await anext(stream)
        await stream.aclose()
        assert canceled.is_set()
    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("value", [0, 65, True, -1])
def test_invalid_concurrency_rejected(value):
    async def worker(item):
        raise AssertionError("must not start")
    async def scenario():
        return [row async for row in completed_cases([1], worker, concurrency=value)]
    with pytest.raises(ValueError):
        asyncio.run(scenario())
