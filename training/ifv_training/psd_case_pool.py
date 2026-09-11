"""Completion-order bounded case scheduling within ONE frozen-policy PSD round.

This is asynchronous sample construction, not stale-policy/asynchronous RL
optimization. A case's own hint -> rollout -> verifier dependency remains serial.
"""
from __future__ import annotations

import asyncio


async def completed_cases(items, worker, *, concurrency=1):
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("case concurrency must be 1..64")
    iterator = iter(enumerate(items))
    pending = {}

    async def guarded(item):
        try:
            return await worker(item), None
        except Exception as exc:
            # Provider exceptions can carry credentials. Return only the type;
            # each case's own bound diagnostic artifacts preserve its context.
            return None, type(exc).__name__

    def fill():
        while len(pending) < concurrency:
            pair = next(iterator, None)
            if pair is None:
                break
            index, item = pair
            pending[asyncio.create_task(guarded(item))] = (index, item)

    fill()
    try:
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            completed = []
            for task in sorted(done, key=lambda task: pending[task][0]):
                index, item = pending.pop(task)
                result, error = task.result()
                completed.append((index, item, result, error))
            # Refill before yielding, so a slow first case cannot block queued
            # work and persistence/processing of another completed case.
            fill()
            for row in completed:
                yield row
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
