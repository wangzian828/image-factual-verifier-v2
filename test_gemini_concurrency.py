"""Tests for the Gemini request gate used by concurrent rollouts."""

from __future__ import annotations

import asyncio

import pytest

from src.eval.gemini_run_guard import GeminiRunGuard
from src.integrations.gemini.interactions import GeminiRequestGate


def test_gemini_request_gate_caps_cross_task_inflight_requests() -> None:
    async def scenario() -> int:
        gate = GeminiRequestGate(2)
        active = 0
        peak = 0

        async def one_request() -> None:
            nonlocal active, peak
            async with gate:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(one_request() for _ in range(8)))
        return peak

    assert asyncio.run(scenario()) == 2


def test_gemini_request_gate_releases_after_exception() -> None:
    async def scenario() -> int:
        gate = GeminiRequestGate(1)
        async with gate:
            try:
                raise RuntimeError("synthetic request failure")
            except RuntimeError:
                pass
        async with gate:
            return 1

    assert asyncio.run(scenario()) == 1


def test_gemini_run_guard_rejects_second_process_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("IFV_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("GEMINI_EVAL_MAX_CONCURRENCY", "4")
    monkeypatch.delenv("IFV_GEMINI_LOCK_DIR", raising=False)

    first = GeminiRunGuard.acquire(
        provider="gemini",
        concurrency=2,
        run_id="first",
    )
    try:
        with pytest.raises(RuntimeError, match="already active"):
            GeminiRunGuard.acquire(
                provider="gemini",
                concurrency=2,
                run_id="second",
            )
    finally:
        first.release()


def test_gemini_run_guard_enforces_rollout_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("IFV_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("GEMINI_EVAL_MAX_CONCURRENCY", "2")
    monkeypatch.delenv("IFV_GEMINI_LOCK_DIR", raising=False)

    with pytest.raises(ValueError, match="exceeds the configured cap"):
        GeminiRunGuard.acquire(provider="gemini", concurrency=3)
