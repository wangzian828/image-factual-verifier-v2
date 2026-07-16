from __future__ import annotations

import pytest

from src.integrations.clock.system_clock import SystemClockClient
from src.orchestrator.pipeline import Orchestrator


def test_runtime_date_override_is_shared_by_clock_and_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IFV_RUNTIME_DATE", "2026-07-16")

    clock = SystemClockClient().now()
    orchestrator = Orchestrator(
        provider="gemini",
        model_name="controlled-date",
        validate_startup=False,
    )

    assert clock["current_date"] == "2026-07-16"
    assert clock["current_datetime"].startswith("2026-07-16T")
    assert clock["source"] == "runtime_date_override"
    assert orchestrator.date_prefix.startswith("Current date: 2026-07-16 ")


def test_invalid_runtime_date_override_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IFV_RUNTIME_DATE", "2026-02-30")

    with pytest.raises(ValueError):
        SystemClockClient().now()
