from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "h20"
    / "gpu_utilization_guard.py"
)
SPEC = importlib.util.spec_from_file_location("gpu_utilization_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def _sample(index: int, utilization: float, memory: int = 90000) -> dict:
    return {
        "index": index,
        "memory_used_mib": memory,
        "memory_total_mib": 97871,
        "utilization_percent": utilization,
    }


def test_low_window_accumulates_and_resets_per_gpu() -> None:
    first = guard.update_state(
        {},
        [_sample(0, 0), _sample(1, 0)],
        now=100.0,
        threshold_percent=10.0,
    )
    second = guard.update_state(
        first,
        [_sample(0, 9), _sample(1, 10)],
        now=200.0,
        threshold_percent=10.0,
    )

    assert second["gpus"]["0"]["below_since"] == 100.0
    assert second["gpus"]["0"]["low_seconds"] == 100.0
    assert second["gpus"]["1"]["below_since"] is None
    assert second["gpus"]["1"]["low_seconds"] == 0.0


def test_at_risk_gpu_ids_uses_configured_window() -> None:
    state = {
        "gpus": {
            "0": {"low_seconds": 5399},
            "1": {"low_seconds": 5400},
            "2": {"low_seconds": 7200},
        }
    }

    assert guard.at_risk_gpu_ids(state, 5400) == [1, 2]


def test_all_gpus_free_requires_low_memory_and_utilization() -> None:
    free = [_sample(index, 0, memory=10) for index in range(4)]
    busy_memory = [*free[:3], _sample(3, 0, memory=2048)]
    busy_compute = [*free[:3], _sample(3, 10, memory=10)]

    assert guard.all_gpus_free(
        free, memory_limit_mib=1024, utilization_threshold_percent=10
    )
    assert not guard.all_gpus_free(
        busy_memory, memory_limit_mib=1024, utilization_threshold_percent=10
    )
    assert not guard.all_gpus_free(
        busy_compute, memory_limit_mib=1024, utilization_threshold_percent=10
    )
