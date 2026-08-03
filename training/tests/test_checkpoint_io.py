from __future__ import annotations

import os
from pathlib import Path
import shutil

from ifv_training.checkpoint_io import (
    checkpoint_io_profile,
    checkpoint_storage_preflight,
)


def _touch(path: Path, size: int, mtime_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_checkpoint_io_profile_separates_model_optimizer_and_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    _touch(checkpoint / "model-00001-of-00002.safetensors", 11, 1_000_000_000)
    _touch(checkpoint / "model-00002-of-00002.safetensors", 13, 2_000_000_000)
    _touch(
        checkpoint / "global_step10" / "zero_pp_rank_0_mp_rank_00_model_states.pt",
        3,
        3_000_000_000,
    )
    _touch(
        checkpoint / "global_step10" / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        101,
        5_000_000_000,
    )
    _touch(checkpoint / "scheduler.pt", 2, 6_000_000_000)
    _touch(checkpoint / "rng_state_0.pth", 2, 6_100_000_000)
    _touch(checkpoint / "trainer_state.json", 2, 6_200_000_000)

    result = checkpoint_io_profile(checkpoint)

    assert result["passed"] is True
    assert result["total_bytes"] == 134
    assert result["optimizer_rank_file_count"] == 1
    assert result["categories"]["model_export"]["bytes"] == 24
    assert result["categories"]["optimizer_state"]["bytes"] == 101
    assert result["pipeline_phases"]["optimizer_state"]["span_seconds"] == 2.0
    assert result["write_window_seconds"] == 5.2


def test_checkpoint_storage_preflight_checks_reserve(tmp_path: Path) -> None:
    free_bytes = int(shutil.disk_usage(tmp_path).free)

    accepted = checkpoint_storage_preflight(
        tmp_path / "accepted",
        estimated_checkpoint_bytes=max(1, free_bytes // 10),
        reserve_multiplier=1.1,
    )
    rejected = checkpoint_storage_preflight(
        tmp_path / "rejected",
        estimated_checkpoint_bytes=free_bytes,
        reserve_multiplier=2.0,
    )

    assert accepted["passed"] is True
    assert rejected["passed"] is False
    assert rejected["checks"]["sufficient_free_space"] is False
