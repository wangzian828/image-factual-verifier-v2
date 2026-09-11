from __future__ import annotations

import asyncio
import json

import pytest

from ifv_training.psd_repair_storage import cached_continuation, load_bound, save_bound
from ifv_training.psd_repair_runtime import ContinuationResult
from src.orchestrator.stage_runner import StageStep


def test_completed_continuation_survives_later_failure_without_resampling(tmp_path):
    calls = []
    async def generate():
        calls.append(True)
        return ContinuationResult(teacher_steps=[StageStep(round=1, stage_name="psd_teacher_repair", action_type="tool_call")],
            student_steps=[], teacher_history=[{"secret_large_image": "not duplicated"}], student_history=[],
            hint="compare", teacher_complete=True, teacher_episode_trace={"state": {"runtime_store": {"runtime_path": "/data/archive"}}})
    path = tmp_path / "hint.json"
    first = asyncio.run(cached_continuation(path, identity={"hint": "sha"}, generate=generate))
    second = asyncio.run(cached_continuation(path, identity={"hint": "sha"}, generate=generate))
    assert len(calls) == 1
    assert first.teacher_steps == second.teacher_steps
    assert first.teacher_episode_trace == second.teacher_episode_trace
    assert "not duplicated" not in path.read_text()


def test_incomplete_provider_call_not_cached(tmp_path):
    async def generate():
        raise TimeoutError("transport failed")
    path = tmp_path / "hint.json"
    with pytest.raises(TimeoutError):
        asyncio.run(cached_continuation(path, identity={}, generate=generate))
    assert not path.exists()


@pytest.mark.parametrize("tamper", ["identity", "payload"])
def test_snapshot_tamper_fails_closed(tmp_path, tamper):
    path = tmp_path / "hint.json"
    save_bound(path, identity={"policy": "a"}, payload={"tokens": [1]})
    value = json.loads(path.read_text())
    value[tamper] = {"changed": True}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="binding changed"):
        load_bound(path, identity={"policy": "a"})
