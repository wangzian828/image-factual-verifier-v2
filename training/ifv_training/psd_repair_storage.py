"""Crash-safe local checkpoints for expensive PSD continuation calls."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .io import load_json
from .psd_gemini_judge import _atomic_json
from .psd_repair import _sha


def save_bound(path: Path, *, identity, payload):
    _atomic_json(path, {"identity": identity, "payload": payload, "payload_sha256": _sha(payload)})


def load_bound(path: Path, *, identity):
    saved = load_json(path)
    if saved.get("identity") != identity or saved.get("payload_sha256") != _sha(saved.get("payload")):
        raise ValueError("PSD continuation checkpoint binding changed")
    return saved["payload"]


async def cached_continuation(path: Path, *, identity, generate):
    from .psd_repair_runtime import ContinuationResult
    from src.orchestrator.stage_runner import StageStep
    if path.exists():
        payload = load_bound(path, identity=identity)
        return ContinuationResult(
            teacher_steps=[StageStep(**step) for step in payload.pop("teacher_steps")],
            student_steps=[StageStep(**step) for step in payload.pop("student_steps")],
            teacher_history=[], student_history=[], **payload)
    result = await generate()
    payload = {"teacher_steps": [asdict(step) for step in result.teacher_steps],
               "student_steps": [asdict(step) for step in result.student_steps],
               "hint": result.hint, "teacher_complete": result.teacher_complete,
               "student_complete": result.student_complete, "stop_reason": result.stop_reason,
               "teacher_episode_trace": result.teacher_episode_trace}
    # Raw history is already immutable in the runtime archive. Do not duplicate
    # all base64 images in a second continuation snapshot.
    save_bound(path, identity=identity, payload=payload)
    return result
