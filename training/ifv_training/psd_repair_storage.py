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


def continuation_from_payload(payload):
    from .psd_repair_runtime import ContinuationResult
    from src.orchestrator.stage_runner import StageStep
    payload = dict(payload)
    return ContinuationResult(
        teacher_steps=[StageStep(**step) for step in payload.pop("teacher_steps")],
        student_steps=[StageStep(**step) for step in payload.pop("student_steps")],
        teacher_history=[], student_history=[], **payload)


def continuation_payload(result):
    return {"teacher_steps": [asdict(step) for step in result.teacher_steps],
               "student_steps": [asdict(step) for step in result.student_steps],
               "hint": result.hint, "teacher_complete": result.teacher_complete,
               "student_complete": result.student_complete, "stop_reason": result.stop_reason,
               "teacher_episode_trace": result.teacher_episode_trace,
               "local_targets": result.local_targets}


async def cached_continuation(path: Path, *, identity, generate):
    if path.exists():
        return continuation_from_payload(load_bound(path, identity=identity))
    result = await generate()
    payload = continuation_payload(result)
    # Raw history is already immutable in the runtime archive. Do not duplicate
    # all base64 images in a second continuation snapshot.
    save_bound(path, identity=identity, payload=payload)
    return result
