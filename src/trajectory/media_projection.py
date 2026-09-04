"""Project runtime-request images into portable trajectory SFT rows."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class TrajectoryMediaProjection:
    """Ordered portable media plus diagnostics for one exported episode."""

    initial_images: list[str] = field(default_factory=list)
    images_after_step: dict[int, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def images(self) -> list[str]:
        result = list(self.initial_images)
        for step_index in sorted(self.images_after_step):
            result.extend(self.images_after_step[step_index])
        return result


def project_trajectory_media(
    trace: Mapping[str, Any],
    *,
    candidate_steps: Sequence[Mapping[str, Any]],
    fallback_image_path: str,
) -> TrajectoryMediaProjection:
    """Recover the exact images seen by policy requests.

    The first request supplies the initial user image. Images newly present in
    the request after tool step ``N`` are attached to that step's
    ``tool_response``. Content hashes suppress repeated original-image
    reinjection without dropping genuinely new crops or candidate images.
    """

    projection = TrajectoryMediaProjection()
    runtime_root = _runtime_root(trace)
    seen_hashes: set[str] = set()

    request_media: list[list[tuple[str, str]]] = []
    for step_index, step in enumerate(candidate_steps):
        request_id = _context_request_id(step)
        if runtime_root is None or not request_id:
            request_media.append([])
            continue
        media, missing = _request_media(runtime_root, request_id)
        request_media.append(media)
        projection.missing.extend(
            f"step[{step_index}] request {request_id}: {item}"
            for item in missing
        )

    if request_media:
        projection.initial_images.extend(
            _new_data_urls(request_media[0], seen_hashes)
        )

    if not projection.initial_images:
        fallback = _fallback_data_url(fallback_image_path)
        if fallback is not None:
            digest, data_url = fallback
            seen_hashes.add(digest)
            projection.initial_images.append(data_url)
        else:
            projection.missing.append(
                f"initial image is unavailable: {fallback_image_path}"
            )

    for next_step_index in range(1, len(request_media)):
        previous_step_index = next_step_index - 1
        if str(candidate_steps[previous_step_index].get("action_type", "")) != (
            "tool_call"
        ):
            continue
        newly_visible = _new_data_urls(
            request_media[next_step_index],
            seen_hashes,
        )
        if newly_visible:
            projection.images_after_step[previous_step_index] = newly_visible

    return projection


def image_markers(count: int) -> str:
    """Return one marker per projected image, ready for message appending."""

    return "\n".join("<image>" for _ in range(max(0, int(count))))


def _runtime_root(trace: Mapping[str, Any]) -> Path | None:
    state = trace.get("state")
    if not isinstance(state, Mapping):
        return None
    runtime_store = state.get("runtime_store")
    if not isinstance(runtime_store, Mapping):
        return None
    raw = str(runtime_store.get("runtime_path", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _context_request_id(step: Mapping[str, Any]) -> str:
    metadata = step.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("context_request_id", "")).strip()


def _request_media(
    runtime_root: Path,
    request_id: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    manifest_path = runtime_root / "context" / f"{request_id}.json"
    if not manifest_path.is_file():
        return [], [f"manifest missing at {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"manifest unreadable: {exc}"]

    artifact_root = (runtime_root / "artifacts" / "sha256").resolve()
    result: list[tuple[str, str]] = []
    missing: list[str] = []
    for context_item in manifest.get("context_items", []) or []:
        if not isinstance(context_item, Mapping):
            continue
        if str(context_item.get("kind", "")) != "input_payload":
            continue
        for descriptor in context_item.get("media", []) or []:
            if not isinstance(descriptor, Mapping):
                continue
            recovered, error = _descriptor_data_url(
                artifact_root,
                descriptor,
            )
            if recovered is not None:
                result.append(recovered)
            elif error:
                missing.append(error)
    return result, missing


def _descriptor_data_url(
    artifact_root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[tuple[str, str] | None, str]:
    relative = str(descriptor.get("artifact_path", "")).strip()
    expected_hash = str(descriptor.get("sha256", "")).strip().lower()
    media_type = str(
        descriptor.get("media_type", "application/octet-stream")
    ).strip()
    if not relative:
        return None, "media descriptor has no artifact_path"
    path = (artifact_root / relative).resolve()
    if artifact_root != path and artifact_root not in path.parents:
        return None, f"media path escapes artifact store: {relative}"
    try:
        content = path.read_bytes()
    except OSError as exc:
        return None, f"media artifact unreadable ({relative}): {exc}"
    actual_hash = hashlib.sha256(content).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        return None, (
            f"media hash mismatch ({relative}): "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return (
        actual_hash,
        f"data:{media_type};base64,"
        + base64.b64encode(content).decode("ascii"),
    ), ""


def _fallback_data_url(image_path: str) -> tuple[str, str] | None:
    path = Path(str(image_path)).expanduser()
    if not path.is_file():
        return None
    try:
        content = path.read_bytes()
    except OSError:
        return None
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.casefold(), "application/octet-stream")
    digest = hashlib.sha256(content).hexdigest()
    return (
        digest,
        f"data:{media_type};base64,"
        + base64.b64encode(content).decode("ascii"),
    )


def _new_data_urls(
    media: Sequence[tuple[str, str]],
    seen_hashes: set[str],
) -> list[str]:
    result: list[str] = []
    for digest, data_url in media:
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        result.append(data_url)
    return result
