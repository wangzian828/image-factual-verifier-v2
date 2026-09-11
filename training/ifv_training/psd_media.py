"""Immutable image tensors shared by PSD teacher scoring and student training."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_file

SCHEMA = "ifv-psd-media-v1"
IMAGE_TOKEN = 248056


def media_digest(media: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(media)).encode()).hexdigest()


def media_from_archive(step, *, processor_path: str, output_dir: Path, prompt_ids):
    from . import _repo_import  # noqa: F401
    from src.orchestrator.runtime_events import reconstruct_archived_request
    from transformers import AutoProcessor
    import json

    path = Path(processor_path)
    if (path / "adapter_config.json").is_file():
        path = Path(json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"])
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    request = reconstruct_archived_request(step["runtime_store_path"], step["context_request_id"])
    return bind_media(request.get("input_payload", request), processor=processor,
                      output_dir=output_dir, prompt_ids=prompt_ids, processor_id=str(path))


def image_bytes(request: Any) -> list[bytes]:
    """Read only image blocks, in order; repeated images remain repeated.

    A reconstructed runtime request contains archived data URLs. Remote URLs
    are deliberately not refetched: their contents may have changed.
    """
    result = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, Mapping):
            kind = value.get("type")
            if kind == "image_url":
                url = value.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("PSD requires archived image bytes, not remote URLs")
                result.append(base64.b64decode(url.split(",", 1)[1], validate=True))
            elif kind == "image":
                result.append(base64.b64decode(value["data"], validate=True))
            elif kind in {"video", "video_url", "input_audio"}:
                raise ValueError("PSD media adapter currently supports images only")
            else:
                for item in value.values():
                    visit(item)

    visit(request)
    return result


def bind_media(
    request: Mapping[str, Any], *, processor: Any, output_dir: Path,
    prompt_ids: Sequence[int], processor_id: str,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    blobs = image_bytes(request)
    if not blobs:
        if IMAGE_TOKEN in prompt_ids:
            raise ValueError("visual prompt has no archived images")
        return {}
    images = []
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            images.append(image.convert("RGB"))
    encoded = processor.image_processor(images=images, return_tensors="pt")
    tensors = {name: encoded[name].detach().cpu().contiguous()
               for name in ("pixel_values", "image_grid_thw")}
    merge_size = int(processor.image_processor.merge_size)
    validate_image_runs(prompt_ids, tensors["image_grid_thw"].tolist(), merge_size)
    provenance = {
        "schema_version": SCHEMA,
        "processor_id": processor_id,
        "processor_config": processor.image_processor.to_dict(),
        "image_sha256": [hashlib.sha256(blob).hexdigest() for blob in blobs],
        "image_grid_thw": tensors["image_grid_thw"].tolist(),
        "merge_size": merge_size,
    }
    identity = hashlib.sha256(canonical_json(provenance).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{identity}.pt"
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        torch.save(tensors, temporary)
        temporary.replace(path)
    return {**provenance, "path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_image_runs(ids: Sequence[int], grids: Sequence[Sequence[int]], merge: int):
    runs = []
    count = 0
    for token in [*ids, -1]:
        if token == IMAGE_TOKEN:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if merge <= 0 or any(len(grid) != 3 or any(int(x) <= 0 for x in grid) for grid in grids):
        raise ValueError("invalid image grids")
    expected = [int(t) * int(h) * int(w) // (merge * merge) for t, h, w in grids]
    if runs != expected:
        raise ValueError(f"PSD image placeholder/grid mismatch: {runs} != {expected}")


def validate_media(media: Mapping[str, Any], ids: Sequence[int]) -> None:
    if media.get("schema_version") != SCHEMA:
        raise ValueError("invalid PSD media schema")
    path = Path(media["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("PSD media tensor artifact missing")
    if sha256_file(path) != media.get("sha256"):
        raise ValueError("PSD media tensor hash mismatch")
    grids = media["image_grid_thw"]
    if len(grids) != len(media["image_sha256"]):
        raise ValueError("PSD media image count mismatch")
    validate_image_runs(ids, grids, int(media["merge_size"]))


def load_media(media: Mapping[str, Any], ids: Sequence[int]) -> dict[str, Any]:
    import torch

    validate_media(media, ids)
    tensors = torch.load(media["path"], map_location="cpu", weights_only=True)
    if set(tensors) != {"pixel_values", "image_grid_thw"}:
        raise ValueError("unexpected PSD media tensor fields")
    if tensors["image_grid_thw"].tolist() != media["image_grid_thw"]:
        raise ValueError("PSD media tensor grid mismatch")
    if not torch.isfinite(tensors["pixel_values"]).all():
        raise ValueError("nonfinite PSD image pixels")
    return tensors
