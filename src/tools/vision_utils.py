from __future__ import annotations

import base64
import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import requests

REMOTE_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}

DEFAULT_VISION_TOOL_MAX_LONG_EDGE = 2048
DEFAULT_VISION_TOOL_JPEG_QUALITY = 92
DEFAULT_VISION_TOOL_IMAGE_MODE = "original"


def _guess_mime_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def _remote_image_to_data_url(image_url: str) -> str:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    with requests.get(
        image_url,
        headers=REMOTE_IMAGE_HEADERS,
        timeout=20,
        proxies=proxies,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        mime = content_type or _guess_mime_from_name(urlparse(image_url).path)
        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime};base64,{encoded}"


def image_to_data_url(image_input: str) -> str:
    if image_input.startswith("data:"):
        return image_input
    if image_input.startswith("http://") or image_input.startswith("https://"):
        return _remote_image_to_data_url(image_input)

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_input}")

    stat = path.stat()
    return _local_image_to_data_url(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


@lru_cache(maxsize=64)
def _local_image_to_data_url(
    path_string: str,
    mtime_ns: int,
    size: int,
) -> str:
    """Serialize an unchanged local image only once per process.

    The file metadata is part of the cache key, so replacing an image at the
    same path cannot silently reuse an old payload. This cache is deliberately
    process-local: the canonical image hash and runtime cache remain the
    cross-process sources of truth.
    """

    path = Path(path_string)
    mime = _guess_mime_from_name(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def controlled_image_to_data_url(
    image_input: str,
    *,
    max_long_edge: int | None = None,
    jpeg_quality: int | None = None,
) -> tuple[str, dict]:
    """Serialize a local image at a bounded resolution with auditable metadata."""

    if image_input.startswith(("data:", "http://", "https://")):
        data_url = image_to_data_url(image_input)
        header, encoded = data_url.split(",", 1)
        content = base64.b64decode(encoded)
        return data_url, {
            "source_kind": "remote_or_inline",
            "sha256": hashlib.sha256(content).hexdigest(),
            "original_size": None,
            "sent_size": None,
            "encoded_bytes": len(content),
            "media_type": header[5:].split(";", 1)[0],
        }

    from PIL import Image, ImageOps

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_input}")
    stat = path.stat()
    edge = _resolve_image_edge(max_long_edge)
    quality = _resolve_image_quality(jpeg_quality)
    data_url, metadata = _controlled_local_image_to_data_url(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        edge,
        quality,
    )
    return data_url, dict(metadata)


def vision_tool_image_to_data_url(image_input: str) -> str:
    """Serialize an image for a VLM tool.

    The default is the historical original-image wire path. The compressed path
    remains available as an explicit opt-in because provider latency and visual
    behavior must not change implicitly when an image serializer changes.
    """

    mode = os.getenv(
        "IFV_VISION_TOOL_IMAGE_MODE",
        DEFAULT_VISION_TOOL_IMAGE_MODE,
    ).strip().lower()
    if mode in {"original", "raw", "direct"}:
        return image_to_data_url(image_input)
    if mode not in {"compressed", "jpeg"}:
        raise ValueError(
            "IFV_VISION_TOOL_IMAGE_MODE must be one of: "
            "original, compressed"
        )

    data_url, _ = controlled_image_to_data_url(
        image_input,
        max_long_edge=int(
            os.getenv(
                "IFV_VISION_TOOL_MAX_LONG_EDGE",
                str(DEFAULT_VISION_TOOL_MAX_LONG_EDGE),
            )
        ),
        jpeg_quality=int(
            os.getenv(
                "IFV_VISION_TOOL_JPEG_QUALITY",
                str(DEFAULT_VISION_TOOL_JPEG_QUALITY),
            )
        ),
    )
    return data_url


def _resolve_image_edge(value: int | None) -> int:
    return max(
        256,
        int(value or os.getenv("IFV_IMAGE_MAX_LONG_EDGE", "1280")),
    )


def _resolve_image_quality(value: int | None) -> int:
    return max(
        55,
        min(95, int(value or os.getenv("IFV_IMAGE_JPEG_QUALITY", "88"))),
    )


@lru_cache(maxsize=32)
def _controlled_local_image_to_data_url(
    path_string: str,
    mtime_ns: int,
    size: int,
    edge: int,
    quality: int,
) -> tuple[str, dict]:
    path = Path(path_string)
    from PIL import Image, ImageOps

    original_bytes = path.read_bytes()
    try:
        with Image.open(io.BytesIO(original_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            original_size = list(image.size)
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            sent_size = list(image.size)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
    except Exception as exc:
        # Deterministic protocol fixtures sometimes use stable non-image bytes.
        # Preserve the old wire behavior for those fixtures; production hash and
        # perception validation still reject truly unusable case media upstream.
        return image_to_data_url(path_string), {
            "source_kind": "local_file",
            "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "sha256": hashlib.sha256(original_bytes).hexdigest(),
            "original_size": None,
            "sent_size": None,
            "encoded_bytes": len(original_bytes),
            "media_type": _guess_mime_from_name(path.name),
            "decode_unavailable": f"{type(exc).__name__}: {exc}",
        }
    sent_bytes = buffer.getvalue()
    return (
        "data:image/jpeg;base64," + base64.b64encode(sent_bytes).decode("ascii"),
        {
            "source_kind": "local_file",
            "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "sha256": hashlib.sha256(sent_bytes).hexdigest(),
            "original_size": original_size,
            "sent_size": sent_size,
            "encoded_bytes": len(sent_bytes),
            "media_type": "image/jpeg",
            "max_long_edge": edge,
            "jpeg_quality": quality,
        },
    )
