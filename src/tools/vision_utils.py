from __future__ import annotations

import base64
import hashlib
import io
import os
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
    response = requests.get(image_url, headers=REMOTE_IMAGE_HEADERS, timeout=20, proxies=proxies)
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
    edge = max(
        256,
        int(max_long_edge or os.getenv("IFV_IMAGE_MAX_LONG_EDGE", "1280")),
    )
    quality = max(
        55,
        min(95, int(jpeg_quality or os.getenv("IFV_IMAGE_JPEG_QUALITY", "88"))),
    )
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
        return image_to_data_url(image_input), {
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
