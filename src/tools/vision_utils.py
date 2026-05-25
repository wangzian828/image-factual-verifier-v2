from __future__ import annotations

import base64
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
