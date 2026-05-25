# -*- coding: utf-8 -*-
"""Image utilities for the ReAct agent.

Downloads images from URLs and converts them to base64 data URLs
for inclusion in multimodal messages. Includes an in-memory LRU cache
to avoid re-downloading the same image within a session.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
import threading
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


# Max image size to download (2MB)
MAX_IMAGE_BYTES = 2 * 1024 * 1024
# Max images to include per tool response
MAX_IMAGES_PER_RESPONSE = 3
# Timeout for image download
DOWNLOAD_TIMEOUT = 10
# Max cache entries
MAX_CACHE_SIZE = 50


class _ImageCache:
    """Thread-safe in-memory LRU cache for downloaded image data URLs.

    Avoids re-downloading the same image URL within a session.
    Keyed by URL hash, stores the base64 data URL string.
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self._cache: Dict[str, Optional[str]] = {}
        self._order: List[str] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, url: str) -> Tuple[bool, Optional[str]]:
        """Return (hit, data_url). hit=True means we have a cached result (even if None)."""
        key = hashlib.sha1(url.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                return True, self._cache[key]
        return False, None

    def put(self, url: str, data_url: Optional[str]) -> None:
        key = hashlib.sha1(url.encode()).hexdigest()
        with self._lock:
            if key not in self._cache:
                if len(self._order) >= self._max_size:
                    # Evict oldest
                    oldest = self._order.pop(0)
                    self._cache.pop(oldest, None)
                self._order.append(key)
            self._cache[key] = data_url


# Global cache instance
_cache = _ImageCache()


def download_image_as_data_url(url: str, max_bytes: int = MAX_IMAGE_BYTES) -> Optional[str]:
    """Download an image URL and return as a base64 data URL.

    Uses an in-memory cache to avoid re-downloading the same URL.

    Args:
        url: HTTP(S) URL of the image.
        max_bytes: Maximum file size to download.

    Returns:
        Data URL string (data:image/...;base64,...) or None if failed.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None

    # Check cache first
    hit, cached = _cache.get(url)
    if hit:
        return cached

    try:
        # Use stream to check content-length before downloading
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "").lower()
        if not content_type.startswith("image/"):
            _cache.put(url, None)
            return None

        # Check content length if available
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            _cache.put(url, None)
            return None

        # Download with size limit
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            total += len(chunk)
            if total > max_bytes:
                _cache.put(url, None)
                return None
            chunks.append(chunk)

        image_data = b"".join(chunks)
        if not image_data:
            _cache.put(url, None)
            return None

        # Determine MIME type
        mime_type = _guess_mime(content_type, url)
        b64 = base64.b64encode(image_data).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        # Cache the result
        _cache.put(url, data_url)
        return data_url

    except (requests.RequestException, ValueError, OSError):
        _cache.put(url, None)
        return None


def extract_image_urls_from_result(result_text: str) -> List[str]:
    """Extract image URLs from formatted tool result text.

    Looks for lines like:
        Image: https://...
    """
    urls = []
    for match in re.finditer(r"Image:\s*(https?://\S+)", result_text):
        url = match.group(1).strip()
        if _is_likely_image_url(url):
            urls.append(url)
    return urls


def extract_image_urls_from_lens_results(results: list) -> List[str]:
    """Extract image URLs from Lens search result dicts."""
    urls = []
    for r in results:
        if isinstance(r, dict):
            img_url = r.get("image_url", "")
            if img_url and _is_likely_image_url(img_url):
                urls.append(img_url)
    return urls


def _is_likely_image_url(url: str) -> bool:
    """Check if URL likely points to an image."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    image_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
    if any(path.endswith(ext) for ext in image_exts):
        return True
    # Common image hosting patterns
    image_hosts = ("imgur.com", "i.imgur.com", "pbs.twimg.com", "upload.wikimedia.org",
                   "images.unsplash.com", "encrypted-tbn", "gstatic.com")
    if any(host in parsed.netloc for host in image_hosts):
        return True
    return False


def _guess_mime(content_type: str, url: str) -> str:
    """Guess MIME type from content-type header or URL."""
    if "jpeg" in content_type or "jpg" in content_type:
        return "image/jpeg"
    if "png" in content_type:
        return "image/png"
    if "gif" in content_type:
        return "image/gif"
    if "webp" in content_type:
        return "image/webp"

    # Fallback to URL extension
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".gif"):
        return "image/gif"
    if path.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"  # default
