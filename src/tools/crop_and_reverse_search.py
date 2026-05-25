from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.integrations.search.serper import SerperLensSearchClient
from src.tools.base import BaseTool


@dataclass
class CropAndReverseSearchTool(BaseTool):
    """Run true reverse image search on an existing local crop or remote image URL."""

    client: Optional[SerperLensSearchClient] = None
    top_k: int = 5
    name: str = "crop_and_reverse_search"
    description: str = "Run reverse image search on a crop image or image URL and return visually related results."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {"type": "string", "description": "Local path or remote URL of the crop image."},
                "crop_name": {"type": "string", "description": "Optional crop label for traceability."},
            },
            "required": ["image_input"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SerperLensSearchClient()

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_input = str(params["image_input"])
        crop_name = str(params.get("crop_name", "") or "")
        image_url = self._prepare_remote_url(image_input)
        results = self.client.search(image_url=image_url, top_k=self.top_k)
        return {
            "crop_name": crop_name,
            "image_input": image_input,
            "image_url_for_search": image_url,
            "results": results,
        }

    def _prepare_remote_url(self, image_input: str) -> str:
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return image_input
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_input}")

        upload_url = os.getenv("IMAGE_UPLOAD_API_URL", "").strip() or "https://litterbox.catbox.moe/resources/internals/api.php"
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
        proxies = {"http": proxy, "https": proxy} if proxy else None
        with path.open("rb") as handle:
            if "catbox" in upload_url or "litterbox" in upload_url:
                response = requests.post(upload_url, data={"reqtype": "fileupload", "time": "1h"}, files={"fileToUpload": (path.name, handle)}, timeout=30, proxies=proxies)
            else:
                response = requests.post(upload_url, files={"file": (path.name, handle)}, timeout=30, proxies=proxies)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            payload = response.json()
            url = str(payload.get("url", "")).strip()
            if url:
                return url
        else:
            text = response.text.strip()
            if text.startswith("http://") or text.startswith("https://"):
                return text

        raise RuntimeError(
            f"Upload service did not return a public URL: {upload_url}"
        )
