from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.integrations.search.serper import SerperImageSearchClient, SerperLensSearchClient
from src.integrations.vlm.factory import build_vlm_client
from src.tools.base import BaseTool


IMAGE_QUERY_PROMPT = """你会看到一张图像。请为这张图像生成适合做网页图片搜索的查询词。
要求：
1. 优先生成实体名、地点名、地标名、事件名、品牌名
2. 不要写成完整句子
3. 尽量简洁，但保留区分性
4. 只输出一个 JSON 对象，格式为：
{
  "query": "搜索词",
  "keywords": ["关键词1", "关键词2"]
}
"""


@dataclass
class ReverseImageSearchTool(BaseTool):
    """Hybrid reverse image search: Google Lens (visual) + VLM query (semantic).

    Combines two strategies:
    1. Serper Lens: true reverse image search (upload image → find visually similar pages)
    2. VLM + Image Search: generate text query from image → search for related images

    This gives both visual matches (exact/near-duplicate detection) and semantic matches.
    """

    vlm_client: Optional[Any] = None
    image_search_client: Optional[SerperImageSearchClient] = None
    lens_client: Optional[SerperLensSearchClient] = None
    provider: str = "lmdeploy"
    model_name: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking"
    qwen_model_name: str = "/gsdata/home/wza/models/Qwen3-VL-8B-Thinking"
    top_k: int = 5
    use_lens: bool = True
    use_vlm_query: bool = True
    name: str = "reverse_image_search"
    description: str = (
        "Reverse image search: finds visually similar web pages and images using Google Lens, "
        "plus generates a semantic search query from the image content. "
        "Returns both visual matches (exact/near-duplicate pages) and semantic matches."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {"type": "string", "description": "Local path, URL, or data URL of the image."},
            },
            "required": ["image_input"],
        }
    )

    def __post_init__(self) -> None:
        if self.vlm_client is None:
            self.vlm_client = build_vlm_client(provider=self.provider, model_name=self.model_name or self.qwen_model_name)
        if self.image_search_client is None:
            self.image_search_client = SerperImageSearchClient()
        if self.lens_client is None:
            self.lens_client = SerperLensSearchClient()

    def search(self, image_input: str) -> Dict[str, Any]:
        """Run hybrid reverse image search."""
        results: Dict[str, Any] = {"image_input": image_input}

        # Strategy 1: Google Lens (true visual reverse search)
        if self.use_lens:
            try:
                image_url = self._prepare_remote_url(image_input)
                lens_results = self.lens_client.search(image_url=image_url, top_k=self.top_k)
                results["lens_results"] = lens_results
                results["lens_image_url"] = image_url
            except Exception as e:
                results["lens_error"] = f"{type(e).__name__}: {str(e)}"
                results["lens_results"] = []

        # Strategy 2: VLM generates query → image search (semantic)
        if self.use_vlm_query:
            try:
                query_payload = self._generate_query(image_input)
                query = query_payload.get("query", "").strip()
                if query:
                    semantic_results = self.image_search_client.search(query=query, top_k=self.top_k)
                    results["vlm_query"] = query
                    results["semantic_results"] = semantic_results
                else:
                    results["semantic_results"] = []
            except Exception as e:
                results["vlm_error"] = f"{type(e).__name__}: {str(e)}"
                results["semantic_results"] = []

        return results

    def _generate_query(self, image_input: str) -> Dict[str, Any]:
        return self.vlm_client.create_image_json(
            system_prompt=IMAGE_QUERY_PROMPT,
            user_text="请根据这张图像生成最适合的图片搜索词。",
            image_input=image_input,
            max_tokens=300,
            model_name=self.model_name or self.qwen_model_name,
        )

    def _prepare_remote_url(self, image_input: str) -> str:
        """Upload local image to get a public URL for Lens search."""
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return image_input
        # data URL — not supported by Lens, skip
        if image_input.startswith("data:"):
            raise ValueError("Data URLs not supported for Lens search; use a file path or HTTP URL.")

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

        raise RuntimeError(f"Upload service did not return a public URL: {upload_url}")

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.search(params["image_input"])
