from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.integrations.search.serper import SerperImageSearchClient, SerperLensSearchClient
from src.integrations.search.visual_search import VisualReverseSearchClient
from src.integrations.vlm.factory import build_vlm_client
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool


IMAGE_QUERY_PROMPT = """You will see one image.
Generate a concise web image search query for this image.

Return one JSON object:
{
  "query": "short search query",
  "keywords": ["keyword1", "keyword2"]
}
"""

IMAGE_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "maxLength": 240},
        "keywords": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 6,
        },
    },
}


@dataclass
class ReverseImageSearchTool(BaseTool):
    """Hybrid reverse image search: visual reverse search plus semantic image search."""

    vlm_client: Optional[Any] = None
    image_search_client: Optional[SerperImageSearchClient] = None
    lens_client: Optional[SerperLensSearchClient] = None
    visual_search_client: Optional[VisualReverseSearchClient] = None
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash"
    qwen_model_name: str = "qwen3.6-plus"
    top_k: int = 5
    use_lens: bool = True
    use_vlm_query: bool = True
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "reverse_image_search"
    description: str = (
        "Reverse image search: finds visually similar web pages and images using a visual search provider, "
        "plus generates a semantic search query from the image content. "
        "Returns both visual matches and semantic matches."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_input": {
                    "type": "string",
                    "description": "Local path, URL, or data URL of the image.",
                },
            },
            "required": ["image_input"],
        }
    )

    def __post_init__(self) -> None:
        if self.vlm_client is None:
            self.vlm_client = build_vlm_client(
                provider=self.provider,
                model_name=self.model_name or self.qwen_model_name,
            )
        if self.image_search_client is None:
            self.image_search_client = SerperImageSearchClient()
        if self.lens_client is None:
            self.lens_client = SerperLensSearchClient()
        if self.visual_search_client is None:
            self.visual_search_client = VisualReverseSearchClient(
                serper_lens_client=self.lens_client,
            )

    def search(self, image_input: str) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        results: Dict[str, Any] = {"status": "success", "image_input": image_input}
        timings: Dict[str, Any] = {}

        if self.use_lens and self.use_vlm_query:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_lens = executor.submit(self._run_lens_branch, image_input)
                future_semantic = executor.submit(self._run_semantic_branch, image_input)
                lens_payload = future_lens.result()
                semantic_payload = future_semantic.result()
            self._merge_lens_payload(results, timings, lens_payload)
            self._merge_semantic_payload(results, timings, semantic_payload)
        else:
            if self.use_lens:
                self._merge_lens_payload(results, timings, self._run_lens_branch(image_input))
            if self.use_vlm_query:
                self._merge_semantic_payload(results, timings, self._run_semantic_branch(image_input))

        lens_results = results.get("lens_results", []) or []
        semantic_results = results.get("semantic_results", []) or []
        policy = self.source_access_policy
        if policy is not None:
            lens_results, lens_blocked = policy.filter_rows(lens_results)
            semantic_results, semantic_blocked = policy.filter_rows(semantic_results)
            results["lens_results"] = lens_results
            results["semantic_results"] = semantic_results
            blocked_count = lens_blocked + semantic_blocked
            if blocked_count:
                results["policy_filtered_count"] = blocked_count
        results["candidate_page_urls"] = self._collect_candidate_page_urls(
            lens_results=lens_results,
            semantic_results=semantic_results,
        )
        results["reference_image_candidates"] = self._collect_reference_image_candidates(
            lens_results=lens_results,
            semantic_results=semantic_results,
        )
        if results["reference_image_candidates"]:
            results["reference_image_url"] = results["reference_image_candidates"][0]
        timings["total_ms"] = round((time.perf_counter() - total_t0) * 1000, 2)
        results["timings"] = timings

        errors = []
        lens_error = str(results.get("lens_error", "")).strip()
        if self.use_lens and lens_error:
            errors.append(lens_error)
        vlm_error = str(results.get("vlm_error", "")).strip()
        if self.use_vlm_query and vlm_error and not lens_results and not semantic_results:
            errors.append(vlm_error)
        if errors:
            return {
                **results,
                "status": "error",
                "error": "; ".join(errors),
            }

        return results

    def _run_lens_branch(self, image_input: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "lens_results": [],
            "timings": {},
        }
        try:
            visual = self.visual_search_client.search(image_input, top_k=self.top_k)
            payload.update(
                {
                    "lens_results": visual.get("results", []),
                    "lens_image_url": visual.get("image_url", ""),
                    "visual_search_provider": visual.get("provider", ""),
                    "upload": visual.get("upload", {}),
                    "visual_search_errors": visual.get("errors", {}) if visual.get("errors") else {},
                }
            )
            visual_status = str(visual.get("status", "")).strip().lower()
            if visual_status == "error" or visual.get("error"):
                payload["lens_error"] = (
                    str(visual.get("error", "")).strip()
                    or "Selected visual search provider returned status=error."
                )
            elif visual.get("errors"):
                payload["lens_error"] = "; ".join(
                    f"{name}: {msg}" for name, msg in visual["errors"].items()
                )
            payload["timings"] = dict(visual.get("timings", {}))
        except Exception as exc:
            payload["lens_error"] = f"{type(exc).__name__}: {exc}"
        payload["timings"]["lens_branch_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return payload

    def _run_semantic_branch(self, image_input: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "semantic_results": [],
            "timings": {},
        }
        try:
            query_t0 = time.perf_counter()
            query_payload = self._generate_query(image_input)
            payload["timings"]["vlm_query_ms"] = round((time.perf_counter() - query_t0) * 1000, 2)
            query = str(query_payload.get("query", "")).strip()
            payload["vlm_query"] = query
            if query:
                semantic_t0 = time.perf_counter()
                payload["semantic_results"] = self.image_search_client.search(
                    query=query,
                    top_k=self.top_k,
                )
                payload["timings"]["semantic_search_ms"] = round((time.perf_counter() - semantic_t0) * 1000, 2)
        except Exception as exc:
            payload["vlm_error"] = f"{type(exc).__name__}: {exc}"
        payload["timings"]["semantic_branch_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return payload

    @staticmethod
    def _merge_lens_payload(results: Dict[str, Any], timings: Dict[str, Any], payload: Dict[str, Any]) -> None:
        results["lens_results"] = payload.get("lens_results", [])
        if payload.get("lens_image_url"):
            results["lens_image_url"] = payload["lens_image_url"]
        if payload.get("visual_search_provider"):
            results["visual_search_provider"] = payload["visual_search_provider"]
        if payload.get("upload"):
            results["upload"] = payload["upload"]
        if payload.get("visual_search_errors"):
            results["visual_search_errors"] = payload["visual_search_errors"]
        if payload.get("lens_error"):
            results["lens_error"] = payload["lens_error"]
        timings.update(payload.get("timings", {}))

    @staticmethod
    def _merge_semantic_payload(results: Dict[str, Any], timings: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if payload.get("vlm_query"):
            results["vlm_query"] = payload["vlm_query"]
        results["semantic_results"] = payload.get("semantic_results", [])
        if payload.get("vlm_error"):
            results["vlm_error"] = payload["vlm_error"]
        timings.update(payload.get("timings", {}))

    def _generate_query(self, image_input: str) -> Dict[str, Any]:
        return self.vlm_client.create_image_json(
            system_prompt=IMAGE_QUERY_PROMPT,
            user_text="Generate the best web image search query for this image.",
            image_input=image_input,
            max_tokens=300,
            model_name=self.model_name or self.qwen_model_name,
            response_schema=IMAGE_QUERY_SCHEMA,
        )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.search(params["image_input"])

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy

    @staticmethod
    def _collect_candidate_page_urls(
        *,
        lens_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
    ) -> List[str]:
        urls: List[str] = []
        seen = set()
        for item in (lens_results or []) + (semantic_results or []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url or url in seen or not ReverseImageSearchTool._is_preferred_page_url(url):
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @staticmethod
    def _collect_reference_image_candidates(
        *,
        lens_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
    ) -> List[str]:
        urls: List[str] = []
        seen = set()
        for item in (lens_results or []) + (semantic_results or []):
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url", "")).strip()
            if not image_url or image_url in seen or not ReverseImageSearchTool._is_image_url(image_url):
                continue
            seen.add(image_url)
            urls.append(image_url)
        return urls

    @staticmethod
    def _is_preferred_page_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        host = parsed.netloc.lower()
        path = parsed.path.lower()
        blocked_hosts = (
            "google.com",
            "www.google.com",
            "images.google.com",
            "facebook.com",
            "m.facebook.com",
            "instagram.com",
            "www.instagram.com",
            "pinterest.com",
            "www.pinterest.com",
            "reddit.com",
            "www.reddit.com",
        )
        blocked_exts = (
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
            ".svg",
            ".pdf",
        )
        if any(host == blocked or host.endswith("." + blocked) for blocked in blocked_hosts):
            return False
        if any(path.endswith(ext) for ext in blocked_exts):
            return False
        return True

    @staticmethod
    def _is_image_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        path = parsed.path.lower()
        image_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
        if any(path.endswith(ext) for ext in image_exts):
            return True
        image_hosts = (
            "imgur.com",
            "i.imgur.com",
            "pbs.twimg.com",
            "upload.wikimedia.org",
            "gstatic.com",
            "ggpht.com",
            "ytimg.com",
        )
        return any(host in parsed.netloc.lower() for host in image_hosts)
