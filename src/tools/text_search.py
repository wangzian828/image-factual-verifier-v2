from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.integrations.browse.jina_reader import JinaReaderClient
from src.integrations.search.serper import SerperTextSearchClient
from src.tools.base import BaseTool


@dataclass
class TextSearchTool(BaseTool):
    """Serper-backed batched web text search."""

    client: Optional[SerperTextSearchClient] = None
    browse_client: Optional[JinaReaderClient] = None
    top_k: int = 10
    visit_top_k: int = 3
    name: str = "text_search"
    description: str = "Search the web for textual evidence related to one or more queries."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "queries": {
                    "type": ["array", "string"],
                    "description": "One query string or a list of query strings.",
                },
                "gl": {"type": "string", "description": "Country code such as us or cn."},
                "hl": {"type": "string", "description": "Language code such as en or zh-cn."},
            },
            "required": ["queries"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SerperTextSearchClient()
        if self.browse_client is None:
            self.browse_client = JinaReaderClient()

    def search(
        self,
        queries: Union[List[str], str],
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return {"status": "error", "error": "At least one non-empty search query is required."}
        responses: List[Dict[str, Any]] = []
        for query in queries:
            responses.append(self._run_single_query(query, gl=gl, hl=hl))
        return self._build_result(responses)

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.search(
            queries=params["queries"],
            gl=params.get("gl"),
            hl=params.get("hl"),
        )

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        queries = params["queries"]
        gl = params.get("gl")
        hl = params.get("hl")
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return {"status": "error", "error": "At least one non-empty search query is required."}
        tasks = [
            asyncio.to_thread(self._run_single_query, query, gl=gl, hl=hl)
            for query in queries
        ]
        return self._build_result(list(await asyncio.gather(*tasks)))

    @staticmethod
    def _build_result(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        errors = [
            str(response.get("visit_error", "")).strip()
            for response in responses
            if str(response.get("visit_error", "")).strip()
        ]
        result: Dict[str, Any] = {"status": "success", "queries": responses}
        if errors:
            result.update({"status": "error", "error": "; ".join(errors)})
        return result

    def _run_single_query(
        self,
        query: str,
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> Dict[str, Any]:
        search_t0 = time.perf_counter()
        response = self.client.search(query, top_k=self.top_k, gl=gl, hl=hl)
        search_duration_ms = round((time.perf_counter() - search_t0) * 1000, 2)
        enriched = self._enrich_with_visits(response, goal=query)
        timings = dict(enriched.get("timings", {}))
        timings["search_ms"] = search_duration_ms
        enriched["timings"] = timings
        return enriched

    def _enrich_with_visits(self, response: Dict[str, Any], *, goal: str) -> Dict[str, Any]:
        enriched = dict(response)
        results = response.get("results", [])
        urls = [
            str(item.get("url", "")).strip()
            for item in results[: self.visit_top_k]
            if isinstance(item, dict) and str(item.get("url", "")).strip()
        ]
        if not urls:
            enriched.update(
                {
                    "visited_pages": [],
                    "evidence": "",
                    "summary": "",
                    "rationale": "No result URLs available for webpage visit.",
                    "relevance": "low",
                    "stance": "unclear",
                    "directness": "none",
                    "artifact_sha256": "",
                    "evidence_span": {},
                    "retrieved_at": "",
                    "injection_flags": [],
                    "evidence_eligible": False,
                    "timings": {"visit_ms": 0.0},
                }
            )
            return enriched

        try:
            visit_t0 = time.perf_counter()
            visit_result = self.browse_client.visit_many(urls, goal)
            visit_duration_ms = round((time.perf_counter() - visit_t0) * 1000, 2)
        except Exception as exc:
            visit_duration_ms = round((time.perf_counter() - visit_t0) * 1000, 2) if "visit_t0" in locals() else 0.0
            visit_result = {
                "visits": [],
                "evidence": "",
                "summary": "",
                "rationale": f"Failed to visit result pages: {exc}",
                "relevance": "low",
                "stance": "unclear",
                "directness": "none",
                "artifact_sha256": "",
                "evidence_span": {},
                "retrieved_at": "",
                "injection_flags": [],
                "evidence_eligible": False,
                "error": str(exc),
            }

        enriched.update(
            {
                "visited_pages": visit_result.get("visits", []),
                "evidence": visit_result.get("evidence", ""),
                "summary": visit_result.get("summary", ""),
                "rationale": visit_result.get("rationale", ""),
                "relevance": visit_result.get("relevance", "low"),
                "stance": visit_result.get("stance", "unclear"),
                "directness": visit_result.get("directness", "none"),
                "artifact_sha256": visit_result.get("artifact_sha256", ""),
                "evidence_span": visit_result.get("evidence_span", {}),
                "retrieved_at": visit_result.get("retrieved_at", ""),
                "injection_flags": visit_result.get("injection_flags", []),
                "evidence_eligible": bool(visit_result.get("evidence_eligible", False)),
                "selected_url": visit_result.get("selected_url", ""),
                "blocked_pages": visit_result.get("blocked_pages", []),
                "timings": {
                    "visit_ms": visit_duration_ms,
                    **(visit_result.get("timings", {}) if isinstance(visit_result.get("timings"), dict) else {}),
                },
            }
        )
        visit_status = str(visit_result.get("status", "")).strip().lower()
        if visit_status == "error" or visit_result.get("error"):
            enriched["visit_error"] = (
                str(visit_result.get("error", "")).strip()
                or "Selected browse provider failed while visiting search results."
            )
        return enriched
