from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.integrations.search.jina_search import JinaSearchClient
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool


@dataclass
class JinaSearchTool(BaseTool):
    """Explicit Jina Search Reader action; it never falls back to Serper."""

    client: Optional[JinaSearchClient] = None
    top_k: int = 10
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "jina_search"
    description: str = (
        "Search the web with Jina Search Reader and return candidate pages plus "
        "content previews. This is an explicit Jina route; failures are reported "
        "and never silently switched to another provider."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "queries": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 1,
                    "description": "Exactly one Jina search query.",
                },
                "goal": {
                    "type": "string",
                    "description": "Immutable task goal recorded with the response.",
                },
            },
            "required": ["queries"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = JinaSearchClient()
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy

    def search(
        self,
        queries: Union[List[str], str],
        *,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(queries, str):
            queries = [queries]
        normalized = [str(item).strip() for item in queries if str(item).strip()]
        if len(normalized) != 1:
            return {
                "status": "error",
                "error": "jina_search accepts exactly one query per action.",
            }
        query = normalized[0]
        if self.source_access_policy is not None:
            blocked = self.source_access_policy.blocked_query_reference(query)
            if blocked:
                return {
                    "status": "error",
                    "error": (
                        "Search query explicitly targets a source excluded by "
                        "the active evaluation policy."
                    ),
                }
        started = time.perf_counter()
        try:
            response = self.client.search(query, top_k=self.top_k)
        except Exception as exc:
            return {
                "status": "error",
                "provider": "jina_search",
                "error": f"{type(exc).__name__}: {exc}",
                "subcalls": [
                    {
                        "kind": "search_query",
                        "provider": "jina_search",
                        "status": "error",
                        "request_count": 1,
                        "result_count": 0,
                    }
                ],
            }
        rows = list(response.get("results", []) or [])
        if self.source_access_policy is not None:
            rows, blocked_count = self.source_access_policy.filter_rows(rows)
        else:
            blocked_count = 0
        response = {
            **response,
            "goal": str(goal or query),
            "results": rows[: self.top_k],
            "subcalls": [
                {
                    "kind": "search_query",
                    "provider": "jina_search",
                    "status": "success",
                    "request_count": 1,
                    "result_count": len(rows),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                }
            ],
        }
        if blocked_count:
            response["policy_filtered_count"] = blocked_count
        return {"status": "success", **response}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.search(
            queries=params["queries"],
            goal=params.get("goal"),
        )

    async def call_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.call, params)
