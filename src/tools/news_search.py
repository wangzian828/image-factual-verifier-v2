from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.integrations.search.serper import SerperNewsSearchClient
from src.tools.base import BaseTool


@dataclass
class NewsSearchTool(BaseTool):
    """Serper-backed news search - optimized for recent/breaking events."""

    client: Optional[SerperNewsSearchClient] = None
    top_k: int = 10
    name: str = "news_search"
    description: str = (
        "Search recent news articles. Use this for events that happened recently "
        "(within days or weeks). Returns news headlines, snippets, sources, and dates."
    )
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
            self.client = SerperNewsSearchClient()

    def search(
        self,
        queries: Union[List[str], str],
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return []
        return [self.client.search(query, top_k=self.top_k, gl=gl, hl=hl) for query in queries]

    def call(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.search(
            queries=params["queries"],
            gl=params.get("gl"),
            hl=params.get("hl"),
        )
