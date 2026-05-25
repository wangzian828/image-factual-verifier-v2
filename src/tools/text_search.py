from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.integrations.search.serper import SerperTextSearchClient
from src.tools.base import BaseTool


@dataclass
class TextSearchTool(BaseTool):
    """Serper-backed batched web text search."""

    client: Optional[SerperTextSearchClient] = None
    top_k: int = 10
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
