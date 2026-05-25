from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.integrations.search.serper import SerperImageSearchClient
from src.tools.base import BaseTool


@dataclass
class ImageSearchTool(BaseTool):
    """Text-to-image web search using Serper Images."""

    client: Optional[SerperImageSearchClient] = None
    top_k: int = 5
    name: str = "image_search"
    description: str = "Search the web for reference images using a textual query."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The image search query."},
                "gl": {"type": "string", "description": "Country code such as us or cn."},
                "hl": {"type": "string", "description": "Language code such as en or zh-cn."},
            },
            "required": ["query"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SerperImageSearchClient()

    def search(
        self,
        query: str,
        *,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.client.search(query, top_k=self.top_k, gl=gl, hl=hl)

    def call(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.search(
            query=params["query"],
            gl=params.get("gl"),
            hl=params.get("hl"),
        )
