from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.integrations.browse.jina_reader import JinaReaderClient
from src.tools.base import BaseTool


@dataclass
class VisitTool(BaseTool):
    """Goal-conditioned webpage visit using Jina Reader."""

    client: Optional[JinaReaderClient] = None
    name: str = "visit"
    description: str = "Visit a webpage with a verification goal and extract a focused snippet."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The webpage URL to visit."},
                "goal": {"type": "string", "description": "The verification goal or target claim."},
            },
            "required": ["url", "goal"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = JinaReaderClient()

    def visit(self, url: str, goal: str) -> dict:
        return self.client.visit(url, goal)

    def call(self, params: dict) -> dict:
        return self.visit(url=params["url"], goal=params["goal"])
