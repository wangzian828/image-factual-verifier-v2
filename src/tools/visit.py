from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.integrations.browse.jina_reader import JinaReaderClient
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool


@dataclass
class VisitTool(BaseTool):
    """Goal-conditioned webpage visit using Jina Reader."""

    client: Optional[JinaReaderClient] = None
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "visit"
    description: str = (
        "Visit one or more webpages with a verification goal and extract "
        "goal-conditioned evidence, rationale, and summary."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": ["string", "array"],
                    "description": "One webpage URL or a list of URLs to visit.",
                },
                "goal": {"type": "string", "description": "The verification goal or target claim."},
            },
            "required": ["url", "goal"],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = JinaReaderClient()
        if self.source_access_policy is not None:
            self.set_source_access_policy(self.source_access_policy)

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        self.source_access_policy = policy
        setter = getattr(self.client, "set_source_access_policy", None)
        if callable(setter):
            setter(policy)

    def visit(self, url: Any, goal: str) -> dict:
        try:
            if isinstance(url, list):
                urls = [str(item) for item in url if str(item).strip()]
                if self.source_access_policy is not None:
                    urls = [item for item in urls if self.source_access_policy.allows(item)]
                if not urls:
                    return {
                        "status": "error",
                        "error": "All requested URLs are blocked by the active source access policy.",
                    }
                result = self.client.visit_many(urls, goal)
            else:
                if self.source_access_policy is not None and not self.source_access_policy.allows(str(url)):
                    return {
                        "status": "error",
                        "error": "URL blocked by the active source access policy.",
                    }
                result = self.client.visit(str(url), goal)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        status = str(result.get("status", "")).strip().lower()
        visits = result.get("visits", [])
        all_visits_failed = bool(visits) and all(
            isinstance(visit, dict)
            and (
                str(visit.get("status", "")).strip().lower() == "error"
                or bool(str(visit.get("error", "")).strip())
                or bool(visit.get("blocked"))
            )
            for visit in visits
        )
        if status == "error" or result.get("error") or all_visits_failed:
            nested_errors = [
                str(visit.get("error", "")).strip()
                for visit in visits
                if isinstance(visit, dict) and str(visit.get("error", "")).strip()
            ]
            error = (
                str(result.get("error", "")).strip()
                or "; ".join(nested_errors)
                or "All webpage visits failed."
            )
            return {**result, "status": "error", "error": error}
        return {**result, "status": "success"}

    def call(self, params: dict) -> dict:
        return self.visit(url=params["url"], goal=params["goal"])
