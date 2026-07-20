from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.integrations.browse.jina_reader import JinaReaderClient
from src.integrations.gemini import RUNTIME_METRICS_KEY, exception_runtime_metrics
from src.orchestrator.source_access import SourceAccessPolicy
from src.tools.base import BaseTool


@dataclass
class VisitTool(BaseTool):
    """Goal-conditioned webpage visit using Jina Reader."""

    client: Optional[JinaReaderClient] = None
    source_access_policy: Optional[SourceAccessPolicy] = None
    name: str = "visit"
    description: str = (
        "Visit one webpage and extract an exact passage for the runtime-bound "
        "image claim and retrieval goal."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 1,
                    "description": (
                        "Exactly one webpage URL. Additional pages require "
                        "separate policy actions."
                    ),
                },
                "image_claim": {
                    "type": "string",
                    "description": "Runtime-bound image claim used only for evidence stance.",
                },
                "retrieval_goal": {
                    "type": "string",
                    "description": "Runtime-bound goal used only to select relevant passages.",
                },
            },
            "required": ["url", "image_claim", "retrieval_goal"],
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

    def visit(
        self,
        url: Any,
        *,
        image_claim: str,
        retrieval_goal: str,
    ) -> dict:
        try:
            if isinstance(url, list):
                urls = [str(item) for item in url if str(item).strip()]
                if len(urls) != 1:
                    return {
                        "status": "error",
                        "error": (
                            "visit accepts exactly one URL per action; inspect "
                            "additional pages only if the core gap remains open."
                        ),
                    }
                if self.source_access_policy is not None:
                    urls = [item for item in urls if self.source_access_policy.allows(item)]
                if not urls:
                    return {
                        "status": "error",
                        "error": "All requested URLs are blocked by the active source access policy.",
                    }
                result = self.client.visit(
                    urls[0],
                    image_claim=image_claim,
                    retrieval_goal=retrieval_goal,
                )
            else:
                if self.source_access_policy is not None and not self.source_access_policy.allows(str(url)):
                    return {
                        "status": "error",
                        "error": "URL blocked by the active source access policy.",
                    }
                result = self.client.visit(
                    str(url),
                    image_claim=image_claim,
                    retrieval_goal=retrieval_goal,
                )
        except Exception as exc:
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            metrics = exception_runtime_metrics(exc)
            if metrics:
                result[RUNTIME_METRICS_KEY] = metrics
            return result

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
        return self.visit(
            url=params["url"],
            image_claim=params["image_claim"],
            retrieval_goal=params["retrieval_goal"],
        )
