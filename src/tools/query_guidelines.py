# -*- coding: utf-8 -*-
"""Query Guidelines Tool: on-demand verification knowledge for the agent.

The agent calls `query_guidelines(guideline_name=<name>)` to retrieve
domain-specific verification instructions. Guidelines are stored as
plain Markdown files under `knowledge/guidelines/`.

This follows the same pattern as GenEvolve's `query_knowledge` tool —
instead of stuffing all rules into the system prompt, the agent loads
relevant guidelines on demand when it encounters specific verification
challenges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


GUIDELINE_NAMES: List[str] = [
    "fake_news_detection",
    "image_manipulation",
    "source_credibility",
    "temporal_verification",
    "quote_attribution",
    "statistical_claims",
    "identity_verification",
    "screenshot_authenticity",
]


def _default_guidelines_dir() -> Path:
    """Return the directory holding the guideline markdown files."""
    return Path(__file__).resolve().parent.parent.parent / "knowledge" / "guidelines"


class GuidelineBank:
    """Loads and caches markdown guideline files."""

    def __init__(self, guidelines_dir: Optional[str] = None) -> None:
        self.guidelines_dir = Path(guidelines_dir).resolve() if guidelines_dir else _default_guidelines_dir()
        self.guidelines: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.guidelines_dir.exists():
            self.guidelines_dir.mkdir(parents=True, exist_ok=True)
        for name in GUIDELINE_NAMES:
            md_path = self.guidelines_dir / f"{name}.md"
            if md_path.exists():
                self.guidelines[name] = md_path.read_text(encoding="utf-8")

    def get(self, name: str) -> Optional[str]:
        return self.guidelines.get(name)

    def available(self) -> List[str]:
        return sorted(self.guidelines.keys())


@dataclass
class QueryGuidelinesTool(BaseTool):
    """Tool that provides verification guidelines to the agent on demand."""

    guideline_bank: Optional[GuidelineBank] = None
    guidelines_dir: Optional[str] = None
    name: str = "query_guidelines"
    description: str = (
        "Get expert verification guidance for a specific type of claim or image. "
        "Use this when you need structured rules for how to verify a particular "
        "type of content (fake news, manipulated images, quotes, statistics, etc.)."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "guideline_name": {
                    "type": "string",
                    "enum": GUIDELINE_NAMES,
                    "description": (
                        "Which verification guideline to retrieve. Options: "
                        + ", ".join(GUIDELINE_NAMES)
                    ),
                },
            },
            "required": ["guideline_name"],
        }
    )

    def __post_init__(self) -> None:
        if self.guideline_bank is None:
            self.guideline_bank = GuidelineBank(guidelines_dir=self.guidelines_dir)

    def call(self, params: Dict[str, Any]) -> str:
        guideline_name = str(params.get("guideline_name", "")).strip()

        if not guideline_name or guideline_name not in GUIDELINE_NAMES:
            available = self.guideline_bank.available() if self.guideline_bank else []
            return (
                f"Unknown guideline '{guideline_name}'. "
                f"Available guidelines: {', '.join(available or GUIDELINE_NAMES)}"
            )

        instructions = self.guideline_bank.get(guideline_name)
        if not instructions:
            return (
                f"No content available for '{guideline_name}'. "
                f"The guideline file has not been created yet."
            )

        return f"## Verification Guideline: {guideline_name}\n\n{instructions}"
