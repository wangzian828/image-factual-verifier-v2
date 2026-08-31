"""Export the active Agent policy prompts without translating or rewriting them."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.orchestrator.unified_prompts import (
    UNIFIED_DISCREPANCY_DECISION_PROMPT_VERSION,
    UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
    UNIFIED_JUDGMENT_PROMPT_VERSION,
    UNIFIED_JUDGMENT_SYSTEM_PROMPT,
    UNIFIED_REACT_PROMPT_VERSION,
    UNIFIED_REACT_SYSTEM_PROMPT,
    UNIFIED_REFLECTION_PROMPT_VERSION,
    UNIFIED_REFLECTION_SYSTEM_PROMPT,
)


def main() -> None:
    output = ROOT / "docs" / "active-agent-system-prompts.md"
    sections = [
        (
            "Unified ReAct",
            UNIFIED_REACT_PROMPT_VERSION,
            UNIFIED_REACT_SYSTEM_PROMPT,
        ),
        (
            "Unified Reflection",
            UNIFIED_REFLECTION_PROMPT_VERSION,
            UNIFIED_REFLECTION_SYSTEM_PROMPT,
        ),
        (
            "Unified Discrepancy Decision",
            UNIFIED_DISCREPANCY_DECISION_PROMPT_VERSION,
            UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
        ),
        (
            "Unified Judgment",
            UNIFIED_JUDGMENT_PROMPT_VERSION,
            UNIFIED_JUDGMENT_SYSTEM_PROMPT,
        ),
    ]
    lines = [
        "# Active Agent System Prompts",
        "",
        "This file is generated from "
        "`src/orchestrator/unified_prompts.py`. These are the exact English "
        "policy prompts sent to the active Agent stages.",
        "",
        "Tool-internal prompts are not included; they remain next to their "
        "mature tool implementations.",
        "",
    ]
    for title, version, prompt in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                f"Prompt version: `{version}`",
                "",
                "```text",
                prompt.rstrip(),
                "```",
                "",
            ]
        )
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
