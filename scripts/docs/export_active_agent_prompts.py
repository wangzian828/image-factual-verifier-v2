"""Export the active Agent policy prompts without translating or rewriting them."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = runpy.run_path(
    str(ROOT / "src" / "orchestrator" / "unified_prompts.py")
)


def main() -> None:
    output = ROOT / "docs" / "active-agent-system-prompts.md"
    sections = [
        (
            "Unified ReAct",
            PROMPTS["UNIFIED_REACT_PROMPT_VERSION"],
            PROMPTS["UNIFIED_REACT_SYSTEM_PROMPT"],
        ),
        (
            "Unified Judgment",
            PROMPTS["UNIFIED_JUDGMENT_PROMPT_VERSION"],
            PROMPTS["UNIFIED_JUDGMENT_SYSTEM_PROMPT"],
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
