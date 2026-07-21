from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GENERATION_PREFILL = "{{- '<|im_start|>assistant\\n<think>\\n' }}"
PARSER_COMPATIBLE_PREFILL = "{{- '<|im_start|>assistant\\n' }}"


def prepare(model: Path, output: Path) -> dict[str, object]:
    tokenizer_config = model / "tokenizer_config.json"
    config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"chat_template is missing from {tokenizer_config}")
    if template.count(GENERATION_PREFILL) != 1:
        raise RuntimeError(
            "expected exactly one Qwen3-VL generation-time <think> prefill, got "
            f"{template.count(GENERATION_PREFILL)}"
        )

    # The Thinking checkpoint emits <think> itself. vLLM 0.11.2's qwen3 parser
    # needs that opening token in generated output, rather than in the prompt.
    rendered = template.replace(GENERATION_PREFILL, PARSER_COMPATIBLE_PREFILL)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return {
        "source": str(tokenizer_config.resolve()),
        "output": str(output.resolve()),
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "removed_generation_think_prefill": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = prepare(args.model, args.output)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
