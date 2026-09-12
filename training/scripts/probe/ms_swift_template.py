from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(path: Path, row_index: int) -> dict:
    if row_index < 0:
        raise IndexError("row index must be non-negative")
    current = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if current == row_index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{current + 1} is not an object")
                return value
            current += 1
    raise IndexError(f"row index {row_index} is outside {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    from swift import get_processor, get_template

    data = _row(args.dataset, args.row)
    processor = get_processor(args.model)
    template = get_template(processor, loss_scale="default+ignore_empty_think")
    template.set_mode("train")
    encoded = template.encode(data, return_template_inputs=True)
    labels = encoded["labels"]
    trainable = sum(int(value != -100) for value in labels)
    template_inputs = encoded.get("template_inputs")
    images = list(getattr(template_inputs, "images", []) or [])
    image_seqlen = encoded.get("image_seqlen")
    if image_seqlen is None and template_inputs is not None:
        image_seqlen = getattr(template_inputs, "image_seqlen", None)
    report = {
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "row": args.row,
        "input_token_count": len(encoded["input_ids"]),
        "trainable_token_count": trainable,
        "image_count": len(images),
        "image_seqlen": image_seqlen,
        "labels_preview": template.safe_decode(labels),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if trainable <= 0:
        raise SystemExit("no trainable assistant tokens")
    if data.get("images") and not images:
        raise SystemExit("multimodal row lost its image in the ms-swift template")


if __name__ == "__main__":
    main()
