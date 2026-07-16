from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    from swift import get_processor, get_template

    lines = [
        line
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data = json.loads(lines[args.row])
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
