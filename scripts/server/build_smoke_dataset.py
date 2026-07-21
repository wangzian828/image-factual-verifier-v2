from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    image_path = args.output / "synthetic-shapes.png"
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 64, 224, 240), fill="royalblue")
    draw.ellipse((288, 256, 464, 432), fill="darkorange")
    draw.text((64, 288), "IFV SMOKE", fill="black")
    image.save(image_path)

    assistant = {
        "scene_description": "A blue square and an orange circle on a white background.",
        "image_type": "synthetic diagram",
        "entities": [
            {"label": "blue square", "bbox": [48, 64, 224, 240]},
            {"label": "orange circle", "bbox": [288, 256, 464, 432]},
        ],
        "positioned_text": [{"text": "IFV SMOKE", "bbox": [64, 288, 200, 320]}],
        "uncertainty": [],
    }
    row = {
        "messages": [
            {
                "role": "user",
                "content": "<image>Return one JSON object describing only literal visible content.",
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True),
                "loss": True,
            },
        ],
        "images": [str(image_path.resolve())],
        "channel": "perception",
        "model_mode": "thinking",
    }
    _write_jsonl(args.output / "train.jsonl", [row, row])
    _write_jsonl(args.output / "validation.jsonl", [row])
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "ifv-qwen3-vl-smoke-dataset-v1",
        "dataset_version": "ifv-qwen3-vl-smoke-v1",
        "synthetic": True,
        "image": {"path": image_path.name, "sha256": digest},
        "splits": {"train": 2, "validation": 1},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
