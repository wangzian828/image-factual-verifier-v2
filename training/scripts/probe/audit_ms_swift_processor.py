from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LENGTH_CANDIDATES = (4096, 8192, 16384, 32768, 65536, 131072)


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{index + 1} is not an object")
                yield index, value


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else 0,
        "mean": statistics.mean(values) if values else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode every accepted SFT row with the real ms-swift processor."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from swift import get_processor, get_template

    processor = get_processor(args.model)
    template = get_template(
        processor,
        loss_scale="default+ignore_empty_think",
    )
    template.set_mode("train")

    datasets = []
    for split in ("train", "validation"):
        datasets.append(("policy", split, args.policy_dir / f"{split}.jsonl"))
        datasets.append(
            ("perception", split, args.perception_dir / f"{split}.jsonl")
        )

    lengths: list[int] = []
    trainable_lengths: list[int] = []
    by_channel: dict[str, list[int]] = defaultdict(list)
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    longest: list[dict[str, Any]] = []
    for dataset_kind, split, path in datasets:
        if not path.is_file():
            errors.append({"path": str(path), "error": "missing dataset"})
            continue
        for row_index, data in _rows(path):
            location = f"{path.name}[{row_index}]"
            channel = str(data.get("channel") or dataset_kind)
            try:
                encoded = template.encode(data, return_template_inputs=True)
                input_ids = encoded["input_ids"]
                labels = encoded["labels"]
                if len(input_ids) != len(labels):
                    raise ValueError("input_ids and labels have different lengths")
                loss_positions = [
                    index for index, value in enumerate(labels) if value != -100
                ]
                if not loss_positions:
                    raise ValueError("row has no trainable assistant token")
                template_inputs = encoded.get("template_inputs")
                encoded_images = list(
                    getattr(template_inputs, "images", []) or []
                )
                expected_images = list(data.get("images") or [])
                if expected_images and not encoded_images:
                    raise ValueError("multimodal row lost its image")
                if not expected_images and encoded_images:
                    raise ValueError("text-only row unexpectedly gained an image")
                token_count = len(input_ids)
                trainable_count = len(loss_positions)
                lengths.append(token_count)
                trainable_lengths.append(trainable_count)
                by_channel[channel].append(token_count)
                counts[f"{split}:{channel}"] += 1
                longest.append(
                    {
                        "dataset_kind": dataset_kind,
                        "split": split,
                        "channel": channel,
                        "row_index": row_index,
                        "input_tokens": token_count,
                        "trainable_tokens": trainable_count,
                        "first_loss_index": loss_positions[0],
                        "last_loss_index": loss_positions[-1],
                        "image_count": len(encoded_images),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "dataset_kind": dataset_kind,
                        "split": split,
                        "channel": channel,
                        "location": location,
                        "error": str(exc),
                    }
                )

    maximum = max(lengths) if lengths else 0
    selected_max_length = next(
        (candidate for candidate in LENGTH_CANDIDATES if maximum <= candidate),
        None,
    )
    if selected_max_length is None:
        errors.append(
            {
                "error": (
                    f"maximum encoded length {maximum} exceeds the frozen "
                    f"candidate ceiling {LENGTH_CANDIDATES[-1]}"
                )
            }
        )
    report = {
        "schema_version": "ifv-ms-swift-processor-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "processor_class": type(processor).__name__,
        "template_class": type(template).__name__,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "row_count": len(lengths),
        "counts": dict(sorted(counts.items())),
        "input_tokens": _distribution(lengths),
        "trainable_tokens": _distribution(trainable_lengths),
        "input_tokens_by_channel": {
            channel: _distribution(values)
            for channel, values in sorted(by_channel.items())
        },
        "length_candidates": list(LENGTH_CANDIDATES),
        "selected_max_length": selected_max_length,
        "longest_rows": sorted(
            longest,
            key=lambda item: int(item["input_tokens"]),
            reverse=True,
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
