#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from ifv_training.encode_cache import (
    CachedEncodeFunction,
    EncodeCacheConfig,
)


def _assert_equal(left: Any, right: Any, path: str = "payload") -> None:
    try:
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(
                right,
                torch.Tensor,
            ):
                raise AssertionError(f"{path}: tensor type mismatch")
            if left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(f"{path}: tensor metadata mismatch")
            if not torch.equal(left, right):
                raise AssertionError(f"{path}: tensor value mismatch")
            return
    except ImportError:
        pass
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise AssertionError(f"{path}: mapping type mismatch")
        if set(left) != set(right):
            raise AssertionError(f"{path}: mapping keys mismatch")
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            raise AssertionError(f"{path}: sequence type mismatch")
        if len(left) != len(right):
            raise AssertionError(f"{path}: sequence length mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_equal(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify the content-addressed ms-swift encode cache."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--loss-scale", required=True)
    parser.add_argument(
        "--add-non-thinking-prefix",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument("--verify-rows", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    from swift import get_processor, get_template

    processor = get_processor(args.model)
    template = get_template(
        processor,
        max_length=args.max_length,
        loss_scale=args.loss_scale,
        add_non_thinking_prefix=args.add_non_thinking_prefix == "true",
    )
    template.set_mode("train")
    cached_encode = CachedEncodeFunction(
        template.encode,
        EncodeCacheConfig.from_environment(),
    )

    direct_seconds = 0.0
    cached_seconds = 0.0
    row_count = 0
    verified_rows = 0
    split_rows: dict[str, int] = {}
    for dataset_path in args.dataset:
        dataset = load_from_disk(str(dataset_path.expanduser().resolve()))
        split_rows[str(dataset_path)] = len(dataset)
        for index, row in enumerate(dataset):
            if verified_rows < args.verify_rows:
                started = time.perf_counter()
                direct = template.encode(row, return_length=True)
                direct_seconds += time.perf_counter() - started
            else:
                direct = None
            started = time.perf_counter()
            cached = cached_encode(row, return_length=True)
            cached_seconds += time.perf_counter() - started
            if direct is not None:
                _assert_equal(direct, cached)
                verified_rows += 1
            row_count += 1

    report = {
        "schema_version": "ifv-ms-swift-encode-cache-prewarm-v1",
        "model": str(Path(args.model).expanduser().resolve()),
        "datasets": split_rows,
        "row_count": row_count,
        "verified_rows": verified_rows,
        "direct_encode_seconds": direct_seconds,
        "cached_call_seconds": cached_seconds,
        "contract_digest": cached_encode.contract_digest,
        "cache_root": str(cached_encode.contract_root),
        "passed": verified_rows == min(args.verify_rows, row_count),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
