"""Benchmark one CPU PaddleOCR profile on a set of local images.

Run one profile per process because the runtime deliberately shares one OCR
reader per process:

    python scripts/benchmark_ocr_profiles.py --profile default image.jpg
    python scripts/benchmark_ocr_profiles.py --profile mobile image.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.ocr_with_position import OCRWithPositionTool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("default", "mobile"),
        required=True,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
    )
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    os.environ["PADDLEOCR_PROFILE"] = args.profile
    tool = OCRWithPositionTool()
    rows = []
    for raw_path in args.images:
        image_path = str(Path(raw_path).expanduser().resolve())
        timings = []
        last_result = {}
        for _ in range(args.repeats):
            started = time.perf_counter()
            last_result = tool.call({"image_input": image_path})
            timings.append(round(time.perf_counter() - started, 4))
        rows.append(
            {
                "image_path": image_path,
                "status": last_result.get("status"),
                "profile": last_result.get("ocr_model_profile", args.profile),
                "regions": last_result.get("total_regions", 0),
                "timings_seconds": timings,
                "median_seconds": statistics.median(timings),
                "full_text": str(last_result.get("full_text", ""))[:500],
            }
        )

    print(
        json.dumps(
            {
                "profile": args.profile,
                "repeats": args.repeats,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
