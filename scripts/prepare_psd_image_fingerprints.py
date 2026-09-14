"""Fingerprint formal-test images for PSD selection; do not evaluate any model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("IFV_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_jsonl, sha256_file, write_json
from src.tools.vision_utils import bounded_pil_image_to_jpeg_bytes


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("storage-root", "test-runtime", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        value.resolve().relative_to(args.storage_root.resolve())
    recipe = {"max_long_edge": 1024, "jpeg_quality": 95}
    tests = load_jsonl(args.test_runtime)

    def normalize(row):
        path = (args.test_runtime.parent / row["image_path"]).resolve()
        path.relative_to(args.storage_root.resolve())
        raw = sha256_file(path)
        if raw != row["image_sha256"]:
            raise ValueError("formal test image changed")
        with Image.open(path) as im:
            encoded, _ = bounded_pil_image_to_jpeg_bytes(im, **recipe)
        return {"case_id": row["case_id"], "image_path": str(path), "raw_image_sha256": raw,
                "normalized_image_sha256": hashlib.sha256(encoded).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        normalized = list(pool.map(normalize, tests))
    result = {"schema_version": "ifv-psd-test-image-fingerprints-v1", "normalization": recipe,
        "test_runtime_sha256": sha256_file(args.test_runtime), "test_cases_checked": len(tests),
        "test_images": normalized, "scope": "Raw and exact 1024/JPEG95 image encoding; not perceptual or semantic deduplication"}
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != "test_images"}, flush=True)


if __name__ == "__main__":
    main()
