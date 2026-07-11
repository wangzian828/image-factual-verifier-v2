#!/usr/bin/env python3
"""Checkpointed deterministic prefilter for real single-image candidates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from src.storage import data_path


ROUTES = ("core", "external_claim_transfer", "excluded", "needs_review")
_CHECKPOINT_LOCK = threading.Lock()


def _default_candidates() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/averimatec/candidates.jsonl",
        "data/benchmark_candidates/real_seed_v0/averimatec/candidates.jsonl",
    )


def _default_output_dir() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/prefilter",
        "data/benchmark_candidates/real_seed_v0/prefilter",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(_default_candidates()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument(
        "--ocr-gpu",
        action="store_true",
        help="Use the visible CUDA device for sequential EasyOCR extraction.",
    )
    parser.add_argument("--check-urls", action="store_true")
    parser.add_argument("--url-workers", type=int, default=8)
    parser.add_argument("--url-timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _load_checkpoint(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl(path):
        value = str(row.get(key, ""))
        if value:
            result[value] = row
    return result


def _append_checkpoint(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(row), ensure_ascii=False) + "\n"
    with _CHECKPOINT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _dhash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    pixels = list(normalized.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def _inspect_asset(asset: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(asset.get("image_path", "")))
    base = {
        "image_path": str(path),
        "sha256": str(asset.get("sha256", "")),
        "valid": False,
        "error": "",
    }
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            return {
                **base,
                "valid": True,
                "width": width,
                "height": height,
                "format": image.format,
                "mode": image.mode,
                "megapixels": round((width * height) / 1_000_000, 4),
                "aspect_ratio": round(width / height, 6) if height else None,
                "dhash": _dhash(image),
                "low_resolution": min(width, height) < 240 or width * height < 120_000,
            }
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"}


def _build_ocr_reader(use_gpu: bool):
    import easyocr

    storage = os.getenv("EASYOCR_MODULE_PATH", "").strip() or None
    kwargs: Dict[str, Any] = {
        "lang_list": ["ch_sim", "en"],
        "gpu": use_gpu,
        "verbose": False,
    }
    if storage:
        kwargs["model_storage_directory"] = str(Path(storage) / "model")
        kwargs["user_network_directory"] = str(Path(storage) / "user_network")
    return easyocr.Reader(**kwargs)


def _ocr_asset(reader: Any, asset: Mapping[str, Any]) -> Dict[str, Any]:
    sha256 = str(asset.get("sha256", ""))
    path = str(asset.get("image_path", ""))
    try:
        raw = reader.readtext(path, detail=1, paragraph=False)
        regions = []
        text_parts = []
        high_confidence = 0
        for bbox, text, confidence in raw[:100]:
            cleaned = " ".join(str(text).split())
            if not cleaned:
                continue
            confidence_value = float(confidence)
            if confidence_value >= 0.4:
                high_confidence += 1
            text_parts.append(cleaned)
            regions.append(
                {
                    "text": cleaned,
                    "confidence": round(confidence_value, 4),
                    "bbox": [[round(float(x), 2), round(float(y), 2)] for x, y in bbox],
                }
            )
        full_text = " ".join(text_parts)[:4000]
        return {
            "sha256": sha256,
            "image_path": path,
            "status": "success",
            "full_text": full_text,
            "character_count": len(full_text.replace(" ", "")),
            "region_count": len(regions),
            "high_confidence_region_count": high_confidence,
            "visible_text_signal": high_confidence > 0 and len(full_text) >= 4,
            "regions": regions,
        }
    except Exception as exc:
        return {
            "sha256": sha256,
            "image_path": path,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "full_text": "",
            "character_count": 0,
            "region_count": 0,
            "high_confidence_region_count": 0,
            "visible_text_signal": False,
            "regions": [],
        }


def _unique_assets(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    assets: Dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        for asset in candidate.get("claim_assets") or []:
            if not isinstance(asset, dict):
                continue
            key = str(asset.get("sha256") or asset.get("image_path") or "")
            if key:
                assets.setdefault(key, asset)
    return assets


def _hamming(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _duplicate_groups(
    inspected: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    valid = {
        key: row
        for key, row in inspected.items()
        if row.get("valid") and row.get("dhash")
    }
    union = _UnionFind(valid)
    keys = sorted(valid)
    for index, left_key in enumerate(keys):
        left = valid[left_key]
        for right_key in keys[index + 1 :]:
            right = valid[right_key]
            if left.get("sha256") and left.get("sha256") == right.get("sha256"):
                union.union(left_key, right_key)
                continue
            left_ratio = float(left.get("aspect_ratio") or 0.0)
            right_ratio = float(right.get("aspect_ratio") or 0.0)
            if not left_ratio or not right_ratio:
                continue
            if abs(math.log(left_ratio / right_ratio)) > 0.04:
                continue
            if _hamming(str(left["dhash"]), str(right["dhash"])) <= 4:
                union.union(left_key, right_key)

    grouped: Dict[str, List[str]] = defaultdict(list)
    for key in keys:
        grouped[union.find(key)].append(key)
    result: Dict[str, Dict[str, Any]] = {}
    group_number = 0
    for members in sorted(grouped.values(), key=lambda values: values[0]):
        if len(members) < 2:
            continue
        group_number += 1
        group_id = f"dup-{group_number:04d}"
        for member_index, member in enumerate(sorted(members)):
            result[member] = {
                "duplicate_group": group_id,
                "duplicate_group_size": len(members),
                "duplicate_representative": member_index == 0,
            }
    return result


def _candidate_urls(candidate: Mapping[str, Any]) -> List[str]:
    values = [candidate.get("source_article_url")]
    values.extend(candidate.get("evidence_urls") or [])
    seen = set()
    result = []
    for value in values:
        url = str(value or "").strip()
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result[:5]


def _check_url(url: str, timeout: float) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            allow_redirects=True,
            stream=True,
            timeout=(10, timeout),
            headers={"User-Agent": "image-factual-verifier/real-seed-prefilter"},
        )
        status_code = response.status_code
        final_url = response.url
        response.close()
        if 200 <= status_code < 400:
            status = "reachable"
        elif status_code in {401, 403, 429}:
            status = "restricted"
        elif status_code in {404, 410}:
            status = "missing"
        else:
            status = "error"
        return {
            "url": url,
            "status": status,
            "status_code": status_code,
            "final_url": final_url,
            "error": "",
        }
    except Exception as exc:
        return {
            "url": url,
            "status": "error",
            "status_code": None,
            "final_url": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _suggest_route(
    candidate: Mapping[str, Any],
    asset_rows: Sequence[Mapping[str, Any]],
    ocr_rows: Sequence[Mapping[str, Any]],
    url_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, List[str], int]:
    reasons: List[str] = []
    routing = candidate.get("benchmark_routing") or {}
    source_metadata = candidate.get("source_metadata") or {}
    valid_assets = [row for row in asset_rows if row.get("valid")]
    if not asset_rows or not valid_assets:
        return "excluded", ["missing_or_invalid_claim_image"], 0
    if routing.get("person_identity_review_required"):
        return "excluded", ["requires_person_identity_review"], 5

    image_questions = int(candidate.get("image_question_count") or 0)
    evidence_count = len(candidate.get("evidence_urls") or [])
    visible_text = any(row.get("visible_text_signal") for row in ocr_rows)
    low_resolution = all(row.get("low_resolution") for row in valid_assets)
    misuse_signal = bool(source_metadata.get("image_misuse_types"))
    transcription = bool(str(source_metadata.get("transcription") or "").strip())
    reachable = sum(row.get("status") == "reachable" for row in url_rows)
    missing = sum(row.get("status") == "missing" for row in url_rows)

    if low_resolution:
        reasons.append("all_claim_images_low_resolution")
    if image_questions:
        reasons.append("image_related_investigation_present")
    else:
        reasons.append("no_image_related_question")
    if misuse_signal:
        reasons.append("image_misuse_annotation_present")
    if visible_text:
        reasons.append("visible_text_signal_present")
    if transcription:
        reasons.append("dataset_transcription_present")
    if evidence_count:
        reasons.append("evidence_urls_present")
    else:
        reasons.append("no_evidence_urls")
    if url_rows and reachable == 0:
        reasons.append("no_checked_url_reachable")
    if missing:
        reasons.append("one_or_more_checked_urls_missing")

    if image_questions == 0 and not misuse_signal and not visible_text:
        return "external_claim_transfer", reasons, 35
    if image_questions > 0 and evidence_count > 0 and visible_text and not low_resolution:
        return "core", reasons, 100
    if image_questions > 0 or misuse_signal:
        return "needs_review", reasons, 75
    return "external_claim_transfer", reasons, 40


def prefilter(
    *,
    candidates_path: Path,
    output_dir: Path,
    run_ocr: bool,
    ocr_gpu: bool,
    check_urls: bool,
    url_workers: int,
    url_timeout: float,
    limit: int | None,
) -> Dict[str, Any]:
    candidates = _load_jsonl(candidates_path)
    if limit is not None:
        candidates = candidates[: max(0, limit)]
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = _unique_assets(candidates)
    inspected = {key: _inspect_asset(asset) for key, asset in assets.items()}
    duplicate_info = _duplicate_groups(inspected)

    ocr_checkpoint_path = output_dir / "ocr_checkpoint.jsonl"
    ocr_cache = _load_checkpoint(ocr_checkpoint_path, "sha256")
    if run_ocr:
        reader = _build_ocr_reader(ocr_gpu)
        for key, asset in assets.items():
            sha256 = str(asset.get("sha256") or key)
            if sha256 in ocr_cache:
                continue
            result = _ocr_asset(reader, asset)
            _append_checkpoint(ocr_checkpoint_path, result)
            ocr_cache[sha256] = result

    url_checkpoint_path = output_dir / "url_checkpoint.jsonl"
    url_cache = _load_checkpoint(url_checkpoint_path, "url")
    if check_urls:
        pending = sorted(
            {
                url
                for candidate in candidates
                for url in _candidate_urls(candidate)
                if url not in url_cache
            }
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, url_workers)
        ) as executor:
            futures = {
                executor.submit(_check_url, url, url_timeout): url for url in pending
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                _append_checkpoint(url_checkpoint_path, result)
                url_cache[result["url"]] = result

    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        asset_keys = [
            str(asset.get("sha256") or asset.get("image_path") or "")
            for asset in candidate.get("claim_assets") or []
            if isinstance(asset, dict)
        ]
        asset_rows = [inspected[key] for key in asset_keys if key in inspected]
        ocr_rows = [ocr_cache[key] for key in asset_keys if key in ocr_cache]
        url_rows = [
            url_cache[url] for url in _candidate_urls(candidate) if url in url_cache
        ]
        route, reasons, priority = _suggest_route(
            candidate, asset_rows, ocr_rows, url_rows
        )
        duplicates = [duplicate_info[key] for key in asset_keys if key in duplicate_info]
        duplicate = duplicates[0] if duplicates else {}
        if duplicate and not duplicate.get("duplicate_representative"):
            reasons.append("near_duplicate_non_representative")
            priority = max(0, priority - 25)
        if candidate.get("ground_truth") == "real":
            priority += 10
        elif candidate.get("ground_truth") == "unverifiable":
            priority += 5

        rows.append(
            {
                "schema_version": "real-seed-prefilter-v1",
                "sample_id": candidate.get("sample_id"),
                "ground_truth": candidate.get("ground_truth"),
                "original_label": candidate.get("original_label"),
                "suggested_route": route,
                "review_priority": priority,
                "reasons": reasons,
                "asset_checks": asset_rows,
                "ocr": ocr_rows,
                "url_checks": url_rows,
                **duplicate,
            }
        )

    rows.sort(key=lambda row: (-int(row["review_priority"]), str(row["sample_id"])))
    prefilter_path = output_dir / "prefilter.jsonl"
    _write_jsonl(prefilter_path, rows)
    summary = {
        "schema_version": "real-seed-prefilter-summary-v1",
        "candidates_path": str(candidates_path.resolve()),
        "prefilter_path": str(prefilter_path.resolve()),
        "num_candidates": len(rows),
        "num_unique_assets": len(assets),
        "route_distribution": dict(Counter(row["suggested_route"] for row in rows)),
        "ground_truth_distribution": dict(Counter(row["ground_truth"] for row in rows)),
        "duplicate_groups": len(
            {row.get("duplicate_group") for row in rows if row.get("duplicate_group")}
        ),
        "ocr_enabled": run_ocr,
        "ocr_completed_assets": len(ocr_cache),
        "url_checks_enabled": check_urls,
        "url_checks_completed": len(url_cache),
        "routes_are_suggestions_only": True,
        "core_cases_frozen": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = prefilter(
        candidates_path=Path(args.candidates),
        output_dir=Path(args.output_dir),
        run_ocr=not args.skip_ocr,
        ocr_gpu=args.ocr_gpu,
        check_urls=args.check_urls,
        url_workers=args.url_workers,
        url_timeout=args.url_timeout,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
