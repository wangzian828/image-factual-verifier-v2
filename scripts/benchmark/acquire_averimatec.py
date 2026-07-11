#!/usr/bin/env python3
"""Download AVerImaTeC on the server and build a real-case audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import requests
from PIL import Image

from src.storage import data_path
from src.orchestrator.source_access import benchmark_policy_from_rows


DATASET_ID = "Rui4416/AVerImaTeC"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
RESOLVE_URL = f"{DATASET_URL}/resolve/main"
SOURCE_FILES = ("README.md", "train.json", "val.json", "images.zip")
LABEL_MAP = {
    "Supported": "real",
    "Refuted": "fake",
    "Not Enough Evidence": "unverifiable",
    "Conflicting Evidence/Cherrypicking": "unverifiable",
}
LICENSE_STATUS = "not_declared_on_hugging_face_dataset_card"


def _default_dataset_root() -> Path:
    return data_path("datasets/averimatec", "data/external/averimatec")


def _default_output_dir() -> Path:
    return data_path(
        "benchmarks/candidates/real_seed_v0/averimatec",
        "data/benchmark_candidates/real_seed_v0/averimatec",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(_default_dataset_root()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use source files already present under dataset-root/raw.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace existing raw source files.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    session: requests.Session,
    filename: str,
    destination: Path,
    *,
    force: bool,
) -> Dict[str, Any]:
    source_url = f"{RESOLVE_URL}/{filename}"
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return {
            "filename": filename,
            "source_url": source_url,
            "path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "status": "existing",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if force and temporary.exists():
        temporary.unlink()

    resume_offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={resume_offset}-"} if resume_offset else {}
    response = session.get(
        source_url,
        headers=headers,
        stream=True,
        timeout=(30, 600),
    )
    if response.status_code == 416 and resume_offset:
        total_match = re.search(
            r"\*/(\d+)", response.headers.get("Content-Range", "")
        )
        if total_match and int(total_match.group(1)) == resume_offset:
            temporary.replace(destination)
            return {
                "filename": filename,
                "source_url": source_url,
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "etag": response.headers.get("ETag"),
                "status": "resumed",
            }
    response.raise_for_status()
    append = resume_offset > 0 and response.status_code == 206
    if resume_offset and response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {resume_offset}-"):
            raise RuntimeError(
                f"Unexpected Content-Range for {filename}: {content_range!r}"
            )
    mode = "ab" if append else "wb"
    size = resume_offset if append else 0
    with temporary.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            size += len(chunk)
    temporary.replace(destination)
    return {
        "filename": filename,
        "source_url": source_url,
        "path": str(destination.resolve()),
        "bytes": size,
        "sha256": _sha256(destination),
        "etag": response.headers.get("ETag"),
        "status": "resumed" if append else "downloaded",
    }


def _extract_images(archive: Path, extraction_parent: Path) -> Path:
    archive_hash = _sha256(archive)
    extraction_root = extraction_parent / f"images-{archive_hash[:16]}"
    marker = extraction_root / ".complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("archive_sha256") == archive_hash:
            return extraction_root

    extraction_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extraction_root.resolve()
    extracted_files = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (extraction_root / member.filename).resolve()
            if root_resolved not in destination.parents and destination != root_resolved:
                raise ValueError(f"Archive member escapes extraction root: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files += 1
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "archive_sha256": archive_hash,
                "extracted_files": extracted_files,
                "completed_at": _utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return extraction_root


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError(f"Expected a JSON list of objects: {path}")
    return payload


def _image_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicate_names: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".complete.json":
            continue
        key = path.name
        if key in index and index[key] != path:
            duplicate_names.add(key)
        else:
            index[key] = path
    if duplicate_names:
        rendered = ", ".join(sorted(duplicate_names)[:10])
        raise ValueError(f"Duplicate image basenames in archive: {rendered}")
    return index


def _asset_record(path: Path, dataset_root: Path) -> Dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
    return {
        "filename": path.name,
        "image_path": str(path.resolve()),
        "dataset_relative_path": path.resolve().relative_to(dataset_root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
    }


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _question_summary(question: Mapping[str, Any]) -> Dict[str, Any]:
    answers = question.get("answers") if isinstance(question.get("answers"), list) else []
    return {
        "question": str(question.get("question", "")).strip(),
        "question_type": list(question.get("question_type") or []),
        "answer_method": str(question.get("answer_method", "")).strip(),
        "input_images": list(question.get("input_images") or []),
        "evidence_urls": _dedupe(
            str(answer.get("source_url", ""))
            for answer in answers
            if isinstance(answer, dict)
        ),
    }


def _has_image_question(question: Mapping[str, Any]) -> bool:
    return any(
        "image" in str(question_type).casefold()
        for question_type in question.get("question_type") or []
    )


def _non_biometric_identity_flag(questions: Sequence[Mapping[str, Any]]) -> bool:
    patterns = ("who is", "who are", "identity of", "identify the person")
    return any(
        _has_image_question(question)
        and any(pattern in str(question.get("question", "")).casefold() for pattern in patterns)
        for question in questions
    )


def _candidate_record(
    row: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
    image_index: Mapping[str, Path],
    dataset_root: Path,
    asset_cache: Dict[Path, Dict[str, Any]],
) -> Dict[str, Any]:
    original_label = str(row.get("label", "")).strip()
    if original_label not in LABEL_MAP:
        raise ValueError(f"Unknown AVerImaTeC label: {original_label!r}")

    image_names = [str(value) for value in row.get("claim_images") or []]
    missing = [name for name in image_names if name not in image_index]
    if missing:
        raise FileNotFoundError(f"Missing claim images for {split}[{source_index}]: {missing}")
    assets: List[Dict[str, Any]] = []
    for name in image_names:
        path = image_index[name]
        if path not in asset_cache:
            asset_cache[path] = _asset_record(path, dataset_root)
        assets.append(asset_cache[path])

    raw_questions = [
        question for question in row.get("questions") or [] if isinstance(question, dict)
    ]
    questions = [_question_summary(question) for question in raw_questions]
    image_questions = [question for question in raw_questions if _has_image_question(question)]
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    transcription = str(metadata.get("transcription", "")).strip()
    evidence_urls = _dedupe(
        url
        for question in questions
        for url in question.get("evidence_urls", [])
    )
    identity_investigation = _non_biometric_identity_flag(raw_questions)
    review_signals = []
    if transcription:
        review_signals.append("dataset_transcription_present")
    if image_questions:
        review_signals.append("image_related_investigation_present")
    if metadata.get("image_misuse_types"):
        review_signals.append("image_misuse_annotation_present")
    if identity_investigation:
        review_signals.append("person_identity_requires_non_biometric_investigation")

    return {
        "schema_version": "real-seed-candidate-v1",
        "sample_id": f"averimatec-{split}-{source_index:04d}",
        "source_dataset": DATASET_ID,
        "source_split": split,
        "source_index": source_index,
        "source_article_url": str(row.get("article", "")).strip(),
        "source_fact_check_date": str(row.get("date", "")).strip(),
        "source_location": str(row.get("location", "")).strip(),
        "original_label": original_label,
        "ground_truth": LABEL_MAP[original_label],
        "gold_claim_text": str(row.get("claim_text", "")).strip(),
        "gold_justification": str(row.get("justification", "")).strip(),
        "claim_assets": assets,
        "primary_image_path": assets[0]["image_path"] if assets else None,
        "question_count": len(questions),
        "image_question_count": len(image_questions),
        "questions": questions,
        "evidence_urls": evidence_urls,
        "source_metadata": {
            "speaker": metadata.get("speaker"),
            "transcription": transcription,
            "media_source": metadata.get("media_source"),
            "original_claim_url": metadata.get("original_claim_url"),
            "reporting_source": metadata.get("reporting_source"),
            "claim_types": list(metadata.get("claim_types") or []),
            "fact_checking_strategies": list(
                metadata.get("fact_checking_strategies") or []
            ),
            "modality": metadata.get("modality"),
            "refuting_reasons": list(metadata.get("refuting_reasons") or []),
            "image_misuse_types": list(metadata.get("image_misuse_types") or []),
        },
        "benchmark_routing": {
            "core_single_image_status": "pending_claim_surface_audit",
            "external_claim_transfer_status": "pending_case_quality_audit",
            "claim_surface_must_be_recovered_from_pixels_for_core": True,
            "gold_claim_must_not_enter_core_runtime_context": True,
            "person_identity_investigation_required": identity_investigation,
            "biometric_identity_matching_allowed": False,
            "review_signals": review_signals,
        },
        "license": {
            "dataset_license_status": LICENSE_STATUS,
            "asset_terms_status": "pending_source_level_review",
            "redistribution_status": "internal_audit_only_until_terms_review",
        },
    }


def _evaluation_record(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Write an evaluator-private row; only runtime fields reach the Agent."""

    return {
        "sample_id": candidate["sample_id"],
        "image_path": candidate["primary_image_path"],
        "ground_truth": candidate["ground_truth"],
        "bucket": "averimatec_real_seed_candidate",
        "user_claim": candidate["gold_claim_text"],
        "source_article_url": candidate["source_article_url"],
    }


def build_candidate_manifest(
    *,
    dataset_root: Path,
    extraction_root: Path,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    acquisition_files: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    image_index = _image_index(extraction_root)
    asset_cache: Dict[Path, Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    for split, rows in split_rows.items():
        for index, row in enumerate(rows):
            candidates.append(
                _candidate_record(
                    row,
                    split=split,
                    source_index=index,
                    image_index=image_index,
                    dataset_root=dataset_root,
                    asset_cache=asset_cache,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "candidates.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    evaluation_path = output_dir / "evaluation.jsonl"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            if candidate.get("primary_image_path"):
                handle.write(
                    json.dumps(_evaluation_record(candidate), ensure_ascii=False) + "\n"
                )

    original_labels = Counter(row["original_label"] for row in candidates)
    mapped_labels = Counter(row["ground_truth"] for row in candidates)
    summary = {
        "schema_version": "real-seed-acquisition-v1",
        "created_at": _utc_now(),
        "source_dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "dataset_license_status": LICENSE_STATUS,
        "dataset_root": str(dataset_root.resolve()),
        "extraction_root": str(extraction_root.resolve()),
        "candidate_manifest": str(manifest_path.resolve()),
        "evaluation_manifest": str(evaluation_path.resolve()),
        "total_candidates": len(candidates),
        "unique_claim_assets": len(asset_cache),
        "original_label_distribution": dict(original_labels),
        "mapped_label_distribution": dict(mapped_labels),
        "with_image_related_questions": sum(
            row["image_question_count"] > 0 for row in candidates
        ),
        "with_dataset_transcription": sum(
            bool(row["source_metadata"]["transcription"]) for row in candidates
        ),
        "requiring_non_biometric_identity_investigation": sum(
            row["benchmark_routing"]["person_identity_investigation_required"]
            for row in candidates
        ),
        "core_cases_frozen": 0,
        "note": (
            "Every row remains a candidate until manual claim-surface and evidence "
            "audit. Label mapping does not make an item a core single-image case."
        ),
        "acquisition_files": list(acquisition_files),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    access_policy = benchmark_policy_from_rows(
        candidates,
        policy_id="averimatec-evaluation",
    )
    policy_path = output_dir / "source_access_policy.json"
    policy_path.write_text(
        json.dumps(access_policy.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["source_access_policy"] = str(policy_path.resolve())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def acquire(
    *,
    dataset_root: Path,
    output_dir: Path,
    skip_download: bool,
    force_download: bool,
) -> Dict[str, Any]:
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "image-factual-verifier/real-seed-acquisition"})

    acquisition_files: List[Dict[str, Any]] = []
    if skip_download:
        for filename in SOURCE_FILES:
            path = raw_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing raw source file: {path}")
            acquisition_files.append(
                {
                    "filename": filename,
                    "source_url": f"{RESOLVE_URL}/{filename}",
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "status": "existing",
                }
            )
    else:
        for filename in SOURCE_FILES:
            acquisition_files.append(
                _download(
                    session,
                    filename,
                    raw_dir / filename,
                    force=force_download,
                )
            )

    extraction_root = _extract_images(
        raw_dir / "images.zip", dataset_root / "extracted"
    )
    split_rows = {
        "train": _load_rows(raw_dir / "train.json"),
        "val": _load_rows(raw_dir / "val.json"),
    }
    return build_candidate_manifest(
        dataset_root=dataset_root,
        extraction_root=extraction_root,
        split_rows=split_rows,
        output_dir=output_dir,
        acquisition_files=acquisition_files,
    )


def main() -> None:
    args = _parse_args()
    summary = acquire(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        skip_download=args.skip_download,
        force_download=args.force_download,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
