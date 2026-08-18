from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.eval.archive_adapter import load_archive_runtime_input


def test_archive_adapter_projects_only_minimal_runtime_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    image = root / "artifacts" / "images" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"archive-image")
    (root / "archive-summary.json").write_text(
        json.dumps({"archive_id": "fixture-archive"}),
        encoding="utf-8",
    )
    (root / "human-review-candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "candidate:0001",
                "archive_image_path": "artifacts/images/0001.jpg",
                "factual_status": "refuted",
                "claim_atom": {"subject": "private"},
                "evidence": {"source": "private"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source = load_archive_runtime_input(root)

    assert source.archive_id == "fixture-archive"
    assert source.case_count == 1
    assert source.rows == [
        {
            "case_id": "candidate:0001",
            "image_path": str(image.resolve()),
            "image_sha256": hashlib.sha256(b"archive-image").hexdigest(),
        }
    ]
    assert set(source.rows[0]) == {"case_id", "image_path", "image_sha256"}


def test_archive_adapter_rejects_missing_or_escaping_images(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "human-review-candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "candidate:0001",
                "archive_image_path": "../outside.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes archive"):
        load_archive_runtime_input(root)


def test_archive_adapter_rejects_duplicate_candidate_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    image = root / "artifacts" / "images" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"archive-image")
    row = {
        "candidate_id": "duplicate",
        "archive_image_path": "artifacts/images/0001.jpg",
    }
    (root / "human-review-candidates.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate archive candidate_id"):
        load_archive_runtime_input(root)


def test_archive_adapter_enforces_runtime_case_id_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    image = root / "artifacts" / "images" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"archive-image")
    (root / "human-review-candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "x" * 201,
                "archive_image_path": "artifacts/images/0001.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime case_id limit"):
        load_archive_runtime_input(root)
