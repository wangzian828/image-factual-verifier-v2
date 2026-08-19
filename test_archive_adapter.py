from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.eval.archive_adapter import (
    load_archive_runtime_input,
    materialize_archive_runtime_rows,
)


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
                "archive_source_version_id": "archive:source:0001",
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
            "case_id": "archive:source:0001",
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


def test_archive_adapter_uses_archive_version_id_for_duplicate_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    image_dir = root / "artifacts" / "images"
    image_dir.mkdir(parents=True)
    first = image_dir / "0001.jpg"
    second = image_dir / "0002.jpg"
    first.write_bytes(b"archive-image-1")
    second.write_bytes(b"archive-image-2")
    rows = [
        {
            "candidate_id": "same-candidate",
            "archive_source_version_id": "source-a:0001",
            "archive_image_path": "artifacts/images/0001.jpg",
        },
        {
            "candidate_id": "same-candidate",
            "archive_source_version_id": "source-b:0002",
            "archive_image_path": "artifacts/images/0002.jpg",
        },
    ]
    (root / "human-review-candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    source = load_archive_runtime_input(root)

    assert [row["case_id"] for row in source.rows] == [
        "source-a:0001",
        "source-b:0002",
    ]


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


def test_archive_adapter_defers_hashing_until_selected_rows_are_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    image = root / "artifacts" / "images" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"archive-image")
    (root / "human-review-candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "candidate:0001",
                "archive_image_path": "artifacts/images/0001.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = 0
    original_sha256 = hashlib.sha256

    def counting_sha256(*args: object, **kwargs: object) -> "hashlib._Hash":
        nonlocal calls
        calls += 1
        return original_sha256(*args, **kwargs)

    monkeypatch.setattr("src.eval.archive_adapter.hashlib.sha256", counting_sha256)

    source = load_archive_runtime_input(root, materialize_image_hashes=False)

    assert source.rows == [
        {
            "case_id": "candidate:0001",
            "image_path": str(image.resolve()),
        }
    ]
    assert calls == 0

    rows = materialize_archive_runtime_rows(source.rows)

    assert rows == [
        {
            "case_id": "candidate:0001",
            "image_path": str(image.resolve()),
            "image_sha256": original_sha256(b"archive-image").hexdigest(),
        }
    ]
    assert calls == 1
