"""Read-only adapter for historical human-review candidate archives.

The benchmark-pipeline archive is intentionally not rewritten into a runtime
release.  This adapter projects only the public runtime fields required by the
Agent and keeps all construction, label, and evidence fields out of the
runtime rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ARCHIVE_ADAPTER_SCHEMA_VERSION = "ifv-hrc-archive-runtime-adapter-v1"
CANDIDATE_FILE_NAME = "human-review-candidates.jsonl"
SUMMARY_FILE_NAME = "archive-summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: Mapping[str, Any], key: str, *, line_number: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} requires non-empty {key}"
        )
    if key == "candidate_id" and len(value) > 200:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} candidate_id exceeds "
            "the runtime case_id limit of 200 characters"
        )
    return value


def _resolve_image(root: Path, value: str, *, line_number: int) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image path must be relative"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image path escapes archive"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{CANDIDATE_FILE_NAME} line {line_number} image does not exist: "
            f"{resolved}"
        )
    return resolved


@dataclass(frozen=True)
class ArchiveRuntimeInput:
    """Minimal, Agent-visible projection of one immutable archive."""

    root: Path
    candidate_path: Path
    archive_id: str
    rows: List[Dict[str, str]]

    @property
    def case_count(self) -> int:
        return len(self.rows)

    @property
    def schema_version(self) -> str:
        return ARCHIVE_ADAPTER_SCHEMA_VERSION


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path.name} line {line_number} must be a JSON object"
                )
            yield line_number, payload


def load_archive_runtime_input(archive_root: Path) -> ArchiveRuntimeInput:
    """Load candidate rows without modifying or reserializing the archive.

    Only ``candidate_id`` and ``archive_image_path`` are read into the runtime
    projection.  Fields such as ``factual_status``, ``claim_atom``, ``evidence``,
    and source/construction metadata are deliberately not copied.
    """

    root = archive_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"archive root does not exist: {root}")

    candidate_path = root / CANDIDATE_FILE_NAME
    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing archive candidate file: {candidate_path}")

    archive_id = ""
    summary_path = root / SUMMARY_FILE_NAME
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{SUMMARY_FILE_NAME} is not valid JSON") from exc
        if not isinstance(summary, Mapping):
            raise ValueError(f"{SUMMARY_FILE_NAME} must be a JSON object")
        archive_id = str(summary.get("archive_id") or "").strip()

    rows: List[Dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for line_number, candidate in _iter_jsonl(candidate_path):
        case_id = _required_text(candidate, "candidate_id", line_number=line_number)
        image_value = _required_text(
            candidate,
            "archive_image_path",
            line_number=line_number,
        )
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate archive candidate_id: {case_id}")
        seen_case_ids.add(case_id)
        image_path = _resolve_image(root, image_value, line_number=line_number)
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
            }
        )

    if not rows:
        raise ValueError(f"archive candidate file is empty: {candidate_path}")

    return ArchiveRuntimeInput(
        root=root,
        candidate_path=candidate_path,
        archive_id=archive_id or root.name,
        rows=rows,
    )
