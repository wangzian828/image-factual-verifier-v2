from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_model_visible
from .io import load_json, load_jsonl, sha256_file


GEMINI_WIRE_PROTOCOL_MARKER = "Native Gemini Interactions protocol:"


def _audit_message(message: Any, location: str) -> None:
    if not isinstance(message, Mapping):
        raise ValueError(f"{location} must be an object")
    role = str(message.get("role", ""))
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"{location} has unsupported role {role!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{location}.content must be a non-empty string")
    if role == "assistant" and message.get("loss") is not True:
        raise ValueError(f"{location} must explicitly set loss=true")


def audit_derived_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest = load_json(dataset_dir / "manifest.json")
    errors: list[str] = []
    roles: Counter[str] = Counter()
    rows_seen = 0
    primary_artifacts = {"train", "validation", "test"}
    for artifact_name, artifact in manifest.get("artifacts", {}).items():
        if artifact_name not in primary_artifacts:
            continue
        path = dataset_dir / str(artifact.get("path", ""))
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        if sha256_file(path) != artifact.get("sha256"):
            errors.append(f"sha256 mismatch: {path.name}")
        rows = load_jsonl(path)
        if len(rows) != artifact.get("rows"):
            errors.append(f"row count mismatch: {path.name}")
        for row_index, row in enumerate(rows):
            location = f"{path.name}[{row_index}]"
            try:
                assert_model_visible(row, location=location)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{location}.messages is invalid")
                for message_index, message in enumerate(messages):
                    _audit_message(message, f"{location}.messages[{message_index}]")
                    roles[str(message["role"])] += 1
                images = row.get("images")
                if images is not None:
                    if not isinstance(images, list) or not images:
                        raise ValueError(f"{location}.images must be a non-empty list")
                    if messages[0].get("role") == "user" and "<image>" not in str(
                        messages[0].get("content", "")
                    ):
                        raise ValueError(f"{location} image row lacks <image> placeholder")
                    for image in images:
                        if not Path(str(image)).is_file():
                            raise FileNotFoundError(
                                f"{location} image is unavailable: {image}"
                            )
                if GEMINI_WIRE_PROTOCOL_MARKER in json.dumps(
                    row,
                    ensure_ascii=False,
                ):
                    raise ValueError(
                        f"{location} contains provider wire instructions"
                    )
            except Exception as exc:
                errors.append(str(exc))
            rows_seen += 1
    return {
        "schema_version": "ifv-ms-swift-dataset-audit-v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "row_count": rows_seen,
        "role_counts": dict(sorted(roles.items())),
    }
