from __future__ import annotations

import json
from pathlib import Path

from ifv_training.psd_datums import build_sparse_topk_package
from ifv_training.psd_preflight import verify_psd_training_input


def _target(target_id: str, kind: str) -> dict:
    return {
        "target_id": target_id,
        "target_status": "complete",
        "kind": kind,
        "student_prompt_ids": [10, 11],
        "completion_ids": [20, 21],
        "teacher_topk_by_position": [
            [[20, 0.75], [120, 0.25]],
            [[21, 0.75], [121, 0.25]],
        ],
        "row_weight": 1.0,
    }


def _package(tmp_path: Path) -> Path:
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _target("repair-1", "repair"),
                _target("preserve-1", "preserve"),
            )
        ),
        encoding="utf-8",
    )
    package = tmp_path / "package"
    build_sparse_topk_package(
        targets_path=targets,
        output_dir=package,
        topk=2,
        max_sequence_length=16,
    )
    return package


def test_psd_training_input_preflight_binds_ready_per_target_package(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)

    result = verify_psd_training_input(
        datums_path=package / "datums.jsonl",
        manifest_path=package / "manifest.json",
        expected_topk=2,
        max_context=16,
    )

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["datums"]["by_kind"] == {"preserve": 1, "repair": 1}
    assert result["datums"]["effective_row_mass_by_kind"] == {
        "preserve": 1.0,
        "repair": 1.0,
    }
    assert result["datums"]["lengths"]["input_tokens"]["max"] == 3


def test_psd_training_input_preflight_detects_post_manifest_mutation(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    datums = package / "datums.jsonl"
    datums.write_text(datums.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = verify_psd_training_input(
        datums_path=datums,
        manifest_path=package / "manifest.json",
        expected_topk=2,
        max_context=16,
    )

    assert result["passed"] is False
    assert "manifest_datums_sha256_mismatch" in result["errors"]
