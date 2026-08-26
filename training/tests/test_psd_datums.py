from __future__ import annotations

import json
from pathlib import Path

from ifv_training.psd_datums import (
    build_sparse_topk_datum,
    build_sparse_topk_package,
)


def _target(
    target_id: str,
    *,
    kind: str = "repair",
    row_weight: float = 1.0,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
) -> dict:
    completion = completion_ids or [20, 21]
    distributions = [
        [[token, 0.75], [token + 100, 0.25]]
        for token in completion
    ]
    return {
        "target_id": target_id,
        "target_status": "complete",
        "kind": kind,
        "student_prompt_ids": prompt_ids or [10, 11],
        "completion_ids": completion,
        "teacher_topk_by_position": distributions,
        "row_weight": row_weight,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sparse_datum_uses_causal_completion_prediction_positions() -> None:
    datum = build_sparse_topk_datum(_target("repair-1"), topk=2)

    assert datum["input_ids"] == [10, 11, 20]
    assert datum["loss_positions"] == [1, 2]
    assert datum["target_tokens"] == [
        [0, 0],
        [20, 120],
        [21, 121],
    ]
    assert datum["weights"] == [
        [0.0, 0.0],
        [0.75, 0.25],
        [0.75, 0.25],
    ]


def test_sparse_package_balances_repair_and_preservation_row_mass(
    tmp_path: Path,
) -> None:
    targets = tmp_path / "targets.jsonl"
    _write_jsonl(
        targets,
        [
            _target("repair-1", kind="repair", row_weight=1.0),
            _target("repair-2", kind="repair", row_weight=1.0),
            _target("preserve-1", kind="preserve", row_weight=0.25),
            _target("preserve-2", kind="preserve", row_weight=0.75),
        ],
    )

    output = tmp_path / "datums"
    manifest = build_sparse_topk_package(
        targets_path=targets,
        output_dir=output,
        topk=2,
    )

    assert manifest["status"] == "ready_for_trainer"
    assert manifest["effective_row_mass_by_kind"] == {
        "preserve": 1.0,
        "repair": 1.0,
    }
    rows = [
        json.loads(line)
        for line in (output / "datums.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["row_weight"] for row in rows] == [0.5, 0.5, 0.25, 0.75]


def test_sparse_package_fails_closed_on_incomplete_or_oversized_target(
    tmp_path: Path,
) -> None:
    targets = tmp_path / "targets.jsonl"
    incomplete = _target("incomplete")
    incomplete["target_status"] = "pending_topk"
    _write_jsonl(
        targets,
        [
            incomplete,
            _target(
                "too-long",
                prompt_ids=[1, 2, 3, 4],
                completion_ids=[5, 6],
            ),
        ],
    )

    output = tmp_path / "datums"
    manifest = build_sparse_topk_package(
        targets_path=targets,
        output_dir=output,
        topk=2,
        max_sequence_length=5,
    )

    assert manifest["status"] == "blocked_invalid_target"
    assert manifest["counts"]["rejections"] == 2
    assert not (output / "datums.jsonl").exists()
