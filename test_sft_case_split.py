from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent / "scripts" / "trajectory" / "build_sft_case_split.py"
SPEC = importlib.util.spec_from_file_location("build_sft_case_split", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    runtime = []
    gold = []
    for index in range(10):
        case_id = f"case_{index:02d}"
        runtime.append(
            {
                "case_id": case_id,
                "image_path": f"runtime_input/assets/{index}.jpg",
                "image_sha256": f"{index:064x}",
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "label": "supported" if index in {0, 1, 2} else "refuted",
                "claim_atom": {
                    "subject": f"subject-{index}",
                    "event": f"event-{index}",
                },
                "certifying_evidence": [
                    {
                        "source_id": "shared" if index in {0, 3} else f"source-{index}",
                        "snapshot_sha256": f"snapshot-{index}",
                    }
                ],
            }
        )
    return runtime, gold


def test_split_is_deterministic_label_targeted_and_group_safe() -> None:
    runtime, gold = _rows()
    first, summary = MODULE.build_split(
        runtime,
        gold,
        validation_count=4,
        validation_supported=1,
        seed="fixture",
    )
    second, _ = MODULE.build_split(
        runtime,
        gold,
        validation_count=4,
        validation_supported=1,
        seed="fixture",
    )

    assert first == second
    assert summary["split_counts"] == {"train": 6, "validation": 4}
    assert summary["split_label_counts"]["validation"] == {
        "refuted": 3,
        "supported": 1,
    }
    by_case = {row["case_id"]: row for row in first}
    assert by_case["case_00"]["split"] == by_case["case_03"]["split"]
    assert all("label" not in row for row in first)


def test_split_rejects_frozen_eval_overlap() -> None:
    runtime, gold = _rows()
    with pytest.raises(ValueError, match="overlap prohibited"):
        MODULE.build_split(
            runtime,
            gold,
            validation_count=4,
            validation_supported=1,
            seed="fixture",
            prohibited_image_hashes={runtime[4]["image_sha256"]},
        )


def test_split_accepts_private_verdict_aliases() -> None:
    runtime = [
        {"case_id": "case-real", "image_sha256": "a" * 64},
        {"case_id": "case-fake", "image_sha256": "b" * 64},
    ]
    gold = [
        {"case_id": "case-real", "expected_verdict": "real"},
        {"case_id": "case-fake", "expected_verdict": "fake"},
    ]

    rows, summary = MODULE.build_split(
        runtime,
        gold,
        validation_count=1,
        validation_supported=1,
        seed="alias-test",
    )

    assert len(rows) == 2
    assert summary["split_label_counts"]["validation"] == {"supported": 1}
