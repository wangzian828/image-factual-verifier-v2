from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ifv_training.audit import audit_derived_dataset
from ifv_training.io import write_json, write_jsonl
from ifv_training.io import sha256_file
from ifv_training.perception import (
    convert_accepted_perception_dataset,
    convert_perception_runs,
    perception_row,
)


def _trace(image: Path) -> dict:
    return {
        "image_id": "case-1",
        "image_path": str(image),
        "state": {
            "image_path": str(image),
            "perception": {
                "entities": [
                    {
                        "name": "red square",
                        "entity_type": "object",
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "confidence": 0.9,
                        "attributes": {},
                    }
                ],
                "text_regions": [],
                "scene_description": "A red square.",
                "image_type": "graphic",
            },
        },
    }


def test_perception_row_is_multimodal_and_json_target(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="red").save(image)

    row = perception_row(_trace(image))

    assert row["images"] == [str(image.resolve())]
    assert row["messages"][1]["content"].startswith("<image>")
    target = json.loads(row["messages"][2]["content"])
    assert target["scene_description"] == "A red square."
    assert set(row) == {"messages", "images"}


def test_perception_conversion_uses_source_split_and_quality_gate(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="red").save(image)
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    write_json(run / "run_manifest.json", {"run_id": "run-1", "git_commit": "a" * 40})
    write_json(run / "traces" / "case-1.json", _trace(image))
    write_jsonl(
        run / "trajectory_scores.jsonl",
        [
            {
                "case_id": "case-1",
                "training_eligible": True,
                "training_exclusion_reasons": [],
            }
        ],
    )
    split_map = tmp_path / "episode_metadata.jsonl"
    write_jsonl(split_map, [{"episode_id": "case-1", "split": "validation"}])

    output = tmp_path / "output"
    manifest = convert_perception_runs(
        [run],
        output,
        split_map_path=split_map,
    )
    audit = audit_derived_dataset(output)

    assert manifest["artifacts"]["validation"]["rows"] == 1
    assert manifest["artifacts"]["train"]["rows"] == 0
    assert audit["passed"] is True


def test_accepted_perception_conversion_uses_frozen_dataset(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="blue").save(image)
    source = tmp_path / "accepted"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-policy-dataset-v2",
            "split_mode": "frozen_teacher_sft",
        },
    )
    row = {
        "dataset_version": "ifv-policy-dataset-v2",
        "episode_id": "case-accepted",
        "source_run_id": "teacher-run",
        "runtime_commit": "a" * 40,
        "image_path": str(image),
        "image_sha256": sha256_file(image),
        "instruction": "Report visible content.",
        "perception_report": {
            "scene_description": "A blue square.",
            "image_type": "graphic",
            "entities": [],
            "text_regions": [],
        },
        "split": "validation",
    }
    write_jsonl(source / "perception.train.jsonl", [])
    write_jsonl(source / "perception.validation.jsonl", [row])
    write_jsonl(source / "perception.test.jsonl", [])

    output = tmp_path / "output"
    manifest = convert_accepted_perception_dataset(source, output)
    audit = audit_derived_dataset(output)

    assert manifest["artifacts"]["validation"]["rows"] == 1
    converted = json.loads(
        (output / "validation.jsonl").read_text(encoding="utf-8")
    )
    assert converted["images"] == [str(image.resolve())]
    assert converted["messages"][1]["content"].startswith("<image>")
    assert audit["passed"] is True


def test_accepted_perception_conversion_accepts_unified_react_v3(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="green").save(image)
    source = tmp_path / "accepted"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "dataset_version": "ifv-trajectory-sft-dataset-v3",
            "split_mode": "frozen_teacher_sft",
        },
    )
    row = {
        "dataset_version": "ifv-trajectory-sft-dataset-v3",
        "episode_id": "case-v3",
        "image_path": str(image),
        "image_sha256": sha256_file(image),
        "instruction": "Report visible content.",
        "perception_report": {
            "scene_description": "A green square.",
            "image_type": "graphic",
            "entities": [],
            "text_regions": [],
        },
        "split": "train",
    }
    write_jsonl(source / "perception.train.jsonl", [row])
    write_jsonl(source / "perception.validation.jsonl", [])
    write_jsonl(source / "perception.test.jsonl", [])

    manifest = convert_accepted_perception_dataset(source, tmp_path / "output")

    assert manifest["example_count"] == 1
