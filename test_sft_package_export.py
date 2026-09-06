from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.trajectory.export_dataset import export_dataset
from scripts.trajectory.build_sft_training_package import _build_default_case_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_action_only_release_keeps_perception_training_example(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"action-only-image")
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    canonical = tmp_path / "accepted-release"
    (canonical / "traces").mkdir(parents=True)
    (canonical / "eligibility").mkdir()
    trace = {
        "image_id": "case-action-only",
        "input_mode": "image_only",
        "termination": "success",
        "decision_policy_version": "unified-react-v1",
        "state": {
            "runtime_case": {
                "case_id": "case-action-only",
                "image_path": str(image_path),
                "image_sha256": image_sha256,
            },
            "input_mode": "image_only",
        },
    }
    trace_path = canonical / "traces" / "case-action-only.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False),
        encoding="utf-8",
    )
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (canonical / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "accepted-release",
                "git_commit": "a" * 40,
                "benchmark": {"release_id": "release-action-only"},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        canonical / "selected_episodes.jsonl",
        [
            {
                "case_id": "case-action-only",
                "episode_id": "case-action-only",
                "source_trace_sha256": trace_sha256,
                "teacher_score": 1.0,
                "deterministic_hard_gate_pass": True,
                "training_buckets": ["action_only", "rl_candidate"],
            }
        ],
    )
    _write_jsonl(
        canonical / "action_only.jsonl",
        [{"episode_id": "case-action-only"}],
    )
    _write_jsonl(
        canonical / "perception_trajectories.jsonl",
        [
            {
                "episode_id": "case-action-only",
                "source_run_id": "accepted-release",
                "runtime_commit": "a" * 40,
                "release_id": "release-action-only",
                "runtime_contract_version": "runtime-v1",
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "instruction": "Report visible image content.",
                "perception_report": {
                    "scene_description": "A controlled test image.",
                    "image_type": "photo",
                    "entities": [],
                    "relations": [],
                    "text_regions": [],
                    "notable_details": [],
                    "uncertainties": [],
                },
            }
        ],
    )
    (canonical / "accepted_release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ifv-accepted-teacher-release-v3",
                "accepted_case_count": 1,
                "rejected_case_count": 0,
            }
        ),
        encoding="utf-8",
    )

    case_split = tmp_path / "case_split.jsonl"
    _write_jsonl(
        case_split,
        [
            {
                "case_id": "case-action-only",
                "image_sha256": image_sha256,
                "split": "validation",
                "split_group_id": "group-action-only",
            }
        ],
    )

    output = tmp_path / "dataset"
    manifest = export_dataset(
        [],
        output,
        accepted_release_path=canonical,
        case_split_path=case_split,
        minimum_accepted_cases=0,
    )

    assert manifest["selected_release_case_count"] == 1
    assert manifest["action_only_episode_count"] == 1
    assert manifest["episode_count"] == 0
    assert manifest["perception_example_counts"]["validation"] == 1
    assert len(
        [
            line
            for line in (output / "perception.validation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    ) == 1
    assert json.loads(
        (output / "action_only.jsonl").read_text(encoding="utf-8")
    )["episode_id"] == "case-action-only"


def test_default_case_split_chooses_validation_from_policy_rows(tmp_path: Path) -> None:
    release = tmp_path / "accepted-release"
    (release / "traces").mkdir(parents=True)
    selected = [
        {
            "case_id": "case-action-only",
            "episode_id": "case-action-only",
            "training_buckets": ["action_only"],
            "trajectory_sft_rows": 0,
        },
        {
            "case_id": "case-policy-a",
            "episode_id": "case-policy-a",
            "training_buckets": ["reasoning_sft"],
            "trajectory_sft_rows": 1,
        },
        {
            "case_id": "case-policy-b",
            "episode_id": "case-policy-b",
            "training_buckets": ["reasoning_sft"],
            "trajectory_sft_rows": 1,
        },
    ]
    (release / "selected_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selected),
        encoding="utf-8",
    )
    for row in selected:
        trace = {
            "state": {
                "runtime_case": {
                    "image_sha256": "a" * 64,
                }
            }
        }
        (release / "traces" / f"{row['episode_id']}.json").write_text(
            json.dumps(trace),
            encoding="utf-8",
        )

    destination = tmp_path / "case-split.jsonl"
    summary = _build_default_case_split(release, destination)
    rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["validation_count"] == 1
    validation = [row for row in rows if row["split"] == "validation"]
    assert validation
    assert validation[0]["case_id"] != "case-action-only"
