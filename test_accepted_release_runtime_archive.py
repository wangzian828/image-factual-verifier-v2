from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.trajectory.export_dataset import export_dataset
from scripts.trajectory.stage_accepted_teacher_release import stage_release
from test_trajectory_media_projection import _trace


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_release_runtime_archive_survives_source_deletion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "rollout"
    trace_path = run_dir / "traces" / "case-1.json"
    trace = _trace(tmp_path)
    trace["verdict"] = "real"
    _write_json(trace_path, trace)
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "ifv-run-manifest-v1",
            "run_id": "runtime-archive-smoke",
            "status": "completed",
            "git_commit": "a" * 40,
            "benchmark": {
                "release_id": "runtime-archive-smoke",
                "runtime_contract_version": "runtime-v1",
            },
        },
    )

    eligibility_dir = tmp_path / "eligibility"
    _write_json(
        eligibility_dir / "case-1.sft_eligibility.json",
        {
            "case_id": "case-1",
            "episode_id": "case-1",
            "source_trace": {
                "sha256": trace_sha256,
                "decision_policy_version": "unified-react-v1",
            },
            "gates": {
                "engineering_valid": True,
                "sft_eligibility_pass": True,
            },
            "metrics": {},
        },
    )

    release_dir = tmp_path / "accepted-release"
    manifest = stage_release(
        [(run_dir, eligibility_dir, None)],
        release_dir,
        minimum_accepted_cases=1,
    )

    selected = [
        json.loads(line)
        for line in (release_dir / "selected_episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert manifest["schema_version"] == "ifv-accepted-teacher-release-v4"
    assert manifest["runtime_store_archive"]["selected_count"] == 1
    assert selected[0]["runtime_store_file_count"] >= 6
    archived_runtime = release_dir / selected[0]["runtime_store_path"]
    assert archived_runtime.is_dir()
    assert hashlib.sha256(
        (release_dir / "traces" / "case-1.json").read_bytes()
    ).hexdigest() == trace_sha256

    source_runtime = Path(
        trace["state"]["runtime_store"]["runtime_path"]
    )
    source_image = Path(trace["state"]["runtime_case"]["image_path"])
    shutil.rmtree(source_runtime)
    source_image.unlink()
    assert not source_runtime.exists()
    assert not source_image.exists()

    case_split = tmp_path / "case-split.jsonl"
    _write_jsonl(
        case_split,
        [
            {
                "case_id": "case-1",
                "image_sha256": trace["state"]["runtime_case"][
                    "image_sha256"
                ],
                "split": "validation",
                "split_group_id": "case-1-group",
            }
        ],
    )
    dataset_dir = tmp_path / "accepted-dataset"
    dataset_manifest = export_dataset(
        [],
        dataset_dir,
        accepted_release_path=release_dir,
        case_split_path=case_split,
        minimum_accepted_cases=1,
    )

    rows = [
        json.loads(line)
        for line in (dataset_dir / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert dataset_manifest["episode_count"] == 1
    assert dataset_manifest["frozen_inputs"]["accepted_release"][
        "schema_version"
    ] == "ifv-accepted-teacher-release-v4"
    assert len(rows) == 1
    assert len(rows[0]["images"]) == 3
    assert sum(
        message["content"].count("<image>")
        for message in rows[0]["messages"]
    ) == 3
