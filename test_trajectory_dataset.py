from __future__ import annotations

import json
from pathlib import Path

from scripts.trajectory.audit_dataset import audit_dataset
from scripts.trajectory.export_dataset import export_dataset
from src.trajectory.exporter import export_policy_examples
from test_image_only_v2_trajectory import (
    test_scripted_image_only_v2_complete_trajectory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_dir(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    test_scripted_image_only_v2_complete_trajectory(fixture)
    trace_path = fixture / "traces" / "case_scripted_v2.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    (run_dir / "traces").mkdir(parents=True)
    copied_trace = run_dir / "traces" / trace_path.name
    copied_trace.write_text(
        json.dumps(trace, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-scripted",
                "git_commit": "a" * 40,
                "benchmark": {"release_id": "release-scripted"},
            }
        ),
        encoding="utf-8",
    )
    examples = export_policy_examples(
        trace,
        source_metadata={
            "source_run_id": "run-scripted",
            "runtime_commit": "a" * 40,
            "release_id": "release-scripted",
            "runtime_contract_version": "ifv-image-only-runtime-v1",
            "process_reference_protocol_version": (
                "ifv-image-only-process-reference-protocol-v1"
            ),
        },
    )
    _write_jsonl(
        run_dir / "policy_trajectories.jsonl",
        [item.model_dump(mode="json") for item in examples],
    )
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": trace["image_id"],
                "total": 4.75,
                "components": {},
            }
        ],
    )
    return run_dir


def test_dataset_export_is_episode_and_source_family_split_safe(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    output = tmp_path / "dataset"

    manifest = export_dataset([run_dir], output)
    report = audit_dataset(output)

    assert manifest["episode_count"] == 1
    assert manifest["example_counts"]["train"] + (
        manifest["example_counts"]["validation"]
    ) + manifest["example_counts"]["test"] == 6
    assert report["passed"] is True
    assert report["example_count"] == 6
    assert report["episode_count"] == 1
    assert report["teacher_score_distribution"]["mean"] == 4.75


def test_dataset_audit_rejects_private_policy_input(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    output = tmp_path / "dataset"
    export_dataset([run_dir], output)
    split_path = next(
        path
        for path in (
            output / "train.jsonl",
            output / "validation.jsonl",
            output / "test.jsonl",
        )
        if path.read_text(encoding="utf-8").strip()
    )
    rows = [
        json.loads(line)
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["policy_input"]["gold"] = {"verdict": "real"}
    _write_jsonl(split_path, rows)

    report = audit_dataset(output)
    codes = {item["code"] for item in report["errors"]}

    assert report["passed"] is False
    assert "PRIVATE_DATA_LEAK" in codes
