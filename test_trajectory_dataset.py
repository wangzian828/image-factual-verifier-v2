from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from scripts.trajectory.audit_dataset import audit_dataset
from scripts.trajectory.export_dataset import (
    _cross_case_source_families,
    export_dataset,
)
from src.trajectory.exporter import export_trajectory_sft_example
from src.trajectory.perception_exporter import export_perception_example
from test_image_only_trajectory import (
    test_scripted_image_only_complete_trajectory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_dir(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    test_scripted_image_only_complete_trajectory(fixture)
    trace_path = fixture / "traces" / "case_scripted_v3.json"
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
    trajectory = export_trajectory_sft_example(
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
        run_dir / "trajectory_sft.jsonl",
        [trajectory.model_dump(mode="json")],
    )
    perception = export_perception_example(
        trace,
        source_metadata={
            "source_run_id": "run-scripted",
            "runtime_commit": "a" * 40,
            "release_id": "release-scripted",
            "runtime_contract_version": "ifv-image-only-runtime-v1",
        },
    )
    _write_jsonl(
        run_dir / "perception_trajectories.jsonl",
        [perception.model_dump(mode="json")],
    )
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": trace["image_id"],
                "total": 4.75,
                "components": {},
                "training_eligible": True,
                "training_exclusion_reasons": [],
            }
        ],
    )
    return run_dir


def test_trajectory_export_uses_qwen_native_thinking_and_tool_calls(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    test_scripted_image_only_complete_trajectory(fixture)
    trace_path = fixture / "traces" / "case_scripted_v3.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    tool_step = next(
        step
        for step in trace["state"]["all_steps"]
        if (
            isinstance(step.get("metadata"), dict)
            and isinstance(step["metadata"].get("policy_action"), dict)
            and step["metadata"]["policy_action"].get("type") == "tool_call"
        )
    )
    tool_step["thought"] = "先核对这个问题，再调用对应工具。"

    trajectory = export_trajectory_sft_example(trace)
    messages = trajectory.messages
    tool_assistant = next(
        message
        for message in messages
        if (
            message.get("role") == "assistant"
            and "<tool_call>" in message.get("content", "")
        )
    )
    tool_response = next(
        message for message in messages if message.get("role") == "tool"
    )

    assert "<think>\n先核对这个问题，再调用对应工具。\n</think>" in (
        tool_assistant["content"]
    )
    assert "<function=" in tool_assistant["content"]
    assert "<parameter=" in tool_assistant["content"]
    assert tool_response["content"]
    assert all(
        message.get("role") not in {"tool_call", "tool_response"}
        for message in messages
    )


def test_dataset_export_is_episode_and_source_family_split_safe(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    output = tmp_path / "dataset"

    manifest = export_dataset(
        [run_dir],
        output,
        short_max_tokens=300_000,
    )
    report = audit_dataset(output)

    assert manifest["episode_count"] == 1
    assert manifest["example_counts"]["train"] + (
        manifest["example_counts"]["validation"]
    ) + manifest["example_counts"]["test"] == 1
    assert report["passed"] is True
    assert report["example_count"] == 1
    assert report["episode_count"] == 1
    assert sum(manifest["perception_example_counts"].values()) == 1
    assert report["teacher_score_distribution"]["mean"] == 4.75


def test_long_trajectory_is_retained_in_holdout_not_short_sft(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    output = tmp_path / "long-holdout"

    manifest = export_dataset(
        [run_dir],
        output,
        short_max_tokens=1,
    )

    assert manifest["accepted_episode_count"] == 1
    assert manifest["long_holdout_episode_count"] == 1
    assert manifest["short_sft_case_count"] == 0
    assert all(
        not (output / f"{split}.jsonl").read_text(encoding="utf-8").strip()
        for split in ("train", "validation", "test")
    )
    holdout = [
        json.loads(line)
        for line in (output / "long_holdout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(holdout) == 1
    assert holdout[0]["holdout_reason"] == (
        "trajectory_exceeds_short_token_budget"
    )


def test_cross_case_source_families_exclude_domain_fallbacks() -> None:
    trace = {
        "state": {
            "investigation_state": {
                "evidence": [
                    {"source_family": "domain:wikipedia.org"},
                    {"source_family": "domain:facebook.com"},
                    {"source_family": "content:" + "a" * 64},
                    {"source_family": "content:" + "a" * 64},
                ]
            }
        }
    }

    assert _cross_case_source_families(trace) == ["content:" + "a" * 64]


def test_dataset_audit_rejects_private_policy_input(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    output = tmp_path / "dataset"
    export_dataset(
        [run_dir],
        output,
        short_max_tokens=300_000,
    )
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
    rows[0]["messages"][0]["gold"] = {"verdict": "real"}
    _write_jsonl(split_path, rows)

    report = audit_dataset(output)
    codes = {item["code"] for item in report["errors"]}

    assert report["passed"] is False
    assert "PRIVATE_DATA_LEAK" in codes


def test_dataset_export_excludes_quality_gate_failures(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": "case_scripted_v3",
                "total": 2.0,
                "components": {},
                "training_eligible": False,
                "training_exclusion_reasons": [
                    "semantic_duplicate_actions"
                ],
            }
        ],
    )
    output = tmp_path / "dataset"

    manifest = export_dataset([run_dir], output)
    report = audit_dataset(output)

    assert manifest["candidate_episode_count"] == 1
    assert manifest["episode_count"] == 0
    assert manifest["excluded_episode_count"] == 1
    assert report["passed"] is True
    assert report["example_count"] == 0
    assert report["excluded_episode_count"] == 1


def _write_frozen_gate_inputs(
    tmp_path: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    trace_path = run_dir / "traces" / "case_scripted_v3.json"
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    split = tmp_path / "case_split.jsonl"
    split.write_text(
        json.dumps(
            {
                "case_id": "case_scripted_v3",
                "image_sha256": json.loads(
                    trace_path.read_text(encoding="utf-8")
                )["state"]["runtime_case"]["image_sha256"],
                "split": "validation",
                "split_group_id": "group-frozen",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eligibility = tmp_path / "eligibility"
    eligibility.mkdir()
    (eligibility / "episode.sft_eligibility.json").write_text(
        json.dumps(
            {
                "case_id": "case_scripted_v3",
                "episode_id": "case_scripted_v3",
                "artifact_id": "sha256:eligibility",
                "source_trace": {"sha256": trace_sha},
                "gates": {
                    "strict_trace_audit_pass": True,
                    "engineering_valid": True,
                    "sft_eligibility_pass": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return split, eligibility


def test_frozen_sft_export_uses_structured_gate_without_semantic_reward(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    split, eligibility = _write_frozen_gate_inputs(tmp_path, run_dir)
    output = tmp_path / "frozen-dataset"
    manifest = export_dataset(
        [run_dir],
        output,
        case_split_path=split,
        eligibility_dir=eligibility,
        minimum_accepted_cases=1,
        short_max_tokens=300_000,
    )

    assert manifest["split_mode"] == "frozen_teacher_sft"
    assert manifest["frozen_inputs"]["semantic_reward_dir"] is None
    assert manifest["accepted_case_count"] == 1
    assert manifest["example_counts"]["validation"] == 1
    assert manifest["example_counts"]["train"] == 0


def test_frozen_sft_export_consumes_canonical_accepted_release(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    split, eligibility = _write_frozen_gate_inputs(tmp_path, run_dir)
    canonical = tmp_path / "accepted-release"
    canonical.mkdir()
    shutil.copy2(run_dir / "run_manifest.json", canonical / "run_manifest.json")
    shutil.copy2(
        run_dir / "trajectory_sft.jsonl",
        canonical / "trajectory_sft.jsonl",
    )
    shutil.copy2(
        run_dir / "perception_trajectories.jsonl",
        canonical / "perception_trajectories.jsonl",
    )
    shutil.copytree(run_dir / "traces", canonical / "traces")
    shutil.copytree(eligibility, canonical / "eligibility")
    trace_path = run_dir / "traces" / "case_scripted_v3.json"
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (canonical / "selected_episodes.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case_scripted_v3",
                "episode_id": "case_scripted_v3",
                "source_trace_sha256": trace_sha,
                "teacher_score": 4.75,
                "deterministic_hard_gate_pass": True,
                "deterministic_fatal_reasons": [],
                "deterministic_red_flags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (canonical / "accepted_release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ifv-accepted-teacher-release-v2",
                "accepted_case_count": 1,
                "rejected_case_count": 0,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "canonical-dataset"
    manifest = export_dataset(
        [],
        output,
        accepted_release_path=canonical,
        case_split_path=split,
        minimum_accepted_cases=1,
    )

    assert manifest["source_mode"] == "canonical_accepted_release"
    assert manifest["accepted_case_count"] == 1
    assert manifest["frozen_inputs"]["accepted_release"]["path"] == str(
        canonical.resolve()
    )


def test_frozen_sft_export_records_nonfatal_deterministic_red_flags(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": "case_scripted_v3",
                "episode_id": "case_scripted_v3",
                "total": 3.0,
                "components": {"result_reward": 1.0},
                "training_eligible": False,
                "training_exclusion_reasons": ["evidence_chain_incomplete"],
            }
        ],
    )
    split, eligibility = _write_frozen_gate_inputs(tmp_path, run_dir)
    output = tmp_path / "frozen-nonfatal"

    manifest = export_dataset(
        [run_dir],
        output,
        case_split_path=split,
        eligibility_dir=eligibility,
        minimum_accepted_cases=1,
        short_max_tokens=300_000,
    )
    metadata = [
        json.loads(line)
        for line in (output / "episode_metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][0]

    assert manifest["accepted_case_count"] == 1
    assert metadata["deterministic_hard_gate_pass"] is True
    assert metadata["deterministic_red_flags"] == [
        "evidence_chain_incomplete"
    ]


def test_frozen_sft_export_records_protocol_rejections_as_nonfatal_red_flags(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": "case_scripted_v3",
                "episode_id": "case_scripted_v3",
                "total": 3.0,
                "components": {"result_reward": 1.0},
                "training_eligible": False,
                "training_exclusion_reasons": ["protocol_rejections"],
            }
        ],
    )
    split, eligibility = _write_frozen_gate_inputs(tmp_path, run_dir)
    output = tmp_path / "frozen-protocol-red-flag"

    manifest = export_dataset(
        [run_dir],
        output,
        case_split_path=split,
        eligibility_dir=eligibility,
        minimum_accepted_cases=1,
        short_max_tokens=300_000,
    )
    metadata = [
        json.loads(line)
        for line in (output / "episode_metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][0]

    assert manifest["accepted_case_count"] == 1
    assert metadata["deterministic_hard_gate_pass"] is True
    assert metadata["deterministic_red_flags"] == ["protocol_rejections"]


def test_frozen_sft_export_rejects_actual_fatal_deterministic_red_flags(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    _write_jsonl(
        run_dir / "trajectory_scores.jsonl",
        [
            {
                "case_id": "case_scripted_v3",
                "episode_id": "case_scripted_v3",
                "total": 3.0,
                "components": {"result_reward": 1.0},
                "training_eligible": False,
                "training_exclusion_reasons": ["engineering_error"],
            }
        ],
    )
    split, eligibility = _write_frozen_gate_inputs(tmp_path, run_dir)

    with pytest.raises(ValueError, match="accepted too few cases"):
        export_dataset(
            [run_dir],
            tmp_path / "frozen-fatal",
            case_split_path=split,
            eligibility_dir=eligibility,
            minimum_accepted_cases=1,
            short_max_tokens=300_000,
        )


def test_frozen_sft_export_rejects_missing_validation_gate(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    split = tmp_path / "case_split.jsonl"
    split.write_text(
        json.dumps(
            {
                "case_id": "case_scripted_v3",
                "image_sha256": "0" * 64,
                "split": "validation",
                "split_group_id": "group-frozen",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="eligibility-dir"):
        export_dataset(
            [run_dir],
            tmp_path / "frozen-dataset",
            case_split_path=split,
        )
