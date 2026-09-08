from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from scripts.trajectory.run_teacher_rollout_autopilot import (
    _attempt_command_log_path,
    _runtime_command,
    _candidate_trace_sources,
    _classify_initial_outcomes,
    _build_quality_reroll_summary,
    _bootstrap_completed_initial_pipeline,
    _has_early_correct_judgment,
    _merge_successful_attempts,
    _quality_reroll_case_ids,
    _read_jsonl,
    _sft_audit_complete,
    _state_model_config,
    _successful_trace_sources,
    _validate_rollout_profile_model,
    prepare_runtime_release,
)
from scripts.trajectory.build_sft_training_package import _assert_new_or_empty
from src.eval.public_release import load_public_release, resolve_image_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_row(case_id: str, *, factual_status: str, image: str) -> dict[str, Any]:
    return {
        "unified_case_id": case_id,
        "split": "train",
        "unified_image_path": image,
        "factual_status": factual_status,
        "target_claim": f"claim for {case_id}",
        "evidence": {"binding": {"source_evidence_span": "private evidence"}},
    }


def _trace(case_id: str, *, early_judgment: bool) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        {
            "stage": "unified_discrepancy_decision",
            "output": {"verdict_proposal": "fake"},
        }
    ]
    if early_judgment:
        steps.append(
            {
                "stage": "unified_judgment",
                "output": {"verdict": "fake"},
            }
        )
    return {
        "image_id": f"{case_id}--teacher-r000",
        "termination": "success",
        "verdict": "fake",
        "state": {
            "runtime_case": {"case_id": case_id},
            "all_steps": steps,
        },
    }


def test_old_pipeline_state_accepts_new_judge_key_environment() -> None:
    args = Namespace(
        rollout_profile="teacher-qwen-server",
        rollout_model="teacher-model",
        sft_judge_provider="qwen_local",
        sft_model="judge-model",
        sft_judge_base_url="http://judge.test/v1",
        sft_judge_api_key_env="PRIVATE_JUDGE_KEY",
        sft_judge_wire_api="chat_completions",
        sft_judge_enable_thinking=True,
    )
    state = {
        "model_config": {
            "rollout": {
                "profile": "teacher-qwen-server",
                "model": "teacher-model",
            },
            "sft_judge": {
                "provider": "qwen_local",
                "model": "judge-model",
                "base_url": "http://judge.test/v1",
                "wire_api": "chat_completions",
                "enable_thinking": True,
            },
        }
    }

    _, judge = _state_model_config(state, args)

    assert judge["api_key_env"] == "PRIVATE_JUDGE_KEY"


def test_rollout_profile_model_must_match_active_qwen_teacher(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QWEN_TEACHER_MODEL", "actual-teacher")
    monkeypatch.setenv("QWEN_TEACHER_BASE_URL", "http://teacher.test/v1")
    monkeypatch.setenv("QWEN_TEACHER_API_KEY", "teacher-key")
    monkeypatch.setenv("QWEN_LOCAL_API_KEY", "teacher-key")
    monkeypatch.delenv("QWEN_TEACHER_VISION_MODEL", raising=False)

    _validate_rollout_profile_model(
        profile="teacher-qwen-server",
        recorded_model="actual-teacher",
    )
    try:
        _validate_rollout_profile_model(
            profile="teacher-qwen-server",
            recorded_model="wrong-teacher",
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched teacher model must fail closed")

    monkeypatch.setenv("QWEN_TEACHER_VISION_MODEL", "different-vision-model")
    try:
        _validate_rollout_profile_model(
            profile="teacher-qwen-server",
            recorded_model="actual-teacher",
        )
    except ValueError as exc:
        assert "same model" in str(exc)
    else:
        raise AssertionError("teacher visual model drift must fail closed")

    monkeypatch.setenv("QWEN_TEACHER_VISION_MODEL", "actual-teacher")
    monkeypatch.setenv("QWEN_LOCAL_API_KEY", "stale-local-key")
    try:
        _validate_rollout_profile_model(
            profile="teacher-qwen-server",
            recorded_model="actual-teacher",
        )
    except ValueError as exc:
        assert "mapped" in str(exc)
    else:
        raise AssertionError("stale local Qwen credentials must fail closed")


def test_prepare_runtime_release_projects_only_runtime_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "unified-dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "one.jpg").write_bytes(b"one")
    (dataset / "images" / "two.jpg").write_bytes(b"two")
    train_manifest = dataset / "train-manifest.jsonl"
    train_manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _manifest_row("case-two", factual_status="supported", image="images/two.jpg"),
                _manifest_row("case-one", factual_status="refuted", image="images/one.jpg"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "pipeline"
    prepared = prepare_runtime_release(
        dataset_root=dataset,
        train_manifest=train_manifest,
        output_dir=output,
        limit=1,
    )

    benchmark = Path(prepared["benchmark"])
    release = load_public_release(benchmark)
    rows = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines()]
    assert [row["case_id"] for row in rows] == ["case-one"]
    assert all(set(row) == {"case_id", "image_path", "image_sha256"} for row in rows)
    assert all("factual_status" not in row for row in rows)
    assert all(
        Path(resolve_image_path(row, release=release, benchmark_path=benchmark)["image_path"]).is_file()
        for row in rows
    )
    private_gold = Path(prepared["private_gold"])
    assert private_gold.is_file()
    assert "private evidence" in private_gold.read_text(encoding="utf-8")
    assert "private evidence" not in benchmark.read_text(encoding="utf-8")
    assert prepared["runtime_cases_sha256"] == _sha256(benchmark)
    assert prepared["limit"] == 1
    assert len(_read_jsonl(Path(str(prepared["private_gold"])))) == 1
    case_split = Path(str(prepared["case_split"]))
    assert case_split.is_file()
    split_rows = _read_jsonl(case_split)
    assert split_rows[0]["split"] == "train"
    assert "label" not in split_rows[0]


def test_prepare_runtime_release_rejects_mutated_frozen_split(tmp_path: Path) -> None:
    dataset = tmp_path / "unified-dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "one.jpg").write_bytes(b"image-one")
    (dataset / "images" / "two.jpg").write_bytes(b"image-two")
    train_manifest = dataset / "train-manifest.jsonl"
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    _manifest_row(
                        "case-one",
                        factual_status="supported",
                        image="images/one.jpg",
                    )
                ),
                json.dumps(
                    _manifest_row(
                        "case-two",
                        factual_status="refuted",
                        image="images/two.jpg",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pipeline"
    prepared = prepare_runtime_release(
        dataset_root=dataset,
        train_manifest=train_manifest,
        output_dir=output,
        validation_count=1,
    )
    Path(str(prepared["case_split"])).write_text(
        '{"case_id":"tampered"}\n',
        encoding="utf-8",
    )

    try:
        prepare_runtime_release(
            dataset_root=dataset,
            train_manifest=train_manifest,
            output_dir=output,
            validation_count=1,
        )
    except ValueError as exc:
        assert "case split" in str(exc)
    else:
        raise AssertionError("mutated frozen split must fail closed")


def test_prepare_runtime_release_allows_precreated_logs_only(tmp_path: Path) -> None:
    dataset = tmp_path / "unified-dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "one.jpg").write_bytes(b"one")
    train_manifest = dataset / "train-manifest.jsonl"
    train_manifest.write_text(
        json.dumps(
            _manifest_row("case-one", factual_status="refuted", image="images/one.jpg")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pipeline"
    (output / "logs").mkdir(parents=True)
    (output / "logs" / "launcher.log").write_text("launched\n", encoding="utf-8")

    prepared = prepare_runtime_release(
        dataset_root=dataset,
        train_manifest=train_manifest,
        output_dir=output,
    )

    assert prepared["case_count"] == 1
    assert (output / "preparation.json").is_file()


def test_bootstrap_completed_initial_preserves_only_matching_audited_traces(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "unified-dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    for case_id in ("case-one", "case-two"):
        (images / f"{case_id}.jpg").write_bytes(case_id.encode("utf-8"))
    train_manifest = dataset / "train-manifest.jsonl"
    train_manifest.write_text(
        "\n".join(
            json.dumps(
                _manifest_row(
                    case_id,
                    factual_status="refuted",
                    image=f"images/{case_id}.jpg",
                )
            )
            for case_id in ("case-two", "case-one")
        )
        + "\n",
        encoding="utf-8",
    )

    source_run = tmp_path / "source-run"
    source_traces = source_run / "traces"
    source_traces.mkdir(parents=True)
    (source_run / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "git_commit": "source-commit",
                "benchmark": {"path": "/source/cases.jsonl"},
                "agent": {"model": "gemini-3.7-flash"},
                "source_access_policy": {"active": False},
            }
        ),
        encoding="utf-8",
    )
    source_eligibility = tmp_path / "source-eligibility"
    source_eligibility.mkdir()
    summary_rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(("case-one", "case-two")):
        trace = _trace(case_id, early_judgment=True)
        trace_path = source_traces / f"{trace['image_id']}.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        passed = index == 0
        artifact = {
            "schema_version": "ifv-sft-eligibility-v4",
            "case_id": case_id,
            "episode_id": trace["image_id"],
            "source_trace": {"sha256": _sha256(trace_path)},
            "metrics": {
                "expected_verdict": "fake",
                "target_scope": "direct_target",
                "decision_support": "supports_fake" if passed else "insufficient",
                "retrieval_quality": "effective" if passed else "poor",
                "decisive_evidence_ids": ["evidence-1"] if passed else [],
                "fatal_errors": [] if passed else ["no_decisive_evidence"],
                "warnings": [],
                "trajectory_conduct": "clean",
                "overclaiming": "none",
                "boundary_assessment": "respected",
            },
            "gates": {"sft_eligibility_pass": passed},
        }
        artifact_path = (
            source_eligibility / f"{trace['image_id']}.sft_eligibility.json"
        )
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        summary_rows.append(
            {
                "status": "success",
                "case_id": case_id,
                "episode_id": trace["image_id"],
            }
        )
    (source_eligibility / "sft_eligibility_summary.json").write_text(
        json.dumps({"rows": summary_rows}),
        encoding="utf-8",
    )

    output = tmp_path / "bootstrap-pipeline"
    args = Namespace(
        dataset_root=dataset,
        train_manifest=train_manifest,
        output_dir=output,
        source_access_policy=None,
        limit=None,
        bootstrap_initial_run=source_run,
        bootstrap_initial_eligibility=source_eligibility,
        rollout_model="gemini-3.7-flash",
        sft_model="gemini-3.7-flash",
        rollout_concurrency=10,
        sft_concurrency=10,
    )
    state, preparation, initial_run, initial_eligibility = (
        _bootstrap_completed_initial_pipeline(args)
    )

    assert state["status"] == "bootstrapped"
    assert preparation["case_count"] == 2
    assert len(list((initial_run / "traces").glob("*.json"))) == 2
    assert _sha256(
        initial_run / "traces" / "case-one--teacher-r000.json"
    ) == _sha256(source_traces / "case-one--teacher-r000.json")
    assert len(
        list(initial_eligibility.glob("*.sft_eligibility.json"))
    ) == 2
    assert (
        output / "classification" / "quality-reroll-case-list.txt"
    ).read_text(encoding="utf-8").splitlines() == ["case-two"]

    resumed, _, _, _ = _bootstrap_completed_initial_pipeline(args)
    assert resumed["bootstrap_initial"] == state["bootstrap_initial"]


def test_attempt_launcher_log_is_outside_run_cases_output(tmp_path: Path) -> None:
    group = tmp_path / "rollouts" / "initial"
    attempt = group / "attempt-01"
    log = _attempt_command_log_path(group, 1)

    assert log == group / "logs" / "attempt-01.log"
    assert attempt not in log.parents


def test_autopilot_uses_portable_server_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = tmp_path / "run-ifv"
    monkeypatch.setenv("IFV_SERVER_RUNNER", str(runner))

    command = _runtime_command("-m", "src.eval.run_cases")

    assert command[0] == str(runner.resolve())
    assert command[1]
    assert command[-2:] == ["-m", "src.eval.run_cases"]


def test_successful_merge_preserves_source_access_policy_metadata(
    tmp_path: Path,
) -> None:
    group = tmp_path / "rollouts" / "initial"
    attempt = group / "attempt-01"
    traces = attempt / "traces"
    traces.mkdir(parents=True)
    trace = _trace("case-one", early_judgment=True)
    (traces / "episode.json").write_text(
        json.dumps(trace),
        encoding="utf-8",
    )
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "git_commit": "commit-1",
                "benchmark": {"path": "/release/runtime_input/cases.jsonl"},
                "agent": {"model": "gemini-3.7-flash"},
                "source_access_policy": {
                    "active": True,
                    "path": "/release/evaluator_private/source_access_policy.json",
                    "policy_id": "fixture-policy",
                },
            }
        ),
        encoding="utf-8",
    )

    merged, manifest = _merge_successful_attempts(
        group_dir=group,
        group_name="initial",
        target_ids=["case-one"],
    )

    assert merged == group / "merged"
    assert manifest["git_commit"] == "commit-1"
    assert manifest["benchmark"]["path"].endswith("cases.jsonl")
    assert manifest["agent"]["model"] == "gemini-3.7-flash"
    assert manifest["source_access_policy"]["active"] is True
    assert manifest["source_access_policy"]["policy_id"] == "fixture-policy"


def test_successful_merge_reports_actual_terminal_episode_count(
    tmp_path: Path,
) -> None:
    group = tmp_path / "rollouts" / "initial"
    attempt = group / "attempt-01"
    traces = attempt / "traces"
    traces.mkdir(parents=True)
    trace = _trace("case-one", early_judgment=True)
    (traces / "episode.json").write_text(json.dumps(trace), encoding="utf-8")
    (attempt / "run_manifest.json").write_text(
        json.dumps({"status": "completed_with_errors"}),
        encoding="utf-8",
    )

    _, manifest = _merge_successful_attempts(
        group_dir=group,
        group_name="initial",
        target_ids=["case-one", "case-two"],
    )

    assert manifest["result"]["num_cases"] == 2
    assert manifest["result"]["num_episodes"] == 1
    assert manifest["result"]["num_errors"] == 1


def test_sft_audit_completion_is_bound_to_trace_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    traces = run_dir / "traces"
    traces.mkdir(parents=True)
    trace = _trace("case-one", early_judgment=True)
    trace_path = traces / "episode.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    eligibility = tmp_path / "eligibility"
    eligibility.mkdir()
    episode_id = trace["image_id"]
    artifact_path = eligibility / f"{episode_id}.sft_eligibility.json"
    artifact = {
        "episode_id": episode_id,
        "source_trace": {"sha256": "wrong"},
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    (eligibility / "sft_eligibility_summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "rows": [{"episode_id": episode_id}],
            }
        ),
        encoding="utf-8",
    )

    assert _sft_audit_complete(eligibility, 1, run_dir=run_dir) is False
    artifact["source_trace"]["sha256"] = _sha256(trace_path)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert _sft_audit_complete(eligibility, 1, run_dir=run_dir) is True


def test_package_output_allows_autopilot_command_log_only(tmp_path: Path) -> None:
    output = tmp_path / "sft-training-package"
    output.mkdir()
    (output / "command.log").write_text("launcher created this first\n", encoding="utf-8")

    _assert_new_or_empty(output)

    (output / "partial-artifact").write_text("do not reuse\n", encoding="utf-8")
    try:
        _assert_new_or_empty(output)
    except FileExistsError as exc:
        assert "partial-artifact" in str(exc)
    else:
        raise AssertionError("partial package output must be rejected")


def test_success_scan_keeps_only_trace_metadata(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-01"
    traces = attempt / "traces"
    traces.mkdir(parents=True)
    trace = _trace("case-one", early_judgment=True)
    trace["large_state_payload"] = "x" * 100_000
    (traces / "episode.json").write_text(
        json.dumps(trace),
        encoding="utf-8",
    )

    selected = _successful_trace_sources([attempt])

    assert selected["case-one"][0] == attempt
    assert selected["case-one"][1].name == "episode.json"
    assert selected["case-one"][2] == {
        "case_id": "case-one",
        "episode_id": "case-one--teacher-r000",
        "termination": "success",
        "verdict": "fake",
    }


def test_success_scan_ignores_provider_error_without_case_identity(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt-01"
    traces = attempt / "traces"
    traces.mkdir(parents=True)
    (traces / "failed-before-case-init.json").write_text(
        json.dumps(
            {
                "termination": "error",
                "verdict": "error",
                "error": "SSLError: [SSL] record layer failure",
            }
        ),
        encoding="utf-8",
    )
    successful = _trace("case-one", early_judgment=True)
    (traces / "successful.json").write_text(
        json.dumps(successful),
        encoding="utf-8",
    )

    selected = _successful_trace_sources([attempt])
    grouped = _candidate_trace_sources([attempt])

    assert set(selected) == {"case-one"}
    assert set(grouped) == {"case-one"}


def test_early_bucket_requires_strict_discrepancy_judgment(tmp_path: Path) -> None:
    proposal_only = _trace("case-proposal", early_judgment=False)
    explicit_judgment = _trace("case-judgment", early_judgment=True)
    assert _has_early_correct_judgment(proposal_only, "fake") is False
    assert _has_early_correct_judgment(explicit_judgment, "fake") is True

    run = tmp_path / "run"
    traces = run / "traces"
    traces.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    for trace in (proposal_only, explicit_judgment):
        (traces / f"{trace['image_id']}.json").write_text(
            json.dumps(trace),
            encoding="utf-8",
        )
    gold = tmp_path / "private-gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": case_id,
                    "factual_status": "refuted",
                    "target_claim": "private",
                }
            )
            for case_id in ("case-proposal", "case-judgment")
        )
        + "\n",
        encoding="utf-8",
    )
    eligibility = tmp_path / "eligibility"
    eligibility.mkdir()
    for trace in (proposal_only, explicit_judgment):
        (eligibility / f"{trace['image_id']}.sft_eligibility.json").write_text(
            json.dumps(
                {
                    "episode_id": trace["image_id"],
                    "gates": {"sft_eligibility_pass": True},
                }
            ),
            encoding="utf-8",
        )

    summary = _classify_initial_outcomes(
        run_dir=run,
        eligibility_dir=eligibility,
        private_gold=gold,
        output_dir=tmp_path / "classification",
    )

    assert summary["early_correct_judgment_count"] == 1
    assert summary["final_only_judgment_count"] == 1
    reroll = (tmp_path / "classification" / "quality-reroll-case-list.txt").read_text(
        encoding="utf-8"
    )
    assert reroll.splitlines() == ["case-proposal"]


def test_quality_reroll_queue_contains_only_sft_rejections_and_incomplete_cases(
    tmp_path: Path,
) -> None:
    classification = tmp_path / "classification"
    classification.mkdir()
    (classification / "sft-rejected.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("case-b", "case-a", "case-b")
        )
        + "\n",
        encoding="utf-8",
    )
    (classification / "incomplete-cases.jsonl").write_text(
        json.dumps({"case_id": "case-c"}) + "\n",
        encoding="utf-8",
    )
    (classification / "final-only-judgment.jsonl").write_text(
        json.dumps({"case_id": "case-passed-final-only"}) + "\n",
        encoding="utf-8",
    )

    assert _quality_reroll_case_ids(classification) == [
        "case-a",
        "case-b",
        "case-c",
    ]


def test_quality_reroll_queue_is_scoped_to_current_round_targets(
    tmp_path: Path,
) -> None:
    classification = tmp_path / "classification"
    classification.mkdir()
    (classification / "classification.json").write_text(
        json.dumps(
            {
                "target_case_ids": ["case-a", "case-b"],
                "source_run": str(tmp_path / "rollouts" / "initial" / "merged"),
            }
        ),
        encoding="utf-8",
    )
    (classification / "sft-rejected.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("case-a", "case-outside")
        )
        + "\n",
        encoding="utf-8",
    )
    (classification / "hard-cases.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("case-a", "case-outside")
        )
        + "\n",
        encoding="utf-8",
    )
    (classification / "incomplete-cases.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("case-b", "case-outside")
        )
        + "\n",
        encoding="utf-8",
    )

    assert _quality_reroll_case_ids(classification) == ["case-a", "case-b"]


def test_quality_reroll_queue_skips_case_with_one_selected_candidate(
    tmp_path: Path,
) -> None:
    classification = tmp_path / "classification"
    classification.mkdir()
    (classification / "classification.json").write_text(
        json.dumps(
            {
                "target_case_ids": ["case-selected", "case-hard"],
                "source_run": str(tmp_path / "rollouts" / "initial" / "merged"),
            }
        ),
        encoding="utf-8",
    )
    (classification / "sft-rejected.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case-selected"}),
                json.dumps({"case_id": "case-hard"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (classification / "hard-cases.jsonl").write_text(
        json.dumps({"case_id": "case-hard"}) + "\n",
        encoding="utf-8",
    )
    (classification / "incomplete-cases.jsonl").write_text("", encoding="utf-8")

    assert _quality_reroll_case_ids(classification) == ["case-hard"]


def test_quality_reroll_summary_selects_one_winner_per_case_across_rounds(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "classification" / "initial"
    reroll = tmp_path / "classification" / "quality-reroll-01"
    for directory in (initial, reroll):
        directory.mkdir(parents=True)

    (initial / "selected-candidates.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "episode_id": "case-a--r000",
                "bucket": "early_correct_judgment",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (initial / "sft-rejected.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-b",
                "episode_id": "case-b--r000",
                "expected_verdict": "fake",
                "rejection_reasons": ["poor_retrieval_quality"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (initial / "incomplete-cases.jsonl").write_text("", encoding="utf-8")

    (reroll / "selected-candidates.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-b",
                "episode_id": "case-b--r001",
                "bucket": "final_only_judgment",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (reroll / "sft-rejected.jsonl").write_text("", encoding="utf-8")
    (reroll / "incomplete-cases.jsonl").write_text("", encoding="utf-8")

    gold = tmp_path / "private-gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps({"case_id": case_id})
            for case_id in ("case-a", "case-b")
        )
        + "\n",
        encoding="utf-8",
    )
    summary = _build_quality_reroll_summary(
        pipeline_dir=tmp_path,
        private_gold=gold,
        classification_dirs=[initial, reroll],
        quality_rounds=[{"round": 1, "selected_case_count": 1}],
    )

    assert summary["selected_case_count"] == 2
    assert summary["early_correct_judgment_count"] == 1
    assert summary["final_only_judgment_count"] == 1
    assert summary["hard_case_count"] == 0


def test_candidate_collection_keeps_four_terminal_traces_per_case(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-01" / "traces"
    attempt.mkdir(parents=True)
    for index in range(4):
        trace = _trace("case-four", early_judgment=True)
        trace["image_id"] = f"case-four--r{index:03d}"
        (attempt / f"{trace['image_id']}.json").write_text(
            json.dumps(trace),
            encoding="utf-8",
        )

    grouped = _candidate_trace_sources([tmp_path / "attempt-01"])

    assert len(grouped["case-four"]) == 4
    assert [item[2]["episode_id"] for item in grouped["case-four"]] == [
        "case-four--r000",
        "case-four--r001",
        "case-four--r002",
        "case-four--r003",
    ]


def test_four_candidate_selection_picks_best_and_marks_all_rejected_hard_case(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    traces = run / "traces"
    traces.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    cases = ("case-best", "case-hard")
    for case_id in cases:
        for index in range(4):
            trace = _trace(case_id, early_judgment=True)
            trace["image_id"] = f"{case_id}--r{index:03d}"
            (traces / f"{trace['image_id']}.json").write_text(
                json.dumps(trace),
                encoding="utf-8",
            )
    gold = tmp_path / "private-gold.jsonl"
    gold.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": case_id,
                    "factual_status": "refuted",
                    "target_claim": "private",
                }
            )
            + "\n"
            for case_id in cases
        ),
        encoding="utf-8",
    )
    eligibility = tmp_path / "eligibility"
    eligibility.mkdir()
    for case_id in cases:
        for index in range(4):
            episode_id = f"{case_id}--r{index:03d}"
            passed = case_id == "case-best" and index in {1, 2}
            metrics = {
                "fact_alignment": "same_image_fact" if passed else "different_fact",
                "decision_support": "supports_fake" if passed else "insufficient",
                "retrieval_quality": "effective" if passed else "poor",
                "trajectory_conduct": "clean" if passed else "unresolved",
                "overclaiming": "none" if passed else "major",
                "boundary_assessment": "respected" if passed else "major_issue",
                "fatal_errors": [] if passed else ["different_image_fact"],
                "warnings": [],
            }
            (eligibility / f"{episode_id}.sft_eligibility.json").write_text(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "gates": {"sft_eligibility_pass": passed},
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )

    summary = _classify_initial_outcomes(
        run_dir=run,
        eligibility_dir=eligibility,
        private_gold=gold,
        output_dir=tmp_path / "classification",
        candidates_per_case=4,
    )

    selected = _read_jsonl(
        tmp_path / "classification" / "selected-candidates.jsonl"
    )
    hard_cases = _read_jsonl(tmp_path / "classification" / "hard-cases.jsonl")
    assert summary["selected_case_count"] == 1
    assert summary["hard_case_count"] == 1
    assert selected[0]["case_id"] == "case-best"
    assert selected[0]["episode_id"] in {"case-best--r001", "case-best--r002"}
    assert hard_cases[0]["case_id"] == "case-hard"
