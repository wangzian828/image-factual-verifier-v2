from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.trajectory.run_teacher_rollout_autopilot import (
    _attempt_command_log_path,
    _candidate_trace_sources,
    _classify_initial_outcomes,
    _build_quality_reroll_summary,
    _has_early_correct_judgment,
    _quality_reroll_case_ids,
    _read_jsonl,
    _successful_trace_sources,
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


def test_attempt_launcher_log_is_outside_run_cases_output(tmp_path: Path) -> None:
    group = tmp_path / "rollouts" / "initial"
    attempt = group / "attempt-01"
    log = _attempt_command_log_path(group, 1)

    assert log == group / "logs" / "attempt-01.log"
    assert attempt not in log.parents


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
