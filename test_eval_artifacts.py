from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.eval import run_eval


def test_classification_prediction_accepts_only_binary_v4_verdicts() -> None:
    sample = {"case_id": "case-binary"}
    assert run_eval._classification_prediction(
        sample,
        {"verdict": "real", "termination": "success", "error": ""},
    ) == {"case_id": "case-binary", "verdict": "real"}
    assert run_eval._classification_prediction(
        sample,
        {"verdict": "fake", "termination": "success", "error": ""},
    ) == {"case_id": "case-binary", "verdict": "fake"}
    assert run_eval._classification_prediction(
        sample,
        {"verdict": "unverifiable", "termination": "success", "error": ""},
    ) is None


def test_private_index_accepts_candidate_id_for_runtime_case_id() -> None:
    indexed = run_eval._private_index(
        [{"candidate_id": "case-candidate", "label": "supported"}],
        key="case_id",
        name="evaluation-gold",
    )

    assert indexed["case-candidate"]["label"] == "supported"


def test_safe_trace_filename_matches_workflow_sanitization() -> None:
    assert run_eval._safe_trace_filename(
        "baseline:historical:generated:case-01"
    ) == "baseline_historical_generated_case-01.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_release(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "release"
    runtime_dir = root / "runtime_input"
    image = runtime_dir / "assets" / "sha256" / "fixture.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"runtime-image-fixture")
    case_id = "case_0123456789abcdef"
    benchmark = runtime_dir / "cases.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "image_path": "assets/sha256/fixture.jpg",
                "image_sha256": _sha256(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold = root / "evaluator_private" / "gold.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "factual_status": "supported",
                "decisive_facts": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(root / "evaluation" / "classification_protocol.json", {})
    _write_json(root / "evaluation" / "process_reference_protocol.json", {})
    licenses = root / "metadata" / "licenses.jsonl"
    licenses.parent.mkdir(parents=True)
    licenses.write_text("", encoding="utf-8")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ifv-image-only-benchmark-release-v0.3",
            "release_id": "image-only-eval-fixture",
            "release_stage": "development_subset",
            "runtime_contract_version": "ifv-image-only-runtime-v1",
            "input_mode": "image_only",
            "decision_policy_version": "reinspect-v2",
            "runtime_contract": {
                "allowed_keys": ["case_id", "image_path", "image_sha256"],
                "private_keys_absent": True,
            },
            "artifacts": {
                "agent_input": "runtime_input/cases.jsonl",
                "evaluation_gold": "evaluator_private/gold.jsonl",
                "classification_protocol": (
                    "evaluation/classification_protocol.json"
                ),
                "process_reference_protocol": (
                    "evaluation/process_reference_protocol.json"
                ),
                "licenses": "metadata/licenses.jsonl",
            },
            "source_access_policy": {"active": False},
        },
    )
    return benchmark, gold


def _build_scoring_release(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "scoring-release"
    image = root / "runtime_input" / "assets" / "sha256" / "fixture.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"scoring-runtime-image-fixture")
    case_id = "case_scoring_fixture"
    benchmark = root / "runtime_input" / "cases.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "image_path": "runtime_input/assets/sha256/fixture.jpg",
                "image_sha256": _sha256(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold = root / "evaluator_private" / "gold.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(
        json.dumps(
            {
                "schema_version": "ifv-scoring-gold-v1",
                "case_id": case_id,
                "label": "supported",
                "decisive_fact": {
                    "statement": "The depicted relation is correct.",
                    "visual_anchors": ["fixture"],
                },
                "claim_atom": {
                    "subject": "fixture",
                    "event": "fixture event",
                    "slot": "depicted_relation",
                    "depicted_value": "correct",
                },
                "key_error": None,
                "evidence_target": {
                    "binding": {
                        "subject": "fixture",
                        "event": "fixture event",
                        "slot": "depicted_relation",
                        "depicted_value": "correct",
                        "match": "same_relation",
                    },
                    "required_directness": "direct",
                    "required_stance": "supports",
                    "statement": "correct",
                },
                "boundary": {
                    "directly_decides": "The depicted relation is correct.",
                    "does_not_prove": ["Every adjacent relation is correct."],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    migration = root / "evaluator_private" / "migration_audit.json"
    _write_json(migration, {"status": "fixture"})
    (root / "SHA256SUMS").write_text("fixture\n", encoding="utf-8")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ifv-existing-eval-package-v1",
            "release_id": "ifv-scoring-fixture",
            "input_mode": "image_only",
            "counts": {"cases": 1, "supported": 1, "refuted": 0},
            "runtime_contract": {
                "allowed_keys": ["case_id", "image_path", "image_sha256"],
                "private_keys_absent": True,
            },
            "artifacts": {
                "runtime_input": "runtime_input/cases.jsonl",
                "gold": "evaluator_private/gold.jsonl",
                "migration_audit": "evaluator_private/migration_audit.json",
            },
        },
    )
    return benchmark, gold


def _args(benchmark: Path, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=str(benchmark),
        provider="gemini",
        model="fixture-model",
        vlm_provider=None,
        vlm_model=None,
        llm_wire_api="interactions",
        vlm_wire_api="interactions",
        output_dir=str(run_dir),
        concurrency=1,
        rollouts_per_case=1,
        base_sampling_seed=1729,
        timeout=30.0,
        limit=None,
        shard_count=1,
        shard_index=0,
        training_prohibited=False,
        source_access_policy=None,
    )


def test_v03_eval_keeps_gold_post_rollout_and_writes_scorer_predictions(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark, gold = _build_release(tmp_path)
    run_dir = tmp_path / "run"
    rollout_finished = False

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config
            assert config.decision_policy_version == "unified-react-v1"

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal rollout_finished
            cases = kwargs["runtime_cases"]
            assert len(kwargs["sampling_seeds"]) == 1
            assert len(cases) == 1
            case = cases[0]
            assert set(case.model_dump()) == {
                "case_id",
                "image_path",
                "image_sha256",
            }
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            episode_id = kwargs["image_ids"][0]
            (trace_dir / f"{episode_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": episode_id,
                        "input_mode": "image_only",
                        "decision_policy_version": "unified-react-v1",
                        "verdict": "real",
                        "termination": "success",
                        "state": {
                            "image_id": episode_id,
                            "input_mode": "image_only",
                            "decision_policy_version": "unified-react-v1",
                            "runtime_case": case.model_dump(),
                            "investigation_state": {
                                "facts": [],
                                "tasks": [],
                                "evidence": [],
                                "findings": [],
                                "decisive_fact_ids": [],
                                "action_count": 0,
                            },
                            "all_steps": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            initial_manifest = json.loads(
                (trace_dir.parent / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            assert initial_manifest["benchmark"]["evaluation_gold"] is None
            rollout_finished = True
            return [
                {
                    "image_id": case.case_id,
                    "image_path": case.image_path,
                    "verdict": "real",
                    "confidence": 0.9,
                    "verdict_basis": {"fact_ids": ["vf-1"]},
                    "termination": "success",
                    "time_taken": 1.0,
                    "total_tool_calls": 2,
                    "llm_api_calls": 3,
                    "token_usage": {"total": 100},
                    "state": {"stage_timings": {"total": 1.0}},
                }
            ]

    real_load = run_eval._load_benchmark

    def guarded_load(path: Path) -> list[dict[str, Any]]:
        if path == gold:
            assert rollout_finished, "private gold loaded before rollout"
        return real_load(path)

    monkeypatch.setattr(run_eval, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_eval, "_load_benchmark", guarded_load)

    summary = asyncio.run(run_eval._run_eval(_args(benchmark, run_dir)))

    assert summary["num_samples"] == 1
    assert summary["num_errors"] == 0
    assert "accuracy" not in summary
    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["case_id"] == "case_0123456789abcdef"
    assert prediction["verdict"] == "real"
    assert set(prediction) == {"case_id", "verdict"}
    run_result = json.loads(
        (run_dir / "run_results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert run_result["verdict_basis"] == {"fact_ids": ["vf-1"]}
    assert run_result["trace_path"] == (
        "traces/case_0123456789abcdef.json"
    )
    assert set(run_result).isdisjoint(
        {"ground_truth", "factual_status", "decisive_facts"}
    )
    assert (run_dir / "process_metrics.jsonl").is_file()
    assert (run_dir / "reference_chain_metrics.jsonl").is_file()
    assert (run_dir / "trajectory_scores.jsonl").is_file()
    assert (run_dir / "trajectory_sft.jsonl").is_file()
    assert (run_dir / "perception_trajectories.jsonl").is_file()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["benchmark"]["release_id"] == "image-only-eval-fixture"
    assert manifest["benchmark"]["input_mode"] == "image_only"
    assert manifest["benchmark"]["decision_policy_version"] == (
        "reinspect-v2"
    )
    assert manifest["agent"]["decision_policy_version"] == (
        "unified-react-v1"
    )
    assert manifest["agent"]["stage_thinking_levels"]["unified_react"] == "low"
    assert manifest["agent"]["stage_thinking_levels"][
        "unified_discrepancy_decision"
    ] == "low"
    assert manifest["agent"]["qwen_stage_enable_thinking"]["unified_react"] == "false"
    assert manifest["agent"]["qwen_stage_enable_thinking"][
        "unified_judgment"
    ] == "false"
    assert manifest["benchmark"]["evaluation_gold"]["sha256"] == _sha256(gold)
    assert manifest["source_access_policy"]["active"] is False
    assert manifest["artifacts"]["run_results"] == "run_results.jsonl"
    assert manifest["artifacts"]["process_metrics"] == "process_metrics.jsonl"
    assert manifest["artifacts"]["reference_chain_metrics"] == (
        "reference_chain_metrics.jsonl"
    )
    assert manifest["artifacts"]["trajectory_sft"] == "trajectory_sft.jsonl"
    assert manifest["artifacts"]["perception_trajectories"] == (
        "perception_trajectories.jsonl"
    )
    assert manifest["artifacts"]["rollout_groups"] == "rollout_groups.jsonl"
    assert manifest["artifacts"]["episode_predictions"] == (
        "episode_predictions.jsonl"
    )
    assert (run_dir / "post_rollout_rewards.jsonl").is_file()


def test_scoring_release_keeps_gold_post_rollout_and_requires_structured_gate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark, gold = _build_scoring_release(tmp_path)
    run_dir = tmp_path / "scoring-run"
    rollout_finished = False

    class ScoringWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal rollout_finished
            case = kwargs["runtime_cases"][0]
            episode_id = kwargs["image_ids"][0]
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{episode_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": episode_id,
                        "decision_policy_version": "unified-react-v1",
                        "verdict": "real",
                        "termination": "success",
                        "state": {
                            "image_id": episode_id,
                            "termination": "success",
                            "runtime_case": case.model_dump(),
                            "all_steps": [],
                            "investigation_state": {
                                "target_facts": [],
                                "evidence": [],
                                "findings": [],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            initial = json.loads(
                (trace_dir.parent / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            assert initial["benchmark"]["evaluation_gold"] is None
            rollout_finished = True
            return [
                {
                    "image_id": episode_id,
                    "image_path": case.image_path,
                    "verdict": "real",
                    "confidence": 0.8,
                    "termination": "success",
                    "time_taken": 1.0,
                    "total_tool_calls": 0,
                    "llm_api_calls": 1,
                    "token_usage": {"total": 10},
                    "state": {},
                }
            ]

    real_load = run_eval._load_benchmark

    def guarded_load(path: Path) -> list[dict[str, Any]]:
        if path == gold:
            assert rollout_finished, "private scoring gold loaded before rollout"
        return real_load(path)

    monkeypatch.setattr(run_eval, "VerificationWorkflow", ScoringWorkflow)
    monkeypatch.setattr(run_eval, "_load_benchmark", guarded_load)
    monkeypatch.setattr(
        run_eval,
        "score_process_trace",
        lambda *args, **kwargs: (
            {
                "schema_version": "ifv-process-metrics-v4",
                "case_id": "case_scoring_fixture",
                "result_correct": True,
                "engineering_error": False,
                "training_eligible": True,
                "training_exclusion_reasons": [],
            },
            {
                "schema_version": "ifv-trajectory-score-v4",
                "case_id": "case_scoring_fixture",
                "training_eligible": True,
                "training_exclusion_reasons": [],
            },
        ),
    )
    monkeypatch.setattr(
        run_eval,
        "audit_trace",
        lambda _path, **_options: SimpleNamespace(
            failures=lambda **_kwargs: []
        ),
    )

    summary = asyncio.run(run_eval._run_eval(_args(benchmark, run_dir)))

    assert summary["num_errors"] == 0
    member = json.loads(
        (run_dir / "rollout_groups.jsonl").read_text(encoding="utf-8").strip()
    )
    assert member["sft_structured_eligibility_required"] is True
    assert member["training_eligible"] is False
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["benchmark"]["evaluation_gold"]["sha256"] == _sha256(gold)


def test_rollout_specs_are_stable_unique_and_grouped() -> None:
    runtime_case = SimpleNamespace(case_id="case-a")
    samples = [{"case_id": "case-a", "image_path": "fixture.jpg"}]
    first = run_eval._rollout_specs(
        samples,
        [runtime_case],
        rollouts_per_case=4,
        base_sampling_seed=7,
        policy_revision="commit-a",
        model="Qwen3.5-9B",
    )
    second = run_eval._rollout_specs(
        samples,
        [runtime_case],
        rollouts_per_case=4,
        base_sampling_seed=7,
        policy_revision="commit-a",
        model="Qwen3.5-9B",
    )
    assert first == second
    assert len({item["episode_id"] for item in first}) == 4
    assert len({item["sampling_seed"] for item in first}) == 4
    assert len({item["prompt_group_id"] for item in first}) == 1


def test_eval_cli_exits_nonzero_for_engineering_errors(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    async def failed_eval(_args: argparse.Namespace) -> dict[str, Any]:
        return {"num_samples": 1, "num_errors": 1}

    monkeypatch.setattr(run_eval, "_parse_args", lambda: argparse.Namespace())
    monkeypatch.setattr(run_eval, "_run_eval", failed_eval)

    try:
        run_eval.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("engineering errors must produce a non-zero process exit")

    assert '"num_errors": 1' in capsys.readouterr().out


def test_eval_selects_explicit_ordered_case_ids_before_limit() -> None:
    samples = [
        {"case_id": "case_a"},
        {"case_id": "case_b"},
        {"case_id": "case_c"},
    ]

    selected = run_eval._select_samples(
        samples,
        requested_case_ids=["case_c", "case_a"],
        limit=2,
    )

    assert [item["case_id"] for item in selected] == ["case_c", "case_a"]


def test_eval_shards_sorted_case_ids_without_overlap() -> None:
    samples = [
        {"case_id": "case_d"},
        {"case_id": "case_b"},
        {"case_id": "case_a"},
        {"case_id": "case_c"},
    ]

    shard_a = run_eval._select_samples(
        samples,
        requested_case_ids=None,
        limit=None,
        shard_count=2,
        shard_index=0,
    )
    shard_b = run_eval._select_samples(
        samples,
        requested_case_ids=None,
        limit=None,
        shard_count=2,
        shard_index=1,
    )

    assert [item["case_id"] for item in shard_a] == ["case_a", "case_c"]
    assert [item["case_id"] for item in shard_b] == ["case_b", "case_d"]
    assert {item["case_id"] for item in shard_a}.isdisjoint(
        {item["case_id"] for item in shard_b}
    )


def test_eval_rejects_unknown_or_duplicate_explicit_case_ids() -> None:
    samples = [{"case_id": "case_a"}, {"case_id": "case_b"}]

    try:
        run_eval._select_samples(
            samples,
            requested_case_ids=["case_a", "case_a"],
            limit=None,
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate explicit case IDs must fail")

    try:
        run_eval._select_samples(
            samples,
            requested_case_ids=["case_missing"],
            limit=None,
        )
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("unknown explicit case IDs must fail")


def test_eval_manifest_commit_rejects_environment_mismatch(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GIT_COMMIT", "configured")
    monkeypatch.setattr(
        run_eval.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="actual\n"),
    )

    try:
        run_eval._git_commit()
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a configured commit must not override actual HEAD")
