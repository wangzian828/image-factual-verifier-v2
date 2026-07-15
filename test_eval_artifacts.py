from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval import run_eval


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
        timeout=30.0,
        limit=None,
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

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal rollout_finished
            cases = kwargs["runtime_cases"]
            assert len(cases) == 1
            case = cases[0]
            assert set(case.model_dump()) == {
                "case_id",
                "image_path",
                "image_sha256",
            }
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{case.case_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": case.case_id,
                        "input_mode": "image_only",
                        "decision_policy_version": "reinspect-v2",
                        "verdict": "real",
                        "termination": "success",
                        "state": {
                            "image_id": case.case_id,
                            "input_mode": "image_only",
                            "decision_policy_version": "reinspect-v2",
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
    assert (run_dir / "policy_trajectories.jsonl").is_file()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["benchmark"]["release_id"] == "image-only-eval-fixture"
    assert manifest["benchmark"]["input_mode"] == "image_only"
    assert manifest["benchmark"]["decision_policy_version"] == "reinspect-v2"
    assert manifest["benchmark"]["evaluation_gold"]["sha256"] == _sha256(gold)
    assert manifest["source_access_policy"]["active"] is False
    assert manifest["artifacts"]["run_results"] == "run_results.jsonl"
    assert manifest["artifacts"]["process_metrics"] == "process_metrics.jsonl"
    assert manifest["artifacts"]["reference_chain_metrics"] == (
        "reference_chain_metrics.jsonl"
    )


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
