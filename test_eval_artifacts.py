from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.eval import run_eval


class FakeWorkflow:
    def __init__(self, config: Any) -> None:
        self.config = config

    async def run_batch(
        self,
        *,
        image_paths: list[str],
        image_ids: list[str],
        user_claims: list[str | None],
        claim_observed_ats: list[str | None],
        verification_cases: list[Any] | None = None,
        concurrency: int,
    ) -> list[dict[str, Any]]:
        assert concurrency == 1
        assert user_claims == [None]
        assert claim_observed_ats == ["2024-01-02"]
        assert verification_cases == [None]
        trace_dir = Path(self.config.output_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{image_ids[0]}.json").write_text(
            json.dumps({"image_id": image_ids[0], "verdict": "real"}),
            encoding="utf-8",
        )
        return [
            {
                "image_id": image_ids[0],
                "image_path": image_paths[0],
                "verdict": "real",
                "confidence": 0.9,
                "overall_assessment": "Grounded fixture result.",
                "termination": "success",
                "time_taken": 1.0,
                "total_tool_calls": 2,
                "llm_api_calls": 3,
                "token_usage": {"total": 100},
                "state": {"stage_timings": {"total": 1.0}},
            }
        ]


def test_eval_writes_compact_canonical_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "bucket": "fixture",
                "ground_truth": "real",
                "image_path": str(tmp_path / "image.jpg"),
                "claim_observed_at": "2024-01-02",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    args = argparse.Namespace(
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
        max_verification_iterations=None,
        max_rounds_verification=12,
        limit=None,
        source_access_policy=None,
        evaluator_private=None,
        evaluation_gold=None,
    )
    monkeypatch.setattr(run_eval, "VerificationWorkflow", FakeWorkflow)

    summary = asyncio.run(run_eval._run_eval(args))

    assert summary["num_samples"] == 1
    assert summary["num_errors"] == 0
    assert {path.name for path in run_dir.iterdir()} == {
        "run_manifest.json",
        "predictions.jsonl",
        "summary.json",
        "traces",
    }
    assert {path.name for path in (run_dir / "traces").iterdir()} == {
        "sample-1.json"
    }
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["benchmark"]["sample_count"] == 1
    assert manifest["agent"]["max_verification_iterations"] == 4
    assert manifest["agent"]["min_verification_iterations"] == 2
    assert manifest["agent"]["low_information_gain_patience"] == 2
    assert manifest["agent"]["verification_max_output_tokens"] == 16384
    assert manifest["agent"]["verification_final_max_output_tokens"] == 32768
    assert manifest["agent"]["verification_final_thinking_level"] == "minimal"
    assert manifest["agent"]["stage_thinking_levels"] == {
        "planning": "minimal",
        "verification": "minimal",
        "replanning": "minimal",
        "judgment": "minimal",
    }
    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["trace_path"] == "traces/sample-1.json"
    assert "plan" not in prediction
    assert "verification" not in prediction
    assert not list(run_dir.rglob("*.html"))
    assert not (run_dir / "failures.json").exists()


def test_v02_release_loads_private_labels_only_after_rollout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    release_root = tmp_path / "release"
    runtime_dir = release_root / "runtime_input"
    private_dir = release_root / "evaluator_private"
    gold_dir = release_root / "evaluation_gold"
    runtime_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)
    runtime_path = runtime_dir / "cases.jsonl"
    runtime_path.write_text(
        json.dumps(
            {
                "case_id": "case-v02",
                "image_path": "assets/sha256/00/image.jpg",
                "image_sha256": "0" * 64,
                "claim_mode": "external_claim",
                "user_claim": "A public test claim.",
                "claim_surface": None,
                "claim_source_region": None,
                "claim_observed_at": "2024-01-02",
                "decision_policy_version": "reinspect-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (private_dir / "run_eval.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "case-v02",
                "image_path": str(runtime_dir / "assets/sha256/00/image.jpg"),
                "user_claim": "A public test claim.",
                "ground_truth": "real",
                "bucket": "real_seed_external_claim",
                "source_article_url": "https://factcheck.example/case-v02",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (gold_dir / "gold.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-v02",
                "factual_status": "supported",
                "target_claim": "A public test claim.",
                "acceptable_evidence": [],
                "core_loop_gold": None,
                "unverifiable_reasons": [],
                "image_origin": "camera",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (private_dir / "source_access_policy.json").write_text(
        json.dumps(
            {
                "schema_version": "source-access-policy-v1",
                "policy_id": "release-test",
                "excluded_domains": ["factcheck.example"],
                "excluded_urls": ["https://factcheck.example/case-v02"],
            }
        ),
        encoding="utf-8",
    )

    rollout_finished = False

    class ReleaseWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal rollout_finished
            cases = kwargs["verification_cases"]
            assert cases[0].case_id == "case-v02"
            assert cases[0].model_dump().keys().isdisjoint(
                {"ground_truth", "factual_status", "acceptable_evidence"}
            )
            rollout_finished = True
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "case-v02.json").write_text(
                json.dumps({"image_id": "case-v02", "verdict": "real"}),
                encoding="utf-8",
            )
            return [
                {
                    "image_id": "case-v02",
                    "image_path": kwargs["image_paths"][0],
                    "verdict": "real",
                    "confidence": 0.9,
                    "overall_assessment": "Release fixture result.",
                    "termination": "success",
                    "time_taken": 1.0,
                    "total_tool_calls": 1,
                    "llm_api_calls": 1,
                    "token_usage": {"total": 10},
                    "state": {"stage_timings": {"total": 1.0}},
                }
            ]

    real_load = run_eval._load_benchmark

    def guarded_load(path: Path) -> list[dict[str, Any]]:
        if path in {
            private_dir / "run_eval.jsonl",
            gold_dir / "gold.jsonl",
        }:
            assert rollout_finished, f"private labels loaded before rollout: {path}"
        return real_load(path)

    run_dir = tmp_path / "run"
    args = argparse.Namespace(
        benchmark=str(runtime_path),
        provider="gemini",
        model="fixture-model",
        vlm_provider=None,
        vlm_model=None,
        llm_wire_api="interactions",
        vlm_wire_api="interactions",
        output_dir=str(run_dir),
        concurrency=1,
        timeout=30.0,
        max_verification_iterations=None,
        max_rounds_verification=12,
        limit=None,
        source_access_policy=None,
        evaluator_private=None,
        evaluation_gold=None,
    )
    monkeypatch.setattr(run_eval, "VerificationWorkflow", ReleaseWorkflow)
    monkeypatch.setattr(run_eval, "_load_benchmark", guarded_load)

    summary = asyncio.run(run_eval._run_eval(args))

    assert summary["evaluation_gold_metrics"]["accuracy"] == 1.0
    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["sample_id"] == "case-v02"
    assert prediction["ground_truth"] == "real"


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
