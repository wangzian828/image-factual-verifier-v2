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
        concurrency: int,
    ) -> list[dict[str, Any]]:
        assert concurrency == 1
        assert user_claims == [None]
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
    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["trace_path"] == "traces/sample-1.json"
    assert "plan" not in prediction
    assert "verification" not in prediction
    assert not list(run_dir.rglob("*.html"))
    assert not (run_dir / "failures.json").exists()
