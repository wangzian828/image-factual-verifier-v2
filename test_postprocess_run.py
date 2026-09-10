from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.eval import postprocess_run, run_cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    runtime_dir = root / "runtime_input"
    image = runtime_dir / "assets" / "sha256" / "fixture.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"postprocess-runtime-image-fixture")
    benchmark = runtime_dir / "cases.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "case_id": "case_postprocess",
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
                "case_id": "case_postprocess",
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
            "release_id": "postprocess-fixture",
            "release_stage": "formal_eval",
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
    return benchmark


def _case_args(benchmark: Path, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=str(benchmark),
        metadata=None,
        profile=None,
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
        case_id=None,
        case_list=None,
        shard_count=1,
        shard_index=0,
        source_access_policy=None,
    )


def _postprocess_args(run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=run_dir,
        benchmark=None,
        training_prohibited=False,
    )


def test_run_cases_then_postprocess_splits_runtime_and_gold_steps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark = _build_release(tmp_path)
    run_dir = tmp_path / "run"

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            case = kwargs["runtime_cases"][0]
            episode_id = kwargs["image_ids"][0]
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{episode_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": episode_id,
                        "input_mode": "image_only",
                        "decision_policy_version": "unified-react-v1",
                        "verdict": "real",
                        "termination": "success",
                        "state": {
                            "stage_timings": {"total": 1.0},
                            "investigation_state": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            return [
                {
                    "image_id": case.case_id,
                    "image_path": case.image_path,
                    "verdict": "real",
                    "confidence": 0.9,
                    "verdict_basis": {"claim_ids": ["claim-1"]},
                    "termination": "success",
                    "time_taken": 1.0,
                    "total_tool_calls": 2,
                    "llm_api_calls": 3,
                    "token_usage": {"total": 100},
                    "state": {"stage_timings": {"total": 1.0}},
                }
            ]

    async def reference_score(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "ifv-reference-chain-metrics-v2",
            "case_id": "case_postprocess",
            "metrics": {"fact_recovery_recall": 1.0},
        }

    monkeypatch.setattr(run_cases, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_cases, "_git_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        postprocess_run,
        "score_process_trace",
        lambda *_args, **_kwargs: (
            {
                "schema_version": "ifv-process-metrics-v2",
                "case_id": "case_postprocess",
                "result_correct": True,
                "engineering_error": False,
                "training_eligible": True,
            },
            {
                "schema_version": "ifv-trajectory-score-v2",
                "case_id": "case_postprocess",
                "components": {},
                "total": 1.0,
                "training_eligible": True,
            },
        ),
    )
    monkeypatch.setattr(postprocess_run, "score_reference_chain_trace", reference_score)
    monkeypatch.setattr(
        postprocess_run,
        "audit_trace",
        lambda *_args, **_kwargs: SimpleNamespace(
            failures=lambda **_failure_kwargs: []
        ),
    )

    asyncio.run(run_cases._run_cases(_case_args(benchmark, run_dir)))
    assert (run_dir / "run_results.jsonl").is_file()
    assert not (run_dir / "process_metrics.jsonl").exists()
    assert not (run_dir / "rollout_groups.jsonl").exists()

    summary = asyncio.run(
        postprocess_run._postprocess_run(_postprocess_args(run_dir))
    )

    assert summary["process_metric_rows"] == 1
    assert summary["trajectory_score_rows"] == 1
    assert (run_dir / "process_metrics.jsonl").is_file()
    assert (run_dir / "reference_chain_metrics.jsonl").is_file()
    assert (run_dir / "trajectory_scores.jsonl").is_file()
    assert (run_dir / "rollout_groups.jsonl").is_file()
    assert (run_dir / "post_rollout_rewards.jsonl").is_file()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["postprocess"]["status"] == "completed"
    assert manifest["postprocess"]["evaluation_gold"]["sha256"] == _sha256(
        benchmark.parent.parent / "evaluator_private" / "gold.jsonl"
    )
