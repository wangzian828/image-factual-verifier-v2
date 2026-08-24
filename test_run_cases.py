from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval import case_selection, run_cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_public_release(tmp_path: Path, *, case_count: int = 1) -> Path:
    root = tmp_path / "release"
    runtime_dir = root / "runtime_input"
    asset_dir = runtime_dir / "assets" / "sha256"
    asset_dir.mkdir(parents=True)
    rows = []
    for index in range(case_count):
        case_id = f"case_{index:02d}"
        image = asset_dir / f"{case_id}.jpg"
        image.write_bytes(f"runtime-image-fixture-{index}".encode("utf-8"))
        rows.append(
            {
                "case_id": case_id,
                "image_path": f"assets/sha256/{case_id}.jpg",
                "image_sha256": _sha256(image),
            }
        )
    benchmark = runtime_dir / "cases.jsonl"
    benchmark.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ifv-image-only-benchmark-release-v0.3",
            "release_id": "public-case-run-fixture",
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
            },
            "source_access_policy": {"active": False},
        },
    )
    return benchmark


def _args(
    benchmark: Path | None,
    run_dir: Path,
    *,
    archive_root: Path | None = None,
    metadata: Path | None = None,
    case_list: Path | None = None,
    case_id: list[str] | None = None,
    skip_preflight_image_hash_verification: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=str(benchmark) if benchmark else None,
        archive_root=str(archive_root) if archive_root else None,
        metadata=str(metadata) if metadata else None,
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
        case_id=case_id,
        case_list=str(case_list) if case_list else None,
        shard_count=1,
        shard_index=0,
        source_access_policy=None,
        skip_preflight_image_hash_verification=skip_preflight_image_hash_verification,
    )


def test_run_cases_writes_minimal_artifacts_without_private_gold(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark = _build_public_release(tmp_path)
    run_dir = tmp_path / "run"
    metadata = tmp_path / "case_metadata.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "case_id": "case_00",
                "image_source_type": "ai_generated",
                "temporary_gold": "fake",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
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
            episode_id = kwargs["image_ids"][0]
            (trace_dir / f"{episode_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": episode_id,
                        "input_mode": "image_only",
                        "decision_policy_version": "discrepancy-first-v4",
                        "verdict": "real",
                        "termination": "success",
                        "state": {"stage_timings": {"total": 1.25}},
                    }
                ),
                encoding="utf-8",
            )
            return [
                {
                    "image_id": case.case_id,
                    "image_path": case.image_path,
                    "verdict": "real",
                    "confidence": 0.8,
                    "verdict_basis": {"claim_ids": ["claim-1"]},
                    "termination": "success",
                    "time_taken": 1.25,
                    "total_tool_calls": 2,
                    "llm_api_calls": 3,
                    "token_usage": {"total": 100},
                    "state": {"stage_timings": {"total": 1.25}},
                }
            ]

    monkeypatch.setattr(run_cases, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_cases, "_git_commit", lambda: "a" * 40)

    summary = asyncio.run(
        run_cases._run_cases(_args(benchmark, run_dir, metadata=metadata))
    )

    assert summary["num_cases"] == 1
    assert summary["num_episodes"] == 1
    assert summary["num_errors"] == 0
    expected_files = {
        "run_manifest.json",
        "run_results.jsonl",
        "predictions.jsonl",
        "summary.json",
        "traces",
    }
    assert expected_files.issubset({path.name for path in run_dir.iterdir()})
    for heavy_name in {
        "process_metrics.jsonl",
        "reference_chain_metrics.jsonl",
        "trajectory_scores.jsonl",
        "trajectory_sft.jsonl",
        "perception_trajectories.jsonl",
        "rollout_groups.jsonl",
        "post_rollout_rewards.jsonl",
        "episode_predictions.jsonl",
    }:
        assert not (run_dir / heavy_name).exists()

    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction == {"case_id": "case_00", "verdict": "real"}
    run_result = json.loads(
        (run_dir / "run_results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert run_result["status"] == "success"
    assert run_result["trace_path"] == "traces/case_00.json"
    assert run_result["metadata"] == {
        "case_id": "case_00",
        "image_source_type": "ai_generated",
        "temporary_gold": "fake",
    }
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "ifv-case-run-v1"
    assert manifest["status"] == "completed"
    assert manifest["metadata"]["sha256"] == _sha256(metadata)
    assert manifest["artifacts"] == {
        "run_results": "run_results.jsonl",
        "predictions": "predictions.jsonl",
        "summary": "summary.json",
        "traces": "traces/",
    }
    assert manifest["execution"]["preflight_image_hash_verification"] == "verified"


def test_run_cases_explicitly_skips_redundant_preflight_rehash(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark = _build_public_release(tmp_path)
    run_dir = tmp_path / "run"

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            episode_id = kwargs["image_ids"][0]
            (trace_dir / f"{episode_id}.json").write_text(
                json.dumps(
                    {
                        "image_id": episode_id,
                        "verdict": "fake",
                        "termination": "success",
                        "state": {},
                    }
                ),
                encoding="utf-8",
            )
            return [
                {
                    "image_id": kwargs["runtime_cases"][0].case_id,
                    "image_path": kwargs["runtime_cases"][0].image_path,
                    "verdict": "fake",
                    "confidence": 0.7,
                    "termination": "success",
                    "time_taken": 0.1,
                    "state": {},
                }
            ]

    monkeypatch.setattr(run_cases, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_cases, "_git_commit", lambda: "d" * 40)
    monkeypatch.setattr(
        run_cases,
        "verify_case_image",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("preflight hash verification must be skipped")
        ),
    )

    asyncio.run(
        run_cases._run_cases(
            _args(
                benchmark,
                run_dir,
                skip_preflight_image_hash_verification=True,
            )
        )
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["preflight_image_hash_verification"] == (
        "skipped_explicitly"
    )


def test_run_cases_supports_case_list_selection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    benchmark = _build_public_release(tmp_path, case_count=3)
    run_dir = tmp_path / "run"
    case_list = tmp_path / "cases.txt"
    case_list.write_text("case_02\ncase_00\n", encoding="utf-8")
    seen: list[str] = []

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for case, episode_id in zip(
                kwargs["runtime_cases"],
                kwargs["image_ids"],
            ):
                seen.append(case.case_id)
                (trace_dir / f"{episode_id}.json").write_text(
                    json.dumps(
                        {
                            "image_id": episode_id,
                            "verdict": "fake",
                            "termination": "success",
                            "state": {},
                        }
                    ),
                    encoding="utf-8",
                )
                results.append(
                    {
                        "image_id": case.case_id,
                        "image_path": case.image_path,
                        "verdict": "fake",
                        "confidence": 0.7,
                        "termination": "success",
                        "time_taken": 0.5,
                        "state": {},
                    }
                )
            return results

    monkeypatch.setattr(run_cases, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_cases, "_git_commit", lambda: "b" * 40)

    asyncio.run(
        run_cases._run_cases(_args(benchmark, run_dir, case_list=case_list))
    )

    assert seen == ["case_02", "case_00"]
    rows = [
        json.loads(line)
        for line in (run_dir / "run_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["case_id"] for row in rows] == ["case_02", "case_00"]


def test_run_cases_reads_archive_without_exposing_candidate_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    archive_root = tmp_path / "archive"
    image = archive_root / "artifacts" / "images" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"archive-image")
    (archive_root / "archive-summary.json").write_text(
        json.dumps({"archive_id": "archive-fixture"}),
        encoding="utf-8",
    )
    (archive_root / "human-review-candidates.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "candidate:0001",
                "archive_image_path": "artifacts/images/0001.jpg",
                "factual_status": "refuted",
                "claim_atom": {"private": True},
                "evidence": {"private": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    seen: list[dict[str, Any]] = []

    class ImageOnlyWorkflow:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run_batch(self, **kwargs: Any) -> list[dict[str, Any]]:
            case = kwargs["runtime_cases"][0]
            seen.append(case.model_dump())
            trace_dir = Path(self.config.output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "candidate_0001.json").write_text(
                json.dumps(
                    {
                        "image_id": "candidate:0001",
                        "input_mode": "image_only",
                        "verdict": "real",
                        "termination": "success",
                        "state": {},
                    }
                ),
                encoding="utf-8",
            )
            return [
                {
                    "image_id": "candidate:0001",
                    "image_path": str(image.resolve()),
                    "verdict": "real",
                    "confidence": 0.5,
                    "termination": "success",
                    "time_taken": 0.1,
                    "state": {},
                }
            ]

    monkeypatch.setattr(run_cases, "VerificationWorkflow", ImageOnlyWorkflow)
    monkeypatch.setattr(run_cases, "_git_commit", lambda: "c" * 40)

    summary = asyncio.run(
        run_cases._run_cases(
            _args(
                None,
                run_dir,
                archive_root=archive_root,
            )
        )
    )

    assert summary["num_cases"] == 1
    assert seen == [
        {
            "case_id": "candidate:0001",
            "image_path": str(image.resolve()),
            "image_sha256": _sha256(image),
        }
    ]
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["benchmark"]["archive_input"] is True
    assert manifest["benchmark"]["archive_id"] == "archive-fixture"
    assert manifest["execution"]["mode"] == "archive_case_run"


def test_run_cases_rejects_duplicate_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"case_id": "case_a"}) + "\n"
        + json.dumps({"case_id": "case_a"}) + "\n",
        encoding="utf-8",
    )

    try:
        case_selection.metadata_index(metadata)
    except ValueError as exc:
        assert "duplicate metadata" in str(exc)
    else:
        raise AssertionError("duplicate metadata rows must fail")
