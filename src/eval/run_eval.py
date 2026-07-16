from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.workflow import VerificationWorkflow, WorkflowConfig
from src.provider_profiles import PROFILE_IDS
from src.storage import default_eval_root
from src.eval.release_adapter import (
    image_only_case_from_runtime_row,
    load_runtime_release,
    resolve_runtime_image_path,
)
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.source_access import SourceAccessPolicy
from src.redaction import sanitize_for_persistence
from src.trajectory.exporter import export_policy_examples
from src.trajectory.perception_exporter import export_perception_example
from src.trajectory.reference_chain import score_reference_chain_trace
from src.trajectory.scoring import score_process_trace


RUN_SCHEMA_VERSION = "ifv-eval-run-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation for the image factual verifier."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to a v0.3 release runtime_input/cases.jsonl.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_IDS,
        default=None,
        help="Explicit teacher/student provider profile.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider for the agent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the agent.",
    )
    parser.add_argument(
        "--vlm-provider",
        default=None,
        help="Optional VLM provider override.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Optional VLM model override.",
    )
    parser.add_argument(
        "--llm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Optional wire API override. Gemini accepts interactions only.",
    )
    parser.add_argument(
        "--vlm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Optional VLM wire API override. Gemini production runs require interactions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for traces, predictions, and summary files.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent image verifications.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-image timeout budget in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "Run only this case_id. Repeat to select an explicit ordered canary "
            "set without exposing labels to the Agent."
        ),
    )
    parser.add_argument(
        "--source-access-policy",
        default=None,
        help=(
            "Evaluation-only JSON policy for blocking benchmark-origin sources. "
            "The policy is never added to model context or traces."
        ),
    )
    return parser.parse_args()


def _load_benchmark(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise TypeError("Benchmark must be a JSON list or JSONL file.")


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    configured = os.getenv("GIT_COMMIT", "").strip()
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(
        sanitize_for_persistence(payload),
        ensure_ascii=False,
        indent=2,
    )
    _write_text(path, serialized + "\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    lines = [
        json.dumps(
            sanitize_for_persistence(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    ]
    _write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def _run_result_record(
    sample: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    state = result.get("state") or {}
    judgment = result.get("judgment")
    verdict_basis = result.get("verdict_basis")
    if verdict_basis is None and isinstance(judgment, dict):
        verdict_basis = judgment.get("verdict_basis")
    return {
        "case_id": sample.get("case_id"),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "verdict_basis": verdict_basis,
        "termination": result.get("termination"),
        "time_taken": result.get("time_taken"),
        "total_tool_calls": result.get("total_tool_calls"),
        "llm_api_calls": result.get("llm_api_calls"),
        "token_usage": result.get("token_usage"),
        "error": result.get("error"),
        "stage_timings": state.get("stage_timings", {}),
        "trace_path": None,
    }


def _classification_prediction(
    sample: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Return a scorer row only for a completed factual judgment."""

    verdict = str(result.get("verdict") or "").strip()
    if (
        verdict not in {"real", "fake", "unverifiable"}
        or str(result.get("termination") or "") != "success"
        or bool(str(result.get("error") or "").strip())
    ):
        return None
    return {
        "case_id": sample.get("case_id"),
        "verdict": verdict,
    }


def _compute_summary(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(run_results)
    verdict_counter: Counter[str] = Counter()
    times: List[float] = []
    tool_calls: List[float] = []
    llm_calls: List[float] = []
    errors = 0

    for row in run_results:
        pred = str(row.get("verdict", ""))
        verdict_counter[pred] += 1
        if (
            pred == "error"
            or str(row.get("termination") or "") == "error"
            or bool(str(row.get("error") or "").strip())
        ):
            errors += 1
        if isinstance(row.get("time_taken"), (int, float)):
            times.append(float(row["time_taken"]))
        if isinstance(row.get("total_tool_calls"), (int, float)):
            tool_calls.append(float(row["total_tool_calls"]))
        if isinstance(row.get("llm_api_calls"), (int, float)):
            llm_calls.append(float(row["llm_api_calls"]))

    return {
        "num_samples": total,
        "num_errors": errors,
        "predicted_verdict_distribution": dict(verdict_counter),
        "avg_time_taken_sec": round(_safe_mean(times), 3),
        "avg_tool_calls": round(_safe_mean(tool_calls), 3),
        "avg_llm_api_calls": round(_safe_mean(llm_calls), 3),
    }


def _private_index(
    rows: List[Dict[str, Any]],
    *,
    key: str,
    name: str,
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rendered = str(row.get(key, "")).strip()
        if not rendered:
            raise ValueError(f"{name} row lacks non-empty {key}")
        if rendered in indexed:
            raise ValueError(f"duplicate {name} row for {rendered}")
        indexed[rendered] = row
    return indexed


def _select_samples(
    samples: List[Dict[str, Any]],
    *,
    requested_case_ids: List[str] | None,
    limit: int | None,
) -> List[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        case_id = str(sample.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("runtime input row lacks case_id")
        if case_id in indexed:
            raise ValueError(
                f"image-only release contains duplicate case_id {case_id}"
            )
        indexed[case_id] = sample
    if requested_case_ids:
        requested = [str(item).strip() for item in requested_case_ids]
        if any(not item for item in requested):
            raise ValueError("--case-id values must be non-empty")
        if len(requested) != len(set(requested)):
            raise ValueError("--case-id values must be unique")
        missing = [case_id for case_id in requested if case_id not in indexed]
        if missing:
            raise ValueError(
                "requested case_id values are absent from release: "
                + ", ".join(missing)
            )
        samples = [indexed[case_id] for case_id in requested]
    if limit is not None:
        samples = samples[:limit]
    return samples


def _file_descriptor(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


async def _run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    release = load_runtime_release(benchmark_path)
    samples = _load_benchmark(benchmark_path)
    samples = _select_samples(
        samples,
        requested_case_ids=getattr(args, "case_id", None),
        limit=args.limit,
    )
    samples = [
        resolve_runtime_image_path(sample, benchmark_path) for sample in samples
    ]
    runtime_cases = [
        image_only_case_from_runtime_row(sample) for sample in samples
    ]
    case_ids = [case.case_id for case in runtime_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("image-only release contains duplicate case_id values")
    for runtime_case in runtime_cases:
        verify_case_image(runtime_case, runtime_case.image_path)

    config = WorkflowConfig(
        profile_id=getattr(args, "profile", None),
        provider=getattr(args, "provider", None),
        model_name=getattr(args, "model", None),
        vlm_provider=getattr(args, "vlm_provider", None),
        vlm_model=getattr(args, "vlm_model", None),
        llm_wire_api=getattr(args, "llm_wire_api", None),
        vlm_wire_api=getattr(args, "vlm_wire_api", None),
        timeout=args.timeout,
        save_traces=True,
        decision_policy_version=release.decision_policy_version,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_eval_root()
        / (
            f"{timestamp}_{config.provider}_"
            f"{str(config.model_name).replace('/', '_')}"
        )
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory must be new or empty: {run_dir}"
        )
    verification_max_output_tokens = _positive_int(
        os.getenv("GEMINI_VERIFICATION_MAX_OUTPUT_TOKENS", "16384"),
        name="verification max output tokens",
    )
    verification_final_max_output_tokens = _positive_int(
        os.getenv("GEMINI_VERIFICATION_FINAL_MAX_OUTPUT_TOKENS", "32768"),
        name="verification final max output tokens",
    )
    stage_thinking_levels = {
        stage.lower(): os.getenv(
            f"GEMINI_{stage}_THINKING_LEVEL",
            os.getenv("GEMINI_AGENT_THINKING_LEVEL", "minimal"),
        ).strip().lower()
        for stage in ("VERIFICATION", "REFLECTION", "JUDGMENT")
    }
    verification_final_thinking_level = os.getenv(
        "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL",
        stage_thinking_levels["verification"],
    ).strip().lower()
    explicit_policy = (
        SourceAccessPolicy.load(args.source_access_policy)
        if args.source_access_policy
        else SourceAccessPolicy.load(release.artifacts.source_access_policy)
        if release.artifacts.source_access_policy is not None
        else None
    )
    config.source_access_policy = explicit_policy

    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_commit": _git_commit(),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "sample_count": len(samples),
            "limit": args.limit,
            "runtime_release": True,
            "release_id": release.release_id,
            "release_stage": release.release_stage,
            "release_manifest": _file_descriptor(release.manifest_path),
            "runtime_contract_version": release.runtime_contract_version,
            "input_mode": release.input_mode,
            "decision_policy_version": release.decision_policy_version,
            "classification_protocol": _file_descriptor(
                release.artifacts.classification_protocol
            ),
            "process_reference_protocol": _file_descriptor(
                release.artifacts.process_reference_protocol
            ),
            "evaluation_gold": None,
        },
        "agent": {
            "profile_id": config.profile_id,
            "provider": config.provider,
            "model": config.model_name,
            "vlm_provider": config.vlm_provider,
            "vlm_model": config.vlm_model,
            "llm_wire_api": config.llm_wire_api,
            "vlm_wire_api": config.vlm_wire_api,
            "timeout_seconds": args.timeout,
            "max_tool_actions": 24,
            "reflection_interval": 4,
            "max_reflections": 6,
            "verification_max_output_tokens": verification_max_output_tokens,
            "verification_final_max_output_tokens": (
                verification_final_max_output_tokens
            ),
            "verification_final_thinking_level": (
                verification_final_thinking_level
            ),
            "stage_thinking_levels": stage_thinking_levels,
            "concurrency": max(1, args.concurrency),
        },
        "source_access_policy": {
            "active": bool(explicit_policy and explicit_policy.active),
            "policy_id": explicit_policy.policy_id if explicit_policy else None,
            "cache_partition": explicit_policy.cache_partition if explicit_policy else None,
        },
        "artifacts": {
            "predictions": "predictions.jsonl",
            "run_results": "run_results.jsonl",
            "process_metrics": "process_metrics.jsonl",
            "reference_chain_metrics": "reference_chain_metrics.jsonl",
            "trajectory_scores": "trajectory_scores.jsonl",
            "policy_trajectories": "policy_trajectories.jsonl",
            "perception_trajectories": "perception_trajectories.jsonl",
            "summary": "summary.json",
            "traces": "traces/",
        },
    }
    _write_json(manifest_path, manifest)

    config.output_dir = str(trace_dir)
    try:
        workflow = VerificationWorkflow(config)
        image_paths = [str(sample["image_path"]) for sample in samples]
        image_ids = [case.case_id for case in runtime_cases]
        results = await workflow.run_batch(
            image_paths=image_paths,
            image_ids=image_ids,
            runtime_cases=runtime_cases,
            concurrency=max(1, args.concurrency),
        )

        evaluation_gold_index = _private_index(
            _load_benchmark(release.artifacts.evaluation_gold),
            key="case_id",
            name="evaluation-gold",
        )
        missing_gold = [
            case_id
            for case_id in case_ids
            if case_id not in evaluation_gold_index
        ]
        if missing_gold:
            raise RuntimeError(
                "evaluation-gold rows missing runtime case IDs: "
                + ", ".join(missing_gold[:10])
            )
        manifest["benchmark"]["evaluation_gold"] = _file_descriptor(
            release.artifacts.evaluation_gold
        )

        predictions: List[Dict[str, Any]] = []
        run_results: List[Dict[str, Any]] = []
        process_metrics: List[Dict[str, Any]] = []
        reference_chain_metrics: List[Dict[str, Any]] = []
        trajectory_scores: List[Dict[str, Any]] = []
        policy_trajectories: List[Dict[str, Any]] = []
        perception_trajectories: List[Dict[str, Any]] = []
        for sample, result in zip(samples, results):
            identity = str(sample.get("case_id") or "")
            record = _run_result_record(sample, result)
            trace_path = trace_dir / f"{identity}.json"
            if trace_path.exists():
                record["trace_path"] = trace_path.relative_to(run_dir).as_posix()
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                trace_state = trace.get("state")
                if isinstance(trace_state, dict) and isinstance(
                    trace_state.get("perception"), dict
                ):
                    perception_trajectories.append(
                        export_perception_example(
                            trace,
                            source_metadata={
                                "source_run_id": manifest["run_id"],
                                "runtime_commit": manifest["git_commit"],
                                "release_id": release.release_id,
                                "runtime_contract_version": (
                                    release.runtime_contract_version
                                ),
                            },
                        ).model_dump(mode="json")
                    )
                metrics, teacher_score = score_process_trace(
                    trace,
                    evaluation_gold_index[identity],
                    score_metadata={
                        "process_reference_protocol": _file_descriptor(
                            release.artifacts.process_reference_protocol
                        ),
                        "evaluation_gold": _file_descriptor(
                            release.artifacts.evaluation_gold
                        ),
                    },
                )
                process_metrics.append(metrics)
                reference_chain_metrics.append(
                    await score_reference_chain_trace(
                        trace,
                        evaluation_gold_index[identity],
                        score_metadata={
                            "process_reference_protocol": _file_descriptor(
                                release.artifacts.process_reference_protocol
                            ),
                            "evaluation_gold": _file_descriptor(
                                release.artifacts.evaluation_gold
                            ),
                        },
                    )
                )
                trajectory_scores.append(teacher_score)
                policy_trajectories.extend(
                    item.model_dump(mode="json")
                    for item in export_policy_examples(
                        trace,
                        source_metadata={
                            "source_run_id": manifest["run_id"],
                            "runtime_commit": manifest["git_commit"],
                            "release_id": release.release_id,
                            "runtime_contract_version": (
                                release.runtime_contract_version
                            ),
                            "process_reference_protocol_version": (
                                release.manifest.get(
                                    "process_reference_protocol_version",
                                    "",
                                )
                            ),
                        },
                    )
                )
            else:
                process_metrics.append(
                    {
                        "schema_version": "ifv-process-metrics-v2",
                        "case_id": identity,
                        "engineering_error": True,
                        "runtime_verdict": result.get("verdict"),
                        "result_correct": False,
                        "first_error": {
                            "error": "canonical trace is missing",
                        },
                    }
                )
                reference_chain_metrics.append(
                    {
                        "schema_version": "ifv-reference-chain-metrics-v2",
                        "case_id": identity,
                        "engineering_error": True,
                        "error": "canonical trace is missing",
                        "metrics": {
                            "fact_recovery_recall": 0.0,
                            "chain_recovery_recall": 0.0,
                            "evidence_recovery_recall": 0.0,
                            "basis_reference_precision": 0.0,
                        },
                    }
                )
                trajectory_scores.append(
                    {
                        "schema_version": "ifv-trajectory-score-v2",
                        "case_id": identity,
                        "components": {
                            "result_reward": 0.0,
                            "grounded_finding_reward": 0.0,
                            "gap_coverage_reward": 0.0,
                            "bridge_reward": 0.0,
                            "stop_calibration_reward": 0.0,
                            "duplicate_action_penalty": 0.0,
                            "invalid_task_penalty": 0.0,
                            "normalized_cost_penalty": 0.0,
                        },
                        "total": 0.0,
                    }
                )
            run_results.append(record)
            prediction = _classification_prediction(sample, result)
            if prediction is not None:
                predictions.append(prediction)

        summary = _compute_summary(run_results)
        summary["run_id"] = manifest["run_id"]
        _write_jsonl(run_dir / "predictions.jsonl", predictions)
        _write_jsonl(run_dir / "run_results.jsonl", run_results)
        _write_jsonl(run_dir / "process_metrics.jsonl", process_metrics)
        _write_jsonl(
            run_dir / "reference_chain_metrics.jsonl",
            reference_chain_metrics,
        )
        _write_jsonl(run_dir / "trajectory_scores.jsonl", trajectory_scores)
        _write_jsonl(run_dir / "policy_trajectories.jsonl", policy_trajectories)
        _write_jsonl(
            run_dir / "perception_trajectories.jsonl",
            perception_trajectories,
        )
        _write_json(run_dir / "summary.json", summary)

        manifest["status"] = (
            "completed_with_errors" if summary["num_errors"] else "completed"
        )
        manifest["completed_at"] = _now_iso()
        manifest["result"] = {
            "num_samples": summary["num_samples"],
            "num_errors": summary["num_errors"],
        }
        _write_json(manifest_path, manifest)
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = _now_iso()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_run_eval(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if int(summary.get("num_errors", 0) or 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
