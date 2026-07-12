from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.workflow import VerificationWorkflow, WorkflowConfig
from src.storage import default_eval_root
from src.orchestrator.source_access import SourceAccessPolicy, benchmark_policy_from_rows
from src.redaction import sanitize_for_persistence


RUN_SCHEMA_VERSION = "ifv-eval-run-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation for the image factual verifier."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON or JSONL file.",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        help="LLM provider for the agent.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash",
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
        "--max-verification-iterations",
        type=int,
        default=None,
        help="Optional hard cap for verification/replanning iterations (default: 4).",
    )
    parser.add_argument(
        "--max-rounds-verification",
        type=int,
        default=12,
        help="Gemini Interactions turns available inside each verification iteration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
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


def _prediction_record(sample: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    state = result.get("state") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "bucket": sample.get("bucket"),
        "ground_truth": sample.get("ground_truth"),
        "image_path": sample.get("image_path"),
        "predicted_verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "overall_assessment": result.get("overall_assessment"),
        "termination": result.get("termination"),
        "time_taken": result.get("time_taken"),
        "total_tool_calls": result.get("total_tool_calls"),
        "llm_api_calls": result.get("llm_api_calls"),
        "token_usage": result.get("token_usage"),
        "error": result.get("error"),
        "judgment": result.get("judgment"),
        "stage_timings": state.get("stage_timings", {}),
        "trace_path": None,
    }


def _compute_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    correct = 0
    bucket_totals: Counter[str] = Counter()
    bucket_correct: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    verdict_counter: Counter[str] = Counter()
    times: List[float] = []
    tool_calls: List[float] = []
    llm_calls: List[float] = []
    errors = 0

    for row in predictions:
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("predicted_verdict", ""))
        bucket = str(row.get("bucket", "unknown"))
        bucket_totals[bucket] += 1
        verdict_counter[pred] += 1
        if pred == "error":
            errors += 1
        confusion[gt][pred] += 1
        if gt == pred:
            correct += 1
            bucket_correct[bucket] += 1
        if isinstance(row.get("time_taken"), (int, float)):
            times.append(float(row["time_taken"]))
        if isinstance(row.get("total_tool_calls"), (int, float)):
            tool_calls.append(float(row["total_tool_calls"]))
        if isinstance(row.get("llm_api_calls"), (int, float)):
            llm_calls.append(float(row["llm_api_calls"]))

    bucket_metrics = {}
    for bucket, count in bucket_totals.items():
        bucket_metrics[bucket] = {
            "count": count,
            "correct": bucket_correct[bucket],
            "accuracy": round(bucket_correct[bucket] / count, 4) if count else 0.0,
        }

    return {
        "num_samples": total,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "num_errors": errors,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "bucket_metrics": bucket_metrics,
        "confusion": {gt: dict(counter) for gt, counter in confusion.items()},
        "predicted_verdict_distribution": dict(verdict_counter),
        "avg_time_taken_sec": round(_safe_mean(times), 3),
        "avg_tool_calls": round(_safe_mean(tool_calls), 3),
        "avg_llm_api_calls": round(_safe_mean(llm_calls), 3),
    }


async def _run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = Path(args.benchmark)
    samples = _load_benchmark(benchmark_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_eval_root()
        / f"{timestamp}_{args.provider}_{args.model.replace('/', '_')}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory must be new or empty: {run_dir}"
        )
    max_verification_iterations = _positive_int(
        args.max_verification_iterations
        if args.max_verification_iterations is not None
        else os.getenv("MAX_VERIFICATION_ITERATIONS", "4"),
        name="max verification iterations",
    )
    min_verification_iterations = min(
        max_verification_iterations,
        _positive_int(
            os.getenv("MIN_VERIFICATION_ITERATIONS", "2"),
            name="minimum verification iterations",
        ),
    )
    low_information_gain_patience = _positive_int(
        os.getenv("LOW_INFORMATION_GAIN_PATIENCE", "2"),
        name="low information-gain patience",
    )
    verification_max_output_tokens = _positive_int(
        os.getenv("GEMINI_VERIFICATION_MAX_OUTPUT_TOKENS", "16384"),
        name="verification max output tokens",
    )
    verification_final_max_output_tokens = _positive_int(
        os.getenv("GEMINI_VERIFICATION_FINAL_MAX_OUTPUT_TOKENS", "32768"),
        name="verification final max output tokens",
    )
    verification_final_thinking_level = os.getenv(
        "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL", "minimal"
    ).strip().lower()
    explicit_policy = (
        SourceAccessPolicy.load(args.source_access_policy)
        if args.source_access_policy
        else None
    )
    looks_like_averimatec = any(
        str(row.get("sample_id", "")).startswith("averimatec-") for row in samples
    )
    sibling_policy = benchmark_path.parent / "source_access_policy.json"
    if explicit_policy is None and looks_like_averimatec and sibling_policy.exists():
        explicit_policy = SourceAccessPolicy.load(sibling_policy)
    provenance_rows = [
        row for row in samples if str(row.get("source_article_url", "")).strip()
    ]
    if explicit_policy is None and provenance_rows and not looks_like_averimatec:
        explicit_policy = benchmark_policy_from_rows(
            provenance_rows,
            policy_id=f"{benchmark_path.stem}-evaluation",
        )
    if looks_like_averimatec and explicit_policy is None:
        raise RuntimeError(
            "AVerImaTeC evaluation requires the benchmark-wide sibling "
            "source_access_policy.json or --source-access-policy. A per-subset policy "
            "does not prevent cross-case fact-check leakage."
        )

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
            "path": str(benchmark_path.resolve()),
            "sha256": _sha256(benchmark_path),
            "sample_count": len(samples),
            "limit": args.limit,
        },
        "agent": {
            "provider": args.provider,
            "model": args.model,
            "vlm_provider": args.vlm_provider or args.provider,
            "vlm_model": args.vlm_model or args.model,
            "llm_wire_api": args.llm_wire_api,
            "vlm_wire_api": args.vlm_wire_api,
            "timeout_seconds": args.timeout,
            "max_verification_iterations": max_verification_iterations,
            "min_verification_iterations": min_verification_iterations,
            "low_information_gain_patience": low_information_gain_patience,
            "max_rounds_verification": args.max_rounds_verification,
            "verification_max_output_tokens": verification_max_output_tokens,
            "verification_final_max_output_tokens": (
                verification_final_max_output_tokens
            ),
            "verification_final_thinking_level": (
                verification_final_thinking_level
            ),
            "concurrency": max(1, args.concurrency),
        },
        "source_access_policy": {
            "active": bool(explicit_policy and explicit_policy.active),
            "policy_id": explicit_policy.policy_id if explicit_policy else None,
            "cache_partition": explicit_policy.cache_partition if explicit_policy else None,
        },
        "artifacts": {
            "predictions": "predictions.jsonl",
            "summary": "summary.json",
            "traces": "traces/",
        },
    }
    _write_json(manifest_path, manifest)

    config = WorkflowConfig(
        provider=args.provider,
        model_name=args.model,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        llm_wire_api=args.llm_wire_api,
        vlm_wire_api=args.vlm_wire_api,
        output_dir=str(trace_dir),
        max_rounds_verification=max(1, args.max_rounds_verification),
        max_verification_iterations=max_verification_iterations,
        min_verification_iterations=min_verification_iterations,
        low_information_gain_patience=low_information_gain_patience,
        timeout=args.timeout,
        save_traces=True,
        source_access_policy=explicit_policy,
    )
    try:
        workflow = VerificationWorkflow(config)
        image_paths = [str(sample["image_path"]) for sample in samples]
        image_ids = [str(sample["sample_id"]) for sample in samples]
        user_claims = [
            (
                str(sample.get("user_claim") or sample.get("runtime_claim") or "").strip()
                or None
            )
            for sample in samples
        ]
        results = await workflow.run_batch(
            image_paths=image_paths,
            image_ids=image_ids,
            user_claims=user_claims,
            concurrency=max(1, args.concurrency),
        )

        predictions = []
        for sample, result in zip(samples, results):
            record = _prediction_record(sample, result)
            trace_path = trace_dir / f"{sample['sample_id']}.json"
            if trace_path.exists():
                record["trace_path"] = trace_path.relative_to(run_dir).as_posix()
            predictions.append(record)

        summary = _compute_summary(predictions)
        summary["run_id"] = manifest["run_id"]
        _write_jsonl(run_dir / "predictions.jsonl", predictions)
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
