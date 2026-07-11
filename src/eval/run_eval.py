from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.workflow import VerificationWorkflow, WorkflowConfig


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
        default=900.0,
        help="Per-image timeout budget in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
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
        "plan": state.get("plan"),
        "verification": state.get("verification"),
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

    for row in predictions:
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("predicted_verdict", ""))
        bucket = str(row.get("bucket", "unknown"))
        bucket_totals[bucket] += 1
        verdict_counter[pred] += 1
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
        else Path("outputs") / "eval_runs" / f"{timestamp}_{args.provider}_{args.model.replace('/', '_')}"
    )
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    config = WorkflowConfig(
        provider=args.provider,
        model_name=args.model,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        llm_wire_api=args.llm_wire_api,
        vlm_wire_api=args.vlm_wire_api,
        output_dir=str(trace_dir),
        timeout=args.timeout,
        save_traces=True,
    )
    workflow = VerificationWorkflow(config)

    image_paths = [str(sample["image_path"]) for sample in samples]
    image_ids = [str(sample["sample_id"]) for sample in samples]
    results = await workflow.run_batch(
        image_paths=image_paths,
        image_ids=image_ids,
        concurrency=max(1, args.concurrency),
    )

    predictions = []
    failures = []
    for sample, result in zip(samples, results):
        record = _prediction_record(sample, result)
        trace_path = trace_dir / f"{sample['sample_id']}.json"
        if trace_path.exists():
            record["trace_path"] = str(trace_path)
        predictions.append(record)
        if record["ground_truth"] != record["predicted_verdict"]:
            failures.append(record)

    summary = _compute_summary(predictions)
    summary.update(
        {
            "benchmark_path": str(benchmark_path),
            "provider": args.provider,
            "model": args.model,
            "vlm_provider": args.vlm_provider or args.provider,
            "vlm_model": args.vlm_model or args.model,
            "llm_wire_api": args.llm_wire_api,
            "vlm_wire_api": args.vlm_wire_api,
            "output_dir": str(run_dir),
            "trace_dir": str(trace_dir),
            "num_failures": len(failures),
        }
    )

    (run_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_run_eval(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
