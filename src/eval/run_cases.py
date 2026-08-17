from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.eval.case_selection import (
    case_list,
    metadata_index,
    select_samples,
)
from src.eval.public_release import load_public_release, resolve_image_path
from src.eval.release_adapter import image_only_case_from_runtime_row
from src.eval.result_records import (
    classification_prediction,
    compute_summary,
    run_result_record,
)
from src.eval.rollout import rollout_specs as build_rollout_specs
from src.eval.run_artifacts import (
    file_descriptor as _file_descriptor,
    load_benchmark,
    now_iso as _now_iso,
    sha256_file as _sha256,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.source_access import SourceAccessPolicy
from src.provider_profiles import PROFILE_IDS
from src.storage import default_eval_root
from src.workflow import (
    AGENT_DECISION_POLICY_VERSION,
    VerificationWorkflow,
    WorkflowConfig,
)


RUN_CASES_SCHEMA_VERSION = "ifv-case-run-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run image-only cases and write minimal reproducible artifacts. "
            "This runner does not load private gold, score traces, or export "
            "training trajectories."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to runtime_input/cases.jsonl.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Optional JSONL sidecar keyed by case_id. Metadata is joined into "
            "run_results.jsonl and never passed to the Agent."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_IDS,
        default=None,
        help="Explicit provider profile.",
    )
    parser.add_argument("--provider", default=None, help="LLM provider.")
    parser.add_argument("--model", default=None, help="LLM model name.")
    parser.add_argument("--vlm-provider", default=None, help="Optional VLM provider.")
    parser.add_argument("--vlm-model", default=None, help="Optional VLM model.")
    parser.add_argument(
        "--image-access-mode",
        choices=["direct_multimodal", "separate_vlm"],
        default="direct_multimodal",
    )
    parser.add_argument(
        "--llm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
    )
    parser.add_argument(
        "--vlm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New or empty directory for manifest, run results, predictions, and traces.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent complete rollout episodes.",
    )
    parser.add_argument(
        "--rollouts-per-case",
        type=int,
        default=1,
        help="Independent complete episodes per case.",
    )
    parser.add_argument(
        "--base-sampling-seed",
        type=int,
        default=1729,
        help="Base seed for reproducible per-episode sampling seeds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-image timeout budget in seconds.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only this case_id. Repeat for an ordered canary set.",
    )
    parser.add_argument(
        "--case-list",
        default=None,
        help="Optional text file with one case_id per non-empty line.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--source-access-policy",
        default=None,
        help="Optional evaluator-side source-access policy override.",
    )
    return parser.parse_args()


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _git_commit() -> str:
    configured = os.getenv("GIT_COMMIT", "").strip()
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
        return configured
    actual = completed.stdout.strip()
    if configured and actual and configured != actual:
        raise RuntimeError(
            "GIT_COMMIT does not match the runtime checkout: "
            f"configured={configured}, actual={actual}"
        )
    return actual or configured


def _selected_cases(args: argparse.Namespace, benchmark_path: Path) -> list[Dict[str, Any]]:
    case_list_path = (
        Path(args.case_list).expanduser().resolve()
        if getattr(args, "case_list", None)
        else None
    )
    if getattr(args, "case_id", None) and case_list_path is not None:
        raise ValueError("--case-id and --case-list are mutually exclusive")
    return select_samples(
        load_benchmark(benchmark_path),
        requested_case_ids=getattr(args, "case_id", None) or case_list(case_list_path),
        limit=getattr(args, "limit", None),
        shard_count=int(getattr(args, "shard_count", 1)),
        shard_index=int(getattr(args, "shard_index", 0)),
    )


def _workflow_config(args: argparse.Namespace) -> WorkflowConfig:
    return WorkflowConfig(
        profile_id=getattr(args, "profile", None),
        provider=getattr(args, "provider", None),
        model_name=getattr(args, "model", None),
        vlm_provider=getattr(args, "vlm_provider", None),
        vlm_model=getattr(args, "vlm_model", None),
        image_access_mode=getattr(
            args,
            "image_access_mode",
            "direct_multimodal",
        ),
        llm_wire_api=getattr(args, "llm_wire_api", None),
        vlm_wire_api=getattr(args, "vlm_wire_api", None),
        timeout=args.timeout,
        save_traces=True,
        decision_policy_version=AGENT_DECISION_POLICY_VERSION,
    )


async def _run_cases(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    metadata_path = (
        Path(args.metadata).expanduser().resolve()
        if getattr(args, "metadata", None)
        else None
    )
    release = load_public_release(benchmark_path)
    samples = [
        resolve_image_path(sample, release=release, benchmark_path=benchmark_path)
        for sample in _selected_cases(args, benchmark_path)
    ]
    metadata_by_case = metadata_index(metadata_path)
    runtime_cases = [image_only_case_from_runtime_row(sample) for sample in samples]
    for runtime_case in runtime_cases:
        verify_case_image(runtime_case, runtime_case.image_path)

    runtime_commit = _git_commit()
    rollouts_per_case = _positive_int(
        getattr(args, "rollouts_per_case", 1),
        name="rollouts per case",
    )
    base_sampling_seed = int(getattr(args, "base_sampling_seed", 1729))
    config = _workflow_config(args)
    explicit_policy = (
        SourceAccessPolicy.load(args.source_access_policy)
        if getattr(args, "source_access_policy", None)
        else SourceAccessPolicy.load(release.source_access_policy_path)
        if release.source_access_policy_path is not None
        else None
    )
    config.source_access_policy = explicit_policy
    rollout_specs = build_rollout_specs(
        samples,
        runtime_cases,
        rollouts_per_case=rollouts_per_case,
        base_sampling_seed=base_sampling_seed,
        policy_revision=runtime_commit,
        model=str(config.model_name),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_eval_root()
        / f"{timestamp}_{config.provider}_{str(config.model_name).replace('/', '_')}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run output directory must be new or empty: {run_dir}")
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(trace_dir)

    manifest_path = run_dir / "run_manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": RUN_CASES_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_commit": runtime_commit,
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "sample_count": len(samples),
            "episode_count": len(rollout_specs),
            "limit": getattr(args, "limit", None),
            "shard_count": int(getattr(args, "shard_count", 1)),
            "shard_index": int(getattr(args, "shard_index", 0)),
            "release_id": release.release_id,
            "release_stage": release.release_stage,
            "schema_version": release.schema_version,
            "runtime_contract_version": release.runtime_contract_version,
            "input_mode": release.input_mode,
            "decision_policy_version": release.decision_policy_version,
            "release_manifest": _file_descriptor(release.manifest_path),
        },
        "metadata": _file_descriptor(metadata_path),
        "agent": {
            "profile_id": config.profile_id,
            "decision_policy_version": config.decision_policy_version,
            "provider": config.provider,
            "model": config.model_name,
            "vlm_provider": config.vlm_provider,
            "vlm_model": config.vlm_model,
            "image_access_mode": config.image_access_mode,
            "base_url": config.llm_base_url,
            "vlm_base_url": config.vlm_base_url,
            "llm_wire_api": config.llm_wire_api,
            "vlm_wire_api": config.vlm_wire_api,
            "timeout_seconds": args.timeout,
            "concurrency": max(1, args.concurrency),
            "rollouts_per_case": rollouts_per_case,
            "base_sampling_seed": base_sampling_seed,
        },
        "execution": {
            "mode": "case_run",
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "source_access_policy": {
            "active": bool(explicit_policy and explicit_policy.active),
            "policy_id": explicit_policy.policy_id if explicit_policy else None,
            "cache_partition": explicit_policy.cache_partition if explicit_policy else None,
        },
        "artifacts": {
            "run_results": "run_results.jsonl",
            "predictions": "predictions.jsonl",
            "summary": "summary.json",
            "traces": "traces/",
        },
    }
    _write_json(manifest_path, manifest)

    try:
        workflow = VerificationWorkflow(config)
        results = await workflow.run_batch(
            image_paths=[str(spec["sample"]["image_path"]) for spec in rollout_specs],
            image_ids=[str(spec["episode_id"]) for spec in rollout_specs],
            runtime_cases=[spec["runtime_case"] for spec in rollout_specs],
            sampling_seeds=[spec["sampling_seed"] for spec in rollout_specs],
            concurrency=max(1, args.concurrency),
        )

        run_results: list[Dict[str, Any]] = []
        predictions: list[Dict[str, Any]] = []
        for spec, result in zip(rollout_specs, results):
            case_id = str(spec["case_id"])
            episode_id = str(spec["episode_id"])
            trace_path = trace_dir / f"{episode_id}.json"
            relative_trace = (
                trace_path.relative_to(run_dir).as_posix()
                if trace_path.exists()
                else None
            )
            run_results.append(
                run_result_record(
                    spec["sample"],
                    result,
                    trace_path=relative_trace,
                    metadata=metadata_by_case.get(case_id),
                    episode_id=episode_id,
                    prompt_group_id=str(spec["prompt_group_id"]),
                    rollout_index=int(spec["rollout_index"]),
                    sampling_seed=int(spec["sampling_seed"]),
                )
            )
            prediction = classification_prediction(spec["sample"], result)
            if prediction is not None and int(spec["rollout_index"]) == 0:
                predictions.append(prediction)

        summary = compute_summary(run_results)
        summary["run_id"] = manifest["run_id"]
        summary["num_cases"] = len(samples)
        _write_jsonl(run_dir / "run_results.jsonl", run_results)
        _write_jsonl(run_dir / "predictions.jsonl", predictions)
        _write_json(run_dir / "summary.json", summary)

        manifest["status"] = (
            "completed_with_errors" if summary["num_errors"] else "completed"
        )
        manifest["completed_at"] = _now_iso()
        manifest["result"] = {
            "num_cases": len(samples),
            "num_episodes": summary["num_episodes"],
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
    summary = asyncio.run(_run_cases(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if int(summary.get("num_errors", 0) or 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
