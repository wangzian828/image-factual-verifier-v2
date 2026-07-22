from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import asdict
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.workflow import (
    AGENT_DECISION_POLICY_VERSION,
    VerificationWorkflow,
    WorkflowConfig,
)
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
from scripts.audit_real_trace import audit_trace


RUN_SCHEMA_VERSION = "ifv-eval-run-v2"
ROLLOUT_MEMBER_SCHEMA_VERSION = "ifv-rollout-group-member-v1"


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
        help="Concurrent complete rollout episodes.",
    )
    parser.add_argument(
        "--rollouts-per-case",
        type=int,
        default=1,
        help="Independent complete episodes per case (recommended for RL: 4).",
    )
    parser.add_argument(
        "--base-sampling-seed",
        type=int,
        default=1729,
        help="Base seed for reproducible per-episode Qwen sampling seeds.",
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
        verdict not in {"real", "fake"}
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


def _rollout_specs(
    samples: List[Dict[str, Any]],
    runtime_cases: List[Any],
    *,
    rollouts_per_case: int,
    base_sampling_seed: int,
    policy_revision: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Expand public cases into isolated, reproducible rollout episodes."""

    if rollouts_per_case < 1:
        raise ValueError("--rollouts-per-case must be at least 1")
    specs: List[Dict[str, Any]] = []
    for sample, runtime_case in zip(samples, runtime_cases):
        case_id = str(runtime_case.case_id)
        group_material = {
            "case_id": case_id,
            "policy_revision": policy_revision,
            "model": model,
            "base_sampling_seed": int(base_sampling_seed),
            "group_size": rollouts_per_case,
        }
        prompt_group_id = "pg-" + hashlib.sha256(
            json.dumps(
                group_material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        for rollout_index in range(rollouts_per_case):
            seed_material = (
                f"{base_sampling_seed}:{case_id}:{rollout_index}"
            ).encode("utf-8")
            sampling_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:4], "big"
            ) & 0x7FFFFFFF
            episode_id = case_id
            if rollouts_per_case > 1:
                episode_id = (
                    f"{case_id[:150]}--{prompt_group_id[3:11]}"
                    f"--r{rollout_index:03d}"
                )
            specs.append(
                {
                    "sample": sample,
                    "runtime_case": runtime_case,
                    "case_id": case_id,
                    "prompt_group_id": prompt_group_id,
                    "episode_id": episode_id,
                    "rollout_index": rollout_index,
                    "group_size": rollouts_per_case,
                    "sampling_seed": sampling_seed,
                }
            )
    return specs


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

    runtime_commit = _git_commit()
    rollouts_per_case = _positive_int(
        getattr(args, "rollouts_per_case", 1),
        name="rollouts per case",
    )
    base_sampling_seed = int(getattr(args, "base_sampling_seed", 1729))
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
        decision_policy_version=AGENT_DECISION_POLICY_VERSION,
    )
    rollout_specs = _rollout_specs(
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
            (
                "high"
                if stage == "PLANNING"
                else os.getenv("GEMINI_AGENT_THINKING_LEVEL", "minimal")
            ),
        ).strip().lower()
        for stage in ("PLANNING", "VERIFICATION", "REFLECTION", "JUDGMENT")
    }
    verification_final_thinking_level = os.getenv(
        "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL",
        stage_thinking_levels["verification"],
    ).strip().lower()
    qwen_stage_thinking = {
        stage.lower(): os.getenv(
            f"QWEN_{stage}_ENABLE_THINKING",
            (
                "true"
                if stage
                in {
                    "PLANNING",
                    "EVIDENCE_DECISION",
                    "REFLECTION",
                    "JUDGMENT",
                }
                else "false"
            ),
        ).strip().lower()
        for stage in (
            "PLANNING",
            "VERIFICATION",
            "EVIDENCE_DECISION",
            "REFLECTION",
            "QUERY_CONCEPT_EXTRACTION",
            "QUERY_REPLAN",
            "JUDGMENT",
        )
    }
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
        "git_commit": runtime_commit,
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "sample_count": len(samples),
            "episode_count": len(rollout_specs),
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
            "decision_policy_version": config.decision_policy_version,
            "provider": config.provider,
            "model": config.model_name,
            "vlm_provider": config.vlm_provider,
            "vlm_model": config.vlm_model,
            "base_url": config.llm_base_url,
            "vlm_base_url": config.vlm_base_url,
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
            "qwen_stage_enable_thinking": qwen_stage_thinking,
            "concurrency": max(1, args.concurrency),
            "rollouts_per_case": rollouts_per_case,
            "base_sampling_seed": base_sampling_seed,
            "sampling_seed_derivation": (
                "sha256(base_seed:case_id:rollout_index)-31bit-v1"
            ),
        },
        "source_access_policy": {
            "active": bool(explicit_policy and explicit_policy.active),
            "policy_id": explicit_policy.policy_id if explicit_policy else None,
            "cache_partition": explicit_policy.cache_partition if explicit_policy else None,
        },
        "artifacts": {
            "predictions": "predictions.jsonl",
            "episode_predictions": "episode_predictions.jsonl",
            "run_results": "run_results.jsonl",
            "process_metrics": "process_metrics.jsonl",
            "reference_chain_metrics": "reference_chain_metrics.jsonl",
            "trajectory_scores": "trajectory_scores.jsonl",
            "policy_trajectories": "policy_trajectories.jsonl",
            "perception_trajectories": "perception_trajectories.jsonl",
            "rollout_groups": "rollout_groups.jsonl",
            "post_rollout_rewards": "post_rollout_rewards.jsonl",
            "summary": "summary.json",
            "traces": "traces/",
        },
    }
    _write_json(manifest_path, manifest)

    config.output_dir = str(trace_dir)
    try:
        workflow = VerificationWorkflow(config)
        image_paths = [
            str(spec["sample"]["image_path"]) for spec in rollout_specs
        ]
        image_ids = [str(spec["episode_id"]) for spec in rollout_specs]
        results = await workflow.run_batch(
            image_paths=image_paths,
            image_ids=image_ids,
            runtime_cases=[spec["runtime_case"] for spec in rollout_specs],
            sampling_seeds=[spec["sampling_seed"] for spec in rollout_specs],
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
        episode_predictions: List[Dict[str, Any]] = []
        run_results: List[Dict[str, Any]] = []
        process_metrics: List[Dict[str, Any]] = []
        reference_chain_metrics: List[Dict[str, Any]] = []
        trajectory_scores: List[Dict[str, Any]] = []
        policy_trajectories: List[Dict[str, Any]] = []
        perception_trajectories: List[Dict[str, Any]] = []
        rollout_members: List[Dict[str, Any]] = []
        post_rollout_rewards: List[Dict[str, Any]] = []
        for spec, result in zip(rollout_specs, results):
            sample = spec["sample"]
            identity = str(spec["case_id"])
            episode_id = str(spec["episode_id"])
            record = _run_result_record(sample, result)
            record.update(
                {
                    "prompt_group_id": spec["prompt_group_id"],
                    "episode_id": episode_id,
                    "rollout_index": spec["rollout_index"],
                    "sampling_seed": spec["sampling_seed"],
                }
            )
            trace_path = trace_dir / f"{episode_id}.json"
            member = {
                "schema_version": ROLLOUT_MEMBER_SCHEMA_VERSION,
                "prompt_group_id": spec["prompt_group_id"],
                "case_id": identity,
                "episode_id": episode_id,
                "rollout_index": spec["rollout_index"],
                "group_size": spec["group_size"],
                "sampling_seed": spec["sampling_seed"],
                "policy_revision": manifest["git_commit"],
                "trace_path": None,
                "training_prohibited": (
                    release.release_stage == "development_subset"
                ),
                "training_eligible": False,
            }
            if trace_path.exists():
                record["trace_path"] = trace_path.relative_to(run_dir).as_posix()
                member["trace_path"] = record["trace_path"]
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
                metrics["episode_id"] = episode_id
                metrics["prompt_group_id"] = spec["prompt_group_id"]
                teacher_score["episode_id"] = episode_id
                teacher_score["prompt_group_id"] = spec["prompt_group_id"]
                strict_failures = audit_trace(trace_path).failures(
                    strict_scheduler=True
                )
                strict_trace_audit_pass = not strict_failures
                metrics["strict_trace_audit_pass"] = strict_trace_audit_pass
                metrics["strict_trace_audit_failures"] = [
                    asdict(item) for item in strict_failures
                ]
                process_metrics.append(metrics)
                reference_score = await score_reference_chain_trace(
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
                reference_score["episode_id"] = episode_id
                reference_score["prompt_group_id"] = spec["prompt_group_id"]
                reference_chain_metrics.append(reference_score)
                trajectory_scores.append(teacher_score)
                try:
                    exported_policy = export_policy_examples(
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
                except ValueError as exc:
                    metrics["training_eligible"] = False
                    reasons = list(
                        metrics.get("training_exclusion_reasons", []) or []
                    )
                    reasons.append(f"policy_export_rejected: {exc}")
                    metrics["training_exclusion_reasons"] = list(
                        dict.fromkeys(reasons)
                    )
                    teacher_score["training_eligible"] = False
                    teacher_score["training_exclusion_reasons"] = list(
                        metrics["training_exclusion_reasons"]
                    )
                    exported_policy = []
                policy_trajectories.extend(
                    item.model_dump(mode="json")
                    for item in exported_policy
                )
                member["training_eligible"] = bool(
                    strict_trace_audit_pass
                    and not member["training_prohibited"]
                )
                post_rollout_rewards.append(
                    {
                        "schema_version": "ifv-post-rollout-deterministic-v1",
                        "prompt_group_id": spec["prompt_group_id"],
                        "case_id": identity,
                        "episode_id": episode_id,
                        "classification_correct": bool(
                            metrics.get("result_correct", False)
                        ),
                        "fatal_engineering_error": bool(
                            metrics.get("engineering_error", False)
                        ),
                        "strict_trace_audit_pass": bool(
                            strict_trace_audit_pass
                        ),
                        "training_prohibited": member["training_prohibited"],
                    }
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
                post_rollout_rewards.append(
                    {
                        "schema_version": "ifv-post-rollout-deterministic-v1",
                        "prompt_group_id": spec["prompt_group_id"],
                        "case_id": identity,
                        "episode_id": episode_id,
                        "classification_correct": False,
                        "fatal_engineering_error": True,
                        "strict_trace_audit_pass": False,
                        "training_prohibited": member["training_prohibited"],
                    }
                )
            run_results.append(record)
            rollout_members.append(member)
            prediction = _classification_prediction(sample, result)
            if prediction is not None:
                episode_predictions.append(
                    {
                        **prediction,
                        "episode_id": episode_id,
                        "prompt_group_id": spec["prompt_group_id"],
                        "rollout_index": spec["rollout_index"],
                    }
                )
                # predictions.jsonl stays consumable by the data-owned
                # classification scorer: one row per public case. Multi-rollout
                # analysis belongs in episode_predictions.jsonl.
                if spec["rollout_index"] == 0:
                    predictions.append(prediction)

        summary = _compute_summary(run_results)
        summary["run_id"] = manifest["run_id"]
        _write_jsonl(run_dir / "predictions.jsonl", predictions)
        _write_jsonl(
            run_dir / "episode_predictions.jsonl",
            episode_predictions,
        )
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
        _write_jsonl(run_dir / "rollout_groups.jsonl", rollout_members)
        _write_jsonl(
            run_dir / "post_rollout_rewards.jsonl",
            post_rollout_rewards,
        )
        _write_json(run_dir / "summary.json", summary)

        manifest["status"] = (
            "completed_with_errors" if summary["num_errors"] else "completed"
        )
        manifest["completed_at"] = _now_iso()
        manifest["result"] = {
            "num_samples": len(samples),
            "num_episodes": summary["num_samples"],
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
