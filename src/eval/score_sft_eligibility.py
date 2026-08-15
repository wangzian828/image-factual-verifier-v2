"""Score completed teacher traces against private structured SFT targets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.audit_real_trace import audit_trace
from src.eval.score_semantic_reward import _resolve_image_path
from src.orchestrator.llm_backend import APIBackend
from src.trajectory.semantic_reward import sha256_file
from src.trajectory.sft_eligibility import (
    SFT_ELIGIBILITY_SCHEMA_VERSION,
    SFTEligibilityJudge,
    build_sft_eligibility_artifact,
    build_sft_eligibility_input,
    classify_sft_audit_failures,
    sft_eligibility_cache_key,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--provider", default="gemini", choices=["gemini"])
    parser.add_argument(
        "--model",
        default=os.getenv("IFV_SFT_ELIGIBILITY_MODEL", "gemini-3.7-flash"),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_index(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        aliases = [
            str(row.get(key, "")).strip()
            for key in ("case_id", "candidate_id", "assignment_id")
        ]
        aliases = [value for value in aliases if value]
        if not aliases:
            raise ValueError(
                f"{path}:{line_number} lacks case_id/candidate_id/assignment_id"
            )
        for alias in aliases:
            if alias in result and result[alias] is not row:
                raise ValueError(f"duplicate private target alias: {alias!r}")
            result[alias] = row
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    pending.replace(path)


def _load_cache(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _json_object(path)
    if value.get("schema_version") != SFT_ELIGIBILITY_SCHEMA_VERSION:
        raise ValueError(f"unsupported SFT eligibility cache entry: {path}")
    return value


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    manifest = _json_object(run_dir / "run_manifest.json")
    if manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise RuntimeError(
            "private gold may be loaded only after the rollout run reaches a "
            "terminal completed status"
        )
    source_policy = manifest.get("source_access_policy")
    source_policy_active = bool(
        isinstance(source_policy, Mapping) and source_policy.get("active") is True
    )
    trace_paths = sorted((run_dir / "traces").glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"no canonical traces under {run_dir / 'traces'}")

    # The completed-manifest check intentionally precedes this private read.
    gold_path = args.gold.expanduser().resolve()
    gold = _jsonl_index(gold_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or output_dir / "cache").expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None

    backend = APIBackend(
        provider=args.provider,
        model_name=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    judge = SFTEligibilityJudge(
        backend,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    rows: list[Dict[str, Any]] = []
    try:
        for trace_path in trace_paths:
            trace = _json_object(trace_path)
            state = trace.get("state") if isinstance(trace.get("state"), Mapping) else {}
            runtime_case = (
                state.get("runtime_case")
                if isinstance(state, Mapping)
                and isinstance(state.get("runtime_case"), Mapping)
                else {}
            )
            case_id = str(runtime_case.get("case_id", "")).strip()
            if case_id not in gold:
                raise ValueError(f"private gold lacks trace case_id {case_id!r}")
            image_path = _resolve_image_path(
                trace,
                explicit_image=None,
                image_root=image_root,
            )
            if image_path is None:
                raise FileNotFoundError(
                    f"cannot resolve original image for teacher trace: {trace_path}"
                )
            packet = build_sft_eligibility_input(
                trace,
                gold[case_id],
                image_path=image_path,
            )
            trace_sha = sha256_file(trace_path)
            report = audit_trace(
                trace_path,
                enforce_source_access_policy=source_policy_active,
            )
            failures = report.failures(strict_scheduler=True)
            fatal_audit_errors, _ = classify_sft_audit_failures(
                [asdict(item) for item in failures]
            )
            engineering_valid = bool(
                str(trace.get("termination", "")) == "success"
                and str(trace.get("verdict", "")) in {"real", "fake"}
            )
            cache_key = sft_eligibility_cache_key(
                trace_sha256=trace_sha,
                packet=packet,
                provider=args.provider,
                model=args.model,
                generation_version=judge.generation_identity,
            )
            cache_path = cache_dir / cache_key[:2] / f"{cache_key}.json"
            artifact = None if args.force else _load_cache(cache_path)
            from_cache = artifact is not None
            if artifact is None:
                judgment = None
                judge_audit = None
                if engineering_valid and not fatal_audit_errors:
                    judgment, judge_audit = await judge.judge(
                        packet,
                        image_path=image_path,
                    )
                artifact = build_sft_eligibility_artifact(
                    trace=trace,
                    trace_sha256=trace_sha,
                    packet=packet,
                    judgment=judgment,
                    judge_audit=judge_audit,
                    strict_trace_audit_pass=not failures,
                    strict_trace_audit_failures=[asdict(item) for item in failures],
                )
                _write_json(cache_path, artifact)
            episode_id = str(artifact.get("episode_id", ""))
            artifact_path = output_dir / f"{episode_id}.sft_eligibility.json"
            _write_json(artifact_path, artifact)
            rows.append(
                {
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "artifact": str(artifact_path),
                    "artifact_id": artifact.get("artifact_id"),
                    "sft_eligibility_pass": artifact.get("gates", {}).get(
                        "sft_eligibility_pass"
                    ),
                    "fact_alignment": artifact.get("metrics", {}).get(
                        "fact_alignment"
                    ),
                    "decision_support": artifact.get("metrics", {}).get(
                        "decision_support"
                    ),
                    "decisive_evidence_ids": artifact.get("metrics", {}).get(
                        "decisive_evidence_ids"
                    ),
                    "fatal_errors": artifact.get("metrics", {}).get(
                        "fatal_errors"
                    ),
                    "warnings": artifact.get("metrics", {}).get("warnings"),
                    "from_cache": from_cache,
                }
            )
    finally:
        await backend.aclose()

    accepted = [row for row in rows if row["sft_eligibility_pass"] is True]
    _write_jsonl(output_dir / "accepted_episodes.jsonl", accepted)
    summary = {
        "schema_version": "ifv-sft-eligibility-summary-v2",
        "run_dir": str(run_dir),
        "private_gold": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
        "provider": args.provider,
        "model": args.model,
        "episode_count": len(rows),
        "passed_count": len(accepted),
        "rows": rows,
    }
    _write_json(output_dir / "sft_eligibility_summary.json", summary)
    return summary


def main() -> None:
    summary = asyncio.run(_run(_parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
