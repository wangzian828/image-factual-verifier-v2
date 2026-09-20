"""Review a completed PSD rollout bank once, stopping after the first pass per case.

This owner is intended for large rollout archives.  It never hashes an archive
file or task image as a separate I/O pass: the source reviewer already binds
the canonical trace content and image bytes it consumes.  Resume checks use
immutable collection metadata plus file stat identities and do not reopen
completed traces.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from ifv_training.io import load_json, load_jsonl
from ifv_training.psd_candidates import _load_train_case_allowlist, _safe_trace_path, _trace_case_id
from ifv_training.psd_case_pool import completed_cases
from ifv_training.psd_collection import require_completed_collection
from ifv_training.psd_gemini_judge import _atomic_json
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_source_review import (
    PROMPT,
    SCHEMA,
    TRACE_PROJECTION,
    VERSION,
    judge_source,
    validate_source_review,
)


OWNER_VERSION = "ifv-psd-lightweight-source-review-v1"


def _stat_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _review_path(root: Path, episode_id: str) -> Path:
    import hashlib
    return root / "reviews" / (hashlib.sha256(episode_id.encode()).hexdigest() + ".json")


def _saved_identity(*, run_identity: str, case_id: str, episode_id: str,
                    trace_path: Path, image_sha256: str, gold: dict) -> dict:
    return {
        "owner_version": OWNER_VERSION,
        "run": run_identity,
        "case_id": case_id,
        "episode_id": episode_id,
        "trace_stat": _stat_identity(trace_path),
        "image_sha256": image_sha256,
        "gold_sha256": _sha(gold),
    }


async def run(*, run_dir: Path, benchmark: Path, train_cases: Path,
              private_gold: Path, output: Path, model: str,
              concurrency: int = 8, client=None,
              retry_error_types: list[str] | None = None) -> dict:
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("invalid lightweight source-review concurrency")
    run_dir, output = run_dir.resolve(), output.resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    require_completed_collection(run_dir, manifest)
    if manifest.get("benchmark", {}).get("training_prohibited"):
        raise ValueError("source review requires training-only rollouts")

    public_rows, gold_rows = load_jsonl(benchmark), load_jsonl(private_gold)
    public = {row["case_id"]: row for row in public_rows}
    gold = {row["case_id"]: row for row in gold_rows}
    allowed = _load_train_case_allowlist(train_cases)
    rows = load_jsonl(run_dir / "run_results.jsonl")
    if not rows or len(public) != len(public_rows) or len(gold) != len(gold_rows):
        raise ValueError("duplicate or empty source-review inputs")

    by_case: dict[str, list[dict]] = defaultdict(list)
    seen_episodes: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        episode_id = str(row.get("episode_id") or case_id)
        if episode_id in seen_episodes:
            raise ValueError("duplicate episode ID")
        seen_episodes.add(episode_id)
        if case_id not in public or case_id not in gold or allowed.get(case_id) != "train":
            raise ValueError("source-review membership is invalid")
        by_case[case_id].append(row)
    if set(by_case) != set(public):
        raise ValueError("source-review coverage is invalid")
    for case_rows in by_case.values():
        case_rows.sort(key=lambda row: (int(row.get("rollout_index", 0)), str(row.get("episode_id", ""))))

    identity = {
        "owner_version": OWNER_VERSION,
        "review_version": VERSION,
        "trace_projection": TRACE_PROJECTION,
        "model": model,
        "prompt_sha256": _sha(PROMPT),
        "schema_sha256": _sha(SCHEMA),
        "inputs": [_stat_identity(path) for path in (
            run_dir / "run_manifest.json", run_dir / "run_results.jsonl",
            benchmark, train_cases, private_gold,
        )],
        "policy": "source_order_first_verified_pass_else_all_rollouts",
    }
    marker = output / "inputs.json"
    if marker.exists():
        load_bound(marker, identity=identity)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("source-review output must be new or already bound")
        save_bound(marker, identity=identity, payload={"training_only": True})
    run_identity = _sha(identity)
    cache_dir = output / "judge-cache"

    async def one_case(item):
        case_id, case_rows = item
        results = []
        for row in case_rows:
            episode_id = str(row.get("episode_id") or case_id)
            trace_path = _safe_trace_path(run_dir, str(row.get("trace_path") or ""), episode_id)
            image = (benchmark.parent / public[case_id]["image_path"]).resolve()
            saved_path = _review_path(output, episode_id)
            binding = _saved_identity(
                run_identity=run_identity,
                case_id=case_id,
                episode_id=episode_id,
                trace_path=trace_path,
                image_sha256=str(public[case_id]["image_sha256"]),
                gold=gold[case_id],
            )
            if saved_path.exists():
                artifact = load_bound(saved_path, identity=binding)
                status = str(artifact["decision"]["status"])
            else:
                trace = load_json(trace_path)
                if _trace_case_id(trace) != case_id:
                    raise ValueError("source trace case mismatch")
                artifact = await judge_source(
                    live_client,
                    trace,
                    gold=gold[case_id],
                    image_path=image,
                    model=model,
                    cache_dir=cache_dir,
                )
                status = validate_source_review(
                    artifact,
                    trace=trace,
                    gold=gold[case_id],
                    trace_canonical_sha256=artifact["source_trace_canonical_sha256"],
                )
                save_bound(saved_path, identity=binding, payload=artifact)
            results.append({
                "case_id": case_id,
                "episode_id": episode_id,
                "rollout_index": int(row.get("rollout_index", 0)),
                "status": status,
                "path": str(saved_path),
            })
            if status == "pass":
                break
        return {"case_id": case_id, "reviews": results,
                "skipped_after_pass": len(case_rows) - len(results)}

    completed: dict[str, dict] = {}
    retry_error_types = list(retry_error_types or [])
    if retry_error_types:
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise ValueError("selective retry requires an existing summary")
        prior = load_json(summary_path)
        if prior.get("schema_version") != OWNER_VERSION:
            raise ValueError("selective retry summary has the wrong schema")
        prior_cases = prior.get("cases")
        if (not isinstance(prior_cases, list)
                or {str(row.get("case_id")) for row in prior_cases} != set(by_case)):
            raise ValueError("selective retry summary coverage changed")
        allowed_errors = set(retry_error_types)
        for row in prior_cases:
            if row.get("error_type") not in allowed_errors:
                completed[str(row["case_id"])] = row

    async def collect():
        work = sorted((case_id, rows) for case_id, rows in by_case.items()
                      if case_id not in completed)
        if retry_error_types and not work:
            raise ValueError("selective retry matched no cases")
        async for _index, item, result, error in completed_cases(work, one_case, concurrency=concurrency):
            if error is None:
                completed[item[0]] = result
            else:
                completed[item[0]] = {"case_id": item[0], "reviews": [],
                    "skipped_after_pass": 0, "error_type": error}
            flattened = [review for case in completed.values() for review in case["reviews"]]
            counts = Counter(review["status"] for review in flattened)
            _atomic_json(output / "progress.json", {
                "schema_version": OWNER_VERSION,
                "selected_cases": len(by_case),
                "retry_cases": len(work) if retry_error_types else 0,
                "completed_cases": len(completed),
                "provider_reviews": len(flattened),
                "skipped_after_pass": sum(case["skipped_after_pass"] for case in completed.values()),
                "case_errors": sum("error_type" in case for case in completed.values()),
                "review_counts": dict(sorted(counts.items())),
            })
        cases = [completed[case_id] for case_id in sorted(completed)]
        flattened = [review for case in cases for review in case["reviews"]]
        counts = Counter(review["status"] for review in flattened)
        summary = {
            "schema_version": OWNER_VERSION,
            "status": "paused_source_review_requires_resolution"
                if any("error_type" in case for case in cases) else "source_reviews_complete",
            "selected_cases": len(by_case),
            "selected_rollouts": len(rows),
            "provider_reviews": len(flattened),
            "skipped_after_pass": sum(case["skipped_after_pass"] for case in cases),
            "case_errors": sum("error_type" in case for case in cases),
            "review_counts": dict(sorted(counts.items())),
            "cases": cases,
            "training_started": False,
        }
        _atomic_json(output / "summary.json", summary)
        return summary

    if client is not None:
        live_client = client
        return await collect()
    from src.integrations.gemini import GeminiInteractionsClient, GeminiRequestGate
    async with GeminiInteractionsClient(
        timeout=240,
        max_retries=2,
        request_gate=GeminiRequestGate(concurrency),
    ) as live_client:
        return await collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "benchmark", "train-cases", "private-gold", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retry-error-type", dest="retry_error_types", action="append",
                        help="Retry only cases with this error type in the existing summary")
    result = asyncio.run(run(**vars(parser.parse_args())))
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     ensure_ascii=False, indent=2))
    if result["case_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
