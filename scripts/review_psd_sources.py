"""Review frozen training rollouts before PSD routing; resumable and GPU-free."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file
from ifv_training.psd_candidates import _load_train_case_allowlist, _safe_trace_path, _trace_case_id
from ifv_training.psd_case_pool import completed_cases
from ifv_training.psd_gemini_judge import _atomic_json
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_source_review import VERSION, TRACE_PROJECTION, PROMPT, SCHEMA, judge_source, validate_source_review
from ifv_training.psd_collection import require_completed_collection


def review_path(root, episode):
    return root / "reviews" / (hashlib.sha256(episode.encode()).hexdigest() + ".json")


async def review_sources(*, run_dir, benchmark, train_cases, private_gold, output,
                         model, concurrency=4, client=None, prefetch=None):
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError('Invalid source review concurrency')
    run_dir, output = run_dir.resolve(), output.resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    require_completed_collection(run_dir, manifest)
    if manifest.get("benchmark", {}).get("training_prohibited"):
        raise ValueError("PSD source review requires completed training-only rollouts")
    cache_dir = output / 'judge-cache'
    if prefetch is not None:
        from scripts.prefetch_psd_source_reviews import validate_prefetch_cache
        cache_dir = validate_prefetch_cache(prefetch, run_dir=run_dir, benchmark=benchmark,
            train_cases=train_cases, private_gold=private_gold, model=model)
        # Prefetch must have drained before the final pass can share its cache.
        from ifv_training.psd_repair_search import search_lock
        with search_lock(Path(prefetch) / 'process-lock'):
            pass
    public_rows, gold_rows = load_jsonl(benchmark), load_jsonl(private_gold)
    public, gold = {r["case_id"]: r for r in public_rows}, {r["case_id"]: r for r in gold_rows}
    allowed = _load_train_case_allowlist(train_cases)
    rows = load_jsonl(run_dir / "run_results.jsonl")
    episodes = [str(r.get("episode_id") or r["case_id"]) for r in rows]
    if (not rows or len(set(episodes)) != len(episodes) or len(public) != len(public_rows)
            or len(gold) != len(gold_rows) or set(public) != {r["case_id"] for r in rows}
            or any(allowed.get(c) != "train" or c not in gold for c in public)):
        raise ValueError("PSD source review membership/coverage is invalid")
    paths = [run_dir / "run_manifest.json", run_dir / "run_results.jsonl", benchmark, train_cases, private_gold]
    identity = {"version": VERSION, "trace_projection": TRACE_PROJECTION,
                "model": model, "prompt": _sha(PROMPT), "schema": _sha(SCHEMA),
                "inputs": {str(p.resolve()): sha256_file(p) for p in paths}}
    marker = output / "inputs.json"
    if marker.exists():
        load_bound(marker, identity=identity)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("source review output must be new or bound")
        save_bound(marker, identity=identity, payload={"training_only": True})

    # Validate every selected case before making any provider call.
    work = []
    for row, episode in zip(rows, episodes):
        case = row["case_id"]
        path = _safe_trace_path(run_dir, str(row.get("trace_path") or ""), episode)
        work.append((case, episode, path))

    async def one(item):
        case, episode, path = item
        saved_path = review_path(output, episode)

        def validate_saved():
            trace = load_json(path)
            if _trace_case_id(trace) != case:
                raise ValueError("source trace case mismatch")
            image = (benchmark.parent / public[case]["image_path"]).resolve()
            binding = {"run": _sha(identity), "episode_id": episode,
                       "trace_sha256": sha256_file(path),
                       "gold_sha256": _sha(gold[case]), "image_sha256": sha256_file(image)}
            artifact = load_bound(saved_path, identity=binding)
            if artifact.get("trace_projection") != TRACE_PROJECTION:
                raise ValueError("source review projection is stale; use a new versioned output")
            status = validate_source_review(artifact, trace=trace, gold=gold[case])
            return {"case_id": case, "episode_id": episode, "status": status,
                    "path": str(saved_path), "sha256": sha256_file(saved_path)}

        # A resumed formal pass may have thousands of already-bound reviews.
        # Their image/archive/hash validation is CPU and filesystem work; doing
        # it synchronously inside this coroutine serializes the entire case
        # pool and makes a small transport-error tail replay every local item at
        # one item per second.  Keep provider concurrency unchanged while using
        # the caller's bounded case pool for independent local revalidation.
        if saved_path.exists():
            return await asyncio.to_thread(validate_saved)
        trace = load_json(path)
        if _trace_case_id(trace) != case:
            raise ValueError("source trace case mismatch")
        image = (benchmark.parent / public[case]["image_path"]).resolve()
        binding = {"run": _sha(identity), "episode_id": episode, "trace_sha256": sha256_file(path),
                   "gold_sha256": _sha(gold[case]), "image_sha256": sha256_file(image)}
        artifact = await judge_source(client, trace, gold=gold[case], image_path=image,
            model=model, cache_dir=cache_dir)
        save_bound(saved_path, identity=binding, payload=artifact)
        if artifact.get("trace_projection") != TRACE_PROJECTION:
            raise ValueError("source review projection is stale; use a new versioned output")
        status = validate_source_review(artifact, trace=trace, gold=gold[case])
        return {"case_id": case, "episode_id": episode, "status": status,
                "path": str(saved_path), "sha256": sha256_file(saved_path)}

    async def collect():
        results = []
        async for index, item, result, error in completed_cases(work, one, concurrency=concurrency):
            results.append(result if error is None else {"case_id": item[0], "episode_id": item[1],
                "status": "pending_error", "error_type": error})
            _atomic_json(output / "progress.json", {"selected": len(work), "completed": len(results),
                "pending": sum(r["status"] == "pending_error" for r in results),
                "abstained": sum(r["status"] == "unresolved" for r in results)})
        results.sort(key=lambda r: r["episode_id"])
        pending = sum(r["status"] == "pending_error" for r in results)
        abstained = sum(r["status"] == "unresolved" for r in results)
        summary = {"schema_version": VERSION, "selected": len(work), "pending": pending,
            "abstained": abstained,
            "status": "paused_source_review_requires_resolution" if pending else
                "source_reviews_complete_with_abstentions" if abstained else "source_reviews_complete",
            "counts": {s: sum(r["status"] == s for r in results) for s in ("pass", "fail", "unresolved", "pending_error")},
            "reviews": results, "training_started": False}
        _atomic_json(output / "summary.json", summary)
        return summary

    if client is not None:
        return await collect()
    from src.integrations.gemini import GeminiInteractionsClient, GeminiRequestGate
    async with GeminiInteractionsClient(timeout=240, max_retries=2,
            request_gate=GeminiRequestGate(concurrency)) as live_client:
        client = live_client
        return await collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "benchmark", "train-cases", "private-gold", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument('--prefetch', type=Path,
        help='Exact-request cache from the completed incremental prefetch process')
    result = asyncio.run(review_sources(**vars(parser.parse_args())))
    print(json.dumps({k: v for k, v in result.items() if k != "reviews"}, ensure_ascii=False, indent=2))
    if result["pending"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
