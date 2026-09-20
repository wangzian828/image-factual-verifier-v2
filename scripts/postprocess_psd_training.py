"""Derive PSD rewards from private training gold without exporting SFT/perception."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_candidates import _load_train_case_allowlist, _trace_case_id
from ifv_training.psd_collection import require_completed_collection, repairable_terminal_model_failure
from scripts.audit_real_trace import audit_trace
from src.eval.postprocess_run import _trace_path
from src.orchestrator.source_access import SourceAccessPolicy
from src.trajectory.exporter import trajectory_policy_step_ids
from src.trajectory.scoring import score_process_trace


@lru_cache(maxsize=4)
def _load_source_policy(path: str) -> SourceAccessPolicy:
    return SourceAccessPolicy.load(Path(path))


def _load_trace_once(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_review_episode_ids(source_reviews: Path) -> set[str]:
    """Return only episodes with a durable review in a completed summary."""
    summary = load_json(source_reviews / "summary.json")
    if not str(summary.get("status", "")).startswith("source_reviews_complete"):
        raise ValueError("reviewed-only postprocess requires completed source reviews")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("source-review summary cases are invalid")
    episode_ids: list[str] = []
    for case in cases:
        reviews = case.get("reviews") if isinstance(case, Mapping) else None
        if not isinstance(reviews, list):
            raise ValueError("source-review summary review list is invalid")
        for review in reviews:
            episode = str(review.get("episode_id") or "") if isinstance(review, Mapping) else ""
            if not episode:
                raise ValueError("source-review summary episode is invalid")
            episode_ids.append(episode)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("source-review summary contains duplicate episodes")
    from scripts.review_psd_sources import review_path
    missing = [episode for episode in episode_ids
               if not review_path(source_reviews, episode).is_file()]
    if missing:
        raise ValueError("source-review summary references missing artifacts")
    return set(episode_ids)


def _saved_audit(
    path: Path,
    *,
    case: str,
    episode: str,
    trace_sha256: str,
    trace_canonical_sha256: str,
    source_access_policy_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    expected = {
        "schema_version": "ifv-psd-trace-audit-v2",
        "case_id": case,
        "episode_id": episode,
        "source_trace_sha256": trace_sha256,
        "source_trace_canonical_sha256": trace_canonical_sha256,
        "source_access_policy_sha256": source_access_policy_sha256,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    failures = value.get("failures")
    if (not isinstance(failures, list) or type(value.get("passed")) is not bool
            or value["passed"] != (not failures)):
        return None
    return value


def _derive_episode(job: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(job["run_dir"])
    row = job["row"]
    case = row["case_id"]
    episode = job["episode"]
    group = str(row.get("prompt_group_id") or case)
    path = _trace_path(run_dir, row)
    member = {"schema_version": "ifv-rollout-member-v1", "case_id": case,
        "episode_id": episode, "prompt_group_id": group, "group_size": job["group_size"],
        "rollout_index": row.get("rollout_index", 0), "sampling_seed": row.get("sampling_seed"),
        "trace_path": path.relative_to(run_dir).as_posix(), "training_prohibited": False}
    reward = {"schema_version": "ifv-post-rollout-deterministic-v1", "case_id": case,
        "episode_id": episode, "prompt_group_id": group, "training_prohibited": False,
        "classification_correct": False, "fatal_engineering_error": True,
        "strict_trace_audit_pass": False, "step_ids": [], "process_components": {}}
    result = {"member": member, "reward": reward, "score": None, "audit": None,
        "audit_reused": False}
    if not path.is_file():
        return result
    trace, trace_sha256 = _load_trace_once(path)
    if _trace_case_id(trace) != case:
        raise ValueError("PSD trace case identity mismatch")
    from ifv_training.psd_repair import _sha
    trace_canonical_sha256 = _sha(trace)
    metrics, score = score_process_trace(trace, job["gold"])
    policy_failure = repairable_terminal_model_failure(trace)
    audit_path = run_dir / "psd-audits" / (path.stem + ".json")
    report = _saved_audit(audit_path, case=case, episode=episode,
        trace_sha256=trace_sha256, trace_canonical_sha256=trace_canonical_sha256,
        source_access_policy_sha256=job["source_access_policy_sha256"])
    if report is None:
        audit = audit_trace(path,
            source_access_policy=_load_source_policy(job["source_access_policy"]),
            payload=trace)
        failures = [asdict(item) for item in audit.failures(strict_scheduler=True)]
        report = {"schema_version": "ifv-psd-trace-audit-v2", "case_id": case,
            "episode_id": episode, "source_trace_sha256": trace_sha256,
            "source_trace_canonical_sha256": trace_canonical_sha256,
            "source_access_policy_sha256": job["source_access_policy_sha256"],
            "passed": not failures, "failures": failures}
        _write_json_atomic(audit_path, report)
    else:
        failures = report["failures"]
        result["audit_reused"] = True
    audit_sha256 = sha256_file(audit_path)
    result["audit"] = {"episode_id": episode, "path": str(audit_path), "sha256": audit_sha256}
    reward["source_audit"] = {"path": str(audit_path.resolve()), "sha256": audit_sha256}
    reward.update(classification_correct=bool(metrics.get("result_correct")),
        fatal_engineering_error=bool(metrics.get("engineering_error")) and not bool(policy_failure),
        source_policy_failure_kind=policy_failure,
        legacy_process_engineering_flag=bool(metrics.get("engineering_error")),
        strict_trace_audit_pass=not failures,
        strict_trace_audit_failure_codes=[item["code"] for item in failures],
        step_ids=trajectory_policy_step_ids(trace, episode_id=episode),
        process_components=score.get("components", {}))
    result["score"] = {**metrics, "episode_id": episode, "prompt_group_id": group}
    member["trace_sha256"] = trace_sha256
    reward["source_task_status"] = "pending"
    if job["source_reviews"] is not None:
        from scripts.review_psd_sources import review_path
        from ifv_training.psd_source_review import source_review_reference, validate_source_review
        review_file = review_path(Path(job["source_reviews"]), episode)
        if review_file.exists():
            reward["source_task_review"] = {"path": str(review_file.resolve()),
                "sha256": sha256_file(review_file)}
            artifact = source_review_reference(reward)
            reward["source_task_status"] = validate_source_review(artifact, trace=trace,
                gold=job["gold"], trace_canonical_sha256=trace_canonical_sha256)
    reward["verified_full_task"] = bool(reward["classification_correct"]
        and reward["strict_trace_audit_pass"] and not reward["fatal_engineering_error"]
        and reward["source_task_status"] == "pass")
    return result


def postprocess(*, run_dir, train_cases, private_gold, source_access_policy,
                source_reviews=None, workers=1, reviewed_only=False):
    run_dir = run_dir.resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    require_completed_collection(run_dir, manifest)
    if manifest.get("benchmark", {}).get("training_prohibited"):
        raise ValueError("PSD requires a completed training-only rollout")
    allowed = _load_train_case_allowlist(train_cases)
    all_rows = load_jsonl(run_dir / "run_results.jsonl")
    gold_rows = load_jsonl(private_gold)
    gold = {row["case_id"]: row for row in gold_rows}
    if len(gold) != len(gold_rows) or not all_rows:
        raise ValueError("empty rollout or duplicate private training references")
    sizes = Counter(str(row.get("prompt_group_id") or row["case_id"]) for row in all_rows)
    if reviewed_only:
        if source_reviews is None:
            raise ValueError("reviewed-only postprocess requires source reviews")
        reviewed = _completed_review_episode_ids(source_reviews.resolve())
        rows = [row for row in all_rows
                if str(row.get("episode_id") or row["case_id"]) in reviewed]
        if len(rows) != len(reviewed):
            raise ValueError("source-review episodes differ from rollout collection")
    else:
        rows = all_rows
    episodes = [str(row.get("episode_id") or row["case_id"]) for row in rows]
    if len(set(episodes)) != len(episodes):
        raise ValueError("duplicate PSD rollout episode")
    for row in rows:
        case = row["case_id"]
        if allowed.get(case) != "train" or case not in gold:
            raise ValueError("case lacks explicit train membership/private reference")
        if gold[case].get("factual_status") not in {"supported", "refuted"}:
            raise ValueError("unsupported private training factual status")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 32:
        raise ValueError("PSD postprocess workers must be in [1, 32]")
    recorded_policy = manifest.get("source_access_policy", {})
    source_access_policy_sha256 = sha256_file(source_access_policy)
    if (not recorded_policy.get("active")
            or sha256_file(Path(recorded_policy["path"])) != source_access_policy_sha256):
        raise ValueError("PSD source policy differs from rollout")
    groups, rewards, scores, audits = [], [], [], []
    jobs = [{"run_dir": str(run_dir), "row": row, "episode": episode,
        "group_size": sizes[str(row.get("prompt_group_id") or row["case_id"])],
        "gold": gold[row["case_id"]], "source_access_policy": str(source_access_policy.resolve()),
        "source_access_policy_sha256": source_access_policy_sha256,
        "source_reviews": str(source_reviews.resolve()) if source_reviews is not None else None}
        for row, episode in zip(rows, episodes)]
    progress_path = run_dir / "psd-postprocess-progress.json"
    _write_json_atomic(progress_path, {"status": "running", "completed": 0,
        "total": len(jobs), "workers": workers, "audit_reused": 0})
    if workers == 1:
        iterator = map(_derive_episode, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        iterator = executor.map(_derive_episode, jobs, chunksize=1)
    reused = 0
    try:
        for completed, derived in enumerate(iterator, start=1):
            groups.append(derived["member"])
            rewards.append(derived["reward"])
            if derived["score"] is not None:
                scores.append(derived["score"])
            if derived["audit"] is not None:
                audits.append(derived["audit"])
            reused += int(derived["audit_reused"])
            if completed % 10 == 0 or completed == len(jobs):
                _write_json_atomic(progress_path, {"status": "running", "completed": completed,
                    "total": len(jobs), "workers": workers, "audit_reused": reused})
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    write_jsonl(run_dir / "rollout_groups.jsonl", groups)
    write_jsonl(run_dir / "post_rollout_rewards.jsonl", rewards)
    write_jsonl(run_dir / "psd-process-metrics.jsonl", scores)
    result = {"schema_version": "ifv-psd-training-postprocess-v1", "episodes": len(rows),
        "collection_episodes": len(all_rows), "reviewed_only": bool(reviewed_only),
        "excluded_unreviewed_episodes": len(all_rows) - len(rows),
        "correct": sum(row["classification_correct"] for row in rewards),
        "strict_pass": sum(row["strict_trace_audit_pass"] for row in rewards),
        "engineering_errors": sum(row["fatal_engineering_error"] for row in rewards),
        "verified_full_task": sum(row.get("verified_full_task") is True for row in rewards),
        "source_review_pending": sum(row.get("source_task_status", "pending") == "pending"
                                     for row in rewards if not row["fatal_engineering_error"]),
        "source_review_abstained": sum(row.get("source_task_status") == "unresolved"
                                       for row in rewards if not row["fatal_engineering_error"]),
        "inputs": {"train_cases_sha256": sha256_file(train_cases), "private_gold_sha256": sha256_file(private_gold),
                   "source_access_policy_sha256": source_access_policy_sha256}, "audits": audits}
    write_json(run_dir / "psd-postprocess.json", result)
    _write_json_atomic(progress_path, {"status": "completed", "completed": len(jobs),
        "total": len(jobs), "workers": workers, "audit_reused": reused})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "train-cases", "private-gold", "source-access-policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-reviews", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reviewed-only", action="store_true")
    print(json.dumps(postprocess(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
