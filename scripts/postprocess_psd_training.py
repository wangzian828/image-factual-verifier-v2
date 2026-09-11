"""Derive PSD rewards from private training gold without exporting SFT/perception."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_candidates import _load_train_case_allowlist, _trace_case_id
from scripts.audit_real_trace import audit_trace
from src.eval.postprocess_run import _trace_path
from src.orchestrator.source_access import SourceAccessPolicy
from src.trajectory.exporter import trajectory_policy_step_ids
from src.trajectory.scoring import score_process_trace


def postprocess(*, run_dir, train_cases, private_gold, source_access_policy):
    run_dir = run_dir.resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed" or manifest.get("benchmark", {}).get("training_prohibited"):
        raise ValueError("PSD requires a completed training-only rollout")
    allowed = _load_train_case_allowlist(train_cases)
    rows = load_jsonl(run_dir / "run_results.jsonl")
    gold_rows = load_jsonl(private_gold)
    gold = {row["case_id"]: row for row in gold_rows}
    if len(gold) != len(gold_rows) or not rows:
        raise ValueError("empty rollout or duplicate private training references")
    episodes = [str(row.get("episode_id") or row["case_id"]) for row in rows]
    if len(set(episodes)) != len(episodes):
        raise ValueError("duplicate PSD rollout episode")
    for row in rows:
        case = row["case_id"]
        if allowed.get(case) != "train" or case not in gold:
            raise ValueError("case lacks explicit train membership/private reference")
        if gold[case].get("factual_status") not in {"supported", "refuted"}:
            raise ValueError("unsupported private training factual status")
    recorded_policy = manifest.get("source_access_policy", {})
    if not recorded_policy.get("active") or sha256_file(Path(recorded_policy["path"])) != sha256_file(source_access_policy):
        raise ValueError("PSD source policy differs from rollout")
    policy = SourceAccessPolicy.load(source_access_policy)
    groups, rewards, scores, audits = [], [], [], []
    sizes = Counter(str(row.get("prompt_group_id") or row["case_id"]) for row in rows)
    for row, episode in zip(rows, episodes):
        case = row["case_id"]
        group = str(row.get("prompt_group_id") or case)
        path = _trace_path(run_dir, row)
        member = {"schema_version": "ifv-rollout-member-v1", "case_id": case,
            "episode_id": episode, "prompt_group_id": group, "group_size": sizes[group],
            "rollout_index": row.get("rollout_index", 0), "sampling_seed": row.get("sampling_seed"),
            "trace_path": path.relative_to(run_dir).as_posix(), "training_prohibited": False}
        groups.append(member)
        reward = {"schema_version": "ifv-post-rollout-deterministic-v1", "case_id": case,
            "episode_id": episode, "prompt_group_id": group, "training_prohibited": False,
            "classification_correct": False, "fatal_engineering_error": True,
            "strict_trace_audit_pass": False, "step_ids": [], "process_components": {}}
        if path.is_file():
            trace = load_json(path)
            if _trace_case_id(trace) != case:
                raise ValueError("PSD trace case identity mismatch")
            metrics, score = score_process_trace(trace, gold[case])
            audit = audit_trace(path, source_access_policy=policy)
            failures = [asdict(item) for item in audit.failures(strict_scheduler=True)]
            report = {"case_id": case, "episode_id": episode, "source_trace_sha256": sha256_file(path),
                      "passed": not failures, "failures": failures}
            audit_path = run_dir / "psd-audits" / (path.stem + ".json")
            write_json(audit_path, report)
            audits.append({"episode_id": episode, "path": str(audit_path), "sha256": sha256_file(audit_path)})
            reward.update(classification_correct=bool(metrics.get("result_correct")),
                fatal_engineering_error=bool(metrics.get("engineering_error")),
                strict_trace_audit_pass=not failures,
                strict_trace_audit_failure_codes=[item["code"] for item in failures],
                step_ids=trajectory_policy_step_ids(trace, episode_id=episode),
                process_components=score.get("components", {}))
            scores.append({**metrics, "episode_id": episode, "prompt_group_id": group})
            member["trace_sha256"] = sha256_file(path)
        rewards.append(reward)
    write_jsonl(run_dir / "rollout_groups.jsonl", groups)
    write_jsonl(run_dir / "post_rollout_rewards.jsonl", rewards)
    write_jsonl(run_dir / "psd-process-metrics.jsonl", scores)
    result = {"schema_version": "ifv-psd-training-postprocess-v1", "episodes": len(rows),
        "correct": sum(row["classification_correct"] for row in rewards),
        "strict_pass": sum(row["strict_trace_audit_pass"] for row in rewards),
        "engineering_errors": sum(row["fatal_engineering_error"] for row in rewards),
        "inputs": {"train_cases_sha256": sha256_file(train_cases), "private_gold_sha256": sha256_file(private_gold),
                   "source_access_policy_sha256": sha256_file(source_access_policy)}, "audits": audits}
    write_json(run_dir / "psd-postprocess.json", result)
    return result


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "train-cases", "private-gold", "source-access-policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(postprocess(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
