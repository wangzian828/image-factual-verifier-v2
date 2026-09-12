"""Audit a completed PSD feedback-search run without replaying provider calls.

The report counts proposal, continuation and verifier outcomes, records token
usage, and verifies the hash-bound search snapshots.  Provider responses are
deduplicated by response ID (or request identity) because the same cached judge
result can be linked from more than one repair round.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from ifv_training.io import canonical_json, load_json, load_jsonl, sha256_file, write_json
from ifv_training.psd_repair import _sha


SCHEMA_VERSION = "ifv-psd-feedback-run-audit-v1"
TERMINAL_STATUSES = {
    "converged",
    "attempt_budget_exhausted",
    "proposal_budget_exhausted",
    "time_budget_exhausted",
    "no_further_grounded_hint",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_bound(path: Path) -> tuple[Mapping[str, Any], Any]:
    saved = load_json(path)
    if saved.get("payload_sha256") != _sha(saved.get("payload")):
        raise ValueError(f"bound payload hash mismatch: {path}")
    identity = saved.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"bound identity is invalid: {path}")
    return identity, saved.get("payload")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _response_key(path: Path, value: Mapping[str, Any]) -> str:
    response = _mapping(value.get("response"))
    response_id = str(response.get("id") or "").strip()
    if response_id:
        return "response:" + response_id
    identity = _mapping(value.get("identity"))
    if identity:
        return "identity:" + _sha(identity)
    return "file:" + sha256_file(path)


def _audit_case(case_dir: Path, *, seen_judge: set[str], seen_attempts: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = case_dir / "manifest.json"
    state_path = case_dir / "search-state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        return {
            "case_key": case_dir.name,
            "passed": False,
            "errors": ["missing manifest.json or search-state.json"],
        }

    manifest = load_json(manifest_path)
    try:
        _, state = _validate_bound(state_path)
    except (TypeError, ValueError) as exc:
        state = {}
        errors.append(str(exc))
    state = _mapping(state)
    rounds = state.get("rounds")
    if not isinstance(rounds, list):
        rounds = []
        errors.append("search-state rounds are invalid")

    status = str(manifest.get("status") or "")
    if status not in TERMINAL_STATUSES:
        errors.append(f"search is not terminal: {status or 'missing'}")
    if manifest.get("proposal_rounds") != len(rounds):
        errors.append("manifest proposal_rounds differs from search-state")

    proposal_requests = 0
    proposal_prompt_tokens = 0
    proposal_completion_tokens = 0
    proposal_passes = 0
    proposal_rejections: Counter[str] = Counter()
    round_directories: set[Path] = set()

    for index, round_state in enumerate(rounds):
        round_state = _mapping(round_state)
        directory_value = str(round_state.get("directory") or "")
        directory = Path(directory_value) if directory_value else case_dir / "missing"
        if not _inside(directory, case_dir):
            errors.append(f"round {index} directory escapes case root")
            continue
        directory = directory.resolve()
        if directory in round_directories:
            errors.append(f"round {index} directory is duplicated")
        round_directories.add(directory)
        if not directory.is_dir():
            errors.append(f"round {index} directory is missing")
            continue

        files = round_state.get("files")
        if not isinstance(files, Mapping):
            errors.append(f"round {index} file snapshot is invalid")
        else:
            for raw_path, expected in files.items():
                path = Path(str(raw_path))
                if not _inside(path, directory):
                    errors.append(f"round {index} snapshot path escapes round root")
                elif not path.is_file():
                    errors.append(f"round {index} snapshot file is missing: {path.name}")
                elif sha256_file(path) != expected:
                    errors.append(f"round {index} snapshot hash changed: {path.name}")

        proposer_path = directory / "proposer-response.json"
        if proposer_path.is_file():
            try:
                _, proposer = _validate_bound(proposer_path)
                proposer = _mapping(proposer)
                proposal_prompt_tokens += _nonnegative_int(
                    proposer.get("prompt_tokens"), field="proposer prompt_tokens"
                )
                proposal_completion_tokens += _nonnegative_int(
                    proposer.get("completion_tokens"), field="proposer completion_tokens"
                )
                proposal_requests += 1
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        else:
            errors.append(f"round {index} lacks cached proposer response")

        audits_path = directory / "proposer-hint-audits.json"
        if audits_path.is_file():
            audits = load_json(audits_path).get("proposals")
            if not isinstance(audits, list):
                errors.append(f"round {index} proposal audits are invalid")
            else:
                for audit in audits:
                    audit = _mapping(audit)
                    if audit.get("passed") is True:
                        proposal_passes += 1
                    else:
                        proposal_rejections[str(audit.get("reason") or "unspecified")] += 1

    attempts = load_jsonl(case_dir / "repair_attempts.jsonl")
    if manifest.get("candidate_count") != len(attempts):
        errors.append("manifest candidate_count differs from repair attempts")
    accepted = sum(row.get("accepted") is True for row in attempts)
    if manifest.get("accepted_count") != accepted:
        errors.append("manifest accepted_count differs from repair attempts")
    if bool(manifest.get("converged")) != bool(accepted):
        errors.append("manifest convergence differs from accepted attempts")

    continuation_rejections: Counter[str] = Counter()
    qwen_prompt_tokens = 0
    qwen_completion_tokens = 0
    case_id = ""
    for row in attempts:
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id:
            errors.append("repair attempt lacks attempt_id")
        elif attempt_id in seen_attempts:
            errors.append("repair attempt ID is duplicated across cases")
        else:
            seen_attempts.add(attempt_id)
        case_id = case_id or str(row.get("case_id") or "")
        prompt_ids = row.get("teacher_prompt_ids")
        completion_ids = row.get("completion_ids")
        if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
            errors.append("repair attempt lacks teacher token IDs")
        else:
            qwen_prompt_tokens += len(prompt_ids)
            qwen_completion_tokens += len(completion_ids)
        if row.get("accepted") is not True:
            reasons = _mapping(row.get("verification")).get("reasons")
            if isinstance(reasons, list) and reasons:
                continuation_rejections.update(str(reason) for reason in reasons)
            else:
                continuation_rejections["unrepairable_without_reason"] += 1

    judge_requests = 0
    duplicate_judge_cache_entries = 0
    judge_usage: Counter[str] = Counter()
    for path in sorted(case_dir.glob("rounds/round-*/judge-cache/*.json")):
        try:
            value = load_json(path)
            key = _response_key(path, value)
            if key in seen_judge:
                duplicate_judge_cache_entries += 1
                continue
            seen_judge.add(key)
            response = _mapping(value.get("response"))
            if response.get("status") != "completed":
                errors.append(f"judge cache is not completed: {path.name}")
                continue
            usage = _mapping(response.get("usage"))
            for field in (
                "total_tokens",
                "total_input_tokens",
                "total_cached_tokens",
                "total_output_tokens",
                "total_thought_tokens",
                "total_tool_use_tokens",
            ):
                judge_usage[field] += _nonnegative_int(
                    usage.get(field, 0), field=f"judge {field}"
                )
            judge_requests += 1
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid judge cache {path.name}: {type(exc).__name__}")

    return {
        "case_key": case_dir.name,
        "case_id": case_id,
        "passed": not errors,
        "errors": errors,
        "status": status,
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "proposal_rounds": len(rounds),
        "proposals": {
            "provider_requests": proposal_requests,
            "prompt_tokens": proposal_prompt_tokens,
            "completion_tokens": proposal_completion_tokens,
            "locally_admissible": proposal_passes,
            "rejections": dict(sorted(proposal_rejections.items())),
        },
        "continuations": {
            "count": len(attempts),
            "accepted": accepted,
            "rejected": len(attempts) - accepted,
            "rejection_reasons": dict(sorted(continuation_rejections.items())),
            "qwen_prompt_tokens": qwen_prompt_tokens,
            "qwen_completion_tokens": qwen_completion_tokens,
        },
        "judge": {
            "unique_completed_requests": judge_requests,
            "duplicate_cache_entries": duplicate_judge_cache_entries,
            "usage": dict(sorted(judge_usage.items())),
        },
    }


def audit_feedback_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    repairs = run_dir / "repairs"
    case_dirs = sorted(path for path in repairs.iterdir() if path.is_dir()) if repairs.is_dir() else []
    seen_judge: set[str] = set()
    seen_attempts: set[str] = set()
    cases = [
        _audit_case(path, seen_judge=seen_judge, seen_attempts=seen_attempts)
        for path in case_dirs
    ]
    errors = [
        {"case_key": case["case_key"], "error": error}
        for case in cases
        for error in case["errors"]
    ]
    totals = {
        "cases": len(cases),
        "converged_cases": sum(case.get("status") == "converged" for case in cases),
        "proposal_rounds": sum(case.get("proposal_rounds", 0) for case in cases),
        "proposal_requests": sum(case.get("proposals", {}).get("provider_requests", 0) for case in cases),
        "proposal_prompt_tokens": sum(case.get("proposals", {}).get("prompt_tokens", 0) for case in cases),
        "proposal_completion_tokens": sum(case.get("proposals", {}).get("completion_tokens", 0) for case in cases),
        "continuations": sum(case.get("continuations", {}).get("count", 0) for case in cases),
        "accepted_continuations": sum(case.get("continuations", {}).get("accepted", 0) for case in cases),
        "qwen_prompt_tokens": sum(case.get("continuations", {}).get("qwen_prompt_tokens", 0) for case in cases),
        "qwen_completion_tokens": sum(case.get("continuations", {}).get("qwen_completion_tokens", 0) for case in cases),
        "unique_judge_requests": sum(case.get("judge", {}).get("unique_completed_requests", 0) for case in cases),
        "duplicate_judge_cache_entries": sum(case.get("judge", {}).get("duplicate_cache_entries", 0) for case in cases),
    }
    judge_usage: Counter[str] = Counter()
    proposal_rejections: Counter[str] = Counter()
    continuation_rejections: Counter[str] = Counter()
    for case in cases:
        judge_usage.update(case.get("judge", {}).get("usage", {}))
        proposal_rejections.update(case.get("proposals", {}).get("rejections", {}))
        continuation_rejections.update(case.get("continuations", {}).get("rejection_reasons", {}))
    totals["judge_usage"] = dict(sorted(judge_usage.items()))
    totals["proposal_rejections"] = dict(sorted(proposal_rejections.items()))
    totals["continuation_rejection_reasons"] = dict(sorted(continuation_rejections.items()))
    if not cases:
        errors.append({"case_key": "", "error": "no repair cases found"})
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
        "run_dir": str(run_dir),
        "errors": errors,
        "totals": totals,
        "cases": cases,
        "privacy": {
            "contains_provider_response_text": False,
            "contains_private_judge_explanations": False,
            "token_cost_is_reported_without_currency_estimate": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_feedback_run(args.run_dir)
    write_json(args.output, report)
    print(canonical_json(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
