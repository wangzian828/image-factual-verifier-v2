"""Finalize an explicitly stopped PSD tail without any new model calls.

Read only compact terminal manifests and their candidate/attempt ledgers.  A
paused or running case is recorded as excluded; its budget and artifacts remain
untouched.  Merged ledgers are written compressed before normal strict bank
materialization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_materialization import materialize_bank


TERMINAL = frozenset({"converged", "passed_without_intervention",
    "attempt_budget_exhausted", "proposal_budget_exhausted",
    "infrastructure_budget_exhausted"})


def ledger(directory: Path, stem: str) -> Path:
    plain, compressed = directory / f"{stem}.jsonl", directory / f"{stem}.jsonl.gz"
    if plain.exists() and compressed.exists():
        raise ValueError(f"duplicate PSD ledger formats: {directory}/{stem}")
    result = plain if plain.exists() else compressed
    if not result.is_file():
        raise ValueError(f"terminal PSD case lacks {stem}: {directory}")
    return result


def collect_terminal(search: Path):
    candidates, attempts, included, excluded = {}, [], [], []
    statuses = Counter()
    expected_accepted = 0
    for manifest_path in sorted((search / "repairs").glob("*/manifest.json")):
        manifest = load_json(manifest_path)
        status = str(manifest.get("status") or "")
        statuses[status] += 1
        key = manifest_path.parent.name
        if status not in TERMINAL:
            excluded.append({"repair_key": key, "status": status})
            continue
        candidate_path = ledger(manifest_path.parent, "repair_candidates")
        attempt_path = ledger(manifest_path.parent, "repair_attempts")
        case_candidates = load_jsonl(candidate_path)
        case_attempts = load_jsonl(attempt_path)
        for row in case_candidates:
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id:
                raise ValueError(f"candidate without id: {candidate_path}")
            if candidate_id in candidates and candidates[candidate_id] != row:
                raise ValueError(f"conflicting duplicate candidate: {candidate_id}")
            candidates[candidate_id] = row
        attempts.extend(case_attempts)
        accepted = sum(row.get("accepted") is True for row in case_attempts)
        if accepted != int(manifest.get("accepted_count") or 0):
            raise ValueError(f"accepted count differs from manifest: {key}")
        expected_accepted += accepted
        included.append({"repair_key": key, "status": status,
            "manifest_sha256": sha256_file(manifest_path)})
    if not included or not expected_accepted:
        raise ValueError("stopped PSD tail has no terminal verified repair")
    attempts.sort(key=lambda row: (str(row.get("case_id") or ""),
        str(row.get("attempt_id") or "")))
    return ([candidates[key] for key in sorted(candidates)], attempts, {
        "schema_version": "ifv-psd-stopped-tail-selection-v1",
        "status_counts": dict(sorted(statuses.items())),
        "included": included, "excluded": excluded,
        "accepted_attempts": expected_accepted,
        "new_agent_or_provider_calls": False})


def finalize(*, search: Path, preservation: Path, control: Path):
    search, preservation, control = search.resolve(), preservation.resolve(), control.resolve()
    control.mkdir(parents=True, exist_ok=False)
    candidates, attempts, selection = collect_terminal(search)
    merge = search / "merged-stopped-tail-v1"
    merge.mkdir(parents=True, exist_ok=False)
    candidates_path = merge / "repair_candidates.jsonl.gz"
    attempts_path = merge / "repair_attempts.jsonl.gz"
    write_jsonl(candidates_path, candidates)
    write_jsonl(attempts_path, attempts)
    selection["inputs"] = {str(path): sha256_file(path) for path in (
        candidates_path, attempts_path, preservation)}
    write_json(control / "selection.json", selection)
    result = materialize_bank(output_dir=search, repair_candidates=candidates_path,
        repair_attempts=attempts_path, preservation_candidates=preservation,
        score_missing_topk=False)
    write_json(control / "result.json", result)
    write_json(control / "state.json", {"phase": result["status"],
        "training_started": False, "new_agent_or_provider_calls": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--preservation", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    print(finalize(**vars(args)))


if __name__ == "__main__":
    main()
