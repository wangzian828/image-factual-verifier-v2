"""Freeze engineering-only replacements, then run a bounded resumable judge.

Never reruns Agent decisions or selects on gold. Existing source runs are immutable.
This controller stops before training; it does not promote evaluation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_rows(path, values):
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("frozen ledger changed")
    else:
        path.write_text(content, encoding="utf-8")


def merge_runs(run_dirs, expected_ids):
    selected, replacements = {}, []
    for index, directory in enumerate(run_dirs):
        seen = set()
        for row in rows(directory / "run_results.jsonl"):
            case = row["case_id"]
            if case in seen or case not in expected_ids:
                raise ValueError("duplicate or unknown case")
            seen.add(case)
            old = selected.get(case)
            if index and (old is None or old["status"] == "success"):
                raise ValueError("retry may only replace an existing engineering failure")
            trace = (directory / row["trace_path"]).resolve()
            trace.relative_to(directory.resolve())
            if not trace.is_file():
                raise ValueError("missing source trace")
            selected[case] = {**row, "source_trace_path": str(trace), "source_trace_sha256": digest(trace)}
            if index:
                replacements.append({"case_id": case, "previous_trace": old["source_trace_path"],
                    "replacement_trace": str(trace), "previous_status": old["status"],
                    "replacement_status": row["status"]})
    if set(selected) != set(expected_ids):
        raise ValueError("incomplete case coverage")
    failed = [case for case, row in selected.items() if row["status"] != "success"]
    if failed:
        raise ValueError(f"{len(failed)} engineering failures still require retry")
    return [selected[case] for case in expected_ids], replacements


def terminal(record):
    if record.get("status") == "not_auditable":
        return True
    if record.get("status") != "completed" or record.get("judge_json_parse_error"):
        return False
    properties = PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA["properties"]
    return all(record.get(key) in properties[key]["enum"] for key in
               ("quality_bucket", "fact_alignment", "reason_quality"))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Linux server lock is process-owned and released after crashes.
    import fcntl
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = {"runs": [str(p.resolve()) for p in [args.run_dir, *args.retry_dir]],
                  "benchmark": digest(args.benchmark), "gold": digest(args.gold),
                  "manifest": digest(args.manifest), "judge_model": args.judge_model,
                  "judge_script": digest(ROOT / "scripts/audit_agent_private_gold.py"),
                  "judge_contract": digest(ROOT / "src/eval/private_gold_judge_contract.py"),
                  "expected_count": args.expected_count}
        binding = output / "inputs.json"
        if binding.exists() and json.loads(binding.read_text()) != inputs:
            raise ValueError("controller input binding changed")
        save(binding, inputs)
        deadline = time.monotonic() + args.wait_seconds
        for directory in [args.run_dir, *args.retry_dir]:
            while not (directory / "summary.json").is_file():
                save(output / "progress.json", {"stage": "waiting_for_rollout", "run": str(directory)})
                if time.monotonic() > deadline:
                    raise TimeoutError("rollout wait budget exhausted")
                time.sleep(15)
        expected = [row["case_id"] for row in rows(args.benchmark)]
        if len(expected) != args.expected_count or len(set(expected)) != len(expected):
            raise ValueError("unexpected evaluation denominator")
        merged, replacements = merge_runs([args.run_dir, *args.retry_dir], expected)
        ledger = output / "frozen-results.jsonl"
        save_rows(ledger, merged)
        save(output / "replacements.json", replacements)
        save(output / "freeze.json", {"count": len(merged), "sha256": digest(ledger),
            "scope": "old1682_exploratory" if args.expected_count == 1682 else "explicit_case_set",
            "source_result_sha256": {str(p): digest(p / "run_results.jsonl")
                                     for p in [args.run_dir, *args.retry_dir]}})
        accepted = {}
        # Canary, full batch, then at most two error-only retries. Preserve all votes.
        for batch, concurrency in enumerate((1, args.concurrency, max(1, args.concurrency // 2), 4)):
            pending = [row for row in merged if row["case_id"] not in accepted]
            if batch == 0:
                pending = pending[:1]
            if not pending:
                break
            directory = output / f"judge-{batch}"
            selection = output / f"judge-{batch}-input.jsonl"
            save_rows(selection, pending)
            cache = directory / "audit-results.jsonl"
            cached = rows(cache) if cache.exists() else []
            cached_ids = {r["case_id"] for r in cached}
            if len(cached_ids) != len(cached) or not cached_ids <= {r["case_id"] for r in pending}:
                raise ValueError("invalid cached judge coverage")
            if directory.exists() and not (directory / "summary.json").exists():
                # Never overwrite partial successful paid judgments. Explicitly
                # stop for an error-only recovery selection in a new directory.
                raise ValueError("partial judge batch requires cache-preserving recovery")
            if not directory.exists():
                if batch > 1:
                    time.sleep(15 * 2 ** (batch - 2))
                save(output / "progress.json", {"stage": "judging", "batch": batch,
                    "selected": len(pending), "concurrency": concurrency, "completed": len(accepted)})
                cmd = [sys.executable, str(ROOT / "scripts/audit_agent_private_gold.py"),
                    "--run-dir", str(args.run_dir), "--results-path", str(selection),
                    "--manifest", str(args.manifest), "--private-gold-sidecar", str(args.gold),
                    "--output-dir", str(directory), "--judge-model", args.judge_model,
                    "--thinking-level", "high", "--max-output-tokens", "8192",
                    "--timeout", "240", "--max-retries", "2", "--concurrency", str(concurrency)]
                with (output / f"judge-{batch}.log").open("ab") as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                cached = rows(cache)
            for record in cached:
                if terminal(record):
                    accepted[record["case_id"]] = record
            if batch == 0 and not accepted:
                raise ValueError("judge canary did not pass; inspect before full dispatch")
        save_rows(output / "judge-final.jsonl", [accepted[k] for k in expected if k in accepted])
        save(output / "progress.json", {"stage": "judge_complete" if len(accepted) == len(expected)
            else "judge_retry_budget_exhausted", "completed": len(accepted), "expected": len(expected),
            "training_started": False, "remaining": [k for k in expected if k not in accepted]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "benchmark", "gold", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--retry-dir", action="append", type=Path, default=[])
    parser.add_argument("--expected-count", type=int, default=1682)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--wait-seconds", type=int, default=2100)
    run(parser.parse_args())
