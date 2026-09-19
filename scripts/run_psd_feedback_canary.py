"""Exercise all fixed failed training cases with bounded feedback search.

Read the previous canary's frozen source bank, never mutate its datums or reuse
its privileged-reference-conditioned hints as new public-only search results.
Server artifacts only. Does not launch training or manage inference services.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_gemini_judge import trace_steps, _atomic_json
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_repairs import assemble_psd_repair_package
from ifv_training.psd import build_psd_target_package
from ifv_training.psd_datums import build_sparse_topk_package
from ifv_training.psd_case_pool import completed_cases
from ifv_training.psd_materialization import materialize_bank
from ifv_training.psd_materialization import completed_package
from ifv_training.psd_collection import build_task_repair_selection_parallel
from scripts.run_psd_repair_driver import _parser, _run


TERMINAL_SEARCH_STATUSES = {
    "converged",
    "passed_without_intervention",
    "attempt_budget_exhausted",
    "proposal_budget_exhausted",
    "infrastructure_budget_exhausted",
}


def _case_ledger(directory: Path, stem: str) -> Path:
    plain, compressed = directory / f"{stem}.jsonl", directory / f"{stem}.jsonl.gz"
    if plain.exists() and compressed.exists():
        raise ValueError(f"duplicate PSD case ledger formats: {directory}/{stem}")
    return plain if plain.exists() else compressed


def _validate_fast_resume_receipt(path, expected_sha256):
    path = Path(path).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("PSD fast-resume receipt digest changed")
    receipt = load_json(path)
    if receipt.get("schema_version") != "ifv-psd-fast-resume-inputs-v1":
        raise ValueError("unknown PSD fast-resume receipt schema")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("PSD fast-resume receipt has no files")
    for name, saved in files.items():
        target = Path(name)
        current = target.stat()
        observed = {
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "inode": current.st_ino,
            "device": current.st_dev,
        }
        if observed != {key: saved.get(key) for key in observed}:
            raise ValueError("PSD fast-resume input metadata changed: " + name)
        digest = saved.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("PSD fast-resume input digest is invalid: " + name)
    return receipt


def _receipt_digest(receipt, path):
    name = str(Path(path).resolve())
    try:
        return receipt["files"][name]["sha256"]
    except KeyError as error:
        raise ValueError("PSD fast-resume receipt is missing: " + name) from error


def _resume_case_plan(repair_sources, previous_progress):
    """Carry terminal cases by their small manifest and schedule only the tail.

    A completed case was fully audited in the invocation that persisted it in
    ``progress.json``.  Outer resume passes therefore compare its bound
    manifest instead of recursively reopening the much larger episode tree.
    Missing, paused, or unknown-status cases remain scheduled.
    """
    rows = list(repair_sources)
    indexes = {}
    for index, candidate in enumerate(rows):
        case_id = candidate["case_id"]
        if case_id in indexes:
            raise ValueError(f"duplicate selected repair case: {case_id}")
        indexes[case_id] = index
    carried = {}
    previous_cases = previous_progress.get("cases") or []
    if not isinstance(previous_cases, list):
        raise ValueError("previous PSD progress cases must be a list")
    for record in previous_cases:
        case_id = str(record.get("case_id") or "")
        if case_id not in indexes or case_id in carried:
            continue
        result = record.get("result") or {}
        if result.get("status") not in TERMINAL_SEARCH_STATUSES:
            continue
        directory = Path(str(record.get("directory") or ""))
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or load_json(manifest_path) != result:
            raise ValueError(f"terminal PSD manifest changed for {case_id}")
        carried[case_id] = {**record, "input_index": indexes[case_id]}
    pending = [
        (index, candidate)
        for index, candidate in enumerate(rows)
        if candidate["case_id"] not in carried
    ]
    return [carried[case_id] for case_id in sorted(carried, key=indexes.get)], pending


def _resume_indexed_case_plan(selected_path, index_path, previous_progress):
    """Seek directly to unfinished selected rows without scanning the JSONL."""
    selected_path, index_path = Path(selected_path), Path(index_path)
    index = load_json(index_path)
    if (index.get("schema_version") != "ifv-psd-selected-offset-index-v1"
            or Path(index.get("selected_path", "")).resolve() != selected_path.resolve()):
        raise ValueError("PSD selected-row offset index binding changed")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("PSD selected-row offset index is empty")
    case_indexes = {}
    for input_index, entry in enumerate(entries):
        case_id = str(entry.get("case_id") or "")
        if (not case_id or case_id in case_indexes
                or entry.get("input_index") != input_index
                or type(entry.get("offset")) is not int or entry["offset"] < 0
                or type(entry.get("length")) is not int or entry["length"] <= 0):
            raise ValueError("PSD selected-row offset index is invalid")
        case_indexes[case_id] = input_index
    carried = {}
    previous_cases = previous_progress.get("cases") or []
    if not isinstance(previous_cases, list):
        raise ValueError("previous PSD progress cases must be a list")
    for record in previous_cases:
        case_id = str(record.get("case_id") or "")
        if case_id not in case_indexes or case_id in carried:
            continue
        result = record.get("result") or {}
        if result.get("status") not in TERMINAL_SEARCH_STATUSES:
            continue
        directory = Path(str(record.get("directory") or ""))
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or load_json(manifest_path) != result:
            raise ValueError(f"terminal PSD manifest changed for {case_id}")
        carried[case_id] = {**record, "input_index": case_indexes[case_id]}
    pending = []
    with selected_path.open("rb") as source:
        for entry in entries:
            case_id = entry["case_id"]
            if case_id in carried:
                continue
            source.seek(entry["offset"])
            raw = source.read(entry["length"])
            if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
                raise ValueError("PSD selected row changed for " + case_id)
            candidate = json.loads(raw)
            if candidate.get("case_id") != case_id:
                raise ValueError("PSD selected-row index case mismatch")
            pending.append((entry["input_index"], candidate))
    return [carried[case_id] for case_id in sorted(carried, key=case_indexes.get)], pending


def _iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


async def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("feedback canary must be separate from the frozen source bank")
    preparation = load_json(source / "prepared.json")["preparation"]
    run_dir = Path(preparation.get("rollout_dir") or source / "on-policy-r1")
    benchmark, split, policy, gold_path = [Path(preparation[k]) for k in (
        "benchmark", "case_split", "source_access_policy", "private_gold")]
    serving, checkpoint = args.snapshot / "serving-profile.json", args.snapshot / "checkpoint-manifest.json"
    profile = load_json(serving)
    candidates_path = source / "candidates/repair_candidates.jsonl"
    preservation_path = source / "candidates/preservation_candidates.jsonl"
    input_paths = [
        source / "prepared.json", candidates_path, preservation_path, benchmark, split, policy,
        gold_path, serving, checkpoint]
    if (source / "datums/datums.jsonl").exists():
        input_paths.append(source / "datums/datums.jsonl")
    proposal_rounds = getattr(args, "proposal_rounds", 12)
    search_only = getattr(args, "search_only", False)
    if type(proposal_rounds) is not int or not args.attempts <= proposal_rounds <= 64:
        raise ValueError("proposal rounds must be between attempt budget and 64")
    receipt_path = getattr(args, "resume_input_receipt", None)
    receipt_sha256 = getattr(args, "resume_input_receipt_sha256", None)
    selection_index = getattr(args, "resume_selection_index", None)
    fast_receipt = None
    if any(value is not None for value in (receipt_path, receipt_sha256, selection_index)):
        if not all(value is not None for value in (receipt_path, receipt_sha256, selection_index)):
            raise ValueError("fast resume requires receipt, digest and selection index")
        fast_receipt = _validate_fast_resume_receipt(receipt_path, receipt_sha256)
        _receipt_digest(fast_receipt, selection_index)
    input_digests = ({str(p.resolve()): _receipt_digest(fast_receipt, p) for p in input_paths}
                     if fast_receipt else
                     {str(p.resolve()): sha256_file(p) for p in input_paths})
    identity = {"inputs": input_digests,
        "attempt_budget": args.attempts, "judge_model": args.judge_model,
        "selection": "all fixed original failed training cases, independent of repair outcomes"}
    task_source_selection = getattr(args, "task_source_selection", "all")
    if task_source_selection not in {"all", "longest_failed"}:
        raise ValueError("unknown PSD task source selection")
    if task_source_selection != "all":
        identity["task_source_selection"] = task_source_selection
    if getattr(args, "repair_mode", "feedback") != "feedback":
        identity["repair_mode"] = args.repair_mode
    if getattr(args, "repair_mode", "feedback") == "slate":
        # A terminal old search must not bypass the child protocol identity check.
        identity["slate_position_policy"] = "observed-decisions-not-localizer-lock-v1"
        from ifv_training.psd_slate import SLATE_REVIEW_POLICY
        identity['slate_review_policy'] = SLATE_REVIEW_POLICY
    # Keep the default identity byte-compatible with the immutable historical
    # canary; only bind fields that change its execution/materialization policy.
    if proposal_rounds != 12:
        identity["proposal_budget"] = proposal_rounds
    if search_only:
        identity["search_only"] = True
    marker = output / "inputs.json"
    if marker.exists():
        load_bound(marker, identity=identity)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be new or bound to this canary")
        save_bound(marker, identity=identity, payload={"created": True})
    if type(args.case_concurrency) is not int or not 1 <= args.case_concurrency <= 64:
        raise ValueError("case concurrency must be 1..64")
    progress = output / "progress.json"
    completed_status = load_json(progress).get("status") if progress.exists() else ""
    if completed_status == "search_complete_datums_materialized":
        # Completed banks (including the historical canary) are immutable and
        # need no live model, API key, repeated judge, or repeated repair call.
        from ifv_training.psd_preflight import verify_psd_training_input
        gate = verify_psd_training_input(datums_path=output / "datums/datums.jsonl",
            manifest_path=output / "datums/manifest.json", expected_topk=20, max_context=131072)
        if not gate["passed"]:
            raise ValueError("completed feedback bank no longer passes verification")
        return load_json(progress)
    if completed_status in {
            "search_complete_unmaterialized_throughput_probe",
            "search_complete_no_verified_repairs"}:
        # A search-only probe is also immutable and resumable without a live
        # model or provider key. Verify all bound child snapshots before reuse.
        if getattr(args, "repair_mode", "feedback") == "slate":
            from ifv_training.psd_slate_search import audit_slate_search
            for case in load_json(progress)["cases"]:
                if not audit_slate_search(Path(case["directory"]))["passed"]:
                    raise ValueError("completed slate search no longer passes verification")
        else:
            from scripts.audit_psd_feedback_run import audit_feedback_run
            if not audit_feedback_run(output)["passed"]:
                raise ValueError("completed feedback search no longer passes verification")
        return load_json(progress)
    previous_progress = load_json(progress) if progress.exists() else {}
    previous_wall_seconds = float(previous_progress.get("search_wall_seconds") or 0)
    previous_invocations = list(previous_progress.get("search_invocations") or [])
    summary = {"status": "repairing", "training_started": False, "cases": [],
               "case_concurrency": args.case_concurrency,
               "scheduling": "completion_order_within_one_frozen_checkpoint"}
    public = {r["case_id"]: r for r in load_jsonl(benchmark)}
    gold = {r["case_id"]: r for r in load_jsonl(gold_path)}
    merged_candidates, attempts = {}, []
    async def repair_case(candidate):
        key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
        inputs, directory = output / "case-inputs" / key, output / "repairs" / key
        case = candidate["case_id"]
        # Completed slate searches are immutable child packages.  Re-audit the
        # package itself, but do not rebuild the orchestrator and recursively
        # verify the (much larger) source trace on every outer resume pass.
        # The audit is synchronous filesystem work, so keep it off the event
        # loop used by unfinished provider/policy calls.
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            if manifest.get("status") in TERMINAL_SEARCH_STATUSES:
                from ifv_training.psd_slate_search import audit_slate_search
                audit = await asyncio.to_thread(audit_slate_search, directory)
                if not audit["passed"]:
                    raise ValueError("terminal PSD slate package failed its bound audit")
                return {"case_id": case, "directory": str(directory), "result": manifest}
        trace = run_dir / candidate["source"]["source_trace_path"]
        for name, data in (("candidate.json", candidate), ("gold.json", gold[case]),
                           ("public.json", {"case_id": case, "source_steps": trace_steps(load_json(trace))})):
            path = inputs / name
            if path.exists():
                if load_json(path) != data:
                    raise ValueError("canary input changed")
            else:
                write_json(path, data)
        cli = ["--trace", str(trace), "--candidate", str(inputs / "candidate.json"),
            "--audit", str(run_dir / "psd-audits" / (trace.stem + ".json")),
            "--image", str(benchmark.parent / public[case]["image_path"]),
            "--gold", str(inputs / "gold.json"), "--private-context", str(inputs / "gold.json"),
            "--public-context", str(inputs / "public.json"), "--train-cases", str(split),
            "--source-access-policy", str(policy), "--output-dir", str(directory),
            "--policy-model", profile["profile_id"], "--policy-base-url", profile["base_url"],
            "--policy-serving-profile", str(serving), "--round-start-checkpoint", profile["model_path"],
            "--round-start-checkpoint-manifest", str(checkpoint), "--hint-constructor-provider", "gemini",
            "--hint-constructor-model", args.judge_model, "--hint-constructor-wire-api", "interactions",
            "--judge-model", args.judge_model, "--repair-attempts", str(args.attempts),
            "--proposal-rounds", str(proposal_rounds),
            "--search-mode", getattr(args, "repair_mode", "feedback")]
        if (directory / "search-state.json").exists() or (directory / "run-inputs.json").exists():
            cli.append("--resume")
        result = await _run(_parser().parse_args(cli))
        return {"case_id": case, "directory": str(directory), "result": result}

    repair_sources = None
    if task_source_selection == "longest_failed":
        selection_root = output / "task-source-selection"
        if fast_receipt:
            for path in (candidates_path, selection_root / "selection.json",
                         selection_root / "selected_candidates.jsonl",
                         output / ".task-source-selection-stage/state.json"):
                _receipt_digest(fast_receipt, path)
        else:
            completed_package(output_dir=selection_root, input_files=[candidates_path],
                build=lambda destination: build_task_repair_selection_parallel(
                    candidates_path=candidates_path, run_dir=run_dir,
                    output_dir=destination, workers=min(args.case_concurrency, 16)))
        selection = load_json(selection_root / "selection.json")
        repair_sources = _iter_jsonl(selection_root / "selected_candidates.jsonl")
        selection_path = output / "task-source-selection.json"
        if selection_path.exists() and load_json(selection_path) != selection:
            raise ValueError("PSD task source selection changed")
        write_json(selection_path, selection)
        summary["task_source_selection"] = {k: v for k, v in selection.items() if k != "tasks"}
    else:
        repair_sources = _iter_jsonl(candidates_path)
    if fast_receipt:
        carried_cases, pending_cases = _resume_indexed_case_plan(
            selection_root / "selected_candidates.jsonl", selection_index, previous_progress)
    else:
        carried_cases, pending_cases = _resume_case_plan(repair_sources, previous_progress)
    summary["cases"] = carried_cases
    summary["resume_scope"] = {
        "selected": len(carried_cases) + len(pending_cases),
        "terminal_manifests_carried": len(carried_cases),
        "scheduled": len(pending_cases),
    }
    search_started = time.monotonic()
    def save_case_error(candidate, diagnostic):
        key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
        _atomic_json(output / "case-errors" / (key + ".json"),
            {"case_id": candidate["case_id"], "time": time.time(), **diagnostic})
    async def repair_planned(planned):
        return await repair_case(planned[1])

    def save_planned_error(planned, diagnostic):
        save_case_error(planned[1], diagnostic)

    async for _, planned, outcome, error in completed_cases(pending_cases,
            repair_planned, concurrency=args.case_concurrency, on_error=save_planned_error):
        index, candidate = planned
        completed_after_seconds = time.monotonic() - search_started
        summary["search_wall_seconds"] = previous_wall_seconds + completed_after_seconds
        summary["search_invocations"] = [*previous_invocations, {
            "wall_seconds": completed_after_seconds,
            "resumed": bool(previous_progress),
            "complete": False,
        }]
        if error:
            summary["cases"].append({"case_id": candidate["case_id"], "input_index": index,
                "completed_after_seconds": completed_after_seconds,
                "result": {"status": "paused_case_exception", "error_type": error}})
            _atomic_json(output / "progress.json", summary)
            continue
        summary["cases"].append({**outcome, "input_index": index,
                                 "completed_after_seconds": completed_after_seconds})
        _atomic_json(output / "progress.json", summary)
    # The original 45-row CPU probe must still be reading exactly the same bytes.
    if fast_receipt:
        # Recheck metadata after the invocation; the one-time attestation holds
        # the full content hashes and the final materialization gate rehashes.
        _validate_fast_resume_receipt(receipt_path, receipt_sha256)
        load_bound(marker, identity=identity)
    else:
        load_bound(marker, identity={**identity, "inputs": {
            name: sha256_file(Path(name)) for name in identity["inputs"]}})
    # Read compact accepted-candidate/attempt ledgers once at the stage boundary;
    # repeated outer resumes must not reopen every terminal episode tree.
    summary["cases"].sort(key=lambda row: row["input_index"])
    for case_record in summary["cases"]:
        directory_value = case_record.get("directory")
        if not directory_value:
            continue
        directory = Path(directory_value)
        candidates_file = _case_ledger(directory, "repair_candidates")
        attempts_file = _case_ledger(directory, "repair_attempts")
        if candidates_file.is_file():
            for row in load_jsonl(candidates_file):
                merged_candidates[row["candidate_id"]] = row
        if attempts_file.is_file():
            attempts.extend(load_jsonl(attempts_file))
    # Completion order is useful progress, not a nondeterministic dataset order.
    attempts.sort(key=lambda row: (row["case_id"], row["attempt_id"]))
    invocation_wall_seconds = time.monotonic() - search_started
    search_wall_seconds = previous_wall_seconds + invocation_wall_seconds
    serial_case_seconds = sum(
        float(row.get("result", {}).get("elapsed_seconds") or 0)
        for row in summary["cases"]
    )
    summary.update(accepted=sum(r.get("accepted") is True for r in attempts),
                   continuations=len(attempts), original_bank_unchanged=True,
                   search_wall_seconds=search_wall_seconds,
                   search_invocations=[*previous_invocations, {
                       "wall_seconds": invocation_wall_seconds,
                       "resumed": bool(previous_progress),
                       "complete": True,
                   }],
                   sum_case_elapsed_seconds=serial_case_seconds,
                   observed_parallel_speedup=(serial_case_seconds / search_wall_seconds
                                              if search_wall_seconds else None))
    if any(row["result"]["status"].startswith("paused_") for row in summary["cases"]):
        summary["status"] = "paused_search_requires_resume"
    elif summary["accepted"] and search_only:
        summary["status"] = "search_complete_unmaterialized_throughput_probe"
    elif summary["accepted"]:
        merge = output / "merged"
        write_jsonl(merge / "repair_candidates.jsonl.gz", [merged_candidates[k] for k in sorted(merged_candidates)])
        write_jsonl(merge / "repair_attempts.jsonl.gz", attempts)
        materialized = materialize_bank(output_dir=output,
            repair_candidates=merge / "repair_candidates.jsonl.gz",
            repair_attempts=merge / "repair_attempts.jsonl.gz",
            preservation_candidates=preservation_path, serving_profile=serving, checkpoint_manifest=checkpoint,
            score_missing_topk=getattr(args, "score_missing_topk", False),
            teacher_device=getattr(args, "teacher_device", "cpu"))
        summary["materialization"] = materialized
        summary["status"] = ("search_complete_datums_materialized" if materialized["status"] == "ready_for_trainer"
                             else materialized["status"])
    else:
        summary["status"] = "search_complete_no_verified_repairs"
    _atomic_json(output / "progress.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--proposal-rounds", type=int, default=12)
    parser.add_argument("--case-concurrency", type=int, default=1,
                        help="Bounded asynchronous case pool; keep 1 while prioritizing the main experiment")
    parser.add_argument("--search-only", action="store_true",
                        help="Measure repair search without rebuilding an already proven datum bank")
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--score-missing-topk", action="store_true")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--task-source-selection", choices=("all", "longest_failed"), default="longest_failed",
        help="Use longest_failed for the published per-task grouped repair search")
    parser.add_argument("--repair-mode", choices=("slate", "feedback"), default="slate")
    parser.add_argument("--resume-input-receipt", type=Path)
    parser.add_argument("--resume-input-receipt-sha256")
    parser.add_argument("--resume-selection-index", type=Path)
    return parser


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(_build_parser().parse_args())), ensure_ascii=False, indent=2))
