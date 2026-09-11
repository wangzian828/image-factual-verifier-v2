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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_gemini_judge import trace_steps, _atomic_json
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_repairs import assemble_psd_repair_package
from ifv_training.psd import build_psd_target_package
from ifv_training.psd_datums import build_sparse_topk_package
from ifv_training.psd_case_pool import completed_cases
from scripts.run_psd_repair_driver import _parser, _run


async def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("feedback canary must be separate from the frozen source bank")
    preparation = load_json(source / "prepared.json")["preparation"]
    benchmark, split, policy, gold_path = [Path(preparation[k]) for k in (
        "benchmark", "case_split", "source_access_policy", "private_gold")]
    serving, checkpoint = args.snapshot / "serving-profile.json", args.snapshot / "checkpoint-manifest.json"
    profile = load_json(serving)
    candidates_path = source / "candidates/repair_candidates.jsonl"
    preservation_path = source / "candidates/preservation_candidates.jsonl"
    identity = {"inputs": {str(p.resolve()): sha256_file(p) for p in (
        source / "prepared.json", candidates_path, preservation_path, benchmark, split, policy,
        gold_path, serving, checkpoint, source / "datums/datums.jsonl")},
        "attempt_budget": args.attempts, "judge_model": args.judge_model,
        "selection": "all fixed original failed training cases, independent of repair outcomes"}
    marker = output / "inputs.json"
    if marker.exists():
        load_bound(marker, identity=identity)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be new or bound to this canary")
        save_bound(marker, identity=identity, payload={"created": True})
    if type(args.case_concurrency) is not int or not 1 <= args.case_concurrency <= 64:
        raise ValueError("case concurrency must be 1..64")
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
        trace = source / "on-policy-r1" / candidate["source"]["source_trace_path"]
        for name, data in (("candidate.json", candidate), ("gold.json", gold[case]),
                           ("public.json", {"case_id": case, "source_steps": trace_steps(load_json(trace))})):
            path = inputs / name
            if path.exists():
                if load_json(path) != data:
                    raise ValueError("canary input changed")
            else:
                write_json(path, data)
        cli = ["--trace", str(trace), "--candidate", str(inputs / "candidate.json"),
            "--audit", str(source / "on-policy-r1/psd-audits" / (trace.stem + ".json")),
            "--image", str(benchmark.parent / public[case]["image_path"]),
            "--gold", str(inputs / "gold.json"), "--private-context", str(inputs / "gold.json"),
            "--public-context", str(inputs / "public.json"), "--train-cases", str(split),
            "--source-access-policy", str(policy), "--output-dir", str(directory),
            "--policy-model", profile["profile_id"], "--policy-base-url", profile["base_url"],
            "--policy-serving-profile", str(serving), "--round-start-checkpoint", profile["model_path"],
            "--round-start-checkpoint-manifest", str(checkpoint), "--hint-constructor-provider", "gemini",
            "--hint-constructor-model", args.judge_model, "--hint-constructor-wire-api", "interactions",
            "--judge-model", args.judge_model, "--repair-attempts", str(args.attempts),
            "--search-mode", "feedback", "--max-suffix-actions", "8"]
        if (directory / "search-state.json").exists():
            cli.append("--resume")
        result = await _run(_parser().parse_args(cli))
        return {"case_id": case, "directory": str(directory), "result": result}

    async for index, candidate, outcome, error in completed_cases(load_jsonl(candidates_path),
            repair_case, concurrency=args.case_concurrency):
        if error:
            summary["cases"].append({"case_id": candidate["case_id"], "input_index": index,
                "result": {"status": "paused_case_exception", "error_type": error}})
            _atomic_json(output / "progress.json", summary)
            continue
        summary["cases"].append({**outcome, "input_index": index})
        directory = Path(outcome["directory"])
        for row in load_jsonl(directory / "repair_candidates.jsonl"):
            merged_candidates[row["candidate_id"]] = row
        attempts.extend(load_jsonl(directory / "repair_attempts.jsonl"))
        _atomic_json(output / "progress.json", summary)
    # The original 45-row CPU probe must still be reading exactly the same bytes.
    load_bound(marker, identity={**identity, "inputs": {name: sha256_file(Path(name)) for name in identity["inputs"]}})
    # Completion order is useful progress, not a nondeterministic dataset order.
    attempts.sort(key=lambda row: (row["case_id"], row["attempt_id"]))
    summary.update(accepted=sum(r.get("accepted") is True for r in attempts),
                   continuations=len(attempts), original_bank_unchanged=True)
    if any(row["result"]["status"].startswith("paused_") for row in summary["cases"]):
        summary["status"] = "paused_search_requires_resume"
    elif summary["accepted"]:
        merge = output / "merged"
        write_jsonl(merge / "repair_candidates.jsonl", [merged_candidates[k] for k in sorted(merged_candidates)])
        write_jsonl(merge / "repair_attempts.jsonl", attempts)
        assembly, targets, datums = output / "assembled", output / "targets", output / "datums"
        if (datums / "manifest.json").exists():
            from ifv_training.psd_preflight import verify_psd_training_input
            summary["datums"] = verify_psd_training_input(datums_path=datums / "datums.jsonl",
                manifest_path=datums / "manifest.json", expected_topk=20, max_context=131072)
        else:
            summary["assembly"] = assemble_psd_repair_package(
                repair_candidates_path=merge / "repair_candidates.jsonl", repair_attempts_path=merge / "repair_attempts.jsonl",
                preservation_candidates_path=preservation_path, output_dir=assembly)
            summary["targets"] = build_psd_target_package(repairs_path=assembly / "repairs.jsonl",
                preservation_path=assembly / "preservation.jsonl", output_dir=targets)
            summary["datums"] = build_sparse_topk_package(targets_path=targets / "targets.jsonl",
                output_dir=datums, topk=20, max_sequence_length=131072, require_both_kinds=True)
        summary["status"] = "search_complete_datums_materialized" if (
            summary["datums"].get("status") == "ready_for_trainer" or summary["datums"].get("passed") is True
        ) else "search_complete_datum_gate_failed"
    else:
        summary["status"] = "search_complete_no_verified_repairs"
    _atomic_json(output / "progress.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--case-concurrency", type=int, default=1,
                        help="Bounded asynchronous case pool; keep 1 while prioritizing the main experiment")
    parser.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))
