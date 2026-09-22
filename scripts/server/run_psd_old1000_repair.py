"""Resume the frozen old-1000 PSD slate search without rescanning its bank.

The first Gemini proposal is cache-only. Every case uses the original selected
row, source trace, private gold, and six-rerun/twelve-proposal budget. This
owner stops at search; teacher scoring and training are separate gates.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

try:
    import fcntl
except ImportError:  # The worker runs on Linux; permit local Windows unit tests.
    fcntl = None

from scripts.run_psd_repair_driver import _parser as repair_parser, _run as repair_run
from scripts.server.precompute_psd_slate_proposals import read_indexed_row


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production1000x4-20260918-v1"
SELECTED = RUN / "processing-lightweight-v1/task-source-selection-v1/selected_candidates.jsonl"
PRECOMPUTE = RUN / "processing-lightweight-v1/slate-precompute-v1"
INDEX = PRECOMPUTE / "selected-offset-index.json"
OUTPUT = RUN / "processing-lightweight-v1/repair-search-v1"
SNAPSHOT = ROOT / "runs/psd-production400x8-20260917-v6/snapshot"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
GOLD = ROOT / "data/psd-candidate-pool-4000-20260914-v3/train/evaluator_private/private_gold.jsonl"
TERMINAL = {"converged", "passed_without_intervention", "attempt_budget_exhausted",
            "proposal_budget_exhausted", "infrastructure_budget_exhausted"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def by_case(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in result:
                raise ValueError("duplicate case in " + str(path))
            result[case_id] = row
    return result


def selected_index() -> list[dict[str, Any]]:
    state = load(PRECOMPUTE / "state.json")
    if state.get("phase") != "complete" or state.get("completed_cases") != 775:
        raise ValueError("old-1000 Gemini slate is not complete")
    index = load(INDEX)
    stat = SELECTED.stat()
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino, "device": stat.st_dev}
    if (index.get("schema_version") != "ifv-psd-slate-precompute-index-v1"
            or index.get("selected_path") != str(SELECTED.resolve())
            or index.get("selected_stat") != identity):
        raise ValueError("frozen old-1000 selection metadata changed")
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != 775:
        raise ValueError("old-1000 selected index must contain 775 cases")
    seen = set()
    for entry in entries:
        case_id = entry.get("case_id")
        if (not isinstance(case_id, str) or case_id in seen
                or type(entry.get("offset")) is not int
                or type(entry.get("length")) is not int
                or entry["offset"] < 0 or entry["length"] <= 0
                or entry["offset"] + entry["length"] > stat.st_size):
            raise ValueError("invalid or duplicate old-1000 selected offset")
        seen.add(case_id)
        receipt = load(PRECOMPUTE / "cases" / case_id / "proposal.json")
        if receipt.get("case_id") != case_id or receipt.get("status") != "ready_for_qwen":
            raise ValueError("old-1000 Gemini proposal receipt changed")
    return entries


def serving_ready() -> None:
    profile = load(SNAPSHOT / "serving-profile.json")
    expected = Path(profile["model_path"]).resolve()
    alias = profile["profile_id"]
    for port in (19002, 19003, 19004, 19005, 19025):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
            data = json.load(response)["data"]
        served_alias = alias if port == 19025 else "ifv-psd-sft3084"
        if not any(row.get("id") == served_alias and Path(row.get("root", "")).resolve() == expected
                   for row in data):
            raise RuntimeError(f"SFT3 model identity mismatch on port {port}")
    with urllib.request.urlopen("http://127.0.0.1:19025/health", timeout=10) as response:
        health = json.load(response)
    if len(health.get("replicas") or []) != 4 or not all(
        row.get("healthy") is True for row in health["replicas"]
    ):
        raise RuntimeError("SFT3 gateway is not four-card healthy")


def no_competing_eval() -> None:
    running = subprocess.run(["ps", "-eo", "pid,args"], text=True,
                             capture_output=True, check=True).stdout
    for line in running.splitlines():
        if ("resume_corrected_agent_eval.py" in line
                or "qwen35-base-agent-full1526-selfextract-20260921-v3" in line):
            raise RuntimeError("three-weight GPU Agent owner is still running")


def case_cli(entry: dict[str, Any], benchmark: dict[str, Any], gold: dict[str, Any]) -> list[str]:
    case_id = entry["case_id"]
    candidate = read_indexed_row(SELECTED, entry)
    proposal = load(PRECOMPUTE / "cases" / case_id / "proposal.json")
    if (candidate.get("candidate_id") != proposal.get("candidate_id")
            or candidate.get("episode_id") != proposal.get("episode_id")):
        raise ValueError("Gemini proposal is not bound to the selected candidate")
    source = candidate["source"]
    trace = RUN / "episodes" / source["source_trace_path"]
    audit = Path(source["source_audit"]["path"])
    image = RUN / "selection/runtime-release/runtime_input" / benchmark[case_id]["image_path"]
    if not all(path.is_file() for path in (trace, audit, image)):
        raise FileNotFoundError("selected old-1000 source, audit, or image is missing")
    key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
    inputs = OUTPUT / "case-inputs" / key
    for name, value in (("gold.json", gold[case_id]), ("public.json", {"case_id": case_id})):
        path = inputs / name
        if path.exists():
            if load(path) != value:
                raise ValueError("frozen repair case input changed")
        else:
            save(path, value)
    directory = OUTPUT / "repairs" / key
    profile = load(SNAPSHOT / "serving-profile.json")
    args = ["--trace", str(trace), "--candidate", str(SELECTED),
            "--candidate-offset", str(entry["offset"]),
            "--candidate-length", str(entry["length"]),
            "--audit", str(audit), "--image", str(image),
            "--gold", str(inputs / "gold.json"),
            "--private-context", str(inputs / "gold.json"),
            "--public-context", str(inputs / "public.json"),
            "--output-dir", str(directory),
            "--policy-model", profile["profile_id"],
            "--policy-base-url", profile["base_url"],
            "--policy-serving-profile", str(SNAPSHOT / "serving-profile.json"),
            "--hint-constructor-provider", "gemini",
            "--hint-constructor-model", proposal["model"],
            "--hint-constructor-wire-api", "interactions",
            "--round-start-checkpoint", profile["model_path"],
            "--round-start-checkpoint-manifest", str(SNAPSHOT / "checkpoint-manifest.json"),
            "--train-cases", str(RUN / "selection/train-cases.jsonl"),
            "--source-access-policy", str(RUN / "selection/runtime-release/evaluator_private/source_access_policy.json"),
            "--judge-model", proposal["model"],
            "--search-mode", "slate", "--repair-attempts", "6",
            "--proposal-rounds", "12",
            "--precomputed-slate-cache", str(PRECOMPUTE / "cases" / case_id / "judge-cache")]
    if (directory / "run-inputs.json").exists():
        args.append("--resume")
    return args


async def cache_probe(entry: dict[str, Any]) -> dict[str, Any]:
    """Prove the precomputed first proposal is an exact cache hit, offline."""
    from ifv_training.psd_gemini_judge import review_images, trace_steps
    from ifv_training.psd_slate import decision_map, propose_slate
    from ifv_training.psd_slate_feedback import checker_feedback, diagnostic_position
    from ifv_training.psd_source_review import source_review_reference

    class NoProvider:
        async def create(self, **kwargs):
            raise AssertionError("cache probe must never call Gemini")

    candidate = read_indexed_row(SELECTED, entry)
    case_id = entry["case_id"]
    proposal = load(PRECOMPUTE / "cases" / case_id / "proposal.json")
    trace = load(RUN / "episodes" / candidate["source"]["source_trace_path"])
    review = source_review_reference(candidate["source"])
    benchmark = by_case(RUN / "selection/runtime-release/runtime_input/cases.jsonl")
    gold = by_case(GOLD)
    image = RUN / "selection/runtime-release/runtime_input" / benchmark[case_id]["image_path"]
    feedback = checker_feedback(review, trace, withhold_invalid_citations=True)
    failed_position = diagnostic_position(feedback, trace)
    images, _ = review_images({"source": trace}, image_path=image)
    public = {"source_steps": trace_steps(trace), "observed_steps": trace_steps(trace),
              "decision_map": decision_map(trace), "checker_feedback": feedback}
    hints, provenance = await propose_slate(
        NoProvider(), public_context=public, previous={}, passing_positions=[],
        failed_position=failed_position, model=proposal["model"],
        cache_dir=PRECOMPUTE / "cases" / case_id / "judge-cache",
        private_context=gold[case_id], images=images, cache_only=True)
    if ({str(position): hint.text for position, hint in hints.items()} != proposal["hints"]
            or provenance.get("response_sha256") != proposal["response_sha256"]
            or failed_position != proposal["failed_position"]):
        raise ValueError("old-1000 first proposal receipt differs from exact cached response")
    return {"case_id": case_id, "cache_hit": True, "model": proposal["model"],
            "provider_calls": 0}


async def worker(max_new_cases: int, concurrency: int) -> dict[str, Any]:
    if not 1 <= concurrency <= 16 or max_new_cases < 0:
        raise ValueError("invalid old-1000 repair worker bounds")
    if fcntl is None:
        raise RuntimeError("old-1000 repair owner requires Linux process locks")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        serving_ready()
        no_competing_eval()
        entries = selected_index()
        benchmark = by_case(RUN / "selection/runtime-release/runtime_input/cases.jsonl")
        gold = by_case(GOLD)
        pending = []
        counts = Counter()
        for entry in entries:
            receipt = OUTPUT / "case-receipts" / (entry["case_id"] + ".json")
            if receipt.is_file() and load(receipt).get("status") in TERMINAL:
                counts["terminal_carried"] += 1
            else:
                pending.append(entry)
        pending_before_limit = len(pending)
        if max_new_cases:
            pending = pending[:max_new_cases]
        save(OUTPUT / "state.json", {"phase": "repairing", "selected": 775,
            "terminal_carried": counts["terminal_carried"],
            "pending_at_start": pending_before_limit, "scheduled": len(pending),
            "concurrency": concurrency, "training_started": False})
        queue = asyncio.Queue()
        for entry in pending:
            queue.put_nowait(entry)

        async def one() -> None:
            while not queue.empty():
                try:
                    entry = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                case_id = entry["case_id"]
                try:
                    args = repair_parser().parse_args(case_cli(entry, benchmark, gold))
                    result = await repair_run(args)
                    status = result.get("status")
                    if status not in TERMINAL:
                        raise RuntimeError("repair case did not reach a terminal manifest")
                    save(OUTPUT / "case-receipts" / (case_id + ".json"), {
                        "case_id": case_id, "status": status,
                        "complete_reruns": result.get("complete_reruns"),
                        "accepted_count": result.get("accepted_count"),
                        "manifest": str(args.output_dir / "manifest.json"),
                        "completed_unix": time.time()})
                    counts[status] += 1
                except Exception as error:
                    save(OUTPUT / "case-errors" / (case_id + ".json"), {
                        "case_id": case_id, "error_type": type(error).__name__,
                        "message": str(error)[:500], "at_unix": time.time()})
                    counts["case_error"] += 1
                finally:
                    queue.task_done()
                    save(OUTPUT / "state.json", {"phase": "repairing", "selected": 775,
                        "terminal_carried": counts["terminal_carried"],
                        "pending_at_start": pending_before_limit, "scheduled": len(pending),
                        "completed_this_pass": sum(counts[s] for s in TERMINAL),
                        "case_errors_this_pass": counts["case_error"],
                        "concurrency": concurrency, "training_started": False})

        await asyncio.gather(*(one() for _ in range(concurrency)))
        remaining = pending_before_limit - sum(counts[s] for s in TERMINAL)
        result = {"phase": "repair_search_complete" if remaining == 0 else "needs_resume",
                  "selected": 775, "terminal_total": 775 - remaining,
                  "remaining": remaining, "statuses_this_pass": dict(counts),
                  "training_started": False}
        save(OUTPUT / "state.json", result)
        return result


def launch(max_new_cases: int, concurrency: int) -> dict[str, Any]:
    from dotenv import dotenv_values

    serving_ready()
    no_competing_eval()
    selected_index()
    current = OUTPUT / "latest-process.json"
    if current.is_file():
        pid = int(load(current)["pid"])
        if (Path("/proc") / str(pid) / "cmdline").exists():
            raise RuntimeError("old-1000 repair owner is already running")
    source = ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py"
    spec = importlib.util.spec_from_file_location("psd_service_owner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen service owner is unavailable")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    env = owner.checked(load(SERVICE / "replica-0.json"))
    env.update({key: value for key, value in dotenv_values(ROOT / "private/runtime.env").items()
                if value is not None})
    if not (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")):
        raise RuntimeError("old-1000 Gemini review credential is missing")
    code_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(code_root) + ":" + str(code_root / "training") + ":" + env.get("PYTHONPATH", "")
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--max-new-cases", str(max_new_cases), "--concurrency", str(concurrency)]
    launches = OUTPUT / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    launch_id = str(int(time.time()))
    receipt = owner.spawn(command, env, launches / (launch_id + ".log"))
    owner.save(launches / (launch_id + ".json"), receipt)
    save(current, {"pid": receipt["pid"], "receipt": str(launches / (launch_id + ".json")),
                   "max_new_cases": max_new_cases, "concurrency": concurrency,
                   "created_unix": time.time()})
    return {"phase": "launched", "pid": receipt["pid"],
            "max_new_cases": max_new_cases, "concurrency": concurrency}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--cache-probe", action="store_true")
    parser.add_argument("--max-new-cases", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.cache_probe:
        result = asyncio.run(cache_probe(selected_index()[0]))
    else:
        result = (launch(args.max_new_cases, args.concurrency) if args.launch
                  else asyncio.run(worker(args.max_new_cases, args.concurrency)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
