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
from ifv_training.psd_repair_storage import load_bound, save_bound
from ifv_training.psd_infrastructure_retry import (
    InfrastructureRetriesExhausted, unusable_chat_response_reason)


ROOT = Path("/volume/ybo/wza")
RUN = ROOT / "runs/psd-production1000x4-20260918-v1"
SELECTED = RUN / "processing-lightweight-v1/task-source-selection-v1/selected_candidates.jsonl"
PRECOMPUTE = RUN / "processing-lightweight-v1/slate-precompute-v1"
INDEX = PRECOMPUTE / "selected-offset-index.json"
DEFAULT_OUTPUT = RUN / "processing-lightweight-v1/repair-search-v1"
OUTPUT = Path(os.environ.get("IFV_OLD1000_REPAIR_OUTPUT", str(DEFAULT_OUTPUT)))
SNAPSHOT = ROOT / "runs/psd-production400x8-20260917-v6/snapshot"
SERVICE = ROOT / "inference/psd-sft3084-20260916"
GOLD = ROOT / "data/psd-candidate-pool-4000-20260914-v3/train/evaluator_private/private_gold.jsonl"
TERMINAL = {"converged", "passed_without_intervention", "attempt_budget_exhausted",
            "proposal_budget_exhausted", "infrastructure_budget_exhausted"}
MAX_WORKER_CONCURRENCY = 28
PREFLIGHT_REJECTION = ("HTTP 400 Bad Request for http://127.0.0.1:19025/v1/chat/completions: "
                       '{"detail":"case-isolated prefix caching requires an opaque 256-bit '
                       'ifv-case-v1- cache_salt"}')


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


def gateway_cache_mode(health: dict[str, Any]) -> str:
    mode = health.get("prefix_cache_policy")
    if mode not in {"case_isolated", "request_isolated"}:
        raise RuntimeError("SFT3 gateway has an unknown prefix-cache protocol")
    if mode == "case_isolated" and (
        health.get("reject_corrupted_responses") is not True
        or health.get("cache_corruption_metric_failures") != 0
    ):
        raise RuntimeError("case-isolated Qwen gateway lacks fail-closed cache admission")
    return mode


def serving_ready() -> str:
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
    return gateway_cache_mode(health)


def no_competing_eval() -> None:
    running = subprocess.run(["ps", "-eo", "pid,args"], text=True,
                             capture_output=True, check=True).stdout
    for line in running.splitlines():
        judge_only = (
            "stream_agent_judges.py" in line
            or "judge-gemini37" in line
        )
        if (("resume_corrected_agent_eval.py" in line
             or "qwen35-base-agent-full1526-selfextract-20260921-v3" in line)
                and not judge_only):
            raise RuntimeError("three-weight GPU Agent owner is still running")


def reconcile_preflight_rejection() -> dict[str, Any]:
    """Classify one proven gateway pre-admission rejection; keep its spent attempt.

    This is intentionally not a generic 400 recovery. The runtime event must
    prove that the first model request was rejected before token generation.
    """
    current = OUTPUT / "latest-process.json"
    if current.exists():
        pid = int(load(current)["pid"])
        if (Path("/proc") / str(pid) / "cmdline").exists():
            raise RuntimeError("cannot reconcile while old-1000 repair owner is active")
    entries = selected_index()
    entry = entries[0]
    if entry["case_id"] != "main-02731":
        raise ValueError("unexpected old-1000 preflight case")
    candidate = read_indexed_row(SELECTED, entry)
    key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
    retry = OUTPUT / "repairs" / key / "slate-rounds/00/infrastructure-attempts"
    marker = retry / "retry-state.json"
    if not marker.is_file():
        raise FileNotFoundError("old-1000 preflight retry ledger missing")
    raw = load(marker)
    identity = raw["identity"]
    state = load_bound(marker, identity=identity)
    attempts = state["attempts"]
    if len(attempts) != 1 or attempts[0].get("index") != 1:
        raise ValueError("preflight attempt history differs from audited one")
    row = attempts[0]
    if (row.get("status") == "infrastructure_failed"
            and row.get("migration_version") == "ifv-old1000-gateway-preflight-v1"):
        return {"case_id": entry["case_id"], "status": "already_reconciled",
                "attempts_charged": 1}
    if row.get("status") != "nonretryable_error" or row.get("error_type") != "RuntimeError":
        raise ValueError("preflight ledger is not the inspected RuntimeError")
    attempt_dir = retry / "attempt-001"
    if Path(row["directory"]).resolve() != attempt_dir.resolve():
        raise ValueError("preflight attempt path differs from ledger")
    if (retry / "result.json").exists() or any(attempt_dir.rglob("result.json")):
        raise RuntimeError("preflight may have completed a Qwen response")
    events = list(attempt_dir.glob("runtime/*/*/events.jsonl"))
    if len(events) != 1:
        raise ValueError("expected exactly one preflight runtime event stream")
    rows = [json.loads(line) for line in events[0].read_text(encoding="utf-8").splitlines()]
    completed = [item for item in rows if item.get("event_type") == "context_request_completed"]
    if ([item.get("event_type") for item in rows] != [
            "case_attempt_started", "context_request_started", "context_request_completed"]
            or len(completed) != 1 or completed[0].get("payload", {}).get("status") != "error"
            or completed[0]["payload"].get("error") != "RuntimeError: " + PREFLIGHT_REJECTION
            or completed[0]["payload"].get("provider_output_tokens") is not None
            or rows[1].get("payload", {}).get("provider_output_tokens") is not None):
        raise ValueError("preflight runtime does not prove zero model output")
    backup = OUTPUT / "recoveries/gateway-cache-contract-preflight-v1/ledger-original.json"
    original = marker.read_bytes()
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("preflight backup differs from current ledger")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
    row.update(status="infrastructure_failed", reason="gateway rejected missing cache salt before model admission",
               legacy_status="nonretryable_error", migration_version="ifv-old1000-gateway-preflight-v1",
               evidence_events=str(events[0]))
    save_bound(marker, identity=identity, payload=state)
    receipt = {"case_id": entry["case_id"], "status": "reconciled", "attempts_charged": 1,
               "ledger": str(marker), "original_sha256": hashlib.sha256(original).hexdigest(),
               "backup": str(backup), "evidence_events": str(events[0]), "time": time.time()}
    save(backup.parent / "receipt.json", receipt)
    return receipt


def reconcile_unusable_completions() -> dict[str, Any]:
    """Reclassify only two audited empty backend choices, charging both attempts."""
    current = OUTPUT / "latest-process.json"
    if current.exists():
        pid = int(load(current)["pid"])
        if (Path("/proc") / str(pid) / "cmdline").exists():
            raise RuntimeError("cannot reconcile while old-1000 repair owner is active")
    targets = {"main-02731": 3, "main-02735": 1}
    entries = {entry["case_id"]: entry for entry in selected_index()}
    reported = []
    for case_id, index in targets.items():
        candidate = read_indexed_row(SELECTED, entries[case_id])
        key = hashlib.sha256(candidate["candidate_id"].encode()).hexdigest()[:16]
        retry = OUTPUT / "repairs" / key / "slate-rounds/00/infrastructure-attempts"
        marker = retry / "retry-state.json"
        raw = load(marker)
        identity = raw["identity"]
        state = load_bound(marker, identity=identity)
        attempts = state["attempts"]
        if len(attempts) != index or attempts[-1].get("index") != index:
            raise ValueError("unusable-response attempt history changed for " + case_id)
        row = attempts[-1]
        if (row.get("status") == "infrastructure_failed"
                and row.get("migration_version") == "ifv-old1000-unusable-choice-v1"):
            reported.append({"case_id": case_id, "status": "already_reconciled"})
            continue
        if row.get("status") != "nonretryable_error" or row.get("error_type") != "RuntimeError":
            raise ValueError("unusable-response ledger is not the inspected RuntimeError")
        attempt_dir = retry / f"attempt-{index:03d}"
        if Path(row["directory"]).resolve() != attempt_dir.resolve():
            raise ValueError("unusable-response attempt directory changed")
        if (retry / "result.json").exists() or any(attempt_dir.rglob("result.json")):
            raise RuntimeError("unusable-response attempt has a completed result")
        events = list(attempt_dir.glob("runtime/*/*/events.jsonl"))
        if len(events) != 1:
            raise ValueError("unusable-response runtime stream is not unique")
        runtime_rows = [json.loads(line) for line in events[0].read_text(encoding="utf-8").splitlines()]
        last = runtime_rows[-1]
        error_text = str(last.get("payload", {}).get("error") or "")
        if (last.get("event_type") != "context_request_completed"
                or last.get("payload", {}).get("status") != "error"
                or not error_text.startswith("RuntimeError: ")
                or unusable_chat_response_reason(RuntimeError(error_text.removeprefix("RuntimeError: ")))
                   != "model_unusable_http_choice"):
            raise ValueError("unusable-response runtime does not prove backend empty choice")
        backup = (OUTPUT / "recoveries/unusable-backend-choice-v1" / case_id /
                  "ledger-original.json")
        original = marker.read_bytes()
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError("unusable-response ledger backup changed")
        else:
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original)
        row.update(status="infrastructure_failed", reason="model_unusable_http_choice",
                   legacy_status="nonretryable_error",
                   migration_version="ifv-old1000-unusable-choice-v1",
                   evidence_events=str(events[0]))
        save_bound(marker, identity=identity, payload=state)
        receipt = {"case_id": case_id, "status": "reconciled", "attempts_charged": index,
                   "ledger": str(marker), "original_sha256": hashlib.sha256(original).hexdigest(),
                   "backup": str(backup), "evidence_events": str(events[0]), "time": time.time()}
        save(backup.parent / "receipt.json", receipt)
        reported.append(receipt)
    return {"schema_version": "ifv-old1000-unusable-choice-v1", "cases": reported,
            "budget_reset": False}


def worker_pythonpath(code_root: Path, launcher: str, replica: str) -> str:
    return ":".join(part for part in (
        str(code_root), str(code_root / "training"), launcher, replica) if part)


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


def parallel_exclusions() -> list[str]:
    """Freeze the smoke's original 16 cases, including ones now terminal."""
    smoke = load(OUTPUT / "latest-process.json")
    if smoke.get("max_new_cases") != 16 or smoke.get("concurrency") != 16:
        raise ValueError("parallel handoff requires the exact frozen smoke")
    created = smoke.get("created_unix")
    if not isinstance(created, (int, float)):
        raise ValueError("smoke start time is unavailable")
    excluded = []
    for entry in selected_index():
        case_id = entry["case_id"]
        receipt = OUTPUT / "case-receipts" / (case_id + ".json")
        if receipt.is_file():
            row = load(receipt)
            if row.get("status") in TERMINAL and row.get("completed_unix", 0) < created:
                continue
        excluded.append(case_id)
        if len(excluded) == 16:
            break
    if len(excluded) != 16:
        raise ValueError("smoke case set cannot be reconstructed")
    return excluded


def select_pending(entries: list[dict[str, Any]], excluded: set[str]) -> tuple[list[dict[str, Any]], int]:
    pending = []
    terminal_carried = 0
    for entry in entries:
        if entry["case_id"] in excluded:
            continue
        receipt = OUTPUT / "case-receipts" / (entry["case_id"] + ".json")
        if receipt.is_file() and load(receipt).get("status") in TERMINAL:
            terminal_carried += 1
        else:
            pending.append(entry)
    return pending, terminal_carried


async def worker(max_new_cases: int, concurrency: int, *, parallel: bool = False) -> dict[str, Any]:
    if not 1 <= concurrency <= MAX_WORKER_CONCURRENCY or max_new_cases < 0:
        raise ValueError("invalid old-1000 repair worker bounds")
    if fcntl is None:
        raise RuntimeError("old-1000 repair owner requires Linux process locks")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ("owner-parallel.lock" if parallel else "owner.lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        serving_ready()
        no_competing_eval()
        entries = selected_index()
        excluded = set(load(OUTPUT / "parallel-binding.json")["excluded_case_ids"]) if parallel else set()
        if parallel and (len(excluded) != 16 or excluded != set(parallel_exclusions())):
            raise ValueError("parallel smoke exclusion binding changed")
        state_path = OUTPUT / ("parallel-state.json" if parallel else "state.json")
        benchmark = by_case(RUN / "selection/runtime-release/runtime_input/cases.jsonl")
        gold = by_case(GOLD)
        pending, terminal_carried = select_pending(entries, excluded)
        counts = Counter()
        counts["terminal_carried"] = terminal_carried
        pending_before_limit = len(pending)
        if max_new_cases:
            pending = pending[:max_new_cases]
        save(state_path, {"phase": "repairing", "selected": 775,
            "terminal_carried": counts["terminal_carried"],
            "pending_at_start": pending_before_limit, "scheduled": len(pending),
            "concurrency": concurrency, "excluded_smoke_cases": len(excluded),
            "training_started": False})
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
                except InfrastructureRetriesExhausted as error:
                    save(OUTPUT / "case-receipts" / (case_id + ".json"), {
                        "case_id": case_id, "status": "infrastructure_budget_exhausted",
                        "reason": str(error),
                        "manifest": None, "completed_unix": time.time()})
                    counts["infrastructure_budget_exhausted"] += 1
                except Exception as error:
                    error_receipt = {
                        "case_id": case_id, "error_type": type(error).__name__,
                        "message": str(error)[:500], "at_unix": time.time()}
                    save(OUTPUT / "case-error-history" / case_id /
                         (str(time.time_ns()) + ".json"), error_receipt)
                    save(OUTPUT / "case-errors" / (case_id + ".json"), error_receipt)
                    counts["case_error"] += 1
                finally:
                    queue.task_done()
                    save(state_path, {"phase": "repairing", "selected": 775,
                        "terminal_carried": counts["terminal_carried"],
                        "pending_at_start": pending_before_limit, "scheduled": len(pending),
                        "completed_this_pass": sum(counts[s] for s in TERMINAL),
                        "case_errors_this_pass": counts["case_error"],
                        "concurrency": concurrency, "excluded_smoke_cases": len(excluded),
                        "training_started": False})

        await asyncio.gather(*(one() for _ in range(concurrency)))
        remaining = pending_before_limit - sum(counts[s] for s in TERMINAL)
        result = {"phase": "repair_search_complete" if remaining == 0 else "needs_resume",
                  "selected": 775, "terminal_total_outside_smoke": 775 - len(excluded) - remaining,
                  "remaining": remaining, "statuses_this_pass": dict(counts),
                  "excluded_smoke_cases": len(excluded), "training_started": False}
        save(state_path, result)
        return result


def launch_parallel(concurrency: int) -> dict[str, Any]:
    from dotenv import dotenv_values

    if concurrency != 16:
        raise ValueError("parallel handoff is frozen at 16 cross-case workers")
    mode = serving_ready()
    no_competing_eval()
    previous = load(OUTPUT / "latest-process.json")
    pid = int(previous["pid"])
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists() or b"run_psd_old1000_repair.py" not in cmdline.read_bytes():
        raise RuntimeError("frozen smoke owner is not running")
    current = OUTPUT / "parallel-process.json"
    if current.exists() and Path(f"/proc/{int(load(current)['pid'])}/cmdline").exists():
        raise RuntimeError("parallel repair owner already running")
    excluded = parallel_exclusions()
    binding = OUTPUT / "parallel-binding.json"
    expected = {"schema_version": "ifv-old1000-disjoint-smoke-v1",
                "smoke_pid": pid, "smoke_created_unix": previous["created_unix"],
                "excluded_case_ids": excluded, "max_new_cases": 0, "concurrency": concurrency}
    if binding.exists():
        if load(binding) != expected:
            raise ValueError("parallel repair binding differs from frozen smoke")
    else:
        save(binding, expected)
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
    env["IFV_PREFIX_CACHE_MODE"] = mode
    env["IFV_OLD1000_REPAIR_OUTPUT"] = str(OUTPUT)
    env["IFV_PSD_GEMINI_REQUEST_RETRIES"] = os.environ.get(
        "IFV_PSD_GEMINI_REQUEST_RETRIES", "3")
    env["GEMINI_MAX_INFLIGHT_REQUESTS"] = os.environ.get(
        "IFV_OLD1000_GEMINI_MAX_INFLIGHT", str(concurrency))
    code_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = worker_pythonpath(
        code_root, os.environ.get("PYTHONPATH", ""), env.get("PYTHONPATH", ""))
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--parallel-remainder", "--max-new-cases", "0",
               "--concurrency", str(concurrency)]
    launches = OUTPUT / "launches-parallel"
    launches.mkdir(parents=True, exist_ok=True)
    launch_id = str(int(time.time()))
    receipt = owner.spawn(command, env, launches / (launch_id + ".log"))
    owner.save(launches / (launch_id + ".json"), receipt)
    save(current, {"pid": receipt["pid"], "receipt": str(launches / (launch_id + ".json")),
                   "binding": str(binding), "created_unix": time.time()})
    return {"phase": "launched_parallel", "pid": receipt["pid"],
            "excluded_smoke_cases": 16, "concurrency": concurrency}


def launch(max_new_cases: int, concurrency: int) -> dict[str, Any]:
    from dotenv import dotenv_values

    cache_mode = serving_ready()
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
    env["IFV_PREFIX_CACHE_MODE"] = cache_mode
    env["IFV_OLD1000_REPAIR_OUTPUT"] = str(OUTPUT)
    env["IFV_PSD_GEMINI_REQUEST_RETRIES"] = os.environ.get(
        "IFV_PSD_GEMINI_REQUEST_RETRIES", "3")
    env["GEMINI_MAX_INFLIGHT_REQUESTS"] = os.environ.get(
        "IFV_OLD1000_GEMINI_MAX_INFLIGHT", str(concurrency))
    code_root = Path(__file__).resolve().parents[2]
    # The serving replica predates this overlay and does not know the frozen
    # base code snapshot. Keep the launcher's explicit package path as well as
    # the checked replica environment; never silently fall back to a mutable
    # checkout when the detached worker imports unchanged modules.
    env["PYTHONPATH"] = worker_pythonpath(
        code_root, os.environ.get("PYTHONPATH", ""), env.get("PYTHONPATH", ""))
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
    parser.add_argument("--parallel-remainder", action="store_true")
    parser.add_argument("--cache-probe", action="store_true")
    parser.add_argument("--reconcile-preflight-rejection", action="store_true")
    parser.add_argument("--reconcile-unusable-completions", action="store_true")
    parser.add_argument("--max-new-cases", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.reconcile_unusable_completions:
        result = reconcile_unusable_completions()
    elif args.reconcile_preflight_rejection:
        result = reconcile_preflight_rejection()
    elif args.cache_probe:
        result = asyncio.run(cache_probe(selected_index()[0]))
    else:
        result = (launch_parallel(args.concurrency) if args.launch and args.parallel_remainder
                  else launch(args.max_new_cases, args.concurrency) if args.launch
                  else asyncio.run(worker(args.max_new_cases, args.concurrency,
                                          parallel=args.parallel_remainder)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
