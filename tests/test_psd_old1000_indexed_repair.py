import argparse
import asyncio
import json
from pathlib import PurePosixPath

import pytest

from scripts.run_psd_repair_driver import _candidate_seed
import scripts.server.run_psd_old1000_repair as old1000
from ifv_training.psd_gemini_judge import _request
from ifv_training.psd_repair_storage import load_bound, save_bound


def test_candidate_slice_reads_only_selected_row(tmp_path):
    selected = tmp_path / "selected.jsonl"
    first = b'{"candidate_id":"first","source":{}}\n'
    second = b'{"candidate_id":"second","source":{}}\n'
    selected.write_bytes(first + second)
    args = argparse.Namespace(candidate=selected, candidate_offset=len(first),
                              candidate_length=len(second))
    seed, binding = _candidate_seed(args)
    assert seed["candidate_id"] == "second"
    assert binding["offset"] == len(first)
    assert binding["length"] == len(second)
    assert len(binding["sha256"]) == 64


def test_candidate_slice_rejects_truncated_or_wrong_record(tmp_path):
    selected = tmp_path / "selected.jsonl"
    selected.write_text('{"candidate_id":"first"}\n', encoding="utf-8")
    args = argparse.Namespace(candidate=selected, candidate_offset=0, candidate_length=3)
    with pytest.raises(ValueError, match="incomplete"):
        _candidate_seed(args)


def test_frozen_gemini_cache_miss_never_calls_provider(tmp_path):
    class Client:
        async def create(self, **kwargs):
            raise AssertionError("must not pay for a precomputed request again")

    with pytest.raises(FileNotFoundError, match="frozen PSD Gemini cache"):
        asyncio.run(_request(Client(), {"source_steps": []}, prompt="prompt",
                             schema={"type": "object"}, model="gemini-3.6-flash",
                             cache_dir=tmp_path, cache_only=True))


def test_old1000_case_cli_keeps_candidate_indexed_and_gemini_model(tmp_path, monkeypatch):
    run = tmp_path / "run"
    selected = tmp_path / "selected.jsonl"
    precompute = tmp_path / "slate-precompute-v1"
    output = tmp_path / "repair-search-v1"
    snapshot = tmp_path / "snapshot"
    case_id = "main-00001"
    candidate = {"case_id": case_id, "candidate_id": "candidate-1",
                 "episode_id": "episode-1", "source": {
                     "source_trace_path": "traces/episode-1.json",
                     "source_audit": {"path": str(tmp_path / "audit.json")}}}
    raw = (json.dumps(candidate) + "\n").encode()
    selected.write_bytes(raw)
    (run / "episodes/traces").mkdir(parents=True)
    (run / "episodes/traces/episode-1.json").write_text("{}")
    (tmp_path / "audit.json").write_text("{}")
    (run / "selection/runtime-release/runtime_input/assets").mkdir(parents=True)
    (run / "selection/runtime-release/runtime_input/assets/image.jpg").write_bytes(b"test")
    (precompute / "cases" / case_id / "judge-cache").mkdir(parents=True)
    (precompute / "cases" / case_id / "proposal.json").write_text(json.dumps({
        "case_id": case_id, "candidate_id": "candidate-1",
        "episode_id": "episode-1", "model": "gemini-3.7-flash"}))
    snapshot.mkdir()
    (snapshot / "serving-profile.json").write_text(json.dumps({
        "profile_id": "sft3", "base_url": "http://127.0.0.1:19025/v1",
        "model_path": "/frozen/sft3"}))
    for name, value in (("RUN", run), ("SELECTED", selected),
                        ("PRECOMPUTE", precompute), ("OUTPUT", output),
                        ("SNAPSHOT", snapshot)):
        monkeypatch.setattr(old1000, name, value)
    cli = old1000.case_cli({"case_id": case_id, "offset": 0, "length": len(raw)},
                           {case_id: {"image_path": "assets/image.jpg"}},
                           {case_id: {"case_id": case_id}})
    assert cli[cli.index("--candidate") + 1] == str(selected)
    assert cli[cli.index("--candidate-length") + 1] == str(len(raw))
    assert cli[cli.index("--hint-constructor-model") + 1] == "gemini-3.7-flash"
    assert cli[cli.index("--precomputed-slate-cache") + 1].endswith("judge-cache")
    assert "--resume" not in cli


def test_detached_worker_keeps_immutable_launch_base_code():
    value = old1000.worker_pythonpath(PurePosixPath("/overlay"), "/frozen/base:/frozen/base/training",
                                       "/old/service")
    assert value.split(":") == ["/overlay", "/overlay/training",
                                 "/frozen/base", "/frozen/base/training", "/old/service"]


def test_main_worker_allows_requested_28_way_concurrency():
    assert old1000.MAX_WORKER_CONCURRENCY == 28


def test_judge_only_recheck_does_not_block_old1000_handoff(monkeypatch):
    class Result:
        stdout = "\\n".join([
            "2177924 python judge-gemini37-rerun-20260922-v1/launch-rerun.py",
            "2178000 python run_cases --output-dir qwen35-base-agent-full1526-selfextract-20260921-v3",
        ])

    monkeypatch.setattr(old1000.subprocess, "run", lambda *args, **kwargs: Result())
    old1000.no_competing_eval()


def test_parallel_handoff_excludes_exact_smoke_cases_after_some_finish(tmp_path, monkeypatch):
    output = tmp_path / "repair-search-v1"
    (output / "case-receipts").mkdir(parents=True)
    (output / "latest-process.json").write_text(json.dumps({
        "pid": 42, "max_new_cases": 16, "concurrency": 16,
        "created_unix": 1000.0}))
    entries = [{"case_id": f"main-{index:05d}"} for index in range(20)]
    for index in (0, 2):
        (output / "case-receipts" / f"main-{index:05d}.json").write_text(json.dumps({
            "status": "converged", "completed_unix": 900.0}))
    for index in (1, 3):
        (output / "case-receipts" / f"main-{index:05d}.json").write_text(json.dumps({
            "status": "infrastructure_budget_exhausted", "completed_unix": 1100.0}))
    monkeypatch.setattr(old1000, "OUTPUT", output)
    monkeypatch.setattr(old1000, "selected_index", lambda: entries)
    expected = [row["case_id"] for row in entries if row["case_id"] not in {
        "main-00000", "main-00002"}][:16]
    assert old1000.parallel_exclusions() == expected
    # Finishing another in-flight smoke case cannot shift the frozen boundary.
    (output / "case-receipts" / "main-00004.json").write_text(json.dumps({
        "status": "converged", "completed_unix": 1200.0}))
    assert old1000.parallel_exclusions() == expected
    pending, carried = old1000.select_pending(entries, set(expected))
    assert carried == 2
    assert [row["case_id"] for row in pending] == ["main-00018", "main-00019"]


def test_cache_mode_requires_fail_closed_case_isolation():
    healthy = {"prefix_cache_policy": "case_isolated",
               "reject_corrupted_responses": True,
               "cache_corruption_metric_failures": 0}
    assert old1000.gateway_cache_mode(healthy) == "case_isolated"
    with pytest.raises(RuntimeError, match="fail-closed"):
        old1000.gateway_cache_mode({**healthy, "reject_corrupted_responses": False})


def test_reconcile_only_exact_zero_output_gateway_rejection(tmp_path, monkeypatch):
    from hashlib import sha256

    candidate = {"candidate_id": "candidate-1"}
    key = sha256(b"candidate-1").hexdigest()[:16]
    output = tmp_path / "repair-search-v1"
    retry = output / "repairs" / key / "slate-rounds/00/infrastructure-attempts"
    attempt = retry / "attempt-001"
    events = attempt / "runtime/main-02731/slate-00/events.jsonl"
    events.parent.mkdir(parents=True)
    identity = {"version": "ifv-psd-infrastructure-retry-v2", "inputs": {}, "max_attempts": 3}
    marker = retry / "retry-state.json"
    save_bound(marker, identity=identity, payload={"attempts": [{"index": 1,
        "directory": str(attempt), "status": "nonretryable_error", "error_type": "RuntimeError"}]})
    event = {"event_type": "context_request_completed", "payload": {
        "status": "error", "error": "RuntimeError: " + old1000.PREFLIGHT_REJECTION,
        "provider_output_tokens": None}}
    events.write_text("\n".join(json.dumps(item) for item in (
        {"event_type": "case_attempt_started"},
        {"event_type": "context_request_started", "payload": {"provider_output_tokens": None}},
        event)) + "\n", encoding="utf-8")
    monkeypatch.setattr(old1000, "OUTPUT", output)
    monkeypatch.setattr(old1000, "selected_index", lambda: [{"case_id": "main-02731"}])
    monkeypatch.setattr(old1000, "read_indexed_row", lambda *args: candidate)
    result = old1000.reconcile_preflight_rejection()
    assert result["status"] == "reconciled"
    assert result["attempts_charged"] == 1
    assert load_bound(marker, identity=identity)["attempts"][0]["status"] == "infrastructure_failed"
    assert old1000.reconcile_preflight_rejection()["status"] == "already_reconciled"


def test_reconcile_rejects_any_generated_output(tmp_path, monkeypatch):
    from hashlib import sha256

    candidate = {"candidate_id": "candidate-1"}
    key = sha256(b"candidate-1").hexdigest()[:16]
    output = tmp_path / "repair-search-v1"
    retry = output / "repairs" / key / "slate-rounds/00/infrastructure-attempts"
    attempt = retry / "attempt-001"
    events = attempt / "runtime/main-02731/slate-00/events.jsonl"
    events.parent.mkdir(parents=True)
    marker = retry / "retry-state.json"
    identity = {"version": "ifv-psd-infrastructure-retry-v2", "inputs": {}, "max_attempts": 3}
    save_bound(marker, identity=identity, payload={"attempts": [{"index": 1,
        "directory": str(attempt), "status": "nonretryable_error", "error_type": "RuntimeError"}]})
    event = {"event_type": "context_request_completed", "payload": {
        "status": "error", "error": "RuntimeError: " + old1000.PREFLIGHT_REJECTION,
        "provider_output_tokens": 1}}
    events.write_text("\n".join(json.dumps(item) for item in (
        {"event_type": "case_attempt_started"},
        {"event_type": "context_request_started", "payload": {"provider_output_tokens": None}},
        event)) + "\n", encoding="utf-8")
    monkeypatch.setattr(old1000, "OUTPUT", output)
    monkeypatch.setattr(old1000, "selected_index", lambda: [{"case_id": "main-02731"}])
    monkeypatch.setattr(old1000, "read_indexed_row", lambda *args: candidate)
    with pytest.raises(ValueError, match="zero model output"):
        old1000.reconcile_preflight_rejection()
    assert load_bound(marker, identity=identity)["attempts"][0]["status"] == "nonretryable_error"


def test_reconcile_unusable_choices_keeps_existing_attempt_counts(tmp_path, monkeypatch):
    from hashlib import sha256

    output = tmp_path / "repair-search-v1"
    identity = {"version": "ifv-psd-infrastructure-retry-v2", "inputs": {}, "max_attempts": 3}
    entries = [{"case_id": "main-02731"}, {"case_id": "main-02735"}]
    candidates = {entry["case_id"]: {"candidate_id": entry["case_id"]} for entry in entries}
    monkeypatch.setattr(old1000, "OUTPUT", output)
    monkeypatch.setattr(old1000, "selected_index", lambda: entries)
    monkeypatch.setattr(old1000, "read_indexed_row", lambda _path, entry: candidates[entry["case_id"]])
    for case_id, index in (("main-02731", 3), ("main-02735", 1)):
        key = sha256(case_id.encode()).hexdigest()[:16]
        retry = output / "repairs" / key / "slate-rounds/00/infrastructure-attempts"
        attempts = []
        for number in range(1, index + 1):
            directory = retry / f"attempt-{number:03d}"
            attempts.append({"index": number, "directory": str(directory),
                             "status": "infrastructure_failed" if number < index else "nonretryable_error",
                             "error_type": "RuntimeError"})
        save_bound(retry / "retry-state.json", identity=identity, payload={"attempts": attempts})
        event_path = (retry / f"attempt-{index:03d}" / "runtime" / case_id /
                      "slate-00/events.jsonl")
        event_path.parent.mkdir(parents=True)
        event_path.write_text(json.dumps({"event_type": "context_request_completed", "payload": {
            "status": "error", "error": "RuntimeError: Chat Completions returned an unusable response: "
            "choices=1, finish_reason=length, content_chars=0, reasoning_chars=514, "
            "reasoning_fallback_requested=False"}}) + "\n", encoding="utf-8")
    assert [item["attempts_charged"] for item in old1000.reconcile_unusable_completions()["cases"]] == [3, 1]
    assert [item["status"] for item in old1000.reconcile_unusable_completions()["cases"]] == [
        "already_reconciled", "already_reconciled"]
