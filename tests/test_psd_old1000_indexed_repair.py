import argparse
import asyncio
import json
from pathlib import PurePosixPath

import pytest

from scripts.run_psd_repair_driver import _candidate_seed
import scripts.server.run_psd_old1000_repair as old1000
from ifv_training.psd_gemini_judge import _request


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


def test_cache_mode_requires_fail_closed_case_isolation():
    healthy = {"prefix_cache_policy": "case_isolated",
               "reject_corrupted_responses": True,
               "cache_corruption_metric_failures": 0}
    assert old1000.gateway_cache_mode(healthy) == "case_isolated"
    with pytest.raises(RuntimeError, match="fail-closed"):
        old1000.gateway_cache_mode({**healthy, "reject_corrupted_responses": False})
