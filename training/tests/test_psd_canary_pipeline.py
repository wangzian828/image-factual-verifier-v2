import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.continue_psd_canary import completed_stage
from scripts.prepare_psd_training_canary import project_frozen_training
from scripts.run_psd_feedback_canary import (
    _build_parser,
    _resume_case_plan,
    _resume_indexed_case_plan,
    _validate_fast_resume_receipt,
)
from src.orchestrator.source_access import benchmark_source_access_policy
from ifv_training.io import load_jsonl
from ifv_training.psd_repair_runtime import QwenContinuationAdapter


def test_feedback_canary_exposes_bounded_search_only_throughput_probe():
    args = _build_parser().parse_args([
        "--source", "source", "--snapshot", "snapshot", "--output", "output",
        "--attempts", "1", "--proposal-rounds", "1", "--case-concurrency", "3",
        "--search-only",
    ])
    assert args.attempts == args.proposal_rounds == 1
    assert args.case_concurrency == 3
    assert args.search_only is True


def test_resume_case_plan_schedules_only_missing_and_paused_cases(tmp_path):
    candidates = [
        {"case_id": "done", "candidate_id": "d"},
        {"case_id": "paused", "candidate_id": "p"},
        {"case_id": "missing", "candidate_id": "m"},
    ]
    done = tmp_path / "done"
    done.mkdir()
    manifest = {"status": "converged", "accepted_count": 1}
    (done / "manifest.json").write_text(json.dumps(manifest))
    previous = {"cases": [
        {"case_id": "done", "directory": str(done), "result": manifest,
         "input_index": 99},
        {"case_id": "paused", "result": {"status": "paused_case_exception"},
         "input_index": 1},
    ]}
    carried, pending = _resume_case_plan(candidates, previous)
    assert [(row["case_id"], row["input_index"]) for row in carried] == [("done", 0)]
    assert [(index, row["case_id"]) for index, row in pending] == [
        (1, "paused"), (2, "missing")]


def test_resume_case_plan_rejects_changed_terminal_manifest(tmp_path):
    directory = tmp_path / "done"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"status": "converged"}))
    previous = {"cases": [{"case_id": "done", "directory": str(directory),
                            "result": {"status": "converged", "accepted_count": 1}}]}
    with pytest.raises(ValueError, match="manifest changed"):
        _resume_case_plan([{"case_id": "done", "candidate_id": "d"}], previous)


def test_indexed_resume_reads_only_pending_rows(tmp_path):
    selected = tmp_path / "selected.jsonl"
    rows = [
        {"case_id": "done", "candidate_id": "d", "padding": "x" * 100},
        {"case_id": "pending", "candidate_id": "p", "padding": "y" * 100},
    ]
    entries, offset = [], 0
    with selected.open("wb") as stream:
        for input_index, row in enumerate(rows):
            raw = (json.dumps(row) + "\n").encode()
            stream.write(raw)
            entries.append({"case_id": row["case_id"], "input_index": input_index,
                "offset": offset, "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()})
            offset += len(raw)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema_version": "ifv-psd-selected-offset-index-v1",
        "selected_path": str(selected.resolve()), "entries": entries}))
    directory = tmp_path / "done"
    directory.mkdir()
    manifest = {"status": "converged"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    carried, pending = _resume_indexed_case_plan(selected, index, {"cases": [{
        "case_id": "done", "directory": str(directory), "result": manifest}]})
    assert [row["case_id"] for row in carried] == ["done"]
    assert [(i, row["case_id"]) for i, row in pending] == [(1, "pending")]


def test_fast_resume_receipt_rejects_metadata_change(tmp_path):
    target = tmp_path / "large.jsonl"
    target.write_bytes(b"original")
    stat = target.stat()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "ifv-psd-fast-resume-inputs-v1",
        "files": {str(target.resolve()): {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "device": stat.st_dev,
        }},
    }))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert _validate_fast_resume_receipt(receipt, digest)["files"]
    target.write_bytes(b"changed-size")
    with pytest.raises(ValueError, match="metadata changed"):
        _validate_fast_resume_receipt(receipt, digest)


def test_completed_proposer_response_cached_before_json_validation(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from src.orchestrator.llm_backend import LLMResponse
    adapter = QwenContinuationAdapter.__new__(QwenContinuationAdapter)
    calls = []
    async def respond(*args, **kwargs):
        calls.append(1)
        return LLMResponse(text="not valid JSON", prompt_tokens=3, completion_tokens=2, raw={"id": "response-id"})
    adapter.hint_constructor_llm = SimpleNamespace(model_name="test", provider="qwen_local",
        wire_api="chat_completions", get_response=respond)
    adapter.runtime_store = None
    adapter.max_output_tokens = 8192
    adapter.hint_constructor_thinking_level = "low"
    adapter.proposer_cache_path = tmp_path / "proposer-response.json"
    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "prompt"}]
    for _ in range(2):
        response = asyncio.run(adapter._call_proposer(messages))
        with pytest.raises(json.JSONDecodeError):
            json.loads(response.text)
    assert calls == [1]


def test_native_history_removes_only_the_identical_transport_system_copy():
    original = [{"role": "system", "content": "original system"}, {"role": "user", "content": "original image"}]
    assert QwenContinuationAdapter._native_history(original, "original system") == original[1:]
    assert len(original) == 2
    with pytest.raises(ValueError, match="differs"):
        QwenContinuationAdapter._native_history(original, "changed instruction")
    with pytest.raises(ValueError, match="interior"):
        QwenContinuationAdapter._native_history(original[1:] + original[:1], "original system")


def test_completed_stage_is_cached_but_mutation_rejected(tmp_path):
    artifact = tmp_path / "artifact.json"
    calls = []
    def action():
        calls.append(1)
        artifact.write_text("{}")
        return {"passed": True}, [artifact]
    assert completed_stage(tmp_path, "test", {"input": "hash"}, action)["passed"]
    assert completed_stage(tmp_path, "test", {"input": "hash"}, action)["passed"]
    assert len(calls) == 1
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        completed_stage(tmp_path, "test", {"input": "hash"}, action)


def test_projection_preserves_frozen_train_split_and_excludes_private_fields(tmp_path):
    staging = tmp_path / "source"
    staging.mkdir()
    image = staging / "image.jpg"
    image.write_bytes(b"unit-test image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    cases = ["a", "b"]
    rows = [{"case_id": case, "unified_image_path": "image.jpg", "source_url": "https://reference.invalid", "factual_status": "supported"} for case in cases]
    split = {case: {"case_id": case, "split": "train", "image_sha256": digest} for case in cases}
    result = project_frozen_training(staging=staging, output_dir=tmp_path / "release",
        rows=rows, gold=rows, split=split,
        policy=benchmark_source_access_policy(["https://reference.invalid"]))
    public = load_jsonl(Path(result["benchmark"]))
    assert len(public) == 2
    assert all(set(row) == {"case_id", "image_path", "image_sha256"} for row in public)
    assert all(row["split"] == "train" for row in load_jsonl(Path(result["case_split"])))
    assert not result["new_validation_split"]
