import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.continue_psd_canary import completed_stage
from scripts.prepare_psd_training_canary import project_frozen_training
from src.orchestrator.source_access import benchmark_source_access_policy
from ifv_training.io import load_jsonl
from ifv_training.psd_repair_runtime import QwenContinuationAdapter


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
