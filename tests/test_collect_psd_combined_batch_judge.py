from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.server import collect_psd_combined_batch_judge as collect


GOOD = {
    "quality_bucket": "usable",
    "fact_alignment": "compatible_subfact",
    "reason_quality": "partially_grounded",
    "failure_modes": [],
    "explanation": "The visual evidence partly supports the verdict.",
}


def expected(case_id="case-1", root: Path | None = None):
    if root is not None:
        trace = root / "trace.json"
        image = root / "image.jpg"
        trace.write_text("{}", encoding="utf-8")
        image.write_bytes(b"image")
        identities = {"trace": collect.stat_identity(trace), "image": collect.stat_identity(image)}
    else:
        identities = {"trace": {"path": "/root/trace.json", "size": 10, "mtime_ns": 1},
                      "image": {"path": "/root/image.jpg", "size": 20, "mtime_ns": 2}}
    return {"case_id": case_id, "source_verdict": "fake", "gold_verdict": "fake",
            **identities}


def succeeded(case_id="case-1", *, image_tokens=10, output=None):
    return {"metadata": {"state": "BATCH_STATE_SUCCEEDED",
                         "batchStats": {"requestCount": "1", "successfulRequestCount": "1"}},
            "response": {"inlinedResponses": {"inlinedResponses": [{
                "metadata": {"case_id": case_id, "key": case_id},
                "response": {"modelVersion": "gemini-3.7-flash",
                             "usageMetadata": {"promptTokensDetails": [
                                 {"modality": "IMAGE", "tokenCount": image_tokens}]},
                             "candidates": [{"finishReason": "STOP", "content": {
                                 "parts": [{"text": json.dumps(output or GOOD)}]}}]}}
            ]}}}


def test_accepts_exact_model_image_and_schema() -> None:
    result = collect.validate_response(succeeded(), [expected()], "batches/example")
    assert len(result) == 1
    assert result[0]["status"] == "completed"
    assert result[0]["image_prompt_tokens"] == 10
    assert result[0]["verdict_matches_gold"] is True
    assert result[0]["quality_bucket"] == "usable"


def test_structured_verdict_excludes_separate_thought_part() -> None:
    payload = succeeded()
    parts = payload["response"]["inlinedResponses"]["inlinedResponses"][0]["response"]["candidates"][0]["content"]["parts"]
    parts.insert(0, {"thought": True, "text": "Thinking through the evidence; this is not JSON."})
    result = collect.validate_response(payload, [expected()], "batches/example")
    assert result[0]["status"] == "completed"
    assert result[0]["judge_output"] == GOOD


@pytest.mark.parametrize("value,reason", [
    (succeeded(image_tokens=0), "no image tokens"),
    (succeeded(output={**GOOD, "quality_bucket": "unknown"}), "invalid quality_bucket"),
    (succeeded(output={**GOOD, "extra": "x"}), "keys differ"),
])
def test_invalid_item_never_becomes_completed(value, reason) -> None:
    item = collect.validate_response(value, [expected()], "batches/example")[0]
    assert item["status"] == "error"
    assert reason in item["error"]


def test_rejects_duplicate_or_mismatched_case_mapping() -> None:
    payload = succeeded()
    item = payload["response"]["inlinedResponses"]["inlinedResponses"][0]
    payload["response"]["inlinedResponses"]["inlinedResponses"].append(item)
    with pytest.raises(ValueError, match="count differs"):
        collect.validate_response(payload, [expected()], "batches/example")
    item["metadata"]["key"] = "wrong"
    payload["response"]["inlinedResponses"]["inlinedResponses"].pop()
    with pytest.raises(ValueError, match="mismatched case key"):
        collect.validate_response(payload, [expected()], "batches/example")


def test_collector_is_restartable_and_never_recreates_batch(tmp_path: Path, monkeypatch) -> None:
    from src.eval.private_gold_judge_contract import (
        PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA,
    )
    plan = {"transport": "generateContent Batch (separate from Interactions)",
            "model": "gemini-3.7-flash",
            "prompt_sha256": hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest(),
            "response_schema_sha256": hashlib.sha256(json.dumps(
                PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest(),
            "shards": [[0]], "prepared": [expected(root=tmp_path)]}
    collect.save(tmp_path / "plan.json", plan)
    shard = tmp_path / "shards/shard-001"
    collect.save(shard / "create-response.json", {
        "http_status": 200, "job_name": "batches/example"})
    credentials = tmp_path / "runtime.env"
    credentials.write_text("GEMINI_API_KEY=fake\n", encoding="utf-8")
    calls = []
    payload = succeeded()
    monkeypatch.setattr(collect, "get_batch", lambda name, key: calls.append(name) or payload)
    first = collect.collect(tmp_path, credentials)
    second = collect.collect(tmp_path, credentials)
    assert first["counts"] == second["counts"] == {"completed": 1, "error": 0}
    assert calls == ["batches/example"]
    assert (shard / "batch-response.json").is_file()
    assert len(json.loads((shard / "records-v2.json").read_text())) == 1
    assert (tmp_path / "collection-status-v2.json").is_file()
