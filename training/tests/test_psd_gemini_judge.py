from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest

from ifv_training import psd_gemini_judge as judge
from ifv_training.io import write_json
from ifv_training.psd_repair import _sha


class Client:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "interaction-test", "status": "completed",
                "outputs": [{"type": "text", "text": json.dumps(self.result)}]}


def packet():
    return {"selected_step_index": 1, "episode_complete": True, "hint": "Compare fields",
            "source_steps": [{"index": 0, "observation": "record says 2019"},
                             {"index": 1, "action": "assert 2018"}],
            "repaired_steps": [{"index": 0, "observation": "record says 2019"},
                               {"index": 1, "action": "compare 2018 against 2019"}]}


def review():
    return {**{key: True for key in judge.CHECKS}, "earliest_error_step": 1,
            "explanation": "The selected decision now compares the contradictory fields.",
            "evidence": [{"trace": "source", "step_index": 1, "quote": "assert 2018"},
                         {"trace": "repaired", "step_index": 1, "quote": "compare 2018 against 2019"}]}


def test_cached_review_is_not_resampled_and_binds_model_packet_images(tmp_path):
    client = Client(review())
    args = {"model": "model-a", "cache_dir": tmp_path, "images": [{"type": "text", "text": "image-sha"}]}
    first = asyncio.run(judge.judge_repair(client, packet(), **args))
    assert first["passed"]
    assert asyncio.run(judge.judge_repair(client, packet(), **args)) == first
    assert len(client.calls) == 1
    asyncio.run(judge.judge_repair(client, packet(), **{**args, "model": "model-b"}))
    asyncio.run(judge.judge_repair(client, packet(), **{**args, "images": []}))
    assert len(client.calls) == 3
    assert client.calls[0]["input"][-1]["text"] == "image-sha"


@pytest.mark.parametrize("field,value", [("grounded_episode", "true"),
    ("earliest_error_step", True), ("explanation", ""), ("evidence", {})])
def test_strict_schema_rejects_wrong_types(field, value):
    data = review()
    data[field] = value
    with pytest.raises(ValueError):
        judge.validate_review(data)


@pytest.mark.parametrize("change", ["invented_quote", "nonexistent_index", "missing_repaired", "invalid_trace"])
def test_positive_review_requires_literal_observed_evidence(change):
    data = review()
    if change == "invented_quote":
        data["evidence"][0]["quote"] = "not actually observed"
    elif change == "nonexistent_index":
        data["evidence"][0]["step_index"] = 9
    elif change == "missing_repaired":
        data["evidence"].pop()
    else:
        data["evidence"][0]["trace"] = "private_reference"
    with pytest.raises(ValueError):
        asyncio.run(judge.judge_repair(Client(data), packet()))


def test_judge_cannot_override_anchor_or_incomplete_episode():
    wrong = packet()
    wrong["selected_step_index"] = 0
    assert not asyncio.run(judge.judge_repair(Client(review()), wrong))["passed"]
    wrong = packet()
    wrong["episode_complete"] = False
    assert not asyncio.run(judge.judge_repair(Client(review()), wrong))["passed"]


def test_rejection_is_cached_without_retry_until_pass(tmp_path):
    data = review()
    data["grounded_episode"] = False
    client = Client(data)
    assert not asyncio.run(judge.judge_repair(client, packet(), cache_dir=tmp_path))["passed"]
    client.result = review()
    assert not asyncio.run(judge.judge_repair(client, packet(), cache_dir=tmp_path))["passed"]
    assert len(client.calls) == 1


def test_malformed_completed_reply_is_not_silently_resampled(tmp_path):
    client = Client({"bad": "schema"})
    for _ in range(2):
        with pytest.raises(ValueError):
            asyncio.run(judge.judge_repair(client, packet(), cache_dir=tmp_path))
    assert len(client.calls) == 1


def test_cache_tamper_fails(tmp_path):
    client = Client(review())
    asyncio.run(judge.judge_repair(client, packet(), cache_dir=tmp_path))
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text())
    saved["response"]["id"] = "replaced"
    write_json(path, saved)
    with pytest.raises(ValueError, match="cache was changed"):
        asyncio.run(judge.judge_repair(client, packet(), cache_dir=tmp_path))


def test_train_case_gate_rejects_evaluation_split(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"case_id": "case-a", "split": "test"}) + "\n")
    with pytest.raises(ValueError, match="training-case"):
        judge.require_training_case({"case_id": "case-a"}, path)


def test_projection_keeps_empty_and_error_observations_but_omits_private_metadata():
    trace = {"state": {"all_steps": [{"action_type": "tool_call", "tool_result": "error: unavailable",
             "metadata": {"policy_action": {"name": "search"}, "policy_token_capture": {"big": [1]}, "gold": "private"}}]}}
    projected = judge.trace_steps(trace)
    assert projected[0]["tool_result"] == "error: unavailable"
    assert "gold" not in json.dumps(projected)
    assert "policy_token_capture" not in json.dumps(projected)


def attempt_fixture():
    source = {"case_id": "case-a", "state": {"all_steps": [
        {"tool_result": "record says 2019"},
        {"stage": "unified_judgment", "output": "assert 2018"}]}}
    episode = copy.deepcopy(source)
    episode["termination"] = "success"
    episode["state"]["all_steps"][1] = {"stage": "unified_judgment", "action_type": "output",
        "output": "compare 2018 against 2019", "metadata": {"policy_token_capture": {
            "prompt_token_ids": [1, 2], "completion_token_ids": [3, 4]}}}
    hint = "Compare fields"
    attempt = {"repair_site": {"source_step_index": 1}, "repair_step_id": "case-a:judgment:1",
               "source_trace_sha256": "a" * 64, "teacher_prompt_ids": [1, 2], "completion_ids": [3, 4],
               "hint_record": {"text": hint, "audit": {"hint_sha256": hashlib.sha256(hint.encode()).hexdigest()}}}
    return source, episode, attempt


def test_judge_artifact_binds_actual_tokens_episode_and_gold(monkeypatch, tmp_path):
    source, episode, attempt = attempt_fixture()
    monkeypatch.setattr(judge, "review_images", lambda *a, **kw: ([], {}))
    data = asyncio.run(judge.judge_attempt(Client(review()), source=source,
        source_trace_sha256="a" * 64, episode=episode, attempt=attempt, gold={"status": "fake"},
        image_path=tmp_path / "unused", model="test-model", cache_dir=tmp_path))
    assert data["passed"]
    assert data["teacher_prompt_sha256"] == _sha([1, 2])
    assert data["teacher_episode_canonical_sha256"] == _sha(episode)
    assert data["private_reference_sha256"] == _sha({"status": "fake"})
    assert "teacher_prompt_ids" not in json.dumps(data["review"])


@pytest.mark.parametrize("tamper", ["prefix", "tokens", "hint", "source"])
def test_judge_refuses_changed_prefix_tokens_hint_source(tamper, tmp_path):
    source, episode, attempt = attempt_fixture()
    if tamper == "prefix":
        episode["state"]["all_steps"][0]["tool_result"] = "different"
    elif tamper == "tokens":
        attempt["completion_ids"] = [99]
    elif tamper == "hint":
        attempt["hint_record"]["text"] = "leaked answer"
    else:
        attempt["source_trace_sha256"] = "b" * 64
    client = Client(review())
    with pytest.raises(ValueError):
        asyncio.run(judge.judge_attempt(client, source=source, source_trace_sha256="a" * 64,
            episode=episode, attempt=attempt, gold={}, image_path=tmp_path / "unused",
            model="test-model", cache_dir=tmp_path))
    assert not client.calls
