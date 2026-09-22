import asyncio
import copy
from types import SimpleNamespace

import pytest

from ifv_training.psd_slate import capture_target, tokenizer_request, validate_slate_revision
from ifv_training.psd_repair import FailureSite, HintProposal
from ifv_training.psd_repair_runtime import QwenContinuationAdapter
import ifv_training.psd_repair_runtime as runtime
from src.orchestrator.stage_runner import StageStep


def hint(text):
    return HintProposal(text=text, level=1, provider="test", model="test", proposal_id=text, audit={})


def test_slate_preserves_working_positions_but_does_not_lock_the_failed_one():
    previous = {0: "first hint", 4: "old later hint"}
    proposed = {0: "first hint", 4: "revised later hint"}
    options = dict(passing_positions=[0], failed_position=4, decision_positions=[0, 1, 4, 5])
    for valid in [proposed, previous, {**proposed, 1: 'earlier cause', 5: 'independent mistake'}]:
        assert validate_slate_revision(previous, valid, **options) == valid
    for bad in [{4: "new"}, {0: "changed", 4: "new"}, {**proposed, 6: "unobserved"}]:
        with pytest.raises(ValueError):
            validate_slate_revision(previous, bad, **options)


def test_tokenizer_request_keeps_tools_thinking_and_processor_config():
    request = {"input_payload": [{"role": "user", "content": "image"}],
        "tool_schema": [{"type": "function", "function": {"name": "search"}}],
        "generation_config": {"enable_thinking": True, "mm_processor_kwargs": {"size": 42}}}
    body = tokenizer_request(request, model="frozen")
    assert body["tools"] == request["tool_schema"]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["mm_processor_kwargs"] == {"size": 42}
    assert body["add_special_tokens"] is False
    native = tokenizer_request({**request, "tool_schema": [{"type": "function", "name": "search",
        "description": "look up", "parameters": {"type": "object"}}]}, model="frozen")
    assert native["tools"][0]["function"]["name"] == "search"


@pytest.mark.parametrize("mismatch", [False, True])
def test_slate_tokens_are_from_corrected_history_and_teacher_render_is_verified(monkeypatch, mismatch):
    import src.orchestrator.runtime_events as events
    prefix = [{"role": "user", "content": "image"},
        {"role": "assistant", "content": "corrected earlier action"},
        {"role": "tool", "content": "new observation"}]
    request = {"input_payload": [{"role": "system", "content": "system"}, *prefix,
        {"role": "user", "content": "local hint"}], "generation_config": {}}
    monkeypatch.setattr(events, "reconstruct_archived_request", lambda *_: request)
    captured = []
    async def tokenize(body):
        captured.append(copy.deepcopy(body))
        if len(captured) == 1:
            return [999] if mismatch else [1, 2, 3]
        return [1, 2]
    step = StageStep(action_type="tool_call", metadata={"context_request_id": "req",
        "policy_token_capture": {"status": "complete", "prompt_token_ids": [1, 2, 3], "completion_token_ids": [4]}})
    call = capture_target(position=4, hint=hint("local hint"), steps=[step], unhinted_prefix=prefix,
        runtime_store=SimpleNamespace(root="archive"), system_instruction="system", tokenize=tokenize, model="frozen")
    if mismatch:
        with pytest.raises(ValueError, match="teacher IDs"):
            asyncio.run(call)
        assert len(captured) == 1
    else:
        row = asyncio.run(call)
        assert row["student_prompt_ids"] == [1, 2]
        assert captured[1]["messages"] == [{"role": "system", "content": "system"}, *prefix]
        assert request["input_payload"][-1]["content"] == "local hint"


def test_live_capture_keeps_json_rendering_tool_order_and_archived_images(monkeypatch):
    import json
    import src.orchestrator.runtime_events as events
    from src.orchestrator.stage_runner import StageRunner
    from src.redaction import sanitize_for_persistence
    from ifv_training.psd_slate import hydrate_live_snapshot
    compact = '{"phase":"unified_react_investigation","count":0}'
    prefix = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": compact}]}]
    messages = [{"role": "system", "content": "system"}, *prefix,
        {"role": "user", "content": "local hint"}]
    tools = [{"type": "function", "name": "search", "parameters": {"type": "object",
        "properties": {"z_query": {"type": "string"}, "a_limit": {"type": "integer"}}}}]
    # Real persistence both formats JSON strings and canonicalizes object keys.
    archived = json.loads(json.dumps(sanitize_for_persistence(
        {"input_payload": messages, "tool_schema": tools}), sort_keys=True))
    assert archived["input_payload"][1]["content"][1]["text"] != compact
    monkeypatch.setattr(events, "reconstruct_archived_request", lambda *_: copy.deepcopy(archived))
    snapshot = StageRunner._policy_input_snapshot(system_instruction="system",
        input_payload=messages[1:], tools=tools, response_format=None)
    step = StageStep(action_type="tool_call", metadata={"context_request_id": "req",
        "policy_input": snapshot, "policy_token_capture": {"status": "complete",
            "prompt_token_ids": [1, 2, 3], "completion_token_ids": [4]}})
    seen = []
    async def tokenize(body):
        seen.append(body)
        assert body["messages"][1]["content"][1]["text"] == compact
        assert body["messages"][1]["content"][0] == prefix[0]["content"][0]
        assert list(body["tools"][0]["function"]["parameters"]["properties"]) == ["z_query", "a_limit"]
        return [1, 2, 3] if len(seen) == 1 else [1, 2]
    row = asyncio.run(capture_target(position=0, hint=hint("local hint"), steps=[step],
        unhinted_prefix=prefix, runtime_store=SimpleNamespace(root="archive"),
        system_instruction="system", tokenize=tokenize, model="frozen"))
    assert row["completion_ids"] == [4] and len(seen) == 2
    with pytest.raises(ValueError, match="content differs"):
        hydrate_live_snapshot(compact, '{"phase":"different"}')


def test_capture_target_binds_accepted_protocol_correction_request(monkeypatch):
    """A rejected JSON response must not make the later correction look stale."""
    import src.orchestrator.runtime_events as events
    from src.orchestrator.stage_runner import StageRunner

    hint_text = "Re-check the visible label."
    original_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "image"},
        {"role": "user", "content": hint_text},
    ]
    correction_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "image"},
        {"role": "assistant", "content": "not valid json"},
        {"role": "user", "content": "Return one corrected complete JSON object."},
        {"role": "user", "content": hint_text},
    ]
    archived = {"input_payload": copy.deepcopy(correction_messages), "tool_schema": []}
    monkeypatch.setattr(events, "reconstruct_archived_request", lambda *_: copy.deepcopy(archived))
    rejected = StageStep(action_type="output_rejected", metadata={
        "policy_input": StageRunner._policy_input_snapshot(
            system_instruction="system", input_payload=original_messages[1:], tools=[], response_format=None),
    })
    accepted = StageStep(action_type="output", metadata={
        "context_request_id": "corrected-request",
        "policy_input": StageRunner._policy_input_snapshot(
            system_instruction="system", input_payload=correction_messages[1:], tools=[], response_format=None),
        "policy_token_capture": {"status": "complete", "prompt_token_ids": [10, 11],
            "completion_token_ids": [12]},
    })
    seen = []

    async def tokenize(body):
        seen.append(copy.deepcopy(body))
        return [10, 11] if len(seen) == 1 else [10]

    row = asyncio.run(capture_target(position=24, hint=hint(hint_text),
        steps=[rejected, accepted], unhinted_prefix=[{"role": "user", "content": "old prefix"}],
        runtime_store=SimpleNamespace(root="archive"), system_instruction="system",
        tokenize=tokenize, model="frozen"))
    assert row["hint_message_index"] == 4
    assert row["teacher_request_kind"] == "accepted_protocol_correction_request"
    assert row["student_prefix_kind"] == "corrected_hint_free_history"
    assert seen[0]["messages"] == correction_messages
    assert seen[1]["messages"] == correction_messages[:4]


@pytest.mark.parametrize("initial_hint", [False, True])
@pytest.mark.parametrize("plain_retry", [False, True])
def test_two_position_runtime_uses_corrected_history_and_removes_each_hint(monkeypatch, initial_hint, plain_retry):
    adapter = QwenContinuationAdapter.__new__(QwenContinuationAdapter)
    adapter.image_path, adapter.runtime_store = "image", None
    adapter.tools_by_name, adapter.source_access_policy = {}, None
    adapter.judgment_system_prompt = "judgment system"
    adapter._site = lambda site: site
    state = SimpleNamespace(action_count=0, stop_reason="")
    adapter._initial_runtime_state = lambda **_: (state, [])
    adapter._excluded_tools = lambda _: []
    async def guard():
        pass
    adapter._guard_policy = guard
    monkeypatch.setattr(runtime, "build_react_runtime_tools", lambda *a, **kw: ["tool"])
    monkeypatch.setattr(runtime, "render_react_runtime_context", lambda s: "context-" + str(s.action_count))
    def record(s, **_):
        s.action_count += 1
        if s.action_count == 2:
            s.stop_reason = "finished"
    monkeypatch.setattr(runtime, "record_react_action", record)
    monkeypatch.setattr(runtime, "compile_react_judgment_basis", lambda *_: {})
    monkeypatch.setattr(runtime, "render_react_judgment_context", lambda *_: "judge context")
    monkeypatch.setattr(runtime, "build_complete_hinted_episode_trace", lambda *a, **kw: {"state": {}, "psd_repair": {}})
    seen = []
    class Runner:
        def __init__(self, history, judgment=False):
            self.history, self.judgment = copy.deepcopy(history), judgment
            if self.history and self.history[0]["role"] == "system":
                self.history.pop(0)
        async def run(self, context):
            seen.append(copy.deepcopy(self.history))
            number = len(seen)
            self.last_native_history = self.history + [{"role": "assistant", "content": f"action-{number}"},
                {"role": "tool", "content": f"observation-{number}"}]
            step = StageStep(stage_name="psd_teacher_judgment" if self.judgment else "psd_teacher_repair",
                action_type="output" if self.judgment else "tool_call", tool_name="search", output={} if self.judgment else None)
            return ({"report": "done"} if self.judgment else None), [step]
    adapter._runner = lambda **kw: Runner(kw["history"])
    adapter._judgment_runner = lambda **kw: Runner(kw["history"], judgment=True)
    site = FailureSite(step_index=0, step_id="step", stage="unified_react", example_type="tool_call",
        policy_input={"system_instruction": "system", "input_payload": [{"role": "user", "content": "image"}]},
        policy_action={}, source_step_index=0)
    async def capture(**kw):
        return {"position": kw["position"], "prefix": copy.deepcopy(kw["unhinted_prefix"])}
    result = asyncio.run(adapter.run_hinted_episode(failure_site=site,
        hint=hint("first advice") if initial_hint and not plain_retry else None, base_trace={},
        hints_by_action={} if plain_retry else {1: hint("second advice")}, capture_local_target=capture))
    if plain_retry:
        assert len(seen) == 3 and result.teacher_complete
        assert result.local_targets == []
        assert result.teacher_episode_trace["psd_repair"]["slate"]["used_positions"] == []
        assert "advice" not in str(seen)
        return
    assert [r["position"] for r in result.local_targets] == ([0, 1] if initial_hint else [1])
    assert "first advice" not in str(seen[1])
    assert "action-1" in str(result.local_targets[-1]["prefix"])
    assert "observation-1" in str(result.local_targets[-1]["prefix"])
    assert "first advice" not in str(seen[2]) and "second advice" not in str(seen[2])
    assert result.teacher_episode_trace["psd_repair"]["slate"]["unused_positions"] == []
