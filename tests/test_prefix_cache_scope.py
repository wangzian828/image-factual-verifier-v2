from __future__ import annotations

import asyncio

import pytest

from src.integrations.llm.prefix_cache import (
    new_case_cache_salt,
    new_rollout_cache_salts,
    validate_case_cache_salt,
)
from src.orchestrator.llm_backend import APIBackend
from src.integrations.vlm.factory import build_vlm_client


class _Response:
    content = b'{"choices":[{"message":{"content":"ok"}}]}'

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class _AsyncHTTPClient:
    def __init__(self):
        self.request = None

    async def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return _Response()


def test_scope_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("IFV_PREFIX_CACHE_MODE", raising=False)
    assert new_case_cache_salt() is None


def test_case_scope_is_random_valid_and_not_shared(monkeypatch):
    monkeypatch.setenv("IFV_PREFIX_CACHE_MODE", "case_isolated")
    first = new_case_cache_salt()
    second = new_case_cache_salt()
    assert validate_case_cache_salt(first) == first
    assert validate_case_cache_salt(second) == second
    assert first != second


def test_agent_and_extractor_use_distinct_private_domains(monkeypatch):
    monkeypatch.setenv("IFV_PREFIX_CACHE_MODE", "case_isolated")
    agent, extractor = new_rollout_cache_salts()
    assert validate_case_cache_salt(agent) == agent
    assert validate_case_cache_salt(extractor) == extractor
    assert agent != extractor


def test_rollout_domains_are_both_disabled_by_default(monkeypatch):
    monkeypatch.delenv("IFV_PREFIX_CACHE_MODE", raising=False)
    assert new_rollout_cache_salts() == (None, None)


def test_local_backend_forwards_explicit_case_scope(monkeypatch):
    salt = "ifv-case-v1-" + "C" * 43
    transport = _AsyncHTTPClient()
    backend = APIBackend(
        provider="qwen_local",
        model_name="policy-under-test",
        base_url="http://127.0.0.1:19025/v1",
        max_retries=0,
        cache_salt=salt,
    )
    monkeypatch.setattr(backend, "_get_shared_client", lambda: transport)
    asyncio.run(backend.get_response([{"role": "user", "content": "hello"}]))
    assert transport.request["json"]["cache_salt"] == salt


def test_local_vision_tool_forwards_private_scope_to_wire(monkeypatch):
    salt = "ifv-case-v1-" + "V" * 43
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_json_completion(self, **kwargs):
            captured["request"] = kwargs
            return '{"scene_description":"ok"}'

        def close(self):
            pass

    monkeypatch.setattr(
        "src.integrations.vlm.openai_vlm.OpenAICompatibleChatClient", FakeChat
    )
    monkeypatch.setattr(
        "src.integrations.vlm.openai_vlm.vision_tool_image_to_data_url",
        lambda _image: "data:image/png;base64,AA==",
    )
    client = build_vlm_client(
        provider="qwen_local",
        model_name="served-model",
        base_url="http://127.0.0.1:19025/v1",
        cache_salt=salt,
    )
    assert client.create_image_json(
        system_prompt="system", user_text="inspect", image_input="image.png",
        max_tokens=100,
    ) == {"scene_description": "ok"}
    assert captured["cache_salt"] == salt
    assert captured["request"]["model_name"] == "served-model"


def test_vision_scope_rejects_public_provider_and_malformed_salt():
    salt = "ifv-case-v1-" + "V" * 43
    with pytest.raises(ValueError, match="local serving"):
        build_vlm_client(provider="openai", api_key="test", cache_salt=salt)
    with pytest.raises(ValueError, match="opaque 256-bit"):
        build_vlm_client(provider="qwen_local", cache_salt="predictable")
