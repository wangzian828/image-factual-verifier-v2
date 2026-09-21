from __future__ import annotations

import asyncio

from src.integrations.llm.prefix_cache import (
    new_case_cache_salt,
    new_rollout_cache_salts,
    validate_case_cache_salt,
)
from src.orchestrator.llm_backend import APIBackend


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
