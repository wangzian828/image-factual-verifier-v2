import json

from src.integrations.llm.openai_compatible import OpenAICompatibleChatClient


class _Response:
    def __init__(self):
        self.payload = {
            "choices": [{"message": {"content": json.dumps({"ok": True})}}]
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _HTTPClient:
    def __init__(self):
        self.request = None

    def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return _Response()


def test_chat_json_completion_forwards_explicit_thinking_contract(monkeypatch):
    transport = _HTTPClient()
    client = OpenAICompatibleChatClient(
        api_key="none",
        base_url="http://127.0.0.1:19025/v1",
        max_retries=0,
        cache_salt="ifv-case-v1-" + "A" * 43,
    )
    monkeypatch.setattr(client, "_get_client", lambda: transport)

    result = client.create_json_completion(
        model_name="policy-under-test",
        messages=[{"role": "user", "content": "return json"}],
        max_tokens=4096,
        response_schema={"type": "object"},
        chat_template_kwargs={"enable_thinking": True},
        thinking_token_budget=2048,
    )

    assert json.loads(result) == {"ok": True}
    assert transport.request["json"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert transport.request["json"]["thinking_token_budget"] == 2048
    assert transport.request["json"]["cache_salt"] == (
        "ifv-case-v1-" + "A" * 43
    )
