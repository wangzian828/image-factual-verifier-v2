import pytest

from src.integrations.browse.jina_reader import JinaReaderClient


def test_qwen_local_extract_model_is_explicitly_receipted(monkeypatch):
    monkeypatch.setenv("BROWSE_EXTRACT_PROVIDER", "qwen_local")
    monkeypatch.setenv("BROWSE_EXTRACT_MODEL", "ifv-sft3-under-test")
    client = JinaReaderClient()
    monkeypatch.setattr(
        client,
        "fetch_page_content",
        lambda _url: ("A normal public page about the requested fact.", "jina_reader"),
    )
    monkeypatch.setattr(
        client,
        "extract_goal_evidence",
        lambda *_args, **_kwargs: {
            "evidence": "Relevant evidence.",
            "evidence_available": True,
            "selected_passage_count": 1,
        },
    )

    result = client.visit(
        "https://example.com/source",
        image_claim="The image makes a factual claim.",
        retrieval_goal="Check that factual claim.",
    )

    extract = [row for row in result["subcalls"] if row["kind"] == "page_extract"]
    assert len(extract) == 1
    assert extract[0]["provider"] == "qwen_local"
    assert extract[0]["model"] == "ifv-sft3-under-test"
    assert extract[0]["status"] == "success"
    assert extract[0]["thinking_enabled"] is False


def test_qwen_local_extract_model_never_falls_back_to_gemini(monkeypatch):
    monkeypatch.delenv("BROWSE_EXTRACT_MODEL", raising=False)
    monkeypatch.delenv("QWEN_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("QWEN35_LOCAL_MODEL", raising=False)
    monkeypatch.setenv("BROWSE_EXTRACT_PROVIDER", "qwen_local")
    monkeypatch.setenv("GEMINI_MODEL", "must-not-be-used")

    client = JinaReaderClient()

    with pytest.raises(RuntimeError, match="qwen_local page extraction requires"):
        client.resolved_extract_model()


def test_qwen_local_extract_thinking_contract_is_explicit(monkeypatch):
    monkeypatch.setenv("BROWSE_EXTRACT_PROVIDER", "qwen_local")
    monkeypatch.setenv("BROWSE_EXTRACT_ENABLE_THINKING", "true")
    monkeypatch.setenv("BROWSE_EXTRACT_THINKING_TOKEN_BUDGET", "2048")
    client = JinaReaderClient()
    assert client.extract_thinking_config() == (True, 2048)
