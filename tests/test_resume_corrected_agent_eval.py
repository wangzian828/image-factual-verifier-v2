import json

from types import SimpleNamespace

from scripts.server.resume_corrected_agent_eval import (
    validate_resume_binding,
    validate_safe_cache_off,
)


def test_validate_resume_binding_accepts_frozen_protocol(tmp_path, monkeypatch):
    binding = {
        "schema_version": "ifv-corrected-self-extract-agent-eval-v1",
        "key": "sft3",
        "profile_model": "ifv-qwen3.5-9b-sft3084-selfextract",
        "runnable_cases": 1526,
        "formal_denominator": 1527,
        "page_extract_provider": "qwen_local",
        "page_extract_model": "ifv-qwen3.5-9b-sft3084-selfextract",
        "page_extract_thinking_enabled": True,
        "page_extract_thinking_token_budget": 2048,
        "large_payload_hashing": False,
    }
    (tmp_path / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    for name in ("smoke-engineering-audit.json", "smoke-extraction-protocol-audit.json"):
        (tmp_path / name).write_text('{"passed": true}', encoding="utf-8")
    assert validate_resume_binding(tmp_path) == binding


def test_validate_resume_binding_rejects_protocol_change(tmp_path):
    (tmp_path / "binding.json").write_text(
        json.dumps(
            {
                "schema_version": "ifv-corrected-self-extract-agent-eval-v1",
                "key": "sft3",
                "profile_model": "ifv-qwen3.5-9b-sft3084-selfextract",
                "runnable_cases": 1526,
                "formal_denominator": 1527,
                "page_extract_provider": "gemini",
                "page_extract_model": "ifv-qwen3.5-9b-sft3084-selfextract",
                "page_extract_thinking_enabled": True,
                "page_extract_thinking_token_budget": 2048,
                "large_payload_hashing": False,
            }
        ),
        encoding="utf-8",
    )
    for name in ("smoke-engineering-audit.json", "smoke-extraction-protocol-audit.json"):
        (tmp_path / name).write_text('{"passed": true}', encoding="utf-8")
    try:
        validate_resume_binding(tmp_path)
    except ValueError as error:
        assert "page_extract_provider" in str(error)
    else:
        raise AssertionError("protocol mismatch was accepted")


def test_validate_safe_cache_off_requires_both_hybrid_flags():
    good = SimpleNamespace(
        originals=[
            {
                "command": [
                    "python",
                    "vllm",
                    "--no-enable-prefix-caching",
                    "--mamba-cache-mode",
                    "none",
                ]
            }
            for _ in range(4)
        ]
    )
    validate_safe_cache_off(good)
    bad = SimpleNamespace(originals=[{"command": ["python", "vllm"]}])
    try:
        validate_safe_cache_off(bad)
    except ValueError as error:
        assert "cache-off" in str(error)
    else:
        raise AssertionError("unsafe cache configuration was accepted")
