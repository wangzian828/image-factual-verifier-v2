import json

from types import SimpleNamespace

from scripts.server.resume_corrected_agent_eval import (
    validate_resume_binding,
    validate_safe_cache_off,
)
from scripts.server.control_corrected_agent_eval import gateway_command_for_source


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


def test_validate_safe_cache_off_requires_both_hybrid_flags(monkeypatch):
    monkeypatch.setenv("IFV_PREFIX_CACHE_MODE", "request_isolated")
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
    bad = SimpleNamespace(
        originals=[
            {
                "command": [
                    "python",
                    "vllm",
                    "--mamba-cache-mode",
                    "none",
                ]
            }
        ]
    )
    try:
        validate_safe_cache_off(bad)
    except ValueError as error:
        assert "cache-off" in str(error)
    else:
        raise AssertionError("unsafe cache configuration was accepted")


def test_validate_safe_cache_off_accepts_validated_case_cache(monkeypatch):
    monkeypatch.setenv("IFV_PREFIX_CACHE_MODE", "case_isolated")
    session = SimpleNamespace(
        originals=[
            {
                "command": [
                    "python",
                    "vllm",
                    "--enable-prefix-caching",
                    "--mamba-cache-mode",
                    "align",
                ]
            }
            for _ in range(4)
        ]
    )
    validate_safe_cache_off(session)


def test_gateway_command_for_source_replaces_only_app_dir(tmp_path):
    command = [
        "python",
        "-m",
        "uvicorn",
        "scripts.server.psd_qwen_gateway:app",
        "--app-dir",
        "/old/code",
        "--port",
        "19025",
    ]
    observed = gateway_command_for_source(command, tmp_path)
    assert observed[observed.index("--app-dir") + 1] == str(tmp_path.resolve())
    assert observed[observed.index("--port") + 1] == "19025"
    assert command[command.index("--app-dir") + 1] == "/old/code"
