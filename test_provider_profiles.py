from __future__ import annotations

import pytest

from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.pipeline import Orchestrator
from src.provider_profiles import resolve_provider_settings
from src.workflow import WorkflowConfig

def test_teacher_profile_is_fixed_to_accepted_gemini_wire() -> None:
    settings = resolve_provider_settings(profile_id="teacher-gemini", environ={})

    assert settings.provider == "gemini"
    assert settings.model_name == "gemini-3.5-flash"
    assert settings.vlm_provider == "gemini"
    assert settings.llm_wire_api == "interactions"
    assert settings.vlm_wire_api == "interactions"

def test_local_student_profile_uses_qwen_without_gemini_fallback() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3-vl-local",
        environ={"QWEN_LOCAL_MODEL": "ifv-qwen3-vl-8b-thinking-smoke"},
    )
    backend = APIBackend(
        provider=settings.provider,
        model_name=settings.model_name,
        wire_api=settings.llm_wire_api,
    )

    assert settings.provider == "qwen_local"
    assert settings.vlm_provider == "qwen_local"
    assert settings.model_name == "ifv-qwen3-vl-8b-thinking-smoke"
    assert backend.provider == "qwen_local"
    assert backend.base_url == "http://127.0.0.1:8899/v1"
    assert backend.wire_api == "chat_completions"


def test_qwen35_local_profile_uses_one_multimodal_model() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3.5-local",
        environ={"QWEN_LOCAL_MODEL": "ifv-qwen3.5-9b-vllm"},
    )

    assert settings.provider == "qwen_local"
    assert settings.vlm_provider == "qwen_local"
    assert settings.model_name == "ifv-qwen3.5-9b-vllm"
    assert settings.vlm_model == "ifv-qwen3.5-9b-vllm"


def test_backend_exposes_bounded_interaction_retry_configuration() -> None:
    backend = APIBackend(
        provider="lmdeploy",
        model_name="/models/qwen-student",
        timeout=90.0,
        max_retries=1,
    )

    assert backend.timeout == 90.0
    assert backend.max_retries == 1

def test_qwen_api_profile_fails_closed_without_selected_model() -> None:
    with pytest.raises(ValueError, match="QWEN_API_MODEL"):
        resolve_provider_settings(profile_id="student-qwen-api", environ={})

def test_profile_rejects_all_loose_overrides() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        WorkflowConfig(
            profile_id="teacher-gemini",
            provider="gemini",
        )

def test_loose_default_remains_gemini() -> None:
    config = WorkflowConfig()

    assert config.profile_id is None
    assert config.provider == "gemini"
    assert config.model_name == "gemini-3.5-flash"


def test_local_qwen_planning_reasoning_is_stage_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "qwen_local"

    assert orchestrator._stage_generation_config("PLANNING") == {
        "enable_thinking": False
    }
    assert orchestrator._stage_generation_config("JUDGMENT") == {
        "enable_thinking": True
    }

    monkeypatch.setenv("QWEN_PLANNING_ENABLE_THINKING", "true")
    assert orchestrator._stage_generation_config("PLANNING") == {
        "enable_thinking": True
    }


def test_qwen35_uses_hybrid_stage_reasoning_defaults() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "qwen_local"
    orchestrator.model_name = "ifv-qwen3.5-9b-vllm"

    for stage in ["PLANNING", "EVIDENCE_DECISION", "REFLECTION", "JUDGMENT"]:
        assert orchestrator._stage_generation_config(stage) == {
            "enable_thinking": True
        }
    for stage in ["VERIFICATION", "QUERY_REPLAN", "QUERY_CONCEPT_EXTRACTION"]:
        assert orchestrator._stage_generation_config(stage) == {
            "enable_thinking": False
        }


def test_local_qwen_verification_has_bounded_tool_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "qwen_local"

    assert orchestrator._stage_output_tokens("VERIFICATION", 16384) == 8192
    assert orchestrator._stage_output_tokens("PLANNING", 8192) == 32768

    monkeypatch.setenv("QWEN_VERIFICATION_MAX_OUTPUT_TOKENS", "4096")
    assert orchestrator._stage_output_tokens("VERIFICATION", 16384) == 4096
