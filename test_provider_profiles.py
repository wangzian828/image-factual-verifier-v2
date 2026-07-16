from __future__ import annotations

import pytest

from src.orchestrator.llm_backend import APIBackend
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
        profile_id="student-qwen-local",
        environ={"LMDEPLOY_MODEL": "/models/qwen-student"},
    )
    backend = APIBackend(
        provider=settings.provider,
        model_name=settings.model_name,
        wire_api=settings.llm_wire_api,
    )

    assert settings.provider == "lmdeploy"
    assert settings.vlm_provider == "lmdeploy"
    assert settings.model_name == "/models/qwen-student"
    assert backend.provider == "lmdeploy"
    assert backend.base_url == "http://127.0.0.1:8899/v1"
    assert backend.wire_api == "chat_completions"

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
