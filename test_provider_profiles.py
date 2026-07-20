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
        profile_id="student-qwen35-local",
        environ={"QWEN_LOCAL_MODEL": "ifv-qwen35-4b-smoke"},
    )
    backend = APIBackend(
        provider=settings.provider,
        model_name=settings.model_name,
        wire_api=settings.llm_wire_api,
    )

    assert settings.provider == "qwen_local"
    assert settings.vlm_provider == "qwen_local"
    assert settings.model_name == "ifv-qwen35-4b-smoke"
    assert backend.provider == "qwen_local"
    assert backend.base_url == "http://127.0.0.1:8899/v1"
    assert backend.wire_api == "chat_completions"


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
