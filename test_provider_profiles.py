from __future__ import annotations

import asyncio

import pytest

from src.orchestrator.llm_backend import APIBackend
from src.orchestrator.pipeline import Orchestrator
from src.provider_profiles import resolve_provider_settings
from src.workflow import VerificationWorkflow, WorkflowConfig

def test_teacher_profile_is_fixed_to_accepted_gemini_wire() -> None:
    settings = resolve_provider_settings(profile_id="teacher-gemini", environ={})

    assert settings.provider == "gemini"
    assert settings.model_name == "gemini-3.7-flash"
    assert settings.vlm_provider == "gemini"
    assert settings.llm_wire_api == "interactions"
    assert settings.vlm_wire_api == "interactions"

def test_local_student_profile_uses_qwen_without_gemini_fallback() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3-vl-local",
        environ={
            "QWEN3_VL_LOCAL_MODEL": "ifv-qwen3-vl-8b-thinking-smoke"
        },
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
    assert settings.base_url == "http://127.0.0.1:8899/v1"
    assert settings.vlm_base_url == settings.base_url


def test_qwen35_local_profile_uses_one_multimodal_model() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3.5-local",
        environ={"QWEN35_LOCAL_MODEL": "ifv-qwen3.5-9b-vllm"},
    )

    assert settings.provider == "qwen_local"
    assert settings.vlm_provider == "qwen_local"
    assert settings.model_name == "ifv-qwen3.5-9b-vllm"
    assert settings.vlm_model == "ifv-qwen3.5-9b-vllm"
    assert settings.base_url == "http://127.0.0.1:8901/v1"
    assert settings.vlm_base_url == settings.base_url


def test_qwen35_profile_endpoint_override_is_profile_scoped() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3.5-local",
        environ={
            "QWEN_LOCAL_BASE_URL": "http://127.0.0.1:8899/v1",
            "QWEN_LOCAL_MODEL": "wrong-shared-model",
            "QWEN35_LOCAL_BASE_URL": "http://127.0.0.1:9001/v1",
            "QWEN35_LOCAL_MODEL": "ifv-qwen3.5-9b-profile",
        },
    )

    assert settings.base_url == "http://127.0.0.1:9001/v1"
    assert settings.model_name == "ifv-qwen3.5-9b-profile"


def test_qwen35_replica_b_profile_isolated_from_primary() -> None:
    settings = resolve_provider_settings(
        profile_id="student-qwen3.5-local-replica-b",
        environ={
            "QWEN35_LOCAL_BASE_URL": "http://127.0.0.1:8901/v1",
            "QWEN35_LOCAL_MODEL": "ifv-qwen3.5-9b",
            "QWEN35_REPLICA_B_LOCAL_BASE_URL": "http://127.0.0.1:9002/v1",
            "QWEN35_REPLICA_B_LOCAL_MODEL": "ifv-qwen3.5-9b-replica-b-test",
        },
    )

    assert settings.provider == "qwen_local"
    assert settings.vlm_provider == "qwen_local"
    assert settings.base_url == "http://127.0.0.1:9002/v1"
    assert settings.vlm_base_url == settings.base_url
    assert settings.model_name == "ifv-qwen3.5-9b-replica-b-test"
    assert settings.vlm_model == settings.model_name


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


def test_profile_config_can_spawn_isolated_seeded_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = VerificationWorkflow(
        WorkflowConfig(profile_id="student-qwen3.5-local", save_traces=False)
    )
    observed: list[tuple[int | None, str | None, str | None]] = []

    async def record_child(
        self: VerificationWorkflow,
        path: str,
        image_id: str = "",
        *,
        runtime_case: object = None,
    ) -> dict[str, object]:
        _ = runtime_case
        observed.append(
            (
                self.config.sampling_seed,
                self.config.profile_id,
                self.config.llm_base_url,
            )
        )
        return {
            "image_id": image_id,
            "image_path": path,
            "verdict": "real",
            "termination": "success",
        }

    monkeypatch.setattr(VerificationWorkflow, "run_single", record_child)
    results = asyncio.run(
        workflow.run_batch(
            ["first.jpg", "second.jpg"],
            image_ids=["episode-0", "episode-1"],
            sampling_seeds=[101, 202],
        )
    )

    assert [result["image_id"] for result in results] == [
        "episode-0",
        "episode-1",
    ]
    assert observed == [
        (101, "student-qwen3.5-local", "http://127.0.0.1:8901/v1"),
        (202, "student-qwen3.5-local", "http://127.0.0.1:8901/v1"),
    ]

def test_loose_default_remains_gemini() -> None:
    config = WorkflowConfig()

    assert config.profile_id is None
    assert config.provider == "gemini"
    assert config.model_name == "gemini-3.7-flash"


def test_gemini_requests_thought_summaries() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "gemini"

    assert orchestrator._stage_generation_config("PLANNING") == {
        "thinking_level": "high",
        "thinking_summaries": "auto",
    }
    assert orchestrator._stage_generation_config("JUDGMENT") == {
        "thinking_level": "low",
        "thinking_summaries": "auto",
    }


def test_gemini_stage_output_budgets_are_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "gemini"

    assert orchestrator._stage_output_tokens("PLANNING", 8192) == 8192
    assert orchestrator._stage_output_tokens("VERIFICATION", 8192) == 8192
    assert orchestrator._stage_output_tokens("EVIDENCE_DECISION", 8192) == 8192
    assert orchestrator._stage_output_tokens("REFLECTION", 8192) == 8192
    assert orchestrator._stage_output_tokens("QUERY_CONCEPT_EXTRACTION", 4096) == 4096
    assert orchestrator._stage_output_tokens("QUERY_REPLAN", 4096) == 4096
    assert orchestrator._stage_output_tokens("ROUTE_LOCAL_REPLAN", 4096) == 4096
    assert orchestrator._stage_output_tokens("JUDGMENT", 8192) == 8192

    monkeypatch.setenv("GEMINI_QUERY_REPLAN_MAX_OUTPUT_TOKENS", "6144")
    assert orchestrator._stage_output_tokens("QUERY_REPLAN", 4096) == 6144


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

    assert orchestrator._stage_generation_config("PLANNING") == {
        "enable_thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "thinking_token_budget": 1024,
    }
    assert orchestrator._stage_generation_config("EVIDENCE_DECISION")[
        "thinking_token_budget"
    ] == 2048
    assert orchestrator._stage_generation_config("REFLECTION")[
        "thinking_token_budget"
    ] == 1536
    for stage in [
        "VERIFICATION",
        "QUERY_REPLAN",
        "QUERY_CONCEPT_EXTRACTION",
        "JUDGMENT",
    ]:
        config = orchestrator._stage_generation_config(stage)
        assert config["enable_thinking"] is False
        assert config["temperature"] == 0.7
        assert config["top_p"] == 0.8
        assert "thinking_token_budget" not in config


def test_local_qwen_verification_has_bounded_tool_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "qwen_local"

    assert orchestrator._stage_output_tokens("VERIFICATION", 16384) == 8192
    orchestrator.model_name = "ifv-qwen3.5-9b-vllm"
    assert orchestrator._stage_output_tokens("PLANNING", 8192) == 8192

    monkeypatch.setenv("QWEN_VERIFICATION_MAX_OUTPUT_TOKENS", "4096")
    assert orchestrator._stage_output_tokens("VERIFICATION", 16384) == 4096


def test_qwen35_thinking_budget_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.provider = "qwen_local"
    orchestrator.model_name = "ifv-qwen3.5-9b-vllm"
    monkeypatch.setenv("QWEN_PLANNING_THINKING_TOKEN_BUDGET", "768")
    assert orchestrator._stage_generation_config("PLANNING")[
        "thinking_token_budget"
    ] == 768
