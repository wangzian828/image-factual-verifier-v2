from __future__ import annotations

from src.orchestrator.react_runtime import (
    REACT_RUNTIME_TOOLS,
    UnifiedReactState,
    validate_react_action,
)
from src.orchestrator.tool_ablation import ReactToolFamilyConfig


def test_default_tool_family_config_preserves_formal_runtime() -> None:
    config = ReactToolFamilyConfig()

    assert config.disabled_tool_names == frozenset()
    assert config.enabled_tool_names == REACT_RUNTIME_TOOLS
    assert config.to_manifest()["exposed_runtime_tools"] == [
        *REACT_RUNTIME_TOOLS,
        "finish_investigation",
    ]


def test_each_ablation_only_removes_current_formal_runtime_tools() -> None:
    assert ReactToolFamilyConfig(
        enable_web_search=False
    ).disabled_tool_names == {"text_search"}
    assert ReactToolFamilyConfig(
        enable_image_retrieval=False
    ).disabled_tool_names == {
        "text_image_search",
        "reverse_image_search",
    }
    assert ReactToolFamilyConfig(
        enable_evidence_inspection=False
    ).disabled_tool_names == {
        "visit",
        "compare_with_reference",
    }

    all_disabled = ReactToolFamilyConfig(
        enable_web_search=False,
        enable_image_retrieval=False,
        enable_evidence_inspection=False,
    ).disabled_tool_names
    assert all_disabled == {
        "text_search",
        "text_image_search",
        "reverse_image_search",
        "visit",
        "compare_with_reference",
    }
    assert not all_disabled.intersection(
        {"crop_and_search", "recall_evidence", "read_evidence"}
    )


def test_action_validator_hard_rejects_disabled_tool() -> None:
    state = UnifiedReactState(
        case_id="case-1",
        image_sha256="0" * 64,
    )

    error = validate_react_action(
        state,
        tool_name="text_search",
        tool_args={"queries": ["example event"]},
        allowed_tool_names={"perceive_scene"},
    )

    assert error == (
        "text_search is disabled by the active tool-family configuration"
    )


def test_finish_action_remains_available_under_every_ablation() -> None:
    state = UnifiedReactState(
        case_id="case-1",
        image_sha256="0" * 64,
    )

    error = validate_react_action(
        state,
        tool_name="finish_investigation",
        tool_args={"rationale": "The retained evidence is sufficient."},
        allowed_tool_names=set(),
    )

    assert error == ""
