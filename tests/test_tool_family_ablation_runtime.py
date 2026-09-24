from src.orchestrator.react_runtime import (
    REACT_RUNTIME_TOOLS,
    UnifiedReactState,
    build_react_runtime_tools,
    validate_react_action,
)
from src.orchestrator.tool_ablation import ReactToolFamilyConfig
from src.tools.base import BaseTool


class DummyTool(BaseTool):
    def __init__(self, name: str):
        self.name = name

    def call(self, params):
        return {}


def test_tool_family_ablations_remove_and_reject_only_the_named_tools():
    state = UnifiedReactState(case_id="audit", image_sha256="0" * 64)
    all_tools = {name: DummyTool(name) for name in REACT_RUNTIME_TOOLS}
    variants = (
        (ReactToolFamilyConfig(enable_web_search=False), {"text_search"}),
        (ReactToolFamilyConfig(enable_image_retrieval=False),
         {"text_image_search", "reverse_image_search"}),
        (ReactToolFamilyConfig(enable_evidence_inspection=False),
         {"visit", "compare_with_reference"}),
    )
    for config, expected_disabled in variants:
        assert config.disabled_tool_names == expected_disabled
        visible = build_react_runtime_tools(
            state, all_tools, excluded_tool_names=config.disabled_tool_names,
        )
        names = {tool.name for tool in visible}
        assert names == set(REACT_RUNTIME_TOOLS) - expected_disabled | {"finish_investigation"}
        for name in expected_disabled:
            assert "disabled by the active tool-family configuration" in validate_react_action(
                state, tool_name=name, tool_args={},
                allowed_tool_names=config.enabled_tool_names,
            )
