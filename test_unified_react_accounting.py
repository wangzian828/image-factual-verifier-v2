from __future__ import annotations

from src.orchestrator.unified_react import is_unified_react_budget_action


def _step(tool_name: str) -> dict[str, str]:
    return {
        "stage": "unified_react",
        "action_type": "tool_call",
        "tool_name": tool_name,
    }


def test_route_local_replan_is_persisted_but_free_for_budget_accounting() -> None:
    assert not is_unified_react_budget_action(_step("perceive_scene"))
    assert not is_unified_react_budget_action(_step("ocr_with_position"))
    assert not is_unified_react_budget_action(_step("route_local_replan"))
    assert is_unified_react_budget_action(_step("text_search"))


def test_reflection_boundary_uses_only_budget_actions() -> None:
    steps = [
        _step("perceive_scene"),
        _step("ocr_with_position"),
        _step("text_search"),
        _step("route_local_replan"),
        _step("visit"),
        {
            "stage": "unified_reflection",
            "action_type": "output",
        },
        _step("compare_with_reference"),
        _step("route_local_replan"),
        _step("text_search"),
    ]

    action_count = 0
    reflection_boundaries: list[int] = []
    for step in steps:
        if is_unified_react_budget_action(step):
            action_count += 1
        elif step.get("stage") == "unified_reflection":
            reflection_boundaries.append(action_count)

    assert action_count == 4
    assert reflection_boundaries == [2]
