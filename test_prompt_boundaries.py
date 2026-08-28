from src.integrations.browse.jina_reader import EXTRACT_PROMPT, EXTRACT_SCHEMA
from src.orchestrator.evidence_policy import (
    neutralize_planning_route_text,
    text_targets_verdict_or_media_origin,
)
from src.orchestrator.unified_prompts import (
    UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
    UNIFIED_JUDGMENT_SYSTEM_PROMPT,
    UNIFIED_REACT_SYSTEM_PROMPT,
    UNIFIED_REFLECTION_SYSTEM_PROMPT,
)


def test_active_prompts_are_current_and_semantically_bounded() -> None:
    prompts = {
        "browse extraction": (EXTRACT_PROMPT, 1400),
        "unified react": (UNIFIED_REACT_SYSTEM_PROMPT, 7000),
        "unified reflection": (UNIFIED_REFLECTION_SYSTEM_PROMPT, 1200),
        "unified decision": (
            UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT,
            2400,
        ),
        "unified judgment": (UNIFIED_JUDGMENT_SYSTEM_PROMPT, 1200),
    }

    for name, (prompt, maximum_length) in prompts.items():
        assert len(prompt) <= maximum_length, name
        assert "\nRules:" not in prompt, name

    react_prompt = " ".join(UNIFIED_REACT_SYSTEM_PROMPT.split())
    decision_prompt = " ".join(
        UNIFIED_DISCREPANCY_DECISION_SYSTEM_PROMPT.split()
    )
    assert "一个持续的调查循环" in react_prompt
    assert "扩展为 2–3 条" in react_prompt
    assert "相似主体、相似地点、同类商品" in react_prompt
    assert "不能仅凭相似性作为当前图片的证据" in react_prompt
    assert "比较参考图只能回答图像是否相同" in react_prompt
    assert "当前 target 仍是 `unresolved`" in react_prompt
    assert "open_gaps" in react_prompt
    assert "同一条路线只换几个词" in react_prompt
    assert "route_local_replan" in react_prompt
    assert "不要求你修改 target" in react_prompt
    assert "合格 Evidence" in decision_prompt
    assert "core target fact is decisive" not in decision_prompt

    browse_prompt = " ".join(EXTRACT_PROMPT.split())
    assert "missing mention is not refutation" in browse_prompt
    assert "same subject-event relation, independent of its value" in browse_prompt
    assert "different_instance requires another occurrence" in browse_prompt
    assert "a competing value for one relation is same_relation" in browse_prompt
    assert "actual value of the disputed relation" in browse_prompt
    assert set(EXTRACT_SCHEMA["properties"]["relation_scope"]["enum"]) == {
        "same_relation",
        "partial_relation",
        "different_instance",
        "unclear",
    }
    assert set(EXTRACT_SCHEMA["properties"]["relation_stance"]["enum"]) == {
        "supports",
        "contradicts",
        "background",
        "unclear",
    }
    assert "stance" not in EXTRACT_SCHEMA["properties"]

    assert (
        neutralize_planning_route_text("AI Governance Framework")
        == "AI Governance Framework"
    )
    assert not text_targets_verdict_or_media_origin(
        "Find the original creator and publication date for the depicted event"
    )
    assert text_targets_verdict_or_media_origin(
        "Determine whether the image was AI generated"
    )
    assert text_targets_verdict_or_media_origin(
        "Find a fact check saying the image is fake"
    )
