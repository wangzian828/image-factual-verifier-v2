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
    assert "one continuing investigation loop" in react_prompt
    assert "canonical target/routes/tasks" in react_prompt
    assert "Similar keywords, subjects, places, or products" in react_prompt
    assert "are not evidence by themselves" in react_prompt
    assert "Reference comparison establishes only image identity/similarity" in react_prompt
    assert "If the target is `unresolved`" in react_prompt
    assert "open_gaps" in react_prompt
    assert "materially different" in react_prompt
    assert "route_local_replan" in react_prompt
    assert "does not require changing the target" in react_prompt or "without changing the target" in react_prompt
    assert "not support for `real`" in react_prompt
    assert "lack of evidence alone is not a verdict" in react_prompt
    assert "overclaiming" in react_prompt
    assert "qualified Evidence" in decision_prompt
    assert "do not support `real`" in decision_prompt
    assert "overclaiming" in decision_prompt
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
