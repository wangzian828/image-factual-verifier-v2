from src.integrations.browse.jina_reader import EXTRACT_PROMPT, EXTRACT_SCHEMA
from src.orchestrator.evidence_policy import (
    neutralize_planning_route_text,
    text_targets_verdict_or_media_origin,
)
from src.orchestrator.unified_prompts import (
    UNIFIED_JUDGMENT_SYSTEM_PROMPT,
    UNIFIED_REACT_SYSTEM_PROMPT,
)


def test_active_prompts_are_current_and_semantically_bounded() -> None:
    prompts = {
        "browse extraction": (EXTRACT_PROMPT, 1400),
        "unified react": (UNIFIED_REACT_SYSTEM_PROMPT, 7000),
        "unified judgment": (UNIFIED_JUDGMENT_SYSTEM_PROMPT, 1200),
    }

    for name, (prompt, maximum_length) in prompts.items():
        assert len(prompt) <= maximum_length, name
        assert "\nRules:" not in prompt, name

    react_prompt = " ".join(UNIFIED_REACT_SYSTEM_PROMPT.split())
    assert "one continuing investigation loop" in react_prompt
    assert "fixed investigation objective" in react_prompt
    assert "perceive_scene" in react_prompt
    assert "ocr_with_position" in react_prompt
    assert "there is no separate planning or replan output" in react_prompt
    assert "no mandatory visual bootstrap gate" in react_prompt
    assert "Background context and a similar subject are not enough" in react_prompt
    assert "unverified candidates, not proof of a match" in react_prompt
    assert "Comparison can establish image similarity" in react_prompt
    assert "Search results, titles, snippets" in react_prompt
    assert "context_only" in react_prompt
    assert "does not choose the final verdict" in react_prompt
    assert "finish_investigation" in react_prompt
    assert "never evidence for either `real` or `fake`" in react_prompt
    assert "raise or lower verdict confidence" in react_prompt
    assert "Never request a generic authenticity" in react_prompt

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
