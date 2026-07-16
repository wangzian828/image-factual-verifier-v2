from src.integrations.browse.jina_reader import EXTRACT_PROMPT
from src.orchestrator.image_only_prompts import (
    REACT_SYSTEM_PROMPT,
    TARGET_PLANNING_SYSTEM_PROMPT,
)


def test_hot_path_prompts_stay_semantic_and_compact() -> None:
    prompts = {
        "browse extraction": (EXTRACT_PROMPT, 1400),
        "investigation": (REACT_SYSTEM_PROMPT, 1200),
        "target planning": (TARGET_PLANNING_SYSTEM_PROMPT, 1300),
    }

    for name, (prompt, maximum_length) in prompts.items():
        assert len(prompt) <= maximum_length, name
        assert "\nRules:" not in prompt, name

    browse_prompt = " ".join(EXTRACT_PROMPT.split())
    react_prompt = " ".join(REACT_SYSTEM_PROMPT.split())
    planning_prompt = " ".join(TARGET_PLANNING_SYSTEM_PROMPT.split())
    assert "missing mention is not a refutation" in browse_prompt
    assert "incompatible with the positive goal" in browse_prompt
    assert "runtime owns task state" in react_prompt
    assert "validates grounding" in planning_prompt
