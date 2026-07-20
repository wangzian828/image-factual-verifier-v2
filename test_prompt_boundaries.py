from src.integrations.browse.jina_reader import EXTRACT_PROMPT
from src.orchestrator.image_only_prompts import (
    DISCREPANCY_DECISION_SYSTEM_PROMPT,
    IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    TARGET_PLANNING_SYSTEM_PROMPT,
)


def test_hot_path_prompts_stay_semantic_and_compact() -> None:
    prompts = {
        "browse extraction": (EXTRACT_PROMPT, 1400),
        "investigation": (REACT_SYSTEM_PROMPT, 1200),
        "target planning": (TARGET_PLANNING_SYSTEM_PROMPT, 1300),
        "image account planning": (
            IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT,
            1900,
        ),
        "discrepancy decision": (
            DISCREPANCY_DECISION_SYSTEM_PROMPT,
            2600,
        ),
    }

    for name, (prompt, maximum_length) in prompts.items():
        assert len(prompt) <= maximum_length, name
        assert "\nRules:" not in prompt, name

    browse_prompt = " ".join(EXTRACT_PROMPT.split())
    react_prompt = " ".join(REACT_SYSTEM_PROMPT.split())
    planning_prompt = " ".join(TARGET_PLANNING_SYSTEM_PROMPT.split())
    image_account_prompt = " ".join(
        IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT.split()
    )
    discrepancy_prompt = " ".join(
        DISCREPANCY_DECISION_SYSTEM_PROMPT.split()
    )
    assert "missing mention is not refutation" in browse_prompt
    assert "conflicting value for the same subject" in browse_prompt
    assert "need not settle every clause" in browse_prompt
    assert "runtime owns task state" in react_prompt
    assert "validates grounding" in planning_prompt
    assert "Hypotheses guide retrieval" in image_account_prompt
    assert "do not own the verdict" in image_account_prompt
    assert "required JSON schema" in image_account_prompt
    assert "reviewed qualified Evidence" in discrepancy_prompt
    assert "MaterialDiscrepancy" in discrepancy_prompt
    assert "otherwise continue" in discrepancy_prompt
