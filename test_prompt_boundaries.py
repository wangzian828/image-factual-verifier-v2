from src.integrations.browse.jina_reader import EXTRACT_PROMPT, EXTRACT_SCHEMA
from src.orchestrator.image_only_prompts import (
    DISCREPANCY_DECISION_SYSTEM_PROMPT,
    DISCREPANCY_REACT_SYSTEM_PROMPT,
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
        "discrepancy investigation": (
            DISCREPANCY_REACT_SYSTEM_PROMPT,
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
    discrepancy_react_prompt = " ".join(
        DISCREPANCY_REACT_SYSTEM_PROMPT.split()
    )
    discrepancy_prompt = " ".join(
        DISCREPANCY_DECISION_SYSTEM_PROMPT.split()
    )
    assert "missing mention is not refutation" in browse_prompt
    assert "same subject-event relation, independent of its value" in browse_prompt
    assert "different_instance requires another occurrence" in browse_prompt
    assert "a competing value for one relation is same_relation" in browse_prompt
    assert "actual value of the disputed relation" in browse_prompt
    assert "never mentions the image's proposed value" in browse_prompt
    assert "does not support its truth" in browse_prompt
    assert "explicit denial refutes it" in browse_prompt
    assert "need not settle every clause" in browse_prompt
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
    assert "runtime owns task state" in react_prompt
    assert "validates grounding" in planning_prompt
    assert "do not own the verdict" in image_account_prompt
    assert "SearchHypotheses ask what actually happened" in image_account_prompt
    assert "one high-salience ImageClaim" in image_account_prompt
    assert "complete subject-event relation" in image_account_prompt
    assert "relation slot, and depicted value" in image_account_prompt
    assert "independently verdict-changing" in image_account_prompt
    assert "Do not inventory details" in image_account_prompt
    assert "positive proposition" in image_account_prompt
    assert "authenticity or a visible integrity anomaly" in image_account_prompt
    assert "what the slot's verified value is" in image_account_prompt
    assert "Image clues guide retrieval but do not restrict it" in image_account_prompt
    assert "Prior knowledge is a lead" in image_account_prompt
    assert "only tool Evidence establishes a fact" in image_account_prompt
    assert "Output: account_summary" in image_account_prompt
    assert "not a boundary on the investigation" in discrepancy_react_prompt
    assert "Frame retrieval around what actually happened" in discrepancy_react_prompt
    assert "identical image can be found" in discrepancy_react_prompt
    assert "only tool Evidence establishes a fact" in discrepancy_react_prompt
    assert "Do not change the ImageClaim" in discrepancy_react_prompt
    assert "select one owned ImageClaim" in discrepancy_react_prompt
    assert "state the passage sought" in discrepancy_react_prompt
    assert "reviewed Evidence" in discrepancy_prompt
    assert "recorded admissible_stances" in discrepancy_prompt
    assert "neutral Evidence cannot" in discrepancy_prompt
    assert "support means it is true" in discrepancy_prompt
    assert "refute means it is false" in discrepancy_prompt
    assert "does not establish semantic coverage" in discrepancy_prompt
    assert "allowed visual anchors" in discrepancy_prompt
    assert "MaterialDiscrepancy" in discrepancy_prompt
    assert "unresolved other Claims do not weaken it" in discrepancy_prompt
    assert "otherwise continue" in discrepancy_prompt
