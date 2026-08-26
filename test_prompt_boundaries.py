from src.integrations.browse.jina_reader import EXTRACT_PROMPT, EXTRACT_SCHEMA
from src.orchestrator.image_only_prompts import (
    DISCREPANCY_DECISION_SYSTEM_PROMPT,
    DISCREPANCY_REACT_SYSTEM_PROMPT,
    IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    TARGET_PLANNING_SYSTEM_PROMPT,
)
from src.orchestrator.evidence_policy import (
    neutralize_planning_route_text,
    text_targets_verdict_or_media_origin,
)


def test_hot_path_prompts_stay_semantic_and_compact() -> None:
    prompts = {
        "browse extraction": (EXTRACT_PROMPT, 1400),
        "investigation": (REACT_SYSTEM_PROMPT, 1200),
        "target planning": (TARGET_PLANNING_SYSTEM_PROMPT, 1300),
        "image account planning": (
            IMAGE_ACCOUNT_PLANNING_SYSTEM_PROMPT,
            3900,
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
        if name != "browse extraction":
            assert "判断图像表达的事实内容是否成立。" in prompt, name

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
    assert "do not decide the verdict" in image_account_prompt
    assert "1. Define the target facts" in image_account_prompt
    assert "2. Design neutral investigation routes" in image_account_prompt
    assert "3. Keep the factual boundary" in image_account_prompt
    assert "4. Prohibited directions" in image_account_prompt
    assert "search_hypotheses as neutral routes" in image_account_prompt
    assert "exactly one high-salience central target" in image_account_prompt
    assert "independently change the verdict" in image_account_prompt
    assert "Do not inventory visible details" in image_account_prompt
    assert "positive real-world proposition" in image_account_prompt
    assert "not candidate verdicts" in image_account_prompt
    assert "investigation context, not target facts" in image_account_prompt
    assert "Every hypothesis must expose an executable first-hop route" in (
        image_account_prompt
    )
    assert "Only tool-produced Evidence can establish a fact" in (
        image_account_prompt
    )
    assert "AI-generation, manipulation, authenticity" in image_account_prompt
    assert "Identity, place, date, creator, platform, publication" in (
        image_account_prompt
    )
    assert "not target facts or verdict grounds" in image_account_prompt
    assert (
        neutralize_planning_route_text("AI Governance Framework")
        == "AI Governance Framework"
    )
    assert (
        neutralize_planning_route_text("AI for Science Fund")
        == "AI for Science Fund"
    )
    assert not text_targets_verdict_or_media_origin(
        "Find the original creator and publication date for the depicted event"
    )
    assert not text_targets_verdict_or_media_origin(
        "Find the AI Governance Framework named in the visible title"
    )
    assert text_targets_verdict_or_media_origin(
        "Determine whether the image was AI generated"
    )
    assert text_targets_verdict_or_media_origin(
        "Find a fact check saying the image is fake"
    )
    assert "Return one JSON object matching the response schema" in (
        image_account_prompt
    )
    assert "``account_summary``" in image_account_prompt
    assert "``target_facts``" in image_account_prompt
    assert "``search_hypotheses``" in image_account_prompt
    assert "not a boundary on the investigation" in discrepancy_react_prompt
    assert "Frame retrieval around what actually happened" in discrepancy_react_prompt
    assert "actual value of the same relation slot" in discrepancy_react_prompt
    assert "identical image can be found" in discrepancy_react_prompt
    assert "visible anchors, relation slots" in discrepancy_react_prompt
    assert "target relation" in discrepancy_react_prompt
    assert "only tool Evidence establishes a fact" in discrepancy_react_prompt
    assert "legacy image-claim record is bookkeeping only" in discrepancy_react_prompt
    assert "select one owned target fact" in discrepancy_react_prompt
    assert "state the passage sought" in discrepancy_react_prompt
    assert "reviewed Evidence" in discrepancy_prompt
    assert "recorded admissible_stances" in discrepancy_prompt
    assert "neutral Evidence cannot" in discrepancy_prompt
    assert "support means it is true" in discrepancy_prompt
    assert "refute means it is false" in discrepancy_prompt
    assert "competing value for the same subject-event relation" in discrepancy_prompt
    assert "does not establish semantic coverage" in discrepancy_prompt
    assert "allowed visual anchors" in discrepancy_prompt
    assert "MaterialDiscrepancy" in discrepancy_prompt
    assert "core target fact is decisive" in discrepancy_prompt
    assert "otherwise continue" in discrepancy_prompt
