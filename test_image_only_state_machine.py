from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from src.orchestrator.bootstrap import build_bootstrap_investigation
from src.orchestrator.coverage import (
    activate_initial_decisive_facts,
    audit_coverage,
    compile_verdict_basis,
)
from src.orchestrator.evidence_adjudication import assess_fact
from src.orchestrator.investigation_models import (
    AttributionFactProposal,
    AttributionOutput,
    FactOrigin,
    Finding,
    InvestigationEvidence,
    InvestigationSegmentOutput,
    ImageOnlyJudgment,
    ReflectionOutput,
    ResearchTask,
    TargetFactProposal,
    TargetPlanningOutput,
    TaskUpdate,
    VisualFact,
)
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.image_only_prompts import (
    pending_discovery_routes,
    render_react_context,
    render_target_refresh_context,
    select_react_tasks,
)
from src.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
)
from src.orchestrator.task_store import (
    apply_attribution,
    apply_reflection,
    apply_target_planning,
    attribution_planning_needed,
    next_action_boundary,
    record_tool_observation,
    state_from_bootstrap,
)


def _runtime_state():
    case = ImageOnlyRuntimeCase(
        case_id="case-state-machine",
        image_path="fixture.jpg",
        image_sha256="a" * 64,
    )
    perception = PerceptionReport(
        scene_description="A marked research vessel is visible on the water.",
        entities=[
            Entity(
                name="NOAA Ship Henry B. Bigelow",
                entity_type="object",
                bbox=[0.1, 0.2, 0.9, 0.9],
                confidence=0.98,
            )
        ],
        text_regions=[
            TextRegion(
                text="HENRY B. BIGELOW",
                bbox_quad=[[0.2, 0.4], [0.7, 0.4], [0.7, 0.5], [0.2, 0.5]],
                confidence=0.98,
            ),
            TextRegion(
                text="R 225",
                bbox_quad=[[0.7, 0.5], [0.82, 0.5], [0.82, 0.58], [0.7, 0.58]],
                confidence=0.96,
            ),
        ],
    )
    state = state_from_bootstrap(
        build_bootstrap_investigation(case, perception)
    )
    activate_initial_decisive_facts(state)
    return case, state


def _screenshot_runtime_state():
    case = ImageOnlyRuntimeCase(
        case_id="case-screenshot-state-machine",
        image_path="screenshot.jpg",
        image_sha256="b" * 64,
    )
    perception = PerceptionReport(
        scene_description="A screenshot of an X post and a reply.",
        image_type="screenshot",
        text_regions=[
            TextRegion(
                text="Major Tom @dingzhen47",
                bbox_quad=[
                    [0.1, 0.1],
                    [0.5, 0.1],
                    [0.5, 0.2],
                    [0.1, 0.2],
                ],
                confidence=0.99,
            ),
            TextRegion(
                text="学生用AI写，学校用AI查",
                bbox_quad=[
                    [0.1, 0.25],
                    [0.9, 0.25],
                    [0.9, 0.4],
                    [0.1, 0.4],
                ],
                confidence=0.99,
                language="zh",
            ),
            TextRegion(
                text="5/18/25",
                bbox_quad=[
                    [0.1, 0.45],
                    [0.3, 0.45],
                    [0.3, 0.5],
                    [0.1, 0.5],
                ],
                confidence=0.98,
            ),
        ],
    )
    state = state_from_bootstrap(
        build_bootstrap_investigation(case, perception)
    )
    scene_fact = next(
        fact
        for fact in state.facts
        if fact.predicate == "appears_to_depict"
    )
    text_facts = [
        fact for fact in state.facts if fact.predicate == "reads"
    ]
    apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The visible Major Tom @dingzhen47 post text "
                        "学生用AI写，学校用AI查 dated 5/18/25 matches an "
                        "original public source record."
                    ),
                    predicate="source_record_matches",
                    parent_fact_ids=[
                        fact.fact_id for fact in text_facts[:3]
                    ],
                    question=(
                        "Does an original public record match the visible account, "
                        "text, date, and reply relation?"
                    ),
                    purpose="Resolve the visible public-record attribution.",
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=[
                        '"学生用AI写，学校用AI查" @dingzhen47'
                    ],
                ),
                TargetFactProposal(
                    statement=(
                        "The visible post layout has no material manipulation "
                        "affecting its displayed account, text, date, or reply."
                    ),
                    kind="internal_consistency",
                    predicate="visual_integrity",
                    parent_fact_ids=[scene_fact.fact_id],
                    question=(
                        "Are visible account, text, date, or reply elements "
                        "materially manipulated?"
                    ),
                    purpose="Inspect the salient visual integrity property.",
                    suggested_tools=[
                        "analyze_visual_anomalies",
                        "check_consistency",
                    ],
                ),
            ]
        ),
    )
    activate_initial_decisive_facts(state)
    return case, state


def _antarctic_butterfly_state():
    case = ImageOnlyRuntimeCase(
        case_id="case-antarctic-butterflies",
        image_path="antarctic-butterflies.jpg",
        image_sha256="c" * 64,
    )
    perception = PerceptionReport(
        scene_description=(
            "Millions of monarch butterflies cover a pine tree in an "
            "Antarctic winter landscape with penguins."
        ),
        entities=[
            Entity(
                name="monarch butterflies",
                entity_type="animal",
                bbox=[0.1, 0.1, 0.7, 0.8],
                confidence=0.98,
            ),
            Entity(
                name="pine tree",
                entity_type="scene_element",
                bbox=[0.2, 0.1, 0.8, 0.9],
                confidence=0.95,
            ),
            Entity(
                name="penguins",
                entity_type="animal",
                bbox=[0.7, 0.6, 0.95, 0.9],
                confidence=0.94,
            ),
        ],
    )
    state = state_from_bootstrap(
        build_bootstrap_investigation(case, perception)
    )
    return case, state


def _step(
    *,
    task_id: str,
    call_id: str,
    tool_name: str,
    result: str,
):
    return SimpleNamespace(
        action_type="tool_call",
        tool_name=tool_name,
        tool_args={"__question_id": task_id},
        tool_result=result,
        metadata={
            "function_call_id": call_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def test_next_action_boundary_advances_after_reflection() -> None:
    assert next_action_boundary(0) == 4
    assert next_action_boundary(1) == 4
    assert next_action_boundary(4) == 8
    assert next_action_boundary(8) == 12
    assert next_action_boundary(23) == 24
    assert next_action_boundary(24) == 24


def test_web_evidence_goal_uses_task_question_not_full_image_claim() -> None:
    _, state = _runtime_state()
    task = next(
        item
        for item in state.tasks
        if set(item.fact_ids) & set(state.decisive_fact_ids)
    )
    fact = next(
        item for item in state.facts if item.fact_id in task.fact_ids
    )
    fact.statement = (
        "The monarch butterfly shown in the image naturally occurs in Antarctica."
    )
    task.question = (
        "Does the natural distribution of monarch butterflies include Antarctica?"
    )

    claims = Orchestrator._image_only_task_claims(state)
    goals = Orchestrator._image_only_task_evidence_goals(state)

    assert claims[task.task_id] == fact.statement
    assert goals[task.task_id] == task.question
    assert goals[task.task_id] != claims[task.task_id]


def test_target_planning_rejects_generic_multi_entity_coexistence() -> None:
    _, state = _antarctic_butterfly_state()
    parent_ids = [
        fact.fact_id
        for fact in state.facts
        if fact.predicate in {"appears_to_depict", "visible_in"}
    ]

    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Monarch butterflies, pine trees, and penguins "
                        "naturally coexist in the same real-world geographic "
                        "location."
                    ),
                    predicate="located_at",
                    parent_fact_ids=parent_ids[:4],
                    question=(
                        "Is there any real-world habitat where monarch "
                        "butterflies, pine trees, and penguins coexist?"
                    ),
                    purpose="Test the depicted-world relation.",
                    suggested_tools=["text_search", "visit"],
                )
            ]
        ),
    )

    assert not update["accepted_fact_ids"]
    assert "not atomic" in update["rejected_reasons"][0]


def test_target_planning_keeps_valid_proposal_when_parallel_target_is_invalid() -> None:
    _, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    parent_ids = [
        fact.fact_id
        for fact in state.facts
        if fact.predicate in {"appears_to_depict", "visible_in"}
    ]

    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Monarch butterflies, pine trees, and penguins coexist "
                        "in one real-world geographic location."
                    ),
                    predicate="located_at",
                    parent_fact_ids=parent_ids[:4],
                    question=(
                        "Is there any habitat where monarch butterflies, pine "
                        "trees, and penguins coexist?"
                    ),
                    purpose="Test the depicted-world relation.",
                    suggested_tools=["text_search", "visit"],
                ),
                TargetFactProposal(
                    statement=(
                        "The depicted monarch butterfly scene is authentic and "
                        "unmodified."
                    ),
                    kind="internal_consistency",
                    predicate="visual_integrity",
                    parent_fact_ids=[scene.fact_id],
                    question="Are the visible pixels authentic and unmodified?",
                    purpose="Inspect visible pixel integrity only.",
                    suggested_tools=["analyze_visual_anomalies"],
                    decision_relevance="supporting",
                ),
            ]
        ),
    )

    assert len(update["accepted_fact_ids"]) == 1
    assert any("not atomic" in reason for reason in update["rejected_reasons"])
    accepted = next(
        fact
        for fact in state.facts
        if fact.fact_id == update["accepted_fact_ids"][0]
    )
    assert accepted.predicate == "visual_integrity"
    assert accepted.decision_relevance == "supporting"


def test_target_planning_accepts_atomic_visible_location_relation() -> None:
    _, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    activate_initial_decisive_facts(state)
    assert scene.fact_id in state.decisive_fact_ids

    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Millions of monarch butterflies naturally overwinter "
                        "in the Antarctic landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question=(
                        "Do millions of monarch butterflies migrate to "
                        "Antarctica during winter?"
                    ),
                    purpose=(
                        "Verify the concrete subject-to-place relation visible "
                        "in the scene."
                    ),
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=[
                        "monarch butterflies Antarctica winter migration"
                    ],
                )
            ]
        ),
    )

    assert len(update["accepted_fact_ids"]) == 1
    fact_id = update["accepted_fact_ids"][0]
    assert state.decisive_fact_ids == [fact_id]
    fact = next(item for item in state.facts if item.fact_id == fact_id)
    assert "Antarctic" in fact.statement
    assert scene.decision_relevance == "supporting"


def test_pending_search_candidate_requires_inspection_before_retrieval() -> None:
    case, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Millions of monarch butterflies naturally overwinter "
                        "in the Antarctic landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question=(
                        "Do millions of monarch butterflies migrate to "
                        "Antarctica during winter?"
                    ),
                    purpose="Verify the visible subject-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                )
            ]
        ),
    )
    task = next(
        item
        for item in state.tasks
        if update["accepted_fact_ids"][0] in item.fact_ids
    )
    record_tool_observation(
        state,
        _step(
            task_id=task.task_id,
            call_id="call-location-search",
            tool_name="text_search",
            result=json.dumps(
                {
                    "status": "success",
                    "queries": [
                        {
                            "query": "monarch butterfly wintering ground",
                            "results": [
                                {
                                    "title": "Monarch migration in Mexico",
                                    "url": (
                                        "https://www.nationalgeographic.com/"
                                        "travel/article/latin-america-"
                                        "butterfly-monarch-migration"
                                    ),
                                    "snippet": (
                                        "Mexico's Central Highlands become "
                                        "the wintering grounds."
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    pending = pending_discovery_routes(
        state,
        task_ids={task.task_id},
    )

    assert len(pending["pages"]) == 1
    assert pending["pages"][0]["task_id"] == task.task_id
    assert Orchestrator._image_only_discovery_route_error(
        state,
        "text_search",
        {"__question_id": task.task_id},
    )
    assert not Orchestrator._image_only_discovery_route_error(
        state,
        "visit",
        {"__question_id": task.task_id},
    )


def test_failed_comparison_does_not_exhaust_task_with_uninspected_page() -> None:
    case, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Millions of monarch butterflies naturally overwinter "
                        "in the Antarctic landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question="Do monarch butterflies migrate to Antarctica?",
                    purpose="Verify the visible subject-to-place relation.",
                    suggested_tools=[
                        "text_search",
                        "reverse_image_search",
                        "compare_with_reference",
                        "visit",
                    ],
                )
            ]
        ),
    )
    task = next(
        item
        for item in state.tasks
        if update["accepted_fact_ids"][0] in item.fact_ids
    )
    for index, (tool_name, result) in enumerate(
        [
            (
                "text_search",
                {
                    "status": "success",
                    "queries": [
                        {
                            "query": "monarch butterfly Antarctica",
                            "results": [
                                {
                                    "title": "Monarch facts",
                                    "url": "https://www.fws.gov/species/monarch",
                                    "snippet": "Official monarch information.",
                                }
                            ],
                        }
                    ],
                },
            ),
            (
                "reverse_image_search",
                {
                    "status": "success",
                    "lens_results": [
                        {
                            "title": "Reference image",
                            "url": "https://example.org/reference-page",
                            "image_url": "https://example.org/reference.jpg",
                        }
                    ],
                },
            ),
            (
                "compare_with_reference",
                {
                    "status": "error",
                    "error": "comparison backend temporarily failed",
                },
            ),
        ]
    ):
        record_tool_observation(
            state,
            _step(
                task_id=task.task_id,
                call_id=f"call-pending-{index}",
                tool_name=tool_name,
                result=json.dumps(result),
            ),
            image_sha256=case.image_sha256,
        )

    assert task.attempt_count == 3
    assert task.status == "active"


def test_pending_discovery_routes_prefer_official_sources() -> None:
    case, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Millions of monarch butterflies naturally overwinter "
                        "in the Antarctic landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question="Do monarch butterflies migrate to Antarctica?",
                    purpose="Verify the visible subject-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                )
            ]
        ),
    )
    task = next(
        item
        for item in state.tasks
        if update["accepted_fact_ids"][0] in item.fact_ids
    )
    record_tool_observation(
        state,
        _step(
            task_id=task.task_id,
            call_id="call-source-ranking",
            tool_name="text_search",
            result=json.dumps(
                {
                    "status": "success",
                    "queries": [
                        {
                            "query": "monarch Antarctica",
                            "results": [
                                {
                                    "title": "Unverified answer",
                                    "url": "https://www.quora.com/example",
                                    "snippet": "No butterflies in Antarctica.",
                                },
                                {
                                    "title": "Monarch butterflies",
                                    "url": (
                                        "https://www.si.edu/spotlight/"
                                        "buginfo/monarch"
                                    ),
                                    "snippet": "Smithsonian monarch record.",
                                },
                            ],
                        }
                    ],
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    pending = pending_discovery_routes(
        state,
        task_ids={task.task_id},
    )

    assert pending["pages"][0]["source_class"] == "official"
    assert pending["pages"][0]["url"].startswith("https://www.si.edu/")


def test_external_target_demotes_parallel_visual_integrity() -> None:
    _, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    activate_initial_decisive_facts(state)

    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The depicted monarch butterfly scene is an authentic, "
                        "unmodified photograph of a real-world scene."
                    ),
                    kind="internal_consistency",
                    predicate="visual_integrity",
                    parent_fact_ids=[scene.fact_id],
                    question="Are the pixels authentic and unmodified?",
                    purpose="Inspect visual integrity.",
                    suggested_tools=["analyze_visual_anomalies"],
                    decision_relevance="decisive",
                ),
                TargetFactProposal(
                    statement=(
                        "Millions of monarch butterflies naturally overwinter "
                        "in the Antarctic landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question="Do monarch butterflies migrate to Antarctica?",
                    purpose="Verify the visible subject-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                    decision_relevance="decisive",
                ),
            ]
        ),
    )

    location_id = next(
        fact_id
        for fact_id in update["accepted_fact_ids"]
        if next(
            fact for fact in state.facts if fact.fact_id == fact_id
        ).predicate
        == "located_at"
    )
    integrity = next(
        fact
        for fact in state.facts
        if fact.predicate == "visual_integrity"
    )
    assert integrity.decision_relevance == "supporting"
    assert state.decisive_fact_ids == [location_id]


def test_target_refresh_can_add_independent_scene_relation() -> None:
    _, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    pine = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "pine" in fact.statement.casefold()
    )
    initial = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Monarch butterflies naturally occur in the Antarctic "
                        "landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question=(
                        "Do monarch butterflies naturally occur in Antarctica?"
                    ),
                    purpose="Verify the visible butterfly-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=["monarch butterflies Antarctica"],
                )
            ]
        ),
    )
    exhausted_task = next(
        task
        for task in state.tasks
        if initial["accepted_fact_ids"][0] in task.fact_ids
    )
    exhausted_task.status = "exhausted"
    assert Orchestrator._image_only_target_refresh_needed(state)

    context = render_target_refresh_context(state)
    assert initial["accepted_fact_ids"][0] in context
    assert "exhausted_decisive_targets" in context

    refreshed = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Pine trees naturally grow in the Antarctic landscape "
                        "shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, pine.fact_id],
                    question="Do pine trees naturally grow in Antarctica?",
                    purpose="Verify the visible tree-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                    suggested_queries=["pine trees Antarctica"],
                )
            ]
        ),
    )

    assert len(refreshed["accepted_fact_ids"]) == 1
    refreshed_fact_id = refreshed["accepted_fact_ids"][0]
    assert refreshed_fact_id in state.decisive_fact_ids
    assert any(
        refreshed_fact_id in task.fact_ids and task.status == "active"
        for task in state.tasks
    )


def test_visual_anomaly_result_does_not_attach_to_depicted_world_fact() -> None:
    case, state = _antarctic_butterfly_state()
    scene = next(
        fact for fact in state.facts if fact.predicate == "appears_to_depict"
    )
    butterfly = next(
        fact
        for fact in state.facts
        if fact.predicate == "visible_in"
        and "monarch" in fact.statement.casefold()
    )
    update = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "Monarch butterflies naturally occur in the Antarctic "
                        "landscape shown in the image."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[scene.fact_id, butterfly.fact_id],
                    question=(
                        "Do monarch butterflies naturally occur in Antarctica?"
                    ),
                    purpose="Verify the visible butterfly-to-place relation.",
                    suggested_tools=["text_search", "visit"],
                )
            ]
        ),
    )
    fact_id = update["accepted_fact_ids"][0]
    task = next(item for item in state.tasks if fact_id in item.fact_ids)
    assert "external source evidence" in Orchestrator._image_only_discovery_route_error(
        state,
        "analyze_visual_anomalies",
        {"__question_id": task.task_id},
    )

    result = record_tool_observation(
        state,
        _step(
            task_id=task.task_id,
            call_id="call-world-anomaly",
            tool_name="analyze_visual_anomalies",
            result=json.dumps(
                {
                    "status": "success",
                    "anomalies": [
                        {"phenomenon": "inconsistent lighting"}
                    ],
                    "overall_authenticity": "likely_ai",
                    "notes": "The scene has synthetic-looking edges.",
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert result["created_evidence_ids"] == []
    assert result["created_finding_ids"] == []
    assert next(fact for fact in state.facts if fact.fact_id == fact_id).status == (
        "active"
    )


def test_react_exposes_only_tasks_blocking_unresolved_decisive_facts() -> None:
    _, state = _runtime_state()
    blocking = select_react_tasks(state)

    assert blocking
    assert all(
        set(task.fact_ids) & set(state.decisive_fact_ids)
        for task in blocking
    )
    assert any(
        not (set(task.fact_ids) & set(state.decisive_fact_ids))
        for task in state.tasks
        if task.status in {"active", "pending"}
    )


def test_non_reflection_coverage_does_not_advance_low_gain_streak() -> None:
    _, state = _runtime_state()

    audit_coverage(state, reflection_checkpoint=True)
    first_low_gain = audit_coverage(
        state,
        reflection_checkpoint=True,
    )
    between_reflections = audit_coverage(state)

    assert first_low_gain.low_gain_intervals == 1
    assert first_low_gain.reflection_checkpoint is True
    assert between_reflections.low_gain_intervals == 1
    assert between_reflections.reflection_checkpoint is False
    assert between_reflections.stop_reason == "continue"


def test_discovery_is_not_evidence_and_reflection_only_reprioritizes() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    step = _step(
        task_id=provenance.task_id,
        call_id="call-ris",
        tool_name="reverse_image_search",
        result=(
            '{"status":"success","candidate_page_urls":["https://example.org/page"],'
            '"reference_image_candidates":["https://example.org/ref.jpg"],'
            '"lens_results":[{"title":"candidate","url":"https://example.org/page",'
            '"snippet":"search snippet","image_url":"https://example.org/ref.jpg"}],'
            '"semantic_results":[]}'
        ),
    )

    update = record_tool_observation(
        state,
        step,
        image_sha256=case.image_sha256,
    )

    assert update["created_discovery_ids"]
    assert not update["created_evidence_ids"]
    assert not state.evidence
    assert not state.findings
    reflection = ReflectionOutput(
        task_updates=[
            TaskUpdate(
                task_id=provenance.task_id,
                priority=2,
                reason="Reprioritize without changing reducer-owned status.",
            )
        ]
    )
    record = apply_reflection(
        state,
        reflection,
        evidence_gain=False,
        decision_gain=False,
    )
    assert record.accepted_task_update_ids == [provenance.task_id]
    assert provenance.priority == 2
    assert provenance.status == "active"
    assert "status" not in TaskUpdate.model_json_schema()["properties"]
    assert "basis_ids" not in TaskUpdate.model_json_schema()["properties"]


def test_discovery_attribution_creates_specific_fact_and_verification_task() -> None:
    case, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    provenance = state.tasks[0]
    update = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-attribution-discovery",
            tool_name="reverse_image_search",
            result=(
                '{"status":"success","candidate_page_urls":'
                '["https://www.si.edu/object/reservation-scene"],'
                '"reference_image_candidates":[],'
                '"lens_results":[{"title":"Reservation Scene by Louise Nez, 1992",'
                '"url":"https://www.si.edu/object/reservation-scene",'
                '"snippet":"The Navajo pictorial weaving Reservation Scene was '
                'created by Louise Nez in 1992.","image_url":""}],'
                '"semantic_results":[]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )
    discovery_id = update["created_discovery_ids"][0]

    applied = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=(
                        'The image depicts the Navajo pictorial weaving '
                        '"Reservation Scene" created by Louise Nez in 1992.'
                    ),
                    predicate="identified_as",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=[discovery_id],
                    suggested_queries=[
                        '"Reservation Scene" "Louise Nez" 1992'
                    ],
                )
            ],
            remaining_attribution_gaps=[
                "Fetch the original collection record."
            ],
        ),
    )

    assert len(applied["accepted_fact_ids"]) == 1
    assert not applied["created_task_ids"]
    specific = next(
        fact
        for fact in state.facts
        if fact.fact_id == applied["accepted_fact_ids"][0]
    )
    assert specific.origin.type == "web_discovery"
    assert specific.status == "active"
    assert specific.decision_relevance == "supporting"
    assert specific.fact_id not in state.decisive_fact_ids
    assert next(
        fact for fact in state.facts if fact.fact_id == parent_fact_id
    ).decision_relevance == "decisive"


def test_official_evidence_supports_promoted_attribution_fact() -> None:
    import json

    case, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    provenance = state.tasks[0]
    comparison = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-attribution-comparison",
            tool_name="compare_with_reference",
            result=json.dumps(
                {
                    "status": "success",
                    "reference_url": (
                        "https://www.si.edu/object/reservation-scene.jpg"
                    ),
                    "resolved_reference_url": (
                        "https://www.si.edu/object/reservation-scene.jpg"
                    ),
                    "source_page_url": (
                        "https://www.si.edu/object/reservation-scene"
                    ),
                    "same_subject_or_scene": True,
                    "same_capture_or_near_duplicate": True,
                    "likely_different_original_capture": False,
                    "edit_evidence_present": False,
                    "overall_observation": (
                        "The input image and museum reference show the same "
                        "original artwork capture."
                    ),
                    "confidence": 0.99,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    statement = (
        'The Smithsonian collection record identifies "Reservation Scene" as '
        "a 1992 Navajo pictorial weaving by Louise Nez."
    )
    update = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-attribution-evidence",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://www.si.edu/object/reservation-scene"
                    ),
                    "url": "https://www.si.edu/object/reservation-scene",
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "f" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    applied = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=(
                        'The image depicts the Navajo pictorial weaving '
                        '"Reservation Scene" created by Louise Nez in 1992.'
                    ),
                    predicate="identified_as",
                    parent_fact_ids=[parent_fact_id],
                    evidence_ids=(
                        comparison["created_evidence_ids"]
                        + update["created_evidence_ids"]
                    ),
                    finding_ids=(
                        comparison["created_finding_ids"]
                        + update["created_finding_ids"]
                    ),
                )
            ]
        ),
    )

    specific_id = applied["accepted_fact_ids"][0]
    specific = next(
        fact for fact in state.facts if fact.fact_id == specific_id
    )
    assert specific.status == "supported"
    assert state.decisive_fact_ids == [specific_id]
    assert all(
        specific_id in item.fact_ids
        for item in state.evidence
        if item.evidence_id
        in (
            comparison["created_evidence_ids"]
            + update["created_evidence_ids"]
        )
    )
    assert all(
        specific_id in item.fact_ids
        for item in state.findings
        if item.finding_id
        in (
            comparison["created_finding_ids"]
            + update["created_finding_ids"]
        )
    )
    verdict, basis = compile_verdict_basis(state)
    assert verdict == "real"
    assert basis.fact_ids == [specific_id]


def test_candidate_attribution_upgrades_after_visual_bridge() -> None:
    case, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    broad_task = next(
        task
        for task in state.tasks
        if parent_fact_id in task.fact_ids
    )
    discovery_update = record_tool_observation(
        state,
        _step(
            task_id=broad_task.task_id,
            call_id="call-candidate-discovery",
            tool_name="reverse_image_search",
            result=(
                '{"status":"success","reference_image_candidates":'
                '["https://www.si.edu/object/reservation-scene.jpg"],'
                '"lens_results":[{"title":"Reservation Scene by Louise Nez, 1992",'
                '"url":"https://www.si.edu/object/reservation-scene",'
                '"snippet":"The Navajo pictorial weaving Reservation Scene was '
                'created by Louise Nez in 1992.",'
                '"image_url":"https://www.si.edu/object/reservation-scene.jpg"}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )
    statement = (
        'The image depicts the Navajo pictorial weaving "Reservation Scene" '
        "created by Louise Nez in 1992."
    )
    candidate_update = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=statement,
                    predicate="identified_as",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=discovery_update[
                        "created_discovery_ids"
                    ],
                    decision_relevance="decisive",
                )
            ]
        ),
    )
    candidate_id = candidate_update["accepted_fact_ids"][0]
    candidate = next(
        fact for fact in state.facts if fact.fact_id == candidate_id
    )
    candidate_task = next(
        task
        for task in state.tasks
        if task.task_id == candidate_update["created_task_ids"][0]
    )
    assert candidate.decision_relevance == "supporting"
    assert state.decisive_fact_ids == [parent_fact_id]

    comparison_update = record_tool_observation(
        state,
        _step(
            task_id=candidate_task.task_id,
            call_id="call-candidate-comparison",
            tool_name="compare_with_reference",
            result=json.dumps(
                {
                    "status": "success",
                    "reference_url": (
                        "https://www.si.edu/object/reservation-scene.jpg"
                    ),
                    "resolved_reference_url": (
                        "https://www.si.edu/object/reservation-scene.jpg"
                    ),
                    "source_page_url": (
                        "https://www.si.edu/object/reservation-scene"
                    ),
                    "same_subject_or_scene": True,
                    "same_capture_or_near_duplicate": True,
                    "likely_different_original_capture": False,
                    "edit_evidence_present": False,
                    "overall_observation": (
                        "The input and museum reference are the same capture."
                    ),
                    "confidence": 0.99,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    upgraded = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=statement,
                    predicate="identified_as",
                    parent_fact_ids=[candidate_id],
                    discovery_ids=discovery_update[
                        "created_discovery_ids"
                    ],
                    evidence_ids=comparison_update[
                        "created_evidence_ids"
                    ],
                    finding_ids=comparison_update[
                        "created_finding_ids"
                    ],
                    decision_relevance="decisive",
                )
            ]
        ),
    )

    assert upgraded["accepted_fact_ids"] == [candidate_id]
    assert candidate.decision_relevance == "decisive"
    assert state.decisive_fact_ids == [candidate_id]
    assert broad_task.status == "superseded"
    assert candidate_task.status == "active"


def test_single_unknown_source_cannot_resolve_promoted_attribution() -> None:
    case, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    provenance = state.tasks[0]
    comparison = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-unknown-attribution-comparison",
            tool_name="compare_with_reference",
            result=json.dumps(
                {
                    "status": "success",
                    "reference_url": "https://example.org/artwork.jpg",
                    "resolved_reference_url": "https://example.org/artwork.jpg",
                    "source_page_url": "https://example.org/artwork",
                    "same_subject_or_scene": True,
                    "same_capture_or_near_duplicate": True,
                    "likely_different_original_capture": False,
                    "edit_evidence_present": False,
                    "overall_observation": (
                        "The input and reference show the same artwork capture."
                    ),
                    "confidence": 0.99,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    statement = (
        'An art blog identifies "Reservation Scene" as a 1992 Navajo '
        "pictorial weaving by Louise Nez."
    )
    assertion = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-unknown-attribution-assertion",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": "https://example.org/artwork",
                    "url": "https://example.org/artwork",
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "e" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    applied = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=(
                        'The image depicts the Navajo pictorial weaving '
                        '"Reservation Scene" created by Louise Nez in 1992.'
                    ),
                    predicate="identified_as",
                    parent_fact_ids=[parent_fact_id],
                    evidence_ids=(
                        comparison["created_evidence_ids"]
                        + assertion["created_evidence_ids"]
                    ),
                    finding_ids=(
                        comparison["created_finding_ids"]
                        + assertion["created_finding_ids"]
                    ),
                )
            ]
        ),
    )

    specific = next(
        fact
        for fact in state.facts
        if fact.fact_id == applied["accepted_fact_ids"][0]
    )
    assert specific.status == "active"
    assert applied["created_task_ids"]


def test_attribution_rejects_unknown_or_ungrounded_records() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    update = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-weak-discovery",
            tool_name="text_search",
            result=(
                '{"status":"success","queries":[{"results":['
                '{"title":"NOAA vessel","url":"https://www.noaa.gov/vessel",'
                '"snippet":"A NOAA research vessel."}]}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )
    parent_fact_id = state.decisive_fact_ids[0]

    unknown = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement="The image depicts an unrelated lunar ceremony.",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=["discovery-does-not-exist"],
                )
            ]
        ),
    )
    ungrounded = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement="The image depicts an unrelated lunar ceremony.",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=update["created_discovery_ids"],
                )
            ]
        ),
    )

    assert not unknown["accepted_fact_ids"]
    assert "unknown discovery/evidence/finding" in unknown["rejected_reasons"][0]
    assert not ungrounded["accepted_fact_ids"]
    assert "not grounded" in ungrounded["rejected_reasons"][0]


def test_decisive_attribution_rejects_negated_image_claim() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    parent_fact_id = state.decisive_fact_ids[0]
    update = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-negative-attribution",
            tool_name="text_search",
            result=(
                '{"status":"success","queries":[{"results":['
                '{"title":"NOAA Ship Henry B. Bigelow",'
                '"url":"https://www.noaa.gov/ship",'
                '"snippet":"The vessel is not located in Antarctica."}]}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )

    applied = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement=(
                        "NOAA Ship Henry B. Bigelow is not located in "
                        "Antarctica."
                    ),
                    predicate="located_at",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=update["created_discovery_ids"],
                    decision_relevance="decisive",
                )
            ]
        ),
    )

    assert not applied["accepted_fact_ids"]
    assert "positive image claim" in applied["rejected_reasons"][0]


def test_duplicate_attribution_updates_existing_fact() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    parent_fact_id = state.decisive_fact_ids[0]
    update = record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-duplicate-attribution",
            tool_name="text_search",
            result=(
                '{"status":"success","queries":[{"results":['
                '{"title":"NOAA Ship Henry B. Bigelow",'
                '"url":"https://www.noaa.gov/ship",'
                '"snippet":"The image shows NOAA Ship Henry B. Bigelow."}]}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )
    proposal = AttributionOutput(
        proposals=[
            AttributionFactProposal(
                statement="The image shows NOAA Ship Henry B. Bigelow.",
                parent_fact_ids=[parent_fact_id],
                discovery_ids=update["created_discovery_ids"],
            )
        ]
    )

    first = apply_attribution(state, proposal)
    fact_count = len(state.facts)
    second = apply_attribution(state, proposal)

    assert first["accepted_fact_ids"] == second["accepted_fact_ids"]
    assert len(state.facts) == fact_count


def test_peripheral_discovery_does_not_trigger_decisive_attribution() -> None:
    case, state = _runtime_state()
    peripheral_task = next(
        task
        for task in state.tasks
        if not any(
            fact_id in state.decisive_fact_ids
            for fact_id in task.fact_ids
        )
    )
    parent_fact_id = peripheral_task.fact_ids[0]
    update = record_tool_observation(
        state,
        _step(
            task_id=peripheral_task.task_id,
            call_id="call-peripheral-discovery",
            tool_name="text_search",
            result=(
                '{"status":"success","queries":[{"results":['
                '{"title":"Peripheral profile picture",'
                '"url":"https://example.org/profile",'
                '"snippet":"A side-object profile picture."}]}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert not attribution_planning_needed(state, update)
    applied = apply_attribution(
        state,
        AttributionOutput(
            proposals=[
                AttributionFactProposal(
                    statement="The side object is a peripheral profile picture.",
                    parent_fact_ids=[parent_fact_id],
                    discovery_ids=update["created_discovery_ids"],
                    decision_relevance="decisive",
                )
            ]
        ),
    )
    assert not applied["accepted_fact_ids"]
    assert "current central fact lineage" in applied[
        "rejected_reasons"
    ][0]


def test_provenance_discovery_triggers_specific_attribution_planning() -> None:
    case, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    planned = apply_target_planning(
        state,
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The marked research vessel has a specific public "
                        "provenance and identity."
                    ),
                    predicate="provenance_matches",
                    parent_fact_ids=[parent_fact_id],
                    question=(
                        "What is the vessel's specific identity and public "
                        "source provenance?"
                    ),
                    purpose=(
                        "Promote a discovered name or source into a checkable "
                        "fact before resolving the broad scene."
                    ),
                    suggested_tools=[
                        "reverse_image_search",
                        "text_search",
                        "visit",
                    ],
                    suggested_queries=[
                        '"HENRY B. BIGELOW" "R 225"'
                    ],
                    decision_relevance="supporting",
                )
            ]
        ),
    )
    planned_fact_id = planned["accepted_fact_ids"][0]
    assert planned_fact_id not in state.decisive_fact_ids
    planned_task = next(
        task
        for task in state.tasks
        if planned_fact_id in task.fact_ids
    )
    update = record_tool_observation(
        state,
        _step(
            task_id=planned_task.task_id,
            call_id="call-provenance-discovery",
            tool_name="reverse_image_search",
            result=(
                '{"status":"success","lens_results":['
                '{"title":"NOAA Ship Henry B. Bigelow",'
                '"url":"https://www.noaa.gov/ship",'
                '"snippet":"Official vessel profile.",'
                '"image_url":"https://www.noaa.gov/ship.jpg"}]}'
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert attribution_planning_needed(state, update)


def test_target_planning_rejects_negative_integrity_and_slotless_provenance() -> None:
    _, state = _runtime_state()
    parent_fact_id = state.decisive_fact_ids[0]
    negative_integrity = apply_target_planning(
        state.model_copy(deep=True),
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement="The image is AI-generated and physically impossible.",
                    kind="internal_consistency",
                    predicate="visual_integrity",
                    parent_fact_ids=[parent_fact_id],
                    question="Is the image visually authentic?",
                    purpose="Check pixel integrity.",
                    suggested_tools=["analyze_visual_anomalies"],
                )
            ]
        ),
    )
    slotless_provenance = apply_target_planning(
        state.model_copy(deep=True),
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement="The image shows a marked research vessel.",
                    predicate="provenance_matches",
                    parent_fact_ids=[parent_fact_id],
                    question="What is the vessel's title, creator, and source?",
                    purpose="Recover the specific identity and provenance.",
                    suggested_tools=["reverse_image_search", "visit"],
                    suggested_queries=['"HENRY B. BIGELOW"'],
                )
            ]
        ),
    )
    negative_world_fact = apply_target_planning(
        state.model_copy(deep=True),
        TargetPlanningOutput(
            proposals=[
                TargetFactProposal(
                    statement=(
                        "The input image is a digitally generated composite "
                        "that does not represent a real-world event."
                    ),
                    predicate="depicts_event",
                    parent_fact_ids=[parent_fact_id],
                    question="Is this a real-world event or an AI composite?",
                    purpose="Determine whether the image is fabricated.",
                    suggested_tools=["text_search"],
                )
            ]
        ),
    )

    assert not negative_integrity["accepted_fact_ids"]
    assert "positive authenticity proposition" in negative_integrity[
        "rejected_reasons"
    ][0]
    assert not slotless_provenance["accepted_fact_ids"]
    assert "omits the identity" in slotless_provenance["rejected_reasons"][0]
    assert not negative_world_fact["accepted_fact_ids"]
    assert "positive image claim" in negative_world_fact["rejected_reasons"][0]


def test_reflection_cannot_spawn_same_fact_search_without_new_grounding() -> None:
    _, state = _runtime_state()
    fact_id = state.decisive_fact_ids[0]
    state.action_count = 4
    record = apply_reflection(
        state,
        ReflectionOutput(
            new_tasks=[
                ResearchTask(
                    task_id="task-ungrounded-follow-up",
                    fact_ids=[fact_id],
                    question="Search the web again for the marked research vessel.",
                    purpose="Try a different generic query for the same fact.",
                    priority=1,
                    status="pending",
                    parent_task_id=state.tasks[0].task_id,
                    origin_ids=[fact_id],
                    suggested_tools=["text_search"],
                    suggested_queries=["research vessel"],
                )
            ]
        ),
        evidence_gain=False,
        decision_gain=False,
    )

    assert record.accepted_new_task_ids == []
    assert any(
        "bare fact cannot justify" in reason
        for reason in record.rejected_reasons
    )


def test_original_social_post_can_support_its_own_source_record_match() -> None:
    import json

    case, state = _screenshot_runtime_state()
    source_task = next(
        task
        for task in state.tasks
        if "public-record attribution" in task.purpose
    )
    parent_fact_id = next(
        fact_id
        for fact_id in source_task.fact_ids
        if fact_id in state.decisive_fact_ids
    )
    statement = (
        "Major Tom @dingzhen47 posted 学生用AI写，学校用AI查 on "
        "May 18, 2025, with the displayed reply thread."
    )
    update = record_tool_observation(
        state,
        _step(
            task_id=source_task.task_id,
            call_id="call-x-original-post",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://x.com/dingzhen47/status/1923790000000000000"
                    ),
                    "url": (
                        "https://x.com/dingzhen47/status/1923790000000000000"
                    ),
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "1" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )
    source_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id == parent_fact_id
    )
    assert source_fact.status == "supported"
    assert next(
        evidence
        for evidence in state.evidence
        if evidence.evidence_id in update["created_evidence_ids"]
    ).source_class == "ugc"


def test_current_mutable_social_metadata_cannot_refute_older_source_record() -> None:
    import json

    case, state = _screenshot_runtime_state()
    source_task = next(
        task
        for task in state.tasks
        if "public-record attribution" in task.purpose
    )
    source_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id in source_task.fact_ids
    )
    statement = "The current X page displays Major Tom @dingzhen_47."
    record_tool_observation(
        state,
        _step(
            task_id=source_task.task_id,
            call_id="call-current-mutable-handle",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "refute",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "2" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert source_fact.status == "active"


def test_related_reply_cannot_support_the_original_source_record() -> None:
    import json

    case, state = _screenshot_runtime_state()
    source_task = next(
        task
        for task in state.tasks
        if "public-record attribution" in task.purpose
    )
    source_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id in source_task.fact_ids
    )
    statement = (
        "@dingzhen_47 并非，学生用ai写会有一种自己真的在写什么的错觉。"
        "所以除非你用假几把插飞机杯也能出来。"
    )
    record_tool_observation(
        state,
        _step(
            task_id=source_task.task_id,
            call_id="call-related-reply",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://x.com/shuiyu_kiger/status/"
                        "1924132199408038381"
                    ),
                    "url": (
                        "https://x.com/shuiyu_kiger/status/"
                        "1924132199408038381"
                    ),
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "before_or_at_cutoff",
                    "artifact_sha256": "5" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert source_fact.status == "active"


def test_temporally_aligned_social_record_can_refute_source_match() -> None:
    import json

    case, state = _screenshot_runtime_state()
    source_task = next(
        task
        for task in state.tasks
        if "public-record attribution" in task.purpose
    )
    source_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id in source_task.fact_ids
    )
    statement = (
        "An archived May 18, 2025 record for the post text "
        "学生用AI写，学校用AI查 displays Major Tom @dingzhen_47."
    )
    record_tool_observation(
        state,
        _step(
            task_id=source_task.task_id,
            call_id="call-archived-handle",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "refute",
                    "directness": "direct",
                    "temporal_alignment": "before_or_at_cutoff",
                    "artifact_sha256": "3" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert source_fact.status == "refuted"


def test_adjacent_date_with_unknown_timezone_cannot_refute_source_match() -> None:
    import json

    case, state = _screenshot_runtime_state()
    source_task = next(
        task
        for task in state.tasks
        if "public-record attribution" in task.purpose
    )
    source_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id in source_task.fact_ids
    )
    statement = "4:24 PM · May 17, 2025 529.6K Views"
    record_tool_observation(
        state,
        _step(
            task_id=source_task.task_id,
            call_id="call-timezone-ambiguous-date",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "url": (
                        "https://x.com/dingzhen_47/status/"
                        "1923776560827597300"
                    ),
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "refute",
                    "directness": "direct",
                    "temporal_alignment": "before_or_at_cutoff",
                    "artifact_sha256": "4" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert source_fact.status == "active"


def test_visible_integrity_fact_is_resolved_by_targeted_anomaly_scan() -> None:
    import json

    case, state = _screenshot_runtime_state()
    integrity_task = next(
        task
        for task in state.tasks
        if "visual integrity property" in task.purpose
    )
    integrity_fact = next(
        fact
        for fact in state.facts
        if fact.fact_id in integrity_task.fact_ids
    )
    update = record_tool_observation(
        state,
        _step(
            task_id=integrity_task.task_id,
            call_id="call-screenshot-integrity",
            tool_name="analyze_visual_anomalies",
            result=json.dumps(
                {
                    "status": "success",
                    "focus_areas": [
                        "author, account, text, date, and reply layout"
                    ],
                    "anomalies": [],
                    "overall_authenticity": "authentic",
                    "confidence": 0.86,
                    "notes": (
                        "Typography, spacing, icon alignment, and reply layout "
                        "are internally consistent with no visible edit seam."
                    ),
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert update["created_finding_ids"]
    assert integrity_fact.status == "supported"


def test_budget_exhaustion_is_reported_as_incomplete_not_high_confidence() -> None:
    _, state = _runtime_state()
    state.stop_reason = "hard_budget_exhausted"
    judgment = ImageOnlyJudgment(
        verdict="unverifiable",
        confidence=0.94,
        selected_fact_ids=list(state.decisive_fact_ids),
        selected_finding_ids=[],
        selected_evidence_ids=[],
        overall_assessment="The remaining fact was not resolved.",
        unresolved_gaps=["Decisive source evidence is absent."],
    )

    normalized = Orchestrator._normalize_incomplete_judgment(
        state,
        judgment,
    )

    assert normalized.confidence == 0.65
    assert normalized.overall_assessment.startswith(
        "Investigation incomplete:"
    )
    assert Orchestrator._investigation_status(state) == (
        "incomplete_budget_exhausted"
    )


def test_semantic_reverse_match_preserves_reference_and_is_rendered() -> None:
    case, state = _runtime_state()
    provenance = state.tasks[0]
    reference_url = "https://www.noaa.gov/media/ship.jpg"
    result = {
        "status": "success",
        "reference_image_candidates": [reference_url],
        "lens_results": [],
        "semantic_results": [
            {
                "title": "Official ship image",
                "url": "https://www.noaa.gov/ship",
                "snippet": "",
                "image_url": reference_url,
            }
        ],
    }
    import json

    record_tool_observation(
        state,
        _step(
            task_id=provenance.task_id,
            call_id="call-semantic-ris",
            tool_name="reverse_image_search",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    discovery = next(
        item for item in state.discoveries if item.candidate_url.endswith("/ship")
    )
    assert discovery.reference_image_url == reference_url
    rendered = render_react_context(state)
    assert "Untested reference images" in rendered
    assert reference_url in rendered
    assert '"source_class": "official"' in rendered


def test_supporting_finding_does_not_close_unresolved_decisive_task() -> None:
    case, state = _runtime_state()
    target_task = next(
        task
        for task in state.tasks
        if any(fact_id in state.decisive_fact_ids for fact_id in task.fact_ids)
    )
    statement = "An independent page identifies a possible vessel context."
    result = {
        "status": "success",
        "selected_url": "https://example.org/possible-context",
        "url": "https://example.org/possible-context",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "support",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "c" * 64,
        "evidence_span": {"start": 0, "end": len(statement)},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "injection_flags": [],
        "evidence_eligible": True,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=target_task.task_id,
            call_id="call-supporting-visit",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    fact = next(item for item in state.facts if item.fact_id in target_task.fact_ids)
    assert update["created_finding_ids"]
    assert fact.status == "active"
    assert target_task.status == "active"
    assert target_task.finding_ids == update["created_finding_ids"]


def test_scene_reference_requires_near_duplicate_not_only_same_subject() -> None:
    case, state = _runtime_state()
    scene_task = next(
        task
        for task in state.tasks
        if any(
            fact.fact_id in task.fact_ids and fact.predicate == "appears_to_depict"
            for fact in state.facts
        )
    )
    result = {
        "status": "success",
        "reference_url": "https://www.noaa.gov/media/different-capture.jpg",
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": False,
        "likely_different_original_capture": True,
        "edit_evidence_present": False,
        "edit_evidence_strength": "none",
        "differences": [],
        "overall_observation": (
            "The same vessel appears in a different original capture and setting."
        ),
        "confidence": 0.95,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-different-capture",
            tool_name="compare_with_reference",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    scene_fact = next(
        fact for fact in state.facts if fact.fact_id in scene_task.fact_ids
    )
    assert update["created_evidence_ids"]
    assert not update["created_finding_ids"]
    assert scene_fact.status == "active"
    assert scene_task.status == "active"


def test_scene_support_requires_near_duplicate_and_source_assertion() -> None:
    case, state = _runtime_state()
    scene_task = next(
        task
        for task in state.tasks
        if any(
            fact.fact_id in task.fact_ids and fact.predicate == "appears_to_depict"
            for fact in state.facts
        )
    )
    result = {
        "status": "success",
        "reference_url": "https://images.example.org/matching-capture.jpg",
        "same_subject_or_scene": True,
        "same_capture_or_near_duplicate": True,
        "likely_different_original_capture": False,
        "edit_evidence_present": False,
        "edit_evidence_strength": "none",
        "differences": [],
        "overall_observation": "The images are the same original capture.",
        "confidence": 0.99,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-near-duplicate",
            tool_name="compare_with_reference",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    scene_fact = next(
        fact for fact in state.facts if fact.fact_id in scene_task.fact_ids
    )
    assert update["created_finding_ids"]
    assert scene_fact.status == "active"
    assert scene_task.status == "active"

    statement = (
        "NOAA's source page identifies this exact image as NOAA Ship Henry B. "
        "Bigelow underway."
    )
    source_update = record_tool_observation(
        state,
        _step(
            task_id=scene_task.task_id,
            call_id="call-source-assertion",
            tool_name="visit",
            result=json.dumps(
                {
                    "status": "success",
                    "selected_url": "https://www.noaa.gov/media/ship-page",
                    "url": "https://www.noaa.gov/media/ship-page",
                    "evidence": statement,
                    "summary": statement,
                    "relevance": "high",
                    "stance": "support",
                    "directness": "direct",
                    "temporal_alignment": "not_applicable",
                    "artifact_sha256": "a" * 64,
                    "evidence_span": {
                        "start": 0,
                        "end": len(statement),
                    },
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "injection_flags": [],
                    "evidence_eligible": True,
                }
            ),
        ),
        image_sha256=case.image_sha256,
    )

    assert scene_fact.status == "supported"
    assert scene_task.status == "resolved"
    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)
    assert verdict == "real"
    assert set(basis.evidence_ids) == set(
        update["created_evidence_ids"]
        + source_update["created_evidence_ids"]
    )
    assert coverage.facts[0].winning_evidence_ids == basis.evidence_ids


def test_generic_official_support_cannot_resolve_scene_without_visual_bridge() -> None:
    case, state = _runtime_state()
    for task in state.tasks:
        for fact_id in task.fact_ids:
            if fact_id in state.decisive_fact_ids:
                target_task = task
                break
        else:
            continue
        break
    else:
        raise AssertionError("no task owns a decisive fact")

    statement = "NOAA identifies the vessel as NOAA Ship Henry B. Bigelow."
    result = {
        "status": "success",
        "selected_url": "https://www.noaa.gov/ship-henry-bigelow",
        "url": "https://www.noaa.gov/ship-henry-bigelow",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "support",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "b" * 64,
        "evidence_span": {"start": 0, "end": len(statement)},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "injection_flags": [],
        "evidence_eligible": True,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=target_task.task_id,
            call_id="call-visit",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    assert update["created_evidence_ids"]
    assert update["created_finding_ids"]
    assert target_task.status == "active"
    segment = InvestigationSegmentOutput(
        segment_summary="Official evidence was recorded.",
    )
    assert segment.ready_for_reflection is True

    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)

    assert coverage.complete is False
    assert verdict == "unverifiable"
    assert basis.fact_ids == state.decisive_fact_ids
    assert basis.evidence_ids == update["created_evidence_ids"]


def test_direct_official_refutation_resolves_scene_and_compiles_fake() -> None:
    case, state = _runtime_state()
    target_task = next(
        task
        for task in state.tasks
        if any(
            fact_id in state.decisive_fact_ids
            for fact_id in task.fact_ids
        )
    )
    statement = (
        "NASA's event record states that this ceremony took place at Johnson "
        "Space Center, not Kennedy Space Center."
    )
    result = {
        "status": "success",
        "selected_url": "https://www.nasa.gov/official-event-record",
        "url": "https://www.nasa.gov/official-event-record",
        "evidence": statement,
        "summary": statement,
        "relevance": "high",
        "stance": "refute",
        "directness": "direct",
        "temporal_alignment": "not_applicable",
        "artifact_sha256": "d" * 64,
        "evidence_span": {"start": 0, "end": len(statement)},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "injection_flags": [],
        "evidence_eligible": True,
    }
    import json

    update = record_tool_observation(
        state,
        _step(
            task_id=target_task.task_id,
            call_id="call-official-refute",
            tool_name="visit",
            result=json.dumps(result),
        ),
        image_sha256=case.image_sha256,
    )

    assert target_task.status == "resolved"
    coverage = audit_coverage(state)
    verdict, basis = compile_verdict_basis(state)

    assert coverage.complete is True
    assert verdict == "fake"
    assert basis.fact_ids == state.decisive_fact_ids
    assert basis.evidence_ids == update["created_evidence_ids"]


def test_visual_integrity_refutation_does_not_hide_unresolved_world_fact() -> None:
    _, state = _runtime_state()
    subject_id = state.facts[0].subject_entity_id
    integrity = VisualFact(
        fact_id="fact-visual-integrity",
        kind="internal_consistency",
        statement="The image is an unmodified authentic photograph.",
        subject_entity_id=subject_id,
        predicate="visual_integrity",
        status="active",
        basis_ids=["anchor-integrity"],
        decision_relevance="decisive",
        origin=FactOrigin(
            type="input_image",
            origin_ids=["anchor-integrity"],
        ),
    )
    world_fact = VisualFact(
        fact_id="fact-world-location",
        kind="relation",
        statement="The depicted subject is located at the visible polar place.",
        subject_entity_id=subject_id,
        predicate="located_at",
        status="active",
        basis_ids=["anchor-location"],
        decision_relevance="decisive",
        origin=FactOrigin(
            type="input_image",
            origin_ids=["anchor-location"],
        ),
    )
    evidence = InvestigationEvidence(
        evidence_id="evidence-integrity-refute",
        task_id="task-integrity",
        fact_ids=[integrity.fact_id],
        function_call_id="call-integrity",
        tool_name="analyze_visual_anomalies",
        evidence_kind="image_region",
        source_url="",
        source_family="visual:integrity",
        source_class="visual",
        exact_text="Direct pixel anomalies refute photographic integrity.",
        image_region=[0.0, 0.0, 1.0, 1.0],
        artifact_sha256="f" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="pixel_observation",
    )
    finding = Finding(
        finding_id="finding-integrity-refute",
        task_id="task-integrity",
        fact_ids=[integrity.fact_id],
        statement="Direct pixel anomalies refute photographic integrity.",
        stance="refute",
        evidence_ids=[evidence.evidence_id],
        source_family_ids=[evidence.source_family],
    )
    state.facts.extend([integrity, world_fact])
    state.decisive_fact_ids = [integrity.fact_id, world_fact.fact_id]
    state.evidence.append(evidence)
    state.findings.append(finding)

    coverage = audit_coverage(state)

    assert coverage.stop_reason == "continue"
    assert state.stop_reason == ""
    assert {
        item.fact_id: item.status for item in coverage.facts
    } == {
        integrity.fact_id: "refuted",
        world_fact.fact_id: "unresolved",
    }


def test_pixel_anomaly_cannot_refute_external_location_fact() -> None:
    fact = VisualFact(
        fact_id="fact-external-location",
        kind="relation",
        statement="Monarch butterflies naturally occur in Antarctica.",
        subject_entity_id="entity-butterfly",
        predicate="located_at",
        status="active",
        basis_ids=["anchor-location"],
        decision_relevance="decisive",
        origin=FactOrigin(
            type="input_image",
            origin_ids=["anchor-location"],
        ),
    )
    evidence = InvestigationEvidence(
        evidence_id="evidence-pixel-anomaly",
        task_id="task-location",
        fact_ids=[fact.fact_id],
        function_call_id="call-pixel-anomaly",
        tool_name="analyze_visual_anomalies",
        evidence_kind="image_region",
        source_url="",
        source_family="visual:location",
        source_class="visual",
        exact_text="The scene has synthetic-looking edges.",
        image_region=[0.0, 0.0, 1.0, 1.0],
        artifact_sha256="a" * 64,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        stance="refute",
        quality="strong",
        directness="direct",
        claim_binding="pixel_observation",
    )
    finding = Finding(
        finding_id="finding-pixel-anomaly",
        task_id="task-location",
        fact_ids=[fact.fact_id],
        statement="The scene has synthetic-looking edges.",
        stance="refute",
        evidence_ids=[evidence.evidence_id],
        source_family_ids=[evidence.source_family],
    )

    assessment = assess_fact(
        fact,
        [finding],
        {evidence.evidence_id: evidence},
        all_fact_evidence=[evidence],
    )

    assert assessment.status == "active"
    assert assessment.refute.score == 0.0


def test_task_exhausts_after_five_attempts_without_resolution() -> None:
    case, state = _runtime_state()
    task = next(
        item
        for item in state.tasks
        if any(
            fact_id in state.decisive_fact_ids
            for fact_id in item.fact_ids
        )
    )
    for index in range(5):
        record_tool_observation(
            state,
            _step(
                task_id=task.task_id,
                call_id=f"call-empty-{index}",
                tool_name="text_search",
                result=(
                    '{"status":"success","queries":['
                    '{"query":"controlled","results":[]}]}'
                ),
            ),
            image_sha256=case.image_sha256,
        )

    assert task.attempt_count == 5
    assert task.status == "exhausted"


def test_conflict_requires_discriminating_evidence_before_resolution() -> None:
    fact = VisualFact(
        fact_id="fact-conflict",
        kind="attribute",
        statement="The input image visibly contains the marked object.",
        subject_entity_id="entity-1",
        predicate="visible_in",
        status="active",
        basis_ids=["entity-1"],
        decision_relevance="decisive",
        origin=FactOrigin(type="input_image", origin_ids=["entity-1"]),
    )
    common = {
        "task_id": "task-conflict",
        "fact_ids": [fact.fact_id],
        "function_call_id": "call-conflict",
        "tool_name": "crop_and_inspect",
        "evidence_kind": "image_region",
        "source_url": "",
        "source_class": "visual",
        "exact_text": "A direct pixel observation.",
        "image_region": [0.0, 0.0, 1.0, 1.0],
        "artifact_sha256": "e" * 64,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "quality": "moderate",
        "directness": "direct",
        "claim_binding": "pixel_observation",
    }
    support = InvestigationEvidence(
        evidence_id="evidence-support",
        source_family="visual:support",
        stance="support",
        **common,
    )
    refute = InvestigationEvidence(
        evidence_id="evidence-refute",
        source_family="visual:refute",
        stance="refute",
        **common,
    )
    findings = [
        Finding(
            finding_id="finding-support",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="Pixel evidence supports the object.",
            stance="support",
            evidence_ids=[support.evidence_id],
            source_family_ids=[support.source_family],
        ),
        Finding(
            finding_id="finding-refute",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="Pixel evidence refutes the object.",
            stance="refute",
            evidence_ids=[refute.evidence_id],
            source_family_ids=[refute.source_family],
        ),
    ]

    tied = assess_fact(
        fact,
        findings,
        {
            support.evidence_id: support,
            refute.evidence_id: refute,
        },
        all_fact_evidence=[support, refute],
    )

    assert tied.status == "conflicted"
    assert (
        tied.conflict_resolution
        == "needs_discriminating_evidence"
    )

    decisive_refute = refute.model_copy(
        update={
            "evidence_id": "evidence-refute-official",
            "source_family": "domain:official.example",
            "source_class": "official",
            "quality": "strong",
        }
    )
    findings.append(
        Finding(
            finding_id="finding-refute-official",
            task_id="task-conflict",
            fact_ids=[fact.fact_id],
            statement="An original direct observation refutes the object.",
            stance="refute",
            evidence_ids=[decisive_refute.evidence_id],
            source_family_ids=[decisive_refute.source_family],
            quality="decisive",
        )
    )
    resolved = assess_fact(
        fact,
        findings,
        {
            support.evidence_id: support,
            refute.evidence_id: refute,
            decisive_refute.evidence_id: decisive_refute,
        },
        all_fact_evidence=[support, refute, decisive_refute],
    )

    assert resolved.status == "refuted"
    assert resolved.conflict_resolution == "refute_wins"
