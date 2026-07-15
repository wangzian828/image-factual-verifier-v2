from __future__ import annotations

from src.orchestrator.route_policy import (
    routes_semantically_equivalent,
    semantic_duplicate_count,
)
from src.orchestrator.stage_runner import StageRunner, StageStep


def test_search_query_paraphrase_is_a_semantic_duplicate() -> None:
    left = {
        "__question_id": "task-1",
        "queries": ["Apollo 14 Moon Tree ceremony Johnson Space Center"],
    }
    right = {
        "__question_id": "task-1",
        "queries": ["Johnson Space Center Apollo 14 moon trees ceremony"],
    }

    assert routes_semantically_equivalent(
        "text_search",
        left,
        "text_search",
        right,
    )


def test_visit_tracking_variants_are_a_semantic_duplicate() -> None:
    left = {
        "__question_id": "task-1",
        "url": "https://www.nasa.gov/event/?utm_source=test",
        "goal": "Where did the ceremony take place?",
    }
    right = {
        "__question_id": "task-1",
        "url": "https://www.nasa.gov/event",
        "goal": "Determine the place of the ceremony.",
    }

    assert routes_semantically_equivalent(
        "visit",
        left,
        "visit",
        right,
    )


def test_same_reference_for_a_different_task_is_still_deduplicated() -> None:
    left = {
        "__question_id": "task-1",
        "reference_url": "https://example.org/reference.jpg",
    }
    right = {
        "__question_id": "task-2",
        "reference_url": "https://example.org/reference.jpg",
    }

    assert routes_semantically_equivalent(
        "compare_with_reference",
        left,
        "compare_with_reference",
        right,
    )
    assert (
        semantic_duplicate_count(
            [
                ("compare_with_reference", left),
                ("compare_with_reference", right),
            ]
        )
        == 1
    )


def test_same_search_goal_is_deduplicated_across_tasks() -> None:
    left = {
        "__question_id": "task-1",
        "queries": ['"学生用AI写，学校用AI查" @dingzhen47'],
        "goal": "Find the original post matching the visible account and text.",
    }
    right = {
        "__question_id": "task-2",
        "queries": ['@dingzhen47 "学生用AI写，学校用AI查"'],
        "goal": "Locate the original post matching its text and account.",
    }

    assert routes_semantically_equivalent(
        "text_search",
        left,
        "text_search",
        right,
    )


def test_same_search_query_with_materially_different_goal_is_allowed() -> None:
    left = {
        "__question_id": "task-1",
        "queries": ["Apple Tysons Corner reopening"],
        "goal": "Identify the store and reopening date.",
    }
    right = {
        "__question_id": "task-2",
        "queries": ["Apple Tysons Corner reopening"],
        "goal": "Find visible image manipulation artifacts.",
    }

    assert not routes_semantically_equivalent(
        "text_search",
        left,
        "text_search",
        right,
    )


def test_stage_runner_blocks_resolved_tasks_and_prior_segment_duplicates() -> None:
    active = {"task-1": False}
    runner = StageRunner(
        llm=object(),
        system_prompt="test",
        tools=[],
        stage_name="verification",
        prior_steps=[
            StageStep(
                action_type="tool_call",
                tool_name="text_search",
                tool_args={
                    "__question_id": "task-1",
                    "queries": ["NOAA Henry Bigelow vessel"],
                },
            )
        ],
        question_is_active=lambda task_id: active.get(task_id, False),
    )
    runner.active_question_ids = ["task-1"]

    assert "already resolved" in runner._question_id_error(
        {"__question_id": "task-1"}
    )
    assert runner._has_duplicate_tool_call(
        [],
        "text_search",
        {
            "__question_id": "task-1",
            "queries": ["Henry Bigelow NOAA vessel"],
        },
    )
