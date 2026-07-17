from __future__ import annotations

import json
from pathlib import Path

from src.trajectory.exporter import export_policy_examples
from src.trajectory.perception_exporter import (
    PERCEPTION_INSTRUCTION,
    export_perception_example,
 )
from src.trajectory.scoring import score_process_trace
from test_image_only_trajectory import (
    test_scripted_image_only_complete_trajectory,
)


def _trace(tmp_path: Path) -> dict:
    test_scripted_image_only_complete_trajectory(tmp_path)
    return json.loads(
        (tmp_path / "traces" / "case_scripted_v3.json").read_text(
            encoding="utf-8"
        )
    )


def test_exporter_uses_actual_policy_boundaries_and_aligned_masks(
    tmp_path: Path,
) -> None:
    examples = export_policy_examples(_trace(tmp_path))

    assert [item.example_type for item in examples] == [
        "planning",
        "react",
        "react",
        "evidence_decision",
        "react",
        "evidence_decision",
        "judgment",
    ]
    assert all(item.tokenizer_id == "utf8-byte-v1" for item in examples)
    assert all(
        len(item.policy_action_token_ids)
        == len(item.policy_action_loss_mask)
        for item in examples
    )
    assert all(set(item.policy_action_loss_mask) == {1} for item in examples)
    react_actions = [
        item.policy_action
        for item in examples
        if item.example_type == "react"
    ]
    assert all(item["type"] == "tool_call" for item in react_actions)
    assert set(react_actions[0]["arguments"]) == {"question_id"}
    assert set(react_actions[1]["arguments"]) == {
        "question_id",
        "reference_url",
        "source_page_url",
        "focus",
    }
    assert set(react_actions[2]["arguments"]) == {
        "question_id",
        "url",
        "goal",
    }
    assert "__claim_text" not in json.dumps(react_actions)
    assert examples[-1].terminated is True


def test_fatal_boundary_is_zero_masked(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    trace["termination"] = "error"
    trace["state"]["termination"] = "error"

    examples = export_policy_examples(trace)

    assert examples[-1].fatal_boundary is True
    assert set(examples[-1].policy_action_loss_mask) == {0}
    assert all(
        set(item.policy_action_loss_mask) == {1}
        for item in examples[:-1]
    )


def test_exporter_rejects_evaluator_private_leak(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    step = next(
        item
        for item in trace["state"]["all_steps"]
        if item.get("metadata", {}).get("policy_input")
    )
    step["metadata"]["policy_input"]["evaluation_gold"] = {"verdict": "real"}

    try:
        export_policy_examples(trace)
    except ValueError as exc:
        assert "evaluation_gold" in str(exc)
    else:
        raise AssertionError("private evaluator data must be rejected")


def test_perception_exporter_keeps_only_public_image_and_report(
    tmp_path: Path,
 ) -> None:
    trace = _trace(tmp_path)
    example = export_perception_example(
        trace,
        source_metadata={
            "source_run_id": "run-1",
            "runtime_commit": "a" * 40,
            "release_id": "release-1",
            "runtime_contract_version": "ifv-image-only-runtime-v1",
        },
    )

    assert example.episode_id == trace["image_id"]
    assert example.instruction == PERCEPTION_INSTRUCTION
    assert example.image_sha256 == (
        trace["state"]["runtime_case"]["image_sha256"]
    )
    assert example.perception_report == trace["state"]["perception"]
    assert "investigation_state" not in example.model_dump()
    assert "judgment" not in example.model_dump()


def test_process_scorer_matches_fact_evidence_and_basis(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    investigation = trace["state"]["investigation_state"]
    runtime_fact = next(
        fact
        for fact in investigation["facts"]
        if fact["fact_id"] in trace["verdict_basis"]["fact_ids"]
    )
    runtime_evidence = next(
        evidence
        for evidence in investigation["evidence"]
        if evidence["evidence_id"] in trace["verdict_basis"]["evidence_ids"]
    )
    gold = {
        "case_id": trace["image_id"],
        "factual_status": "supported",
        "decisive_facts": [
            {
                "fact_id": "gold-fact-1",
                "kind": runtime_fact["kind"],
                "statement": runtime_fact["statement"],
                "expected_status": "supported",
                "visual_anchor": {"type": "scene"},
                "acceptable_evidence": [
                    {
                        "canonical_url": runtime_evidence["source_url"],
                        "exact_span": runtime_evidence["exact_text"],
                        "stance": runtime_evidence["stance"],
                        "source_family": runtime_evidence["source_family"],
                        "artifact_sha256": (
                            runtime_evidence["artifact_sha256"]
                        ),
                    }
                ],
            }
        ],
    }
    trace["verdict_basis"]["fact_ids"] = [runtime_fact["fact_id"]]
    trace["verdict_basis"]["finding_ids"] = [
        finding["finding_id"]
        for finding in investigation["findings"]
        if runtime_fact["fact_id"] in finding["fact_ids"]
    ]
    trace["verdict_basis"]["evidence_ids"] = [
        runtime_evidence["evidence_id"]
    ]

    metrics, score = score_process_trace(
        trace,
        gold,
        score_metadata={
            "process_reference_protocol": {"sha256": "a" * 64},
            "evaluation_gold": {"sha256": "b" * 64},
        },
    )

    assert metrics["decisive_fact_alignment"] == 1.0
    assert metrics["evidence_to_vision_bridge_completion"] == 1.0
    assert metrics["verdict_basis_alignment"] == 1.0
    assert metrics["score_metadata"]["evaluation_gold"]["sha256"] == "b" * 64
    assert score["components"]["result_reward"] == 1.0
    assert score["total"] > 4.0


def test_process_scorer_accepts_semantic_refutation_with_text_sufficient_bridge() -> None:
    fact_id = "fact-monarch-antarctic-relation"
    task_id = "task-monarch-range"
    evidence_id = "evidence-butterflies-except-antarctica"
    finding_id = "finding-monarch-refuted"
    call_id = "call-monarch-source"
    runtime_fact = {
        "fact_id": fact_id,
        "kind": "relation",
        "statement": (
            "The monarch butterfly naturally co-occurs with a penguin in an "
            "Antarctic environment."
        ),
        "predicate": "depicts_relation",
        "status": "refuted",
    }
    evidence = {
        "evidence_id": evidence_id,
        "task_id": task_id,
        "fact_ids": [fact_id],
        "function_call_id": call_id,
        "tool_name": "visit",
        "source_url": "https://zoo.example/butterfly",
        "source_family": "domain:zoo.example",
        "source_class": "unknown",
        "exact_text": (
            "Butterflies are found on every continent except Antarctica."
        ),
        "stance": "refute",
        "quality": "moderate",
        "directness": "indirect",
        "claim_binding": "source_assertion",
        "risk_flags": [],
    }
    trace = {
        "image_id": "case-monarch-semantic-match",
        "verdict": "fake",
        "termination": "success",
        "verdict_basis": {
            "fact_ids": [fact_id],
            "finding_ids": [finding_id],
            "evidence_ids": [evidence_id],
        },
        "state": {
            "all_steps": [
                {
                    "stage": "image_only_investigation",
                    "action_type": "tool_call",
                    "tool_name": "visit",
                    "tool_args": {
                        "__question_id": task_id,
                        "url": ["https://zoo.example/butterfly"],
                    },
                    "tool_result": json.dumps({"status": "success"}),
                    "metadata": {
                        "function_call_id": call_id,
                        "investigation_state_update": {
                            "created_evidence_ids": [evidence_id],
                            "created_finding_ids": [finding_id],
                            "fact_statuses": {fact_id: "refuted"},
                        },
                    },
                }
            ],
            "investigation_state": {
                "facts": [runtime_fact],
                "decisive_fact_ids": [fact_id],
                "tasks": [
                    {
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "suggested_tools": ["text_search", "visit"],
                    }
                ],
                "evidence": [evidence],
                "findings": [
                    {
                        "finding_id": finding_id,
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "evidence_ids": [evidence_id],
                    }
                ],
                "evidence_decisions": [
                    {
                        "output": {
                            "active_fact_id": fact_id,
                            "assessment": "refuted",
                            "selected_evidence_ids": [evidence_id],
                            "binding_requirement": "text_sufficient",
                        }
                    }
                ],
                "coverage_audits": [
                    {
                        "facts": [
                            {
                                "fact_id": fact_id,
                                "status": "refuted",
                                "winning_evidence_ids": [evidence_id],
                                "conflict_resolution": "refute_wins",
                            }
                        ]
                    }
                ],
                "action_count": 1,
            },
        },
    }
    gold = {
        "case_id": "case-monarch-semantic-match",
        "factual_status": "refuted",
        "decisive_facts": [
            {
                "fact_id": "gold-monarch-antarctica",
                "kind": "relation",
                "statement": (
                    "Millions of monarch butterflies migrate to Antarctica "
                    "during the winter."
                ),
                "expected_status": "refuted",
                "visual_anchor": {
                    "type": "scene_or_text",
                    "description": "The butterflies are in Antarctica.",
                },
            }
        ],
    }

    metrics, score = score_process_trace(trace, gold)

    assert metrics["decisive_fact_alignment"] == 1.0
    assert metrics["evidence_to_vision_bridge_completion"] == 1.0
    assert metrics["verdict_basis_alignment"] == 1.0
    assert score["training_eligible"] is True


def test_process_scorer_uses_only_basis_same_capture_source_context() -> None:
    fact_id = "fact-visible-store-event"
    task_id = "task-visible-store-event"
    evidence_id = "evidence-same-capture"
    finding_id = "finding-store-event"
    reference_url = "https://images.example.org/store-opening.jpg"
    source_url = (
        "https://company.example/news/2023/store-tysons-corner-reopens"
    )
    trace = {
        "image_id": "case-source-context",
        "verdict": "real",
        "termination": "success",
        "verdict_basis": {
            "fact_ids": [fact_id],
            "finding_ids": [finding_id],
            "evidence_ids": [evidence_id],
        },
        "state": {
            "all_steps": [
                {
                    "stage": "image_only_investigation",
                    "action_type": "tool_call",
                    "tool_name": "compare_with_reference",
                    "tool_args": {
                        "__question_id": task_id,
                        "reference_url": reference_url,
                    },
                    "tool_result": json.dumps({"status": "success"}),
                    "metadata": {
                        "function_call_id": "call-same-capture",
                        "investigation_state_update": {
                            "created_evidence_ids": [evidence_id],
                            "created_finding_ids": [finding_id],
                            "fact_statuses": {fact_id: "supported"},
                        },
                    },
                }
            ],
            "investigation_state": {
                "facts": [
                    {
                        "fact_id": fact_id,
                        "kind": "relation",
                        "statement": (
                            "The image depicts an Apple Store reopening event "
                            "with employees greeting first customers."
                        ),
                        "predicate": "depicts_event",
                        "status": "supported",
                    }
                ],
                "decisive_fact_ids": [fact_id],
                "tasks": [
                    {
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "suggested_tools": [
                            "reverse_image_search",
                            "compare_with_reference",
                        ],
                    }
                ],
                "discoveries": [
                    {
                        "discovery_id": "discovery-selected-source",
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "candidate_url": source_url,
                        "reference_image_url": reference_url,
                        "title": (
                            "Apple Tysons Corner reopens in Virginia in 2023"
                        ),
                        "snippet": "",
                    },
                    {
                        "discovery_id": "discovery-unselected-noise",
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "candidate_url": (
                            "https://noise.example/unrelated-regent-street"
                        ),
                        "reference_image_url": (
                            "https://noise.example/unselected.jpg"
                        ),
                        "title": "Apple Regent Street opening in London",
                        "snippet": "",
                    },
                ],
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "function_call_id": "call-same-capture",
                        "tool_name": "compare_with_reference",
                        "source_url": reference_url,
                        "source_family": "domain:images.example.org",
                        "source_class": "unknown",
                        "exact_text": "The images are the same original capture.",
                        "stance": "support",
                        "quality": "moderate",
                        "directness": "direct",
                        "claim_binding": "same_capture",
                        "same_capture_or_near_duplicate": True,
                        "likely_different_original_capture": False,
                        "risk_flags": [],
                    }
                ],
                "findings": [
                    {
                        "finding_id": finding_id,
                        "task_id": task_id,
                        "fact_ids": [fact_id],
                        "evidence_ids": [evidence_id],
                    }
                ],
                "coverage_audits": [
                    {
                        "facts": [
                            {
                                "fact_id": fact_id,
                                "status": "supported",
                                "winning_evidence_ids": [evidence_id],
                                "conflict_resolution": "support_wins",
                            }
                        ]
                    }
                ],
                "action_count": 1,
            },
        },
    }
    gold = {
        "case_id": "case-source-context",
        "factual_status": "supported",
        "decisive_facts": [
            {
                "fact_id": "gold-tysons-corner",
                "kind": "relation",
                "statement": (
                    "This photograph shows Apple Tysons Corner in Virginia "
                    "during its 2023 reopening."
                ),
                "expected_status": "supported",
                "visual_anchor": {"type": "visible_anchors"},
            }
        ],
    }

    metrics, score = score_process_trace(trace, gold)

    assert metrics["decisive_fact_alignment"] == 1.0
    assert metrics["evidence_to_vision_bridge_completion"] == 1.0
    assert metrics["verdict_basis_alignment"] == 1.0
    assert score["training_eligible"] is True

    trace["verdict_basis"]["evidence_ids"] = []
    without_selected_bridge, _ = score_process_trace(trace, gold)
    assert without_selected_bridge["decisive_fact_alignment"] == 0.0
