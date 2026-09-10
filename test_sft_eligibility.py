from __future__ import annotations

import asyncio
from argparse import Namespace
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.orchestrator.llm_backend import LLMResponse
from src.orchestrator.react_runtime import REACT_RUNTIME_SCHEMA_VERSION
from src.eval import score_sft_eligibility
from src.eval.score_sft_eligibility import (
    _api_key_from_env,
    _default_storage_dir,
    _jsonl_index,
)
from src.trajectory.sft_eligibility import (
    SFT_ELIGIBILITY_SYSTEM_PROMPT,
    SFTEligibilityJudge,
    SFTEligibilityJudgment,
    build_sft_eligibility_artifact,
    build_sft_eligibility_input,
    build_sft_target,
    sft_eligibility_metrics,
    sft_eligibility_passes,
)
from scripts.trajectory.stage_accepted_teacher_release import (
    _eligible as stage_eligible,
    sft_candidate_rank,
)


def test_sft_judge_uses_explicit_api_key_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVATE_JUDGE_KEY", "judge-secret")

    assert _api_key_from_env("PRIVATE_JUDGE_KEY") == "judge-secret"
    assert _api_key_from_env(None) is None


def test_sft_judge_rejects_empty_explicit_api_key_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_JUDGE_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_JUDGE_KEY"):
        _api_key_from_env("MISSING_JUDGE_KEY")


def _trace(*, verdict: str = "fake") -> dict[str, Any]:
    return {
        "image_id": "case-1--teacher-r000",
        "verdict": verdict,
        "termination": "success",
        "decision_policy_version": "unified-react-v1",
        "verdict_basis": {
            "schema_version": "ifv-verdict-basis-raw-history-v1",
            "decision_mode": "raw_history",
            "observation_ids": ["obs-visual", "obs-search", "obs-visit"],
            "verdict_target": "The displayed winner is A, but the official winner is B.",
            "unresolved_gaps": [],
        },
        "judgment": {
            "verdict": verdict,
            "verdict_observation_ids": ["obs-visit"],
            "fact_check_report": {
                "summary": "The official result names B, not A."
            },
        },
        "state": {
            "image_id": "case-1--teacher-r000",
            "runtime_case": {
                "case_id": "case-1",
                "image_sha256": "a" * 64,
            },
            "investigation_state": {
                "schema_version": REACT_RUNTIME_SCHEMA_VERSION,
                "objective": "Verify the pictured event and its relationship.",
                "open_questions": [],
            },
            "all_steps": [
                {
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "tool_name": "focused_visual_inspection",
                    "thought": "Inspect the displayed winner before checking the event.",
                    "tool_args": {"question": "Who is displayed as the winner?"},
                    "tool_result": json.dumps(
                        {
                            "status": "success",
                            "observation": "The scoreboard displays A as the winner.",
                        }
                    ),
                    "metadata": {
                        "function_call_id": "obs-visual",
                        "tool_success": True,
                    },
                },
                {
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "tool_name": "text_search",
                    "thought": "Search for the official result of the named final.",
                    "tool_args": {
                        "queries": ["official 2026 final result"],
                        "goal": "Find the official final result.",
                    },
                    "tool_result": json.dumps(
                        {
                            "status": "success",
                            "queries": [
                                {
                                    "query": "official 2026 final result",
                                    "results": [
                                        {"url": "https://example.org/final"},
                                        {"url": "https://example.org/news"},
                                    ],
                                }
                            ],
                        }
                    ),
                    "metadata": {
                        "function_call_id": "obs-search",
                        "tool_success": True,
                    },
                },
                {
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "tool_name": "visit",
                    "thought": "Read the official result page for the winner.",
                    "tool_args": {
                        "url": ["https://example.org/final"],
                        "question": "Who won the 2026 final?",
                    },
                    "tool_result": json.dumps(
                        {
                            "status": "success",
                            "url": "https://example.org/final",
                            "evidence_records": [
                                {
                                    "url": "https://example.org/final",
                                    "evidence": "B won the 2026 final.",
                                    "directness": "direct",
                                }
                            ],
                        }
                    ),
                    "metadata": {
                        "function_call_id": "obs-visit",
                        "tool_success": True,
                    },
                },
                {
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "tool_name": "text_search",
                    "thought": "A later query failed at the provider boundary.",
                    "tool_args": {"queries": ["retry official result"]},
                    "tool_result": json.dumps(
                        {"status": "error", "error": "temporary provider failure"}
                    ),
                    "metadata": {
                        "function_call_id": "obs-error",
                        "tool_success": False,
                    },
                },
            ],
        },
    }


def _gold() -> dict[str, Any]:
    return {
        "schema_version": "ifv-route-aware-human-review-candidate-v1",
        "candidate_id": "case-1",
        "factual_status": "refuted",
        "target_claim": "The image claims A won the 2026 final.",
        "claim_atom": {
            "subject": "2026 final",
            "event_or_context": "2026 final",
            "relation_slot": "winner",
            "depicted_value": "A",
        },
        "decisive_visual_atom": "The scoreboard displays A as the winner.",
        "visible_scene_facts": ["A is displayed as the winner."],
        "evidence": {
            "binding": {
                "target_claim": "A won the 2026 final.",
                "source_evidence_span": "B won the 2026 final.",
                "source_url": "https://example.org/final",
                "match": "same_event",
            },
            "used_exact_spans": [
                {"exact_span": "B won the 2026 final.", "role": "event_fact"}
            ],
        },
    }


def _judgment(**updates: Any) -> SFTEligibilityJudgment:
    payload: Dict[str, Any] = {
        "target_scope": "direct_target",
        "decision_support": "supports_fake",
        "retrieval_quality": "effective",
        "decisive_observation_ids": ["obs-visit"],
        "supporting_observation_ids": [],
        "overclaiming": "none",
        "boundary_assessment": "respected",
        "trajectory_conduct": "clean",
        "confidence": 0.72,
        "explanation": "The retained source observation directly establishes the incorrect winner.",
    }
    payload.update(updates)
    return SFTEligibilityJudgment.model_validate(payload)


def _packet() -> dict[str, Any]:
    packet = build_sft_eligibility_input(_trace(), _gold())
    packet["image"]["available_to_judge"] = True
    return packet


def test_default_storage_isolated_by_eligibility_output_version(tmp_path: Any) -> None:
    run_dir = tmp_path / "data" / "runs" / "eval" / "teacher-run"
    v2_dir = run_dir / "sft-eligibility-v2"
    v3_dir = run_dir / "sft-eligibility-v3"

    v2_storage = _default_storage_dir(run_dir, v2_dir)
    v3_storage = _default_storage_dir(run_dir, v3_dir)

    assert v2_storage != v3_storage
    assert v2_storage.name == "teacher-run--sft-eligibility-v2"
    assert v3_storage.name == "teacher-run--sft-eligibility-v3"


def test_private_gold_index_ignores_ambiguous_legacy_aliases(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    rows = [
        {
            "case_id": "canonical-a",
            "candidate_id": "reused-candidate",
            "assignment_id": "reused-assignment",
        },
        {
            "case_id": "canonical-b",
            "candidate_id": "reused-candidate",
            "assignment_id": "reused-assignment",
        },
        {
            "case_id": "canonical-c",
            "candidate_id": "unique-candidate",
            "assignment_id": "unique-assignment",
        },
    ]
    gold_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    indexed = _jsonl_index(gold_path)

    assert indexed["canonical-a"]["case_id"] == "canonical-a"
    assert indexed["canonical-b"]["case_id"] == "canonical-b"
    assert "reused-candidate" not in indexed
    assert "reused-assignment" not in indexed
    assert indexed["unique-candidate"]["case_id"] == "canonical-c"
    assert indexed["unique-assignment"]["case_id"] == "canonical-c"


def test_sft_audit_requeues_only_failed_trace(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class _AuditReport:
        def failures(self, *, strict_scheduler: bool) -> list[dict[str, Any]]:
            assert strict_scheduler is True
            return []

        def warnings(self, *, strict_scheduler: bool) -> list[dict[str, Any]]:
            assert strict_scheduler is True
            return []

    class _Backend:
        async def aclose(self) -> None:
            return None

    attempts: Dict[str, int] = {}

    class _Judge:
        generation_identity = "test-generation"

        async def judge(
            self,
            packet: Dict[str, Any],
            *,
            image_path: Path | None = None,
        ) -> tuple[SFTEligibilityJudgment, Dict[str, Any]]:
            assert image_path is not None
            case_id = str(packet["case_id"])
            attempts[case_id] = attempts.get(case_id, 0) + 1
            if case_id == "case-2" and attempts[case_id] == 1:
                raise RuntimeError("temporary Gemini 500")
            return _judgment(), {"calls": []}

    run_dir = tmp_path / "runs" / "eval" / "teacher"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"not-read-by-the-mocked-judge")

    gold_rows: list[dict[str, Any]] = []
    for case_id in ("case-1", "case-2"):
        trace = _trace()
        trace["image_id"] = f"{case_id}--teacher-r000"
        trace["state"]["image_id"] = trace["image_id"]
        trace["state"]["runtime_case"]["case_id"] = case_id
        (trace_dir / f"{case_id}.json").write_text(
            json.dumps(trace),
            encoding="utf-8",
        )
        gold = _gold()
        gold["candidate_id"] = case_id
        gold_rows.append(gold)

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "".join(json.dumps(row) + "\n" for row in gold_rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eligibility"

    monkeypatch.setattr(score_sft_eligibility, "APIBackend", lambda **_: _Backend())
    monkeypatch.setattr(
        score_sft_eligibility,
        "SFTEligibilityJudge",
        lambda *_args, **_kwargs: _Judge(),
    )
    monkeypatch.setattr(
        score_sft_eligibility,
        "_resolve_image_path",
        lambda *_args, **_kwargs: image_path,
    )
    monkeypatch.setattr(score_sft_eligibility, "audit_trace", lambda *_args, **_kwargs: _AuditReport())
    monkeypatch.setattr(
        score_sft_eligibility,
        "stage_release",
        lambda *_args, **_kwargs: {
            "accepted_case_count": 2,
            "rejected_case_count": 0,
        },
    )

    summary = asyncio.run(
        score_sft_eligibility._run(
            Namespace(
                run_dir=run_dir,
                gold=gold_path,
                output_dir=output_dir,
                cache_dir=None,
                image_root=None,
                storage_dir=tmp_path / "storage",
                provider="gemini",
                model="gemini-3.7-flash",
                max_tokens=4096,
                provider_retries=0,
                trace_retries=1,
                trace_retry_delay=0.0,
                timeout=180.0,
                concurrency=2,
                force=False,
            )
        )
    )

    assert summary["episode_count"] == 2
    assert attempts == {"case-1": 1, "case-2": 2}
    errors = [
        json.loads(line)
        for line in (output_dir / "sft_eligibility_errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(errors) == 1
    assert errors[0]["case_id"] == "case-2"
    assert errors[0]["retry_round"] == 0


class _MockBackend:
    provider = "mock"
    model_name = "frozen-mock"

    def __init__(self, judgment: SFTEligibilityJudgment | None = None) -> None:
        self.judgment = judgment or _judgment()
        self.messages: List[List[Dict[str, Any]]] = []

    async def get_response(
        self, messages: List[Dict[str, Any]], **_: Any
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(
            text=self.judgment.model_dump_json(),
            prompt_tokens=120,
            completion_tokens=40,
        )


def test_target_adapter_uses_one_generic_image_fact_shape() -> None:
    target = build_sft_target(_gold())

    assert target["schema_version"] == "ifv-sft-target-v2"
    assert target["expected_verdict"] == "fake"
    assert target["image_fact"]["statement"].startswith("The image claims")
    assert target["image_fact"]["visible_anchors"]
    assert target["reference_facts"][0]["evidence_text"]


def test_packet_includes_compact_retrieval_and_rejection_history() -> None:
    trace = _trace()
    trace["state"]["all_steps"] = [
        {
            "action_type": "tool_call",
            "tool_name": "text_search",
            "tool_args": {
                "queries": "official 2026 final result",
                "goal": "Find the official final result.",
            },
            "tool_result": json.dumps(
                {
                    "status": "success",
                    "queries": [
                        {
                            "results": [
                                {"url": "https://example.org/final"},
                                {"url": "https://example.org/news"},
                            ]
                        }
                    ],
                }
            ),
        },
        {
            "action_type": "tool_call",
            "tool_name": "visit",
            "tool_args": {
                "url": ["https://example.org/final"],
                "goal": "Read the official result.",
            },
            "tool_result": json.dumps(
                {
                    "status": "success",
                    "url": "https://example.org/final",
                    "evidence": "B won the 2026 final.",
                }
            ),
        },
        {
            "action_type": "format_error",
            "tool_name": "text_search",
            "tool_args": {
                "queries": [],
                "goal": "Find an official result.",
            },
            "metadata": {"error_class": "protocol_error"},
            "tool_result": json.dumps(
                {"status": "error", "error": "duplicate tool call"}
            ),
        },
    ]

    packet = build_sft_eligibility_input(trace, _gold())

    assert packet["schema_version"] == "ifv-sft-eligibility-input-v12"
    assert packet["candidate"]["retrieval_history"] == [
        {
            "tool": "text_search",
            "goal": "Find the official final result.",
            "queries": ["official 2026 final result"],
            "status": "success",
            "result_count": 2,
            "search_error": "",
        },
        {
            "tool": "visit",
            "goal": "Read the official result.",
            "source_urls": ["https://example.org/final"],
            "status": "success",
            "page_text_extracted": True,
            "fetch_error": "",
        },
    ]
    assert packet["candidate"]["rejection_history"] == {
        "count": 1,
        "by_action_type": {"format_error": 1},
        "by_stage": {"unknown": 1},
        "events": [
            {
                "step_index": 2,
                "stage": "unknown",
                "action_type": "format_error",
                "tool": "text_search",
                "error_class": "protocol_error",
                "reason": "duplicate tool call",
                "goal": "Find an official result.",
                "queries": [],
            }
        ],
    }


def test_target_adapter_accepts_web_chain_without_claim_atom() -> None:
    row = {
        "candidate_id": "web-1",
        "factual_status": "refuted",
        "target_claim": "The image shows a real event.",
        "event_identity": "the claimed event",
        "relation_identity": "the image depicts the event",
        "evidence": {
            "binding": {
                "chain": [
                    {
                        "binding": {
                            "target_claim": "The image shows a real event.",
                            "source_url": "https://example.org/check",
                        },
                        "exact_span": "The image was digitally altered.",
                        "role": "contradiction",
                    }
                ]
            }
        },
    }

    target = build_sft_target(row)

    assert target["expected_verdict"] == "fake"
    assert target["image_fact"]["relation"] == (
        "the image depicts the event"
    )
    assert target["reference_facts"][0]["role"] == "contradiction"
    assert target["reference_facts"][0]["evidence_text"] == (
        "The image was digitally altered."
    )


def test_packet_exposes_raw_observations_and_success_statuses() -> None:
    packet = _packet()
    observations = {
        row["observation_id"]: row
        for row in packet["candidate"]["raw_observations"]
    }

    assert set(observations) == {
        "obs-visual",
        "obs-search",
        "obs-visit",
        "obs-error",
    }
    assert packet["candidate"]["successful_observation_ids"] == [
        "obs-visual",
        "obs-search",
        "obs-visit",
    ]
    assert packet["candidate"]["unsuccessful_observation_ids"] == ["obs-error"]
    assert observations["obs-error"]["tool_success"] is False


def test_unified_packet_exposes_ordered_react_actions_and_image_search_candidates() -> None:
    trace = _trace()
    trace["verdict_basis"] = {
        "objective": "Verify the pictured event and its relationship.",
        "evidence_ids": [],
        "open_questions": ["Whether the pictured event occurred as shown."],
    }
    trace["state"]["investigation_state"] = {
        "schema_version": REACT_RUNTIME_SCHEMA_VERSION,
        "objective": "Verify the pictured event and its relationship.",
        "visual_memory": {
            "scene_description": "A person stands beside a marked vehicle.",
            "entities": [{"name": "marked vehicle"}],
            "relations": [{"description": "The person stands beside the vehicle."}],
            "text_regions": [{"text": "EVENT 2026"}],
        },
        "discoveries": [
            {
                "discovery_id": "discovery-image-1",
                "tool_name": "text_image_search",
                "candidate_url": "https://example.org/event-page",
                "reference_image_url": "https://cdn.example.org/event.jpg",
                "title": "Event 2026 image",
                "snippet": "A page containing a related event photograph.",
                "source_query": "EVENT 2026 marked vehicle",
                "candidate_status": "unverified",
                "match_status": "unverified",
            }
        ],
        "evidence": [],
        "open_questions": ["Whether the pictured event occurred as shown."],
    }
    trace["state"]["all_steps"] = [
        {
            "stage": "unified_react",
            "action_type": "tool_call",
            "tool_name": "text_image_search",
            "thought": "The image contains a named event. I will find image-bearing pages for that event.",
            "tool_args": {
                "query": "EVENT 2026 marked vehicle",
                "investigation_progress": {
                    "status": "investigating",
                    "basis": "The event relationship remains unresolved.",
                },
            },
            "tool_result": json.dumps(
                {
                    "status": "success",
                    "query": "EVENT 2026 marked vehicle",
                    "observation_status": "has_results",
                    "results": [
                        {
                            "rank": 1,
                            "title": "Event 2026 image",
                            "url": "https://example.org/event-page",
                            "image_url": "https://cdn.example.org/event.jpg",
                            "snippet": "A related event photograph.",
                        }
                    ],
                    "candidate_page_urls": ["https://example.org/event-page"],
                    "reference_image_candidates": ["https://cdn.example.org/event.jpg"],
                }
            ),
            "metadata": {
                "function_call_id": "obs-image-search",
                "tool_success": True,
            },
        },
    ]

    packet = build_sft_eligibility_input(trace, _gold())

    assert packet["candidate"]["raw_observations"][0]["tool"] == (
        "text_image_search"
    )
    action = packet["candidate"]["raw_observations"][0]
    assert action["observation"]["results"][0]["image_url"].endswith(
        "event.jpg"
    )
    assert packet["candidate"]["runtime_mode"] == "raw_history"
    assert packet["candidate"]["raw_observation_ids"]
    assert packet["candidate"]["retrieval_history"][0]["result_count"] == 1
    assert packet["candidate"]["retrieval_history"][0]["candidate_image_count"] == 1


def test_unified_packet_keeps_nested_visit_evidence_records() -> None:
    trace = _trace()
    trace["state"]["investigation_state"] = {
        "schema_version": REACT_RUNTIME_SCHEMA_VERSION,
        "objective": "Verify the pictured event.",
        "visual_memory": {},
        "discoveries": [],
        "evidence": [],
        "open_questions": [],
    }
    trace["state"]["all_steps"] = [
        {
            "stage": "unified_react",
            "action_type": "tool_call",
            "tool_name": "visit",
            "tool_args": {
                "url": ["https://example.org/page"],
                "question": "What event does the page document?",
            },
            "tool_result": json.dumps(
                {
                    "status": "success",
                    "visits": [
                        {
                            "url": "https://example.org/page",
                            "evidence_records": [
                                {
                                    "url": "https://example.org/page",
                                    "evidence": "The page documents the event.",
                                    "evidence_context": "The surrounding paragraph names the date.",
                                    "stance": "support",
                                    "directness": "direct",
                                }
                            ],
                        }
                    ],
                }
            ),
            "metadata": {"tool_success": True},
        }
    ]

    packet = build_sft_eligibility_input(trace, _gold())

    action = packet["candidate"]["raw_observations"][0]
    assert action["observation"]["evidence_records"][0]["evidence"] == (
        "The page documents the event."
    )
    assert action["observation"]["visited_pages"][0]["record_count"] == 1


def test_decisive_subfact_can_pass_without_claim_relation_matching() -> None:
    packet = _packet()
    judgment = _judgment(
        target_scope="decisive_subfact",
        decision_support="supports_fake",
        decisive_observation_ids=["obs-visit"],
    )
    metrics = sft_eligibility_metrics(packet, judgment)

    assert metrics["verdict_correct"] is True
    assert metrics["fatal_errors"] == []
    assert sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=False,
        engineering_valid=True,
    )


def test_related_but_incomplete_is_rejected_without_calling_it_unrelated() -> None:
    metrics = sft_eligibility_metrics(
        _packet(),
        _judgment(target_scope="related_but_incomplete"),
    )

    assert metrics["target_scope"] == "related_but_incomplete"
    assert "key_target_condition_unchecked" in metrics["fatal_errors"]
    assert "unrelated_image_fact" not in metrics["fatal_errors"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_unrelated_fact_is_rejected_as_unrelated_image_fact() -> None:
    metrics = sft_eligibility_metrics(
        _packet(),
        _judgment(target_scope="unrelated_fact"),
    )

    assert "unrelated_image_fact" in metrics["fatal_errors"]
    assert "key_target_condition_unchecked" not in metrics["fatal_errors"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_candidate_rank_uses_new_target_scope() -> None:
    direct = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=_packet(),
        judgment=_judgment(target_scope="direct_target"),
        judge_audit={},
        strict_trace_audit_pass=True,
    )
    subfact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="c" * 64,
        packet=_packet(),
        judgment=_judgment(target_scope="decisive_subfact"),
        judge_audit={},
        strict_trace_audit_pass=True,
    )

    assert sft_candidate_rank(direct) > sft_candidate_rank(subfact)


def test_nonfatal_audit_warning_does_not_veto_sft() -> None:
    packet = _packet()
    artifact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=_judgment(),
        judge_audit={},
        strict_trace_audit_pass=False,
        strict_trace_audit_failures=[
            {
                "code": "BASIS_OMITS_VALID_EVIDENCE",
                "category": "hard",
                "message": "A valid Evidence row was not selected in basis.",
            }
        ],
    )

    assert artifact["gates"]["sft_eligibility_pass"] is True
    assert artifact["gates"]["fatal_audit_errors"] == []
    assert artifact["gates"]["audit_warnings"]


def test_recovered_protocol_error_does_not_block_sft_or_staging() -> None:
    packet = _packet()
    artifact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=_judgment(),
        judge_audit={},
        strict_trace_audit_pass=False,
        strict_trace_audit_failures=[
            {
                "code": "PROTOCOL_ERROR",
                "category": "protocol",
                "message": "A corrected intermediate policy attempt.",
            }
        ],
    )

    assert artifact["gates"]["sft_eligibility_pass"] is True
    assert artifact["gates"]["fatal_audit_errors"] == []
    assert artifact["gates"]["audit_warnings"]
    assert stage_eligible(
        _trace(),
        "b" * 64,
        artifact,
    )


def test_repeated_or_unresolved_conduct_blocks_sft() -> None:
    packet = _packet()
    for conduct in ("degraded_repetition", "unresolved"):
        artifact = build_sft_eligibility_artifact(
            trace=_trace(),
            trace_sha256="b" * 64,
            packet=packet,
            judgment=_judgment(trajectory_conduct=conduct),
            judge_audit={},
            strict_trace_audit_pass=True,
        )

        assert artifact["gates"]["sft_eligibility_pass"] is False
        assert f"trajectory_conduct_{conduct}" in artifact["metrics"][
            "fatal_errors"
        ]


def test_tool_argument_format_error_does_not_block_sft() -> None:
    packet = _packet()
    artifact = build_sft_eligibility_artifact(
        trace=_trace(),
        trace_sha256="b" * 64,
        packet=packet,
        judgment=_judgment(),
        judge_audit={},
        strict_trace_audit_pass=True,
        strict_trace_audit_warnings=[
            {
                "code": "TOOL_ARGUMENT_FORMAT_ERROR",
                "category": "format",
                "message": "text_search requires exactly one non-empty query.",
            }
        ],
    )

    assert artifact["gates"]["sft_eligibility_pass"] is True
    assert artifact["gates"]["fatal_audit_errors"] == []
    assert artifact["gates"]["audit_warnings"]


def test_invalid_selected_evidence_id_blocks_sft() -> None:
    packet = _packet()
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(decisive_observation_ids=["missing-observation"]),
    )

    assert metrics["invalid_judge_observation_ids"] == ["missing-observation"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_no_decisive_observation_blocks_sft() -> None:
    packet = _packet()
    metrics = sft_eligibility_metrics(
        packet,
        _judgment(
            decision_support="supporting_only",
            decisive_observation_ids=[],
            supporting_observation_ids=["obs-visual"],
        ),
    )

    assert "no_decisive_observation" in metrics["fatal_errors"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_poor_retrieval_quality_blocks_sft() -> None:
    metrics = sft_eligibility_metrics(
        _packet(),
        _judgment(retrieval_quality="poor"),
    )

    assert "poor_retrieval_quality" in metrics["fatal_errors"]
    assert not sft_eligibility_passes(
        metrics,
        strict_trace_audit_pass=True,
        engineering_valid=True,
    )


def test_out_of_range_diagnostic_confidence_is_normalized() -> None:
    metrics = sft_eligibility_metrics(
        _packet(),
        _judgment(confidence=4.5),
    )

    assert metrics["raw_confidence"] == 4.5
    assert metrics["confidence"] == 1.0
    assert "confidence_normalized_out_of_range" in metrics["warnings"]


def test_judge_is_one_post_rollout_call_and_does_not_request_human_review() -> None:
    async def run() -> None:
        packet = _packet()
        backend = _MockBackend()
        judgment, audit = await SFTEligibilityJudge(backend).judge(packet)

        assert judgment.target_scope == "direct_target"
        assert len(backend.messages) == 1
        request = json.dumps(backend.messages[0], ensure_ascii=False)
        assert "decision_paths" not in request
        assert "human_review" not in request

        artifact = build_sft_eligibility_artifact(
            trace=_trace(),
            trace_sha256="b" * 64,
            packet=packet,
            judgment=judgment,
            judge_audit=audit,
            strict_trace_audit_pass=True,
        )
        assert artifact["gates"]["sft_eligibility_pass"] is True
        assert "human_review_required" not in artifact["gates"]
        assert len(artifact["judge"]["calls"]) == 1

    asyncio.run(run())


def test_prompt_is_image_fact_based_not_claim_path_based() -> None:
    prompt = " ".join(SFT_ELIGIBILITY_SYSTEM_PROMPT.split())
    prompt_lower = prompt.lower()

    assert "factual content expressed by the supplied image" in prompt
    assert "Do not require the teacher to reproduce the target wording" in prompt
    assert "First assess target_scope independently from correctness" in prompt
    assert "related_but_incomplete" in prompt
    assert (
        "Do not call a trajectory unrelated_fact merely because it missed a key "
        "condition" in prompt
    )
    assert "Retrieval history describes what the teacher actually investigated" in prompt
    assert "generic web search failure" in prompt
    assert "Assess retrieval_quality" in prompt
    assert "trajectory_conduct" in prompt
    assert "do not create human-review work" in prompt_lower
