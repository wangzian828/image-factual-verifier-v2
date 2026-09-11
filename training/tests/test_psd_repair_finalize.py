from __future__ import annotations

import json
from pathlib import Path

from ifv_training.io import sha256_file
from ifv_training.psd_repair import FailureSite, VerificationResult, _sha
from ifv_training.psd_repair_finalize import finalize_psd_repair_run
from ifv_training.psd_repair_verifier import (
    build_complete_hinted_episode_trace,
)
from scripts.audit_real_trace import audit_trace
from src.orchestrator.stage_runner import StageStep


def _base_trace() -> dict:
    policy_input = {
        "system_instruction": "system",
        "input_payload": [{"role": "user", "content": "inspect"}],
        "tools": [],
    }
    policy_action = {
        "type": "tool_call",
        "name": "text_search",
        "arguments": {"query": "wrong direction"},
    }
    runtime_case = {
        "case_id": "case-1",
        "image_path": "/data/image.jpg",
        "image_sha256": "a" * 64,
    }
    return {
        "image_id": "episode-1",
        "case_id": "case-1",
        "image_path": "/data/image.jpg",
        "input_mode": "image_only",
        "decision_policy_version": "unified-react-v1",
        "termination": "success",
        "state": {
            "image_path": "/data/image.jpg",
            "image_id": "episode-1",
            "runtime_case": runtime_case,
            "input_mode": "image_only",
            "decision_policy_version": "unified-react-v1",
            "investigation_state": {
                "schema_version": "ifv-unified-react-raw-history-v1",
                "case_id": "case-1",
                "image_sha256": "a" * 64,
                "objective": "Verify the image claim.",
                "action_count": 1,
                "stop_reason": "model_finished",
                "finish_rationale": "done",
            },
            "all_steps": [
                {
                    "round": 1,
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "tool_name": "text_search",
                    "tool_args": {"query": "wrong direction"},
                    "tool_result": json.dumps(
                        {"status": "success", "items": []}
                    ),
                    "tokens": {"prompt": 10, "completion": 3, "thought": 1},
                    "metadata": {
                        "function_call_id": "failed-call",
                        "policy_input": policy_input,
                        "policy_action": policy_action,
                    },
                },
                {
                    "round": 1,
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "output": {"verdict": "real"},
                    "metadata": {
                        "policy_input": {
                            "system_instruction": "judge",
                            "input_payload": [],
                        },
                        "policy_action": {"verdict": "real"},
                    },
                },
            ],
            "stage_timings": {},
            "total_tool_calls": 1,
            "total_tool_subcalls": 0,
            "tool_subcalls_by_kind": {},
            "llm_api_calls": 2,
            "token_usage": {"prompt": 10, "completion": 3, "thought": 1},
            "termination": "success",
            "errors": [],
            "tool_health": {},
        },
    }


def _site(trace: dict) -> FailureSite:
    step = trace["state"]["all_steps"][0]
    return FailureSite(
        step_index=0,
        source_step_index=0,
        step_id="episode-1:unified_react:failed-call",
        stage="unified_react",
        example_type="react",
        policy_input=step["metadata"]["policy_input"],
        policy_action=step["metadata"]["policy_action"],
    )


def _teacher_steps() -> list[StageStep]:
    capture = {
        "status": "complete",
        "prompt_token_ids": [1, 2, 3, 4],
        "completion_token_ids": [5, 6],
    }
    return [
        StageStep(
            round=1,
            stage_name="psd_teacher_repair",
            action_type="tool_call",
            tool_name="text_search",
            tool_args={"query": "event relation"},
            tool_result=json.dumps({"status": "success", "items": []}),
            tokens={"prompt": 12, "completion": 4, "thought": 1},
            metadata={
                "function_call_id": "repaired-call",
                "policy_action": {
                    "type": "tool_call",
                    "name": "text_search",
                    "arguments": {"query": "event relation"},
                },
                "policy_token_capture": capture,
            },
        ),
        StageStep(
            round=1,
            stage_name="psd_teacher_judgment",
            action_type="output",
            output={
                "verdict": "fake",
                "confidence": 0.9,
                "verdict_observation_ids": ["repaired-call"],
                "overall_assessment": "The claim is unsupported.",
                "fact_check_report": {
                    "headline": "Unsupported claim",
                    "claim_under_review": "The depicted event occurred.",
                    "verdict_summary": "No matching event was found.",
                    "key_findings": ["Search produced no matching event."],
                    "evidence_summary": "The retained search observation has no match.",
                    "remaining_uncertainties": [],
                },
            },
            tokens={"prompt": 20, "completion": 8, "thought": 1},
            metadata={"policy_action": {"verdict": "fake"}},
        ),
    ]


def test_complete_episode_replaces_failed_suffix_and_passes_trace_audit(
    tmp_path: Path,
) -> None:
    source = _base_trace()
    trace = build_complete_hinted_episode_trace(
        source,
        failure_site=_site(source),
        teacher_steps=_teacher_steps(),
        stop_reason="psd_suffix_action_limit",
    )
    assert trace["verdict"] == "fake"
    assert [
        step["metadata"].get("function_call_id")
        for step in trace["state"]["all_steps"]
        if step["action_type"] == "tool_call"
    ] == ["repaired-call"]
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    assert audit_trace(path).failures(strict_scheduler=True) == []


def test_offline_finalizer_never_replays_provider_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "repair-run"
    (run_dir / "episodes").mkdir(parents=True)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_base_trace()) + "\n", encoding="utf-8")
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({"factual_status": "fake"}) + "\n")
    source_sha = sha256_file(source_path)
    prompt_ids = [1, 2, 3, 4]
    completion_ids = [5, 6]
    hint_sha = "b" * 64
    episode_path = run_dir / "episodes" / "hint-00-teacher.json"
    episode_path.write_text(
        json.dumps({"image_id": "episode-1"}) + "\n", encoding="utf-8"
    )
    attempt = {
        "schema_version": "ifv-psd-repair-attempt-v1",
        "repair_step_id": "episode-1:unified_react:failed-call",
        "source_trace_sha256": source_sha,
        "hint_record": {"audit": {"hint_sha256": hint_sha}},
        "teacher_prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "continuation": {
            "hinted_teacher_episode_trace": "episodes/hint-00-teacher.json",
            "hinted_teacher_episode_trace_sha256": sha256_file(episode_path),
        },
        "accepted": False,
    }
    (run_dir / "repair_attempts.jsonl").write_text(
        json.dumps(attempt) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ifv-psd-repair-driver-result-v1",
                "artifacts": {"repair_attempts": "repair_attempts.jsonl"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local = {
        "schema_version": "ifv-psd-local-verification-v1",
        "passed": True,
    }
    bundle_path = tmp_path / "verification.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "ifv-psd-repair-verification-bundle-v1",
                "attempts": [
                    {
                        "hint_index": 0,
                        "hint_sha256": hint_sha,
                        "local_verification": local,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def verified(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["teacher_prompt_sha256"] == _sha(prompt_ids)
        assert kwargs["teacher_completion_sha256"] == _sha(completion_ids)
        return VerificationResult(
            source_rollout_failed=True,
            hinted_local_pass=True,
            hinted_episode_pass=True,
            hinted_strict_trace_audit_pass=True,
            hinted_recorded_verdict="fake",
            expected_verdict="fake",
            repair_tier="causal_episode_pass",
        )

    monkeypatch.setattr(
        "ifv_training.psd_repair_finalize.verify_causal_episode", verified
    )
    result = finalize_psd_repair_run(
        run_dir=run_dir,
        source_trace_path=source_path,
        gold_path=gold_path,
        verification_bundle_path=bundle_path,
        require_all=True,
    )
    assert result["status"] == "ready_for_assembly"
    assert result["provider_calls"] == 0
    assert calls == 1
    finalized = json.loads(
        (run_dir / "repair_attempts.jsonl").read_text(encoding="utf-8")
    )
    assert finalized["accepted"] is True
    assert (run_dir / "repair_attempts.pre-finalize.jsonl").is_file()
