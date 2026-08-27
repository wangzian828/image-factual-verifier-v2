from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from scripts import run_real_canary
from scripts.audit_real_trace import discover_trace_files


def test_explicit_canary_cases_are_forwarded_in_order() -> None:
    args = argparse.Namespace(
        benchmark="release/runtime_input/cases.jsonl",
        output_dir="run",
        model="gemini-3.5-flash",
        limit=2,
        concurrency=10,
        case_id=["case_refuted", "case_supported"],
        source_access_policy=None,
    )

    command = run_real_canary._command(args)

    assert command[command.index("--limit") + 1] == "2"
    assert command[-4:] == [
        "--case-id",
        "case_refuted",
        "--case-id",
        "case_supported",
    ]


def test_real_canary_rejects_configured_commit_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "configured")
    monkeypatch.setattr(
        run_real_canary,
        "_git_output",
        lambda *args: "actual" if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        run_real_canary._require_clean_runtime_checkout()


def test_real_canary_rejects_dirty_checkout(monkeypatch) -> None:
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        run_real_canary,
        "_git_output",
        lambda *args: (
            "actual"
            if args == ("rev-parse", "HEAD")
            else " M src/orchestrator/pipeline.py"
        ),
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        run_real_canary._require_clean_runtime_checkout()


def test_real_canary_binds_clean_actual_commit(monkeypatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "actual")
    monkeypatch.setattr(
        run_real_canary,
        "_git_output",
        lambda *args: "actual" if args == ("rev-parse", "HEAD") else "",
    )

    assert run_real_canary._require_clean_runtime_checkout() == "actual"


def test_real_canary_accepts_visual_or_page_evidence_routes() -> None:
    assert run_real_canary._missing_required_tool_classes(
        {"reverse_image_search", "compare_with_reference"}
    ) == []
    assert run_real_canary._missing_required_tool_classes(
        {"text_search", "visit"}
    ) == []


def test_real_canary_rejects_retrieval_without_evidence_inspection() -> None:
    assert run_real_canary._missing_required_tool_classes(
        {"reverse_image_search", "ocr_with_position"}
    ) == ["evidence inspection"]
    assert run_real_canary._missing_required_tool_classes(
        {"visit", "compare_with_reference"}
    ) == ["search"]


def _write_v4_canary_artifacts(
    tmp_path,
    *,
    data_policy: str = "reinspect-v2",
    agent_policy: str = "unified-react-v1",
    trace_policy: str = "unified-react-v1",
    decision_mode: str = "evidence_determined",
) -> None:
    (tmp_path / "traces").mkdir()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "agent": {
                    "provider": "gemini",
                    "profile_id": "teacher-gemini",
                    "model": "gemini-2.5-flash",
                    "decision_policy_version": agent_policy,
                },
                "benchmark": {
                    "input_mode": "image_only",
                    "decision_policy_version": data_policy,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"num_errors": 0}),
        encoding="utf-8",
    )
    (tmp_path / "traces" / "case-v4.json").write_text(
        json.dumps(
            {
                "image_id": "case-v4",
                "input_mode": "image_only",
                "decision_policy_version": trace_policy,
                "verdict": "fake",
                "verdict_basis": {
                    "decision_mode": decision_mode,
                    "claim_ids": ["claim-v4"],
                    "discrepancy_ids": (
                        ["discrepancy-v4"]
                        if decision_mode == "evidence_determined"
                        else []
                    ),
                    "unresolved_gaps": (
                        []
                        if decision_mode == "evidence_determined"
                        else ["The material route ended without decisive Evidence."]
                    ),
                },
                "termination": "success",
                "llm_api_calls": 2,
                "state": {
                    "investigation_state": {
                        "core_verdict_fact_id": None,
                        "stop_reason": (
                            "verdict_determined"
                            if decision_mode == "evidence_determined"
                            else "meaningful_routes_exhausted"
                        ),
                        "target_facts": [
                            {"claim_id": "claim-v4", "salience": "high"}
                        ],
                    },
                    "all_steps": [
                        {
                            "action_type": "tool_call",
                            "tool_name": "text_search",
                            "tool_result": json.dumps({"status": "success"}),
                        },
                        {
                            "action_type": "tool_call",
                            "tool_name": "visit",
                            "tool_result": json.dumps({"status": "success"}),
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_real_canary_accepts_unified_react_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    _write_v4_canary_artifacts(tmp_path)
    monkeypatch.setattr(
        run_real_canary,
        "audit_trace",
        lambda path: SimpleNamespace(failures=lambda strict_scheduler: []),
    )

    result = run_real_canary._require_real_run_artifacts(tmp_path)

    assert result["passed"] is True
    assert result["data_pipeline_decision_policy_version"] == "reinspect-v2"
    assert result["agent_decision_policy_version"] == "unified-react-v1"
    assert result["successful_tools"] == ["text_search", "visit"]


def test_trace_discovery_excludes_nested_runtime_json_artifacts(tmp_path) -> None:
    _write_v4_canary_artifacts(tmp_path)
    runtime_artifact = (
        tmp_path
        / "traces"
        / "runtime"
        / "case-v4"
        / "artifacts"
        / "sha256"
        / "ab"
        / "abcdef.json"
    )
    runtime_artifact.parent.mkdir(parents=True)
    runtime_artifact.write_text(json.dumps({"status": "success"}), encoding="utf-8")

    assert discover_trace_files(tmp_path / "traces") == [
        tmp_path / "traces" / "case-v4.json"
    ]
    assert discover_trace_files(tmp_path) == [
        tmp_path / "traces" / "case-v4.json"
    ]


def test_real_canary_accepts_bounded_binary_fake_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    _write_v4_canary_artifacts(
        tmp_path,
        decision_mode="bounded_binary_judgment",
    )
    monkeypatch.setattr(
        run_real_canary,
        "audit_trace",
        lambda path: SimpleNamespace(failures=lambda strict_scheduler: []),
    )

    result = run_real_canary._require_real_run_artifacts(tmp_path)

    assert result["passed"] is True


def test_real_canary_rejects_agent_policy_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _write_v4_canary_artifacts(
        tmp_path,
        agent_policy="reinspect-v2",
    )
    monkeypatch.setattr(
        run_real_canary,
        "audit_trace",
        lambda path: SimpleNamespace(failures=lambda strict_scheduler: []),
    )

    with pytest.raises(RuntimeError, match="unified-react-v1"):
        run_real_canary._require_real_run_artifacts(tmp_path)


def test_real_canary_rejects_data_pipeline_policy_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _write_v4_canary_artifacts(
        tmp_path,
        data_policy="unified-react-v1",
    )
    monkeypatch.setattr(
        run_real_canary,
        "audit_trace",
        lambda path: SimpleNamespace(failures=lambda strict_scheduler: []),
    )

    with pytest.raises(RuntimeError, match="data-pipeline.*reinspect-v2"):
        run_real_canary._require_real_run_artifacts(tmp_path)
