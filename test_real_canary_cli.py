from __future__ import annotations

import argparse

import pytest

from scripts import run_real_canary


def test_explicit_canary_cases_are_forwarded_in_order() -> None:
    args = argparse.Namespace(
        benchmark="release/runtime_input/cases.jsonl",
        output_dir="run",
        model="gemini-3.5-flash",
        limit=2,
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
