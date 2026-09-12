from __future__ import annotations

import json
from pathlib import Path

from ifv_training.io import sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_storage import save_bound
from ifv_training.psd_repair_search import revision_context
from scripts.audit_psd_feedback_run import audit_feedback_run


def _make_run(root: Path) -> tuple[Path, Path]:
    case = root / "repairs" / "case-key"
    round_dir = case / "rounds" / "round-00"
    write_json(round_dir / "manifest.json", {"status": "generated"})
    write_jsonl(round_dir / "repair_candidates.jsonl", [{"candidate_id": "candidate-1"}])
    attempt = {
        "attempt_id": "attempt-1",
        "case_id": "case-1",
        "accepted": False,
        "teacher_prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5],
        "verification": {"reasons": ["hinted_local_verifier_failed"]},
    }
    write_jsonl(round_dir / "repair_attempts.jsonl", [attempt])
    write_json(
        round_dir / "proposer-hint-audits.json",
        {
            "proposals": [
                {"passed": True, "audit": {"passed": True}},
                {"passed": False, "reason": "binary_verdict_leak"},
            ]
        },
    )
    save_bound(
        round_dir / "proposer-response.json",
        identity={"request": "proposal-1"},
        payload={"prompt_tokens": 100, "completion_tokens": 7},
    )
    judge = {
        "identity": {"request": "judge-1"},
        "response": {
            "id": "judge-response-1",
            "status": "completed",
            "usage": {
                "total_tokens": 40,
                "total_input_tokens": 30,
                "total_cached_tokens": 10,
                "total_output_tokens": 3,
                "total_thought_tokens": 7,
                "total_tool_use_tokens": 0,
            },
        },
    }
    write_json(round_dir / "judge-cache" / "a.json", judge)
    write_json(round_dir / "judge-cache" / "duplicate.json", judge)
    files = {
        str(path.resolve()): sha256_file(path)
        for path in (
            round_dir / "manifest.json",
            round_dir / "repair_candidates.jsonl",
            round_dir / "repair_attempts.jsonl",
            round_dir / "proposer-hint-audits.json",
            round_dir / "proposer-response.json",
        )
    }
    save_bound(
        case / "search-state.json",
        identity={"case": "case-1"},
        payload={
            "status": "attempt_budget_exhausted",
            "rounds": [{"directory": str(round_dir.resolve()), "files": files}],
        },
    )
    write_jsonl(case / "repair_attempts.jsonl", [attempt])
    write_json(
        case / "manifest.json",
        {
            "status": "attempt_budget_exhausted",
            "converged": False,
            "candidate_count": 1,
            "accepted_count": 0,
            "proposal_rounds": 1,
            "elapsed_seconds": 12.5,
        },
    )
    return case, round_dir


def test_feedback_audit_counts_costs_rejections_and_deduplicates_judges(tmp_path: Path) -> None:
    _make_run(tmp_path)
    report = audit_feedback_run(tmp_path)
    assert report["passed"] is True
    assert report["totals"] == {
        "cases": 1,
        "converged_cases": 0,
        "proposal_rounds": 1,
        "proposal_requests": 1,
        "proposal_prompt_tokens": 100,
        "proposal_completion_tokens": 7,
        "continuations": 1,
        "accepted_continuations": 0,
        "qwen_prompt_tokens": 3,
        "qwen_completion_tokens": 2,
        "unique_judge_requests": 1,
        "duplicate_judge_cache_entries": 1,
        "judge_usage": {
            "total_cached_tokens": 10,
            "total_input_tokens": 30,
            "total_output_tokens": 3,
            "total_thought_tokens": 7,
            "total_tokens": 40,
            "total_tool_use_tokens": 0,
        },
        "proposal_rejections": {"binary_verdict_leak": 1},
        "continuation_rejection_reasons": {"hinted_local_verifier_failed": 1},
    }
    assert "judge-response-1" not in json.dumps(report)
    assert report["feedback_context"] == {
        "stored_bytes": 0,
        "v2_projected_bytes": 0,
        "projected_requests": 0,
        "projected_reduction_fraction": None,
        "measurement": "canonical JSON bytes; provider token count depends on tokenizer",
    }


def test_feedback_audit_rejects_changed_bound_round_artifact(tmp_path: Path) -> None:
    _, round_dir = _make_run(tmp_path)
    write_json(round_dir / "manifest.json", {"status": "changed"})
    report = audit_feedback_run(tmp_path)
    assert report["passed"] is False
    assert any("snapshot hash changed" in row["error"] for row in report["errors"])


def test_feedback_audit_measures_bound_request_context(tmp_path: Path) -> None:
    case, _ = _make_run(tmp_path)
    save_bound(case / "requests" / "round-00.json", identity={"round": 0},
               payload=revision_context([]))
    report = audit_feedback_run(tmp_path)
    context = report["feedback_context"]
    assert report["passed"] is True
    assert context["projected_requests"] == 1
    assert context["stored_bytes"] == context["v2_projected_bytes"] > 0
    assert context["projected_reduction_fraction"] == 0
