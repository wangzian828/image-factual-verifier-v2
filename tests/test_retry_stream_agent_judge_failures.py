import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.server.retry_stream_agent_judge_failures import (
    TailRepair,
    retryable_failure,
    retryable_rejection_directories,
    unresolved_failed_directories,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_failed(directory: Path, *, error_type: str = "GeminiInteractionsHTTPError", error: str = "HTTP 503") -> None:
    write(directory / "attempt-01.intent.json", {"state": "in_flight"})
    write(directory / "attempt-01.result.json", {"status": "failed"})
    write(
        directory / "failed.json",
        {
            "case_id": "case-1",
            "state": "judge_attempt_failed",
            "error_type": error_type,
            "error": error,
        },
    )


def test_retryable_failure_accepts_explicit_503(tmp_path: Path) -> None:
    make_failed(tmp_path)
    assert retryable_failure(tmp_path)["case_id"] == "case-1"


def test_retryable_failure_accepts_explicit_429(tmp_path: Path) -> None:
    make_failed(tmp_path, error="HTTP 429")
    assert retryable_failure(tmp_path)["case_id"] == "case-1"


def test_retryable_failure_rejects_ambiguous_timeout(tmp_path: Path) -> None:
    make_failed(tmp_path, error_type="ReadTimeout", error="")
    with pytest.raises(ValueError, match="HTTP 429/503"):
        retryable_failure(tmp_path)


def test_retryable_failure_rejects_existing_success(tmp_path: Path) -> None:
    make_failed(tmp_path)
    write(tmp_path / "result.json", {"status": "completed"})
    with pytest.raises(ValueError, match="accepted result"):
        retryable_failure(tmp_path)


def test_unresolved_failures_excludes_repaired_cases(tmp_path: Path) -> None:
    class Judge:
        output = tmp_path

    first = tmp_path / "cases/a"
    second = tmp_path / "cases/b"
    make_failed(first)
    make_failed(second)
    write(second / "result.json", {"status": "completed"})
    assert unresolved_failed_directories(Judge()) == [first]


def test_retryable_subset_excludes_ambiguous_timeout(tmp_path: Path) -> None:
    rejected = tmp_path / "rejected503"
    rate_limited = tmp_path / "rejected429"
    timeout = tmp_path / "timeout"
    make_failed(rejected)
    make_failed(rate_limited, error="HTTP 429")
    make_failed(timeout, error_type="ReadTimeout", error="")
    assert retryable_rejection_directories([rejected, rate_limited, timeout]) == [
        rejected, rate_limited
    ]


def test_tail_repair_does_not_mark_ambiguous_cases_complete(tmp_path: Path) -> None:
    class Judge:
        output = tmp_path
        expected = ["case-1", "case-2"]
        success_rows = {case: ({}, tmp_path) for case in ("case-1", "case-2")}

        @staticmethod
        def case_dir(case_id: str) -> Path:
            return tmp_path / "cases" / case_id

    write(tmp_path / "cases/case-1/result.json", {"case_id": "case-1", "status": "completed"})
    write(tmp_path / "cases/case-2/ambiguous.json", {"case_id": "case-2"})
    repair = TailRepair.__new__(TailRepair)
    repair.judge = Judge()
    repair.args = Namespace(formal_denominator=3)
    repair.failures = []
    repair.all_failures = []
    repair.finalize({})
    state = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert state["phase"] == "judge_tail_retry_incomplete"
    assert state["judge_completed"] == 1
    assert state["judge_ambiguous"] == 1
