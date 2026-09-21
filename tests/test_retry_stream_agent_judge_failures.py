import json
from pathlib import Path

import pytest

from scripts.server.retry_stream_agent_judge_failures import (
    retryable_failure,
    retryable_503_directories,
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


def test_retryable_failure_rejects_non_503(tmp_path: Path) -> None:
    make_failed(tmp_path, error="HTTP 429")
    with pytest.raises(ValueError, match="HTTP 503"):
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
    rejected = tmp_path / "rejected"
    timeout = tmp_path / "timeout"
    make_failed(rejected)
    make_failed(timeout, error_type="ReadTimeout", error="")
    assert retryable_503_directories([rejected, timeout]) == [rejected]
