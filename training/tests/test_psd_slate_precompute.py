import json
import os

import pytest

from scripts.server.precompute_psd_slate_proposals import (
    build_offset_index,
    pending_entries,
    read_indexed_row,
)
from scripts.server.run_psd_slate_precompute_controller import (
    alive,
    retry_delay_seconds,
)


def test_offset_index_is_stat_bound_and_resume_does_not_scan(monkeypatch, tmp_path):
    selected = tmp_path / "selected.jsonl"
    rows = [
        {"case_id": "case-a", "candidate_id": "a"},
        {"case_id": "case-b", "candidate_id": "b"},
    ]
    selected.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    index = tmp_path / "offsets.json"
    entries = build_offset_index(selected, index)
    assert [read_indexed_row(selected, row) for row in entries] == rows

    original_open = type(selected).open

    def fail_selected_open(path, *args, **kwargs):
        if path == selected:
            raise AssertionError("resume scanned selected JSONL")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(selected), "open", fail_selected_open)
    assert build_offset_index(selected, index) == entries


def test_offset_index_rejects_duplicate_case(tmp_path):
    selected = tmp_path / "selected.jsonl"
    selected.write_text(
        '{"case_id":"same"}\n{"case_id":"same"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unique case IDs"):
        build_offset_index(selected, tmp_path / "offsets.json")


def test_smoke_limit_does_not_count_unselected_pending_cases_as_complete(tmp_path):
    entries = [{"case_id": f"case-{index}"} for index in range(3)]
    receipt = tmp_path / "cases/case-0/proposal.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    pending, pending_before_limit, completed = pending_entries(entries, tmp_path, 1)
    assert pending == [{"case_id": "case-1"}]
    assert pending_before_limit == 2
    assert completed == 1


def test_controller_alive_rejects_missing_pid():
    assert alive(999_999_999) is False


@pytest.mark.skipif(os.name == "nt", reason="/proc is unavailable on Windows")
def test_controller_alive_accepts_current_process():
    assert alive(os.getpid()) is True


def test_controller_uses_bounded_exponential_retry_delay():
    assert [
        retry_delay_seconds(attempt, base_seconds=2, max_seconds=10)
        for attempt in range(1, 6)
    ] == [2, 4, 8, 10, 10]
