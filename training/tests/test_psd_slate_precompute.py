import json

import pytest

from scripts.server.precompute_psd_slate_proposals import (
    build_offset_index,
    read_indexed_row,
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
