import json

import pytest

from ifv_training.psd_target_balance import balance_psd_targets


def _write(source, repairs, preserve_by_case):
    with source.open("w", encoding="utf-8") as handle:
        for index in range(repairs):
            handle.write(json.dumps({"target_id": f"r{index}", "kind": "repair",
                                     "case_id": f"repair-{index}", "row_weight": 1.0}) + "\n")
        for case_id, count in preserve_by_case.items():
            for index in range(count):
                handle.write(json.dumps({"target_id": f"p{case_id}-{index}",
                                         "kind": "preserve", "case_id": case_id,
                                         "row_weight": 1.0, "step_index": index}) + "\n")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_count_balance_keeps_all_repairs_and_spreads_preservation(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, 6, {"a": 8, "b": 8, "c": 8})
    result = balance_psd_targets(source=source, output_dir=tmp_path / "balanced")
    rows = _rows(tmp_path / "balanced" / "targets.jsonl")
    assert result["counts"] == {
        "repair_targets": 6, "source_preservation_targets": 24,
        "source_preservation_cases": 3, "selected_preservation_targets": 6,
        "selected_targets": 12, "excluded_preservation_targets": 18}
    assert {row["target_id"] for row in rows if row["kind"] == "repair"} == {
        f"r{index}" for index in range(6)}
    groups = {case_id: [row["step_index"] for row in rows
                        if row["kind"] == "preserve" and row["case_id"] == case_id]
              for case_id in ("a", "b", "c")}
    assert groups == {"a": [2, 6], "b": [2, 6], "c": [2, 6]}
    assert all(row["row_weight"] == 1 for row in rows)
    assert result["per_target_weights_unchanged"] is True
    assert result["provider_calls"] == 0
    with pytest.raises(FileExistsError):
        balance_psd_targets(source=source, output_dir=tmp_path / "balanced")


def test_real_bank_shape_assigns_two_or_three_steps_per_case(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, 626, {f"case-{index:03d}": 14 for index in range(246)})
    result = balance_psd_targets(source=source, output_dir=tmp_path / "balanced")
    assert result["counts"]["selected_targets"] == 1252
    assert result["counts"]["selected_preservation_targets"] == 626
    assert result["preservation_targets_per_case"] == {2: 112, 3: 134}


def test_preservation_shortage_fails_without_creating_output(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, 3, {"a": 2})
    with pytest.raises(ValueError, match="preservation cap"):
        balance_psd_targets(source=source, output_dir=tmp_path / "balanced")
    assert not (tmp_path / "balanced").exists()
