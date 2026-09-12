from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ifv_training.io import load_json, write_jsonl
from scripts.prepare_psd_paired_eval import prepare


def test_paired_eval_selection_is_deterministic_and_outcome_free(tmp_path):
    benchmark = tmp_path / "cases.jsonl"
    write_jsonl(benchmark, [
        {"case_id": f"case-{index}", "image_path": f"{index}.jpg"}
        for index in range(10)
    ])
    first = prepare(
        benchmark=benchmark,
        output_dir=tmp_path / "first",
        count=4,
        salt="fixed",
    )
    second = prepare(
        benchmark=benchmark,
        output_dir=tmp_path / "second",
        count=4,
        salt="fixed",
    )
    assert first["case_ids"] == second["case_ids"]
    assert first["selection_before_outcomes"] is True
    assert (tmp_path / "first" / "case-list.txt").read_text().splitlines() == first["case_ids"]
    assert load_json(tmp_path / "first" / "selection.json")["benchmark_cases"] == 10


def test_paired_eval_selection_rejects_duplicate_cases(tmp_path):
    benchmark = tmp_path / "cases.jsonl"
    write_jsonl(benchmark, [{"case_id": "same"}, {"case_id": "same"}])
    with pytest.raises(ValueError, match="unique"):
        prepare(
            benchmark=benchmark,
            output_dir=tmp_path / "selection",
            count=1,
            salt="fixed",
        )
