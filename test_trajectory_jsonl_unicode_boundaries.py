import json

import pytest

from scripts.trajectory import (
    audit_dataset,
    build_sft_case_split,
    build_sft_training_package,
    export_dataset,
    stage_accepted_teacher_release,
)


@pytest.mark.parametrize(
    "loader",
    [
        audit_dataset._load_jsonl,
        build_sft_case_split._load_jsonl,
        build_sft_training_package._load_jsonl,
        export_dataset._load_jsonl,
        stage_accepted_teacher_release._load_jsonl,
    ],
)
def test_trajectory_jsonl_loaders_preserve_unicode_separators(tmp_path, loader):
    rows = [
        {"case_id": "one", "text": "before\u2028middle\u2029after"},
        {"case_id": "two", "text": "before\u0085after"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert loader(path) == rows
