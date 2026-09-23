import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server/plan_psd_combined_preservation.py"
spec = importlib.util.spec_from_file_location("plan_psd_combined_preservation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _rows(source):
    rows = []
    for i in range(16):
        steps = 5 + i % 4
        row = {"case_id": f"{source}-{i}", "episode_id": f"e-{i}",
               "step_count": steps, "max_prompt_tokens": 100 + 10*i,
               "tool_sequence": ["visit", "text_search" if i % 2 else "ocr_with_position"],
               "selection_source": source}
        if source == "smallbank":
            row.update(verified_full_task=True, strict_trace_audit_pass=True,
                       image_steps=steps)
        else:
            row["step_ids"] = [f"e-{i}:react:{j}" for j in range(steps-1)] + [f"e-{i}:judgment:99"]
        rows.append(row)
    return rows


def test_proposal_keeps_complete_episodes_and_covers_tools():
    small, old = _rows("smallbank"), _rows("old1000")
    result = module.propose(small, old, {"smallbank": 48, "old1000": 48}, time_limit=10)
    selected = result["selected"]
    assert result["metadata_coverage_gate_pass"]
    assert len(selected) == len({row["case_id"] for row in selected})
    assert {row["source"] for row in selected} == {"smallbank", "old1000"}
    assert all(row["step_count"] in (5, 6, 7, 8) for row in selected)
    assert result["represented_native_tools"] == result["all_native_tools"]


def test_rejects_cross_source_case_overlap():
    small, old = _rows("smallbank"), _rows("old1000")
    old[0]["case_id"] = small[0]["case_id"]
    with pytest.raises(ValueError, match="overlap"):
        module.propose(small, old, {"smallbank": 48, "old1000": 48})


def test_read_rejects_unverified_and_incomplete(tmp_path):
    row = _rows("smallbank")[0]
    path = tmp_path / "index.jsonl"
    row["verified_full_task"] = False
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unverified"):
        module._read(path, "smallbank")
    row = _rows("old1000")[0]
    row["step_ids"] = row["step_ids"][:-1]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="step IDs"):
        module._read(path, "old1000")


def test_quarantines_ambiguous_old_judgment_without_dropping_other_cases(tmp_path):
    first, second = _rows("old1000")[:2]
    first["step_ids"][-2] = f"{first['episode_id']}:judgment:4"
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    excluded = []
    kept = module._read(path, "old1000", excluded)
    assert [r["case_id"] for r in kept] == [second["case_id"]]
    assert excluded == [{"source": "old1000", "case_id": first["case_id"],
                         "episode_id": first["episode_id"],
                         "reason": "ambiguous_judgment_steps", "judgment_steps": 2,
                         "preservation_steps": first["step_count"]}]
