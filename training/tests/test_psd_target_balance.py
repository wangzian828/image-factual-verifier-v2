import json

import pytest

from ifv_training.psd_target_balance import balance_psd_targets


def _write_fixture(tmp_path, *, repairs=12, counts=None, truncate=None):
    counts = counts or {f"case-{i}": n for i, n in enumerate([4, 5, 6, 7, 8, 9, 10, 11])}
    source = tmp_path / "targets.jsonl"
    assembled = tmp_path / "assembled.jsonl"
    with source.open("w", encoding="utf-8") as targets, assembled.open("w", encoding="utf-8") as episodes:
        for i in range(repairs):
            targets.write(json.dumps({"target_id": f"r{i}", "kind": "repair", "case_id": f"r{i}"}) + "\n")
        for case_id, count in counts.items():
            episode_id = f"{case_id}-episode"
            steps = [{"step_id": f"{episode_id}:react:{i}"} for i in range(count - 1)]
            steps.append({"step_id": f"{episode_id}:judgment:{count - 1}"})
            episodes.write(json.dumps({"case_id": case_id, "episode_id": episode_id,
                                       "verified_full_task": True, "strict_trace_audit_pass": True,
                                       "preservation_steps": steps}) + "\n")
            for i, step in enumerate(steps):
                if truncate == (case_id, i):
                    continue
                targets.write(json.dumps({"target_id": f"p-{case_id}-{i}", "kind": "preserve",
                                          "case_id": case_id, "episode_id": episode_id,
                                          "repair_site": {"step_id": step["step_id"]},
                                          "verification": {"local_pass": True, "full_episode_pass": True,
                                                           "strict_trace_audit_pass": True},
                                          "step_index": i}) + "\n")
    return source, assembled


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_selects_only_complete_episodes_and_all_repairs(tmp_path):
    source, assembled = _write_fixture(tmp_path)
    output = tmp_path / "selected"
    result = balance_psd_targets(source=source, preservation_episodes_source=assembled,
                                 output_dir=output)
    rows = _rows(output / "targets.jsonl")
    assert sum(row["kind"] == "repair" for row in rows) == 12
    selected = {row["case_id"] for row in rows if row["kind"] == "preserve"}
    for case_id in selected:
        source_ids = [row["target_id"] for row in _rows(source)
                      if row["kind"] == "preserve" and row["case_id"] == case_id]
        chosen_ids = [row["target_id"] for row in rows
                      if row["kind"] == "preserve" and row["case_id"] == case_id]
        assert chosen_ids == source_ids
        assert any(":judgment:" in row["repair_site"]["step_id"] for row in rows
                   if row["kind"] == "preserve" and row["case_id"] == case_id)
    assert result["counts"]["selected_preservation_cases"] == len(selected)
    assert result["status"] == "prepared_for_audit_not_authorized_to_train"
    assert result["provider_calls"] == 0
    with pytest.raises(FileExistsError):
        balance_psd_targets(source=source, preservation_episodes_source=assembled,
                            output_dir=output)


def test_deterministic_seed_and_no_step_level_partial_episode(tmp_path):
    source, assembled = _write_fixture(tmp_path, repairs=20)
    a = balance_psd_targets(source=source, preservation_episodes_source=assembled,
                            output_dir=tmp_path / "a", seed="fixed")
    b = balance_psd_targets(source=source, preservation_episodes_source=assembled,
                            output_dir=tmp_path / "b", seed="fixed")
    assert a["selected_episode_ids"] == b["selected_episode_ids"]
    assert (tmp_path / "a" / "targets.jsonl").read_bytes() == (tmp_path / "b" / "targets.jsonl").read_bytes()


def test_missing_step_fails_before_creating_output(tmp_path):
    source, assembled = _write_fixture(tmp_path, truncate=("case-3", 2))
    with pytest.raises(ValueError, match="incomplete or out of order"):
        balance_psd_targets(source=source, preservation_episodes_source=assembled,
                            output_dir=tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_preservation_shortage_fails_before_creating_output(tmp_path):
    source, assembled = _write_fixture(tmp_path, repairs=100)
    with pytest.raises(ValueError, match="preservation cap"):
        balance_psd_targets(source=source, preservation_episodes_source=assembled,
                            output_dir=tmp_path / "selected")
    assert not (tmp_path / "selected").exists()
