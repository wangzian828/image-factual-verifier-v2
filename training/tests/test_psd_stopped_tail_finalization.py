import pytest

from ifv_training.io import write_json, write_jsonl
from scripts.server.finalize_psd_stopped_tail import collect_terminal


def case(root, key, status, accepted):
    directory = root / "repairs" / key
    write_json(directory / "manifest.json", {"status": status, "accepted_count": accepted})
    write_jsonl(directory / "repair_candidates.jsonl", ([{"candidate_id": key}] if accepted else []))
    write_jsonl(directory / "repair_attempts.jsonl", ([{"attempt_id": key,
        "case_id": key, "accepted": True}] if accepted else []))


def test_collect_terminal_excludes_tail_without_resampling(tmp_path):
    case(tmp_path, "good", "converged", 1)
    case(tmp_path, "spent", "attempt_budget_exhausted", 0)
    case(tmp_path, "tail", "generating", 0)
    candidates, attempts, selection = collect_terminal(tmp_path)
    assert candidates == [{"candidate_id": "good"}]
    assert attempts[0]["accepted"] is True
    assert selection["accepted_attempts"] == 1
    assert selection["new_agent_or_provider_calls"] is False
    assert selection["excluded"] == [{"repair_key": "tail", "status": "generating"}]


def test_collect_terminal_rejects_manifest_ledger_disagreement(tmp_path):
    case(tmp_path, "bad", "converged", 2)
    with pytest.raises(ValueError, match="accepted count differs"):
        collect_terminal(tmp_path)
