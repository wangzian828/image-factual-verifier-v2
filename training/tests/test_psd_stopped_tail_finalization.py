import pytest

from ifv_training.io import write_json, write_jsonl
from scripts.server.finalize_psd_stopped_tail import collect_terminal, finalize


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


def test_finalization_resume_reuses_bound_merge_without_recollecting(tmp_path, monkeypatch):
    search, control = tmp_path / "search", tmp_path / "control"
    merge = search / "merged-stopped-tail-v1"
    merge.mkdir(parents=True)
    candidates, attempts = merge / "repair_candidates.jsonl.gz", merge / "repair_attempts.jsonl.gz"
    preservation = tmp_path / "preservation.jsonl"
    write_jsonl(candidates, [{"candidate_id": "kept"}])
    write_jsonl(attempts, [{"attempt_id": "kept", "accepted": True}])
    write_jsonl(preservation, [])
    from ifv_training.io import sha256_file
    write_json(control / "selection.json", {
        "schema_version": "ifv-psd-stopped-tail-selection-v1",
        "inputs": {str(path.resolve()): sha256_file(path) for path in (
            candidates, attempts, preservation)}})
    monkeypatch.setattr("scripts.server.finalize_psd_stopped_tail.collect_terminal",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not rescan cases")))
    monkeypatch.setattr("scripts.server.finalize_psd_stopped_tail.materialize_bank",
        lambda **_: {"status": "requires_frozen_teacher_topk"})
    result = finalize(search=search, preservation=preservation, control=control)
    assert result["status"] == "requires_frozen_teacher_topk"
