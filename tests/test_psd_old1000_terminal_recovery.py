import json


def test_recovery_binding_selects_only_terminal_nonaccepted_cases(tmp_path, monkeypatch):
    import scripts.server.run_psd_old1000_terminal_recovery as recovery

    source_output = tmp_path / "source"
    receipts = source_output / "case-receipts"
    receipts.mkdir(parents=True)
    (receipts / "main-00001.json").write_text(json.dumps({
        "case_id": "main-00001", "status": "infrastructure_budget_exhausted",
        "accepted_count": 0,
    }))
    (receipts / "main-00002.json").write_text(json.dumps({
        "case_id": "main-00002", "status": "converged", "accepted_count": 1,
    }))
    (receipts / "main-00003.json").write_text(json.dumps({
        "case_id": "main-00003", "status": "attempt_budget_exhausted",
        "accepted_count": 0,
    }))
    monkeypatch.setattr(recovery, "SOURCE_OUTPUT", source_output)
    monkeypatch.setattr(recovery, "RECOVERY", source_output / "recoveries")
    monkeypatch.setattr(recovery.source, "selected_index", lambda: [
        {"case_id": "main-00001"}, {"case_id": "main-00002"}, {"case_id": "main-00003"}
    ])

    rows = recovery.bind_or_validate()
    assert [row["case_id"] for row in rows] == ["main-00001", "main-00003"]
    binding = json.loads((source_output / "recoveries" / "binding.json").read_text())
    assert binding["original_budget_unchanged"] is True


def test_recovery_args_use_separate_output_and_do_not_resume_original(tmp_path, monkeypatch):
    import scripts.server.run_psd_old1000_terminal_recovery as recovery

    source_output = tmp_path / "source"
    monkeypatch.setattr(recovery, "RECOVERY", tmp_path / "recoveries")
    monkeypatch.setattr(recovery.source, "case_cli", lambda *args: [
        "--output-dir", str(source_output / "repairs" / "abc"), "--resume"
    ])
    args = recovery.recovery_args({"case_id": "main-00001"}, {}, {})
    assert str(tmp_path / "recoveries" / "repairs" / "abc") in args
    assert "--resume" not in args
