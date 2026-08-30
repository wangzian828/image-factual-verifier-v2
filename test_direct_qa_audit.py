from scripts.audit_direct_qa_baseline import _latest_records


def test_latest_records_discards_recovered_direct_qa_error_rows() -> None:
    rows = [
        {"case_id": "case-1", "status": "error", "error_type": "ReadTimeout"},
        {"case_id": "case-2", "status": "completed", "predicted_verdict": "fake"},
        {"case_id": "case-1", "status": "completed", "predicted_verdict": "real"},
    ]

    latest = _latest_records(rows)

    assert set(latest) == {"case-1", "case-2"}
    assert latest["case-1"]["status"] == "completed"
    assert latest["case-1"]["predicted_verdict"] == "real"
