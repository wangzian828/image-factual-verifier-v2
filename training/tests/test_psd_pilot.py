import pytest

from scripts.prepare_psd_pilot import select_pilot


def test_pilot_keeps_existing_canary_and_balances_without_outcome_selection():
    rows = [{"case_id": str(i)} for i in range(100)]
    labels = {str(i): "real" if i < 30 else "fake" for i in range(100)}
    args = dict(labels=labels, required_ids=["0", "99"], count=20)
    selected = select_pilot(rows, **args)
    assert selected == select_pilot(list(reversed(rows)), **args)
    assert {"0", "99"} <= {r["case_id"] for r in selected}
    assert sum(labels[r["case_id"]] == "real" for r in selected) == 10
    for change in [{"required_ids": ["absent"]}, {"required_ids": ["0", "0"]}, {"count": 21}, {"count": 80}]:
        with pytest.raises(ValueError):
            select_pilot(rows, **{**args, **change})
