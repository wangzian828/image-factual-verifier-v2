import pytest

from scripts.prepare_psd_experiment_plan import select_balanced


def test_pre_outcome_balanced_selection_is_deterministic_and_capped():
    rows = [{"case_id": str(i), "label": "real" if i < 30 else "fake"} for i in range(100)]
    args = dict(count=40, seed="fixed-before-results", label=lambda r: r["label"], case_id=lambda r: r["case_id"])
    first = select_balanced(rows, **args)
    assert first == select_balanced(list(reversed(rows)), **args)
    assert sum(r["label"] == "real" for r in first) == 20
    assert len({r["case_id"] for r in first}) == 40
    assert select_balanced(rows, **{**args, "count": 80}).count(first[0]) == 1
    with pytest.raises(ValueError):
        select_balanced(rows, **{**args, "count": 101})
    with pytest.raises(ValueError, match="duplicate"):
        select_balanced(rows + [rows[0]], **args)
