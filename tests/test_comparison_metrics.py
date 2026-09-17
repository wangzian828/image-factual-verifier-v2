import copy
import json
from fractions import Fraction

import pytest

from scripts.rebuild_comparison_metrics import DEFAULT_COUNTS, ROOT, binary_metrics, markdown, recover


def test_missing_is_false_negative_not_opposite_prediction():
    result = binary_metrics([[1, 0, 1], [0, 2, 0]], support=(2, 2))
    assert result["macro_precision"] == 1
    assert result["bacc"] == .75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["missing_or_invalid"] == 1


def test_macro_f1_is_mean_of_class_f1_not_f1_of_macro_precision_recall():
    result = binary_metrics([[1, 1, 0], [0, 8, 0]], support=(2, 8))
    assert result["macro_f1"] == pytest.approx((2 / 3 + 16 / 17) / 2)
    p, r = result["macro_precision"], result["bacc"]
    assert result["macro_f1"] != pytest.approx(2 * p * r / (p + r))


def test_no_valid_predictions():
    result = binary_metrics([[0, 0, 377], [0, 0, 1150]])
    assert all(result[key] == 0 for key in ("bacc", "macro_precision", "macro_f1"))


@pytest.mark.parametrize("matrix", [
    [[1, 0], [0, 1]], [[377, 0, 0], [0, 1149, 0]],
    [[377, -1, 1], [0, 1150, 0]], [[377.0, 0, 0], [0, 1150, 0]],
    [[True, 375, 1], [0, 1150, 0]],
])
def test_invalid_counts_rejected(matrix):
    with pytest.raises(ValueError):
        binary_metrics(matrix)


def test_all_twenty_experiments_recovered_without_changing_existing_metrics():
    rows = recover(json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8")))
    assert len(rows) == 20
    assert sum(r["group"] == "agent" for r in rows) == 9
    assert sum(r["group"] == "direct_qa" for r in rows) == 11
    external = next(r for r in rows if r["id"] == "gpt55_agent")
    assert external["metrics"]["macro_f1"] == pytest.approx(.7206050120430545)
    assert next(r for r in rows if r["id"] == "sft2056")["sesr_existing_percent"] == "暂停新提交"


def test_duplicate_experiment_rejected():
    doc = json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8"))
    doc["models"].append(copy.deepcopy(doc["models"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        recover(doc)


def test_main_table_keeps_user_selected_epoch3_sft_and_preserves_history():
    rows = recover(json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8")))
    assert sum(r["main_table"] for r in rows) == 17
    sft = [r for r in rows if r["id"].startswith("sft")]
    assert len(sft) == 4
    selected = [r for r in sft if r["main_table"]]
    assert [r["id"] for r in selected] == ["sft3084"]
    assert selected[0]["sesr_existing_percent"] == "44.66"
    assert f"{100 * selected[0]['metrics']['bacc']:.2f}" == "79.08"
    assert "缓存缺陷记录" in selected[0]["model"]
    assert markdown(rows, "agent").count("SFT-") == 1
    assert markdown(rows, "agent", include_history=True).count("SFT-") == 4


def test_unknown_main_table_exclusion_rejected():
    doc = json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8"))
    doc["main_table_excluded_ids"].append("not-a-recorded-run")
    with pytest.raises(ValueError, match="unknown experiment"):
        recover(doc)


def test_main_tables_match_recovered_counts():
    rows = recover(json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8")))
    report = (ROOT / "docs/evaluation-comparison-20260909.md").read_text(encoding="utf-8")
    for group in ("agent", "direct_qa"):
        assert markdown(rows, group) in report


def test_independent_per_example_set_and_exact_fraction_calculation():
    """Expand counts to labels, independently count sets, and use per-class P/R."""
    rows = recover(json.loads(DEFAULT_COUNTS.read_text(encoding="utf-8")))
    for row in rows:
        truth, predictions = [], []
        for gold, counts in enumerate(row["matrix"]):
            for prediction, count in enumerate(counts):
                truth.extend([gold] * count)
                predictions.extend([prediction] * count)
        precisions, recalls, f1s = [], [], []
        assert len(truth) == len(predictions) == 1527
        for label in (0, 1):
            expected = {i for i, value in enumerate(truth) if value == label}
            actual = {i for i, value in enumerate(predictions) if value == label}
            correct = len(expected & actual)
            precision = Fraction(correct, len(actual)) if actual else Fraction(0)
            recall = Fraction(correct, len(expected))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        for name, values in (("macro_precision", precisions), ("bacc", recalls), ("macro_f1", f1s)):
            assert row["metrics"][name] == pytest.approx(float(sum(values) / 2), abs=1e-12)
