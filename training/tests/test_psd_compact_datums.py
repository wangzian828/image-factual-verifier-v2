import json
import pytest
from ifv_training.psd_datums import compact_datum, expand_datum_targets, build_sparse_topk_datum
from ifv_training.psd_preflight import _datum_error


def dense():
    return build_sparse_topk_datum({"target_id": "case", "kind": "repair", "target_status": "complete",
        "student_prompt_ids": [5] * 1000, "completion_ids": [7, 8], "row_weight": 0.5,
        "teacher_topk_by_position": [[[7, 0.75], [9, 0.25]], [[8, 0.8], [9, 0.2]]]}, topk=2)


def test_compact_bank_preserves_exact_causal_tensors_and_old_banks():
    original = dense()
    compact = compact_datum(original)
    assert expand_datum_targets(compact) == (original["target_tokens"], original["weights"])
    assert expand_datum_targets(original) == (original["target_tokens"], original["weights"])
    assert len(json.dumps(compact)) < len(json.dumps(original)) / 4
    for row in (original, compact):
        assert not _datum_error(row, topk=2, max_context=131072)


@pytest.mark.parametrize("positions", [[0, 1], [999, 999], [1000, 999], [-1, 1000]])
def test_compact_positions_cannot_shift_supervision_into_prompt(positions):
    row = compact_datum(dense())
    row["loss_positions"] = positions
    assert _datum_error(row, topk=2, max_context=131072) == "datum_causal_prediction_positions_invalid"


def test_compact_zero_or_duplicate_teacher_targets_rejected():
    row = compact_datum(dense())
    row["sparse_weights"][0] = [0, 0]
    assert _datum_error(row, topk=2, max_context=131072)
    row = compact_datum(dense())
    row["sparse_target_tokens"][0] = [7, 7]
    assert _datum_error(row, topk=2, max_context=131072) == "datum_duplicate_topk_token"
