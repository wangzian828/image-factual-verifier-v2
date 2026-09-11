import pytest

from ifv_training.psd_datums import build_sparse_topk_datum
from ifv_training.psd_modality import require_text_only_psd


@pytest.mark.parametrize("token", [248053, 248054, 248056, 248057])
def test_rejects_visual_placeholder_without_pixels(token):
    with pytest.raises(ValueError, match="psd_multimodal_not_supported"):
        require_text_only_psd({}, [1, token, 2])


def test_rejects_media_even_when_placeholder_is_missing():
    with pytest.raises(ValueError, match="psd_multimodal_not_supported"):
        require_text_only_psd({"images": ["case.png"]}, [1, 2])


def test_text_only_target_still_allowed():
    require_text_only_psd({}, [1, 2], [3, 4])


def test_datum_builder_cannot_silently_strip_pixels():
    with pytest.raises(ValueError, match="psd_multimodal_not_supported"):
        build_sparse_topk_datum({
            "target_status": "complete", "target_id": "visual", "kind": "repair",
            "student_prompt_ids": [1, 248056], "completion_ids": [2],
            "teacher_topk_by_position": [[[2, 1.0]]],
        }, topk=1)
