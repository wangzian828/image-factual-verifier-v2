import base64
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ifv_training.psd_media import bind_media, load_media, image_bytes, validate_image_runs
from ifv_training.psd_datums import build_sparse_topk_datum


def request():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    block = {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}
    return {"input_payload": [{"role": "user", "content": [block, block]}]}


class ImageProcessor:
    merge_size = 2

    def to_dict(self):
        return {"merge_size": 2}

    def __call__(self, images, return_tensors):
        assert len(images) == 2
        return {"pixel_values": torch.ones(8, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}


def test_repeated_images_survive_datum_and_hash_validation(tmp_path):
    ids = [1, 248056, 2, 248056, 3]
    media = bind_media(request(), processor=SimpleNamespace(image_processor=ImageProcessor()),
                       output_dir=tmp_path, prompt_ids=ids, processor_id="test")
    assert len(media["image_sha256"]) == 2
    assert media["image_sha256"][0] == media["image_sha256"][1]
    datum = build_sparse_topk_datum({
        "target_status": "complete", "target_id": "multi", "kind": "repair",
        "student_prompt_ids": ids, "completion_ids": [10, 11],
        "teacher_topk_by_position": [[[10, 1.0]], [[11, 1.0]]], "psd_media": media,
    }, topk=1)
    assert datum["loss_positions"] == [4, 5]
    assert load_media(datum["psd_media"], datum["input_ids"])["pixel_values"].shape == (8, 12)
    with open(media["path"], "ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_media(media, ids)


def test_rejects_reordered_grid_sizes():
    with pytest.raises(ValueError, match="placeholder/grid mismatch"):
        validate_image_runs([248056, 0, 248056, 248056], [[1, 2, 4], [1, 2, 2]], 2)


def test_never_refetches_remote_image():
    with pytest.raises(ValueError, match="archived image bytes"):
        image_bytes({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})
