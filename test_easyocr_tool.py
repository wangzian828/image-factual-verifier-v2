from __future__ import annotations

import pytest

from src.tools.ocr_with_position import OCRWithPositionTool


class FakeEasyOCRReader:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def readtext(self, image, **kwargs):
        self.calls.append((image, kwargs))
        if self.error is not None:
            raise self.error
        return list(self.results)


def patch_reader(
    monkeypatch: pytest.MonkeyPatch,
    reader: FakeEasyOCRReader,
) -> None:
    monkeypatch.setattr(
        OCRWithPositionTool,
        "_get_reader",
        lambda _self: reader,
    )
    monkeypatch.setattr(
        OCRWithPositionTool,
        "_ocr_model",
        lambda _self: "easyocr-test",
    )


def test_easyocr_filters_low_confidence_regions(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    reader = FakeEasyOCRReader(
        [
            ([[0, 0], [20, 0], [20, 10], [0, 10]], "garbage", 0.08),
            (
                [[20, 0], [90, 0], [90, 10], [20, 10]],
                "Tysons Corner",
                0.92,
            ),
        ]
    )
    patch_reader(monkeypatch, reader)

    result = OCRWithPositionTool(min_confidence=0.5).call(
        {"image_input": str(image_path)}
    )

    assert result["status"] == "success"
    assert result["full_text"] == "Tysons Corner"
    assert [item["text"] for item in result["text_regions"]] == [
        "Tysons Corner"
    ]
    assert [item["text"] for item in result["rejected_text_regions"]] == [
        "garbage"
    ]
    assert result["ocr_backend"] == "easyocr"
    assert result["artifact_sha256"]
    assert reader.calls[0][1] == {"detail": 1, "paragraph": False}


def test_easyocr_maps_crop_coordinates(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "crop.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    reader = FakeEasyOCRReader(
        [
            (
                [[6, 5], [54, 5], [54, 20], [6, 20]],
                "Tysons Corner",
                0.92,
            )
        ]
    )
    patch_reader(monkeypatch, reader)

    result = OCRWithPositionTool().call(
        {
            "image_input": str(image_path),
            "bbox": [0.2, 0.2, 0.8, 0.8],
        }
    )

    assert result["status"] == "success"
    assert result["full_text"] == "Tysons Corner"
    assert result["text_regions"][0]["bbox"] == [0.26, 0.25, 0.74, 0.4]
    assert result["requested_bbox"] == [0.2, 0.2, 0.8, 0.8]


def test_easyocr_accepts_tuple_position_output(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "tuple-box.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    reader = FakeEasyOCRReader(
        [
            (
                ((0, 0), (80, 0), (80, 20), (0, 20)),
                "tuple box",
                0.91,
            )
        ]
    )
    patch_reader(monkeypatch, reader)

    result = OCRWithPositionTool().call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["text_regions"][0]["bbox"] == [0.0, 0.0, 0.8, 0.5]


def test_easyocr_accepts_numpy_position_output(tmp_path, monkeypatch) -> None:
    import numpy as np
    from PIL import Image

    image_path = tmp_path / "numpy-box.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    reader = FakeEasyOCRReader(
        [
            (
                np.asarray(
                    [
                        [np.int32(0), np.int32(0)],
                        [np.int32(80), np.int32(0)],
                        [np.int32(80), np.int32(20)],
                        [np.int32(0), np.int32(20)],
                    ]
                ),
                "numpy box",
                np.float64(0.91),
            )
        ]
    )
    patch_reader(monkeypatch, reader)

    result = OCRWithPositionTool().call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["text_regions"][0]["bbox"] == [0.0, 0.0, 0.8, 0.5]


def test_easyocr_failure_is_explicit(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "failure.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    patch_reader(
        monkeypatch,
        FakeEasyOCRReader(error=RuntimeError("reader failed")),
    )

    result = OCRWithPositionTool().call({"image_input": str(image_path)})

    assert result["status"] == "error"
    assert "EasyOCR failed" in result["error"]
    assert result["backend_attempts"][0]["backend"] == "easyocr"
    assert result["subcalls"][0]["provider"] == "easyocr"


def test_easyocr_empty_result_is_success(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "image-only.png"
    Image.new("RGB", (100, 40), "white").save(image_path)
    patch_reader(monkeypatch, FakeEasyOCRReader())

    result = OCRWithPositionTool().call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["text_regions"] == []
    assert result["total_regions"] == 0
    assert result["full_text"] == ""
