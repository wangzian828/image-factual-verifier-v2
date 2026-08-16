from __future__ import annotations

from src.integrations.ocr.baidu import BaiduOCRClient
from src.tools.ocr_with_position import OCRWithPositionTool


class FakeResponse:
    def __init__(self, payload, *, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _ocr_payload(*, text="Tysons Corner", left=6, top=5):
    return {
        "words_result": [
            {
                "words": text,
                "location": {
                    "left": left,
                    "top": top,
                    "width": 48,
                    "height": 15,
                },
                "vertexes_location": [
                    {"x": left, "y": top},
                    {"x": left + 48, "y": top},
                    {"x": left + 48, "y": top + 15},
                    {"x": left, "y": top + 15},
                ],
                "probability": {"average": 0.92},
            }
        ]
    }


def test_baidu_client_caches_access_token():
    session = FakeSession(
        [
            FakeResponse({"access_token": "token-a", "expires_in": 3600}),
        ]
    )
    client = BaiduOCRClient(
        api_key="test-key-token-cache",
        secret_key="test-secret",
        session=session,
    )

    first, first_hit = client.get_access_token()
    second, second_hit = client.get_access_token()

    assert first == second == "token-a"
    assert first_hit is False
    assert second_hit is True
    assert len(session.calls) == 1
    assert session.calls[0][1]["params"]["grant_type"] == "client_credentials"


def test_baidu_ocr_maps_position_and_uses_form_request(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "baidu.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    session = FakeSession(
        [
            FakeResponse({"access_token": "token-b", "expires_in": 3600}),
            FakeResponse(_ocr_payload()),
        ]
    )
    client = BaiduOCRClient(
        api_key="test-key-region",
        secret_key="test-secret",
        session=session,
    )

    result = OCRWithPositionTool(
        backend="baidu",
        baidu_client=client,
    ).call({"image_input": str(image_path)})

    assert result["status"] == "success"
    assert result["ocr_backend"] == "baidu"
    assert result["ocr_model"] == "baidu-general"
    assert result["full_text"] == "Tysons Corner"
    assert result["text_regions"][0]["bbox"] == [0.06, 0.05, 0.54, 0.2]
    assert len(result["subcalls"]) == 2
    assert session.calls[1][1]["params"]["access_token"] == "token-b"
    assert session.calls[1][1]["data"]["language_type"] == "CHN_ENG"
    assert session.calls[1][1]["data"]["vertexes_location"] == "true"


def test_baidu_ocr_maps_crop_position_back_to_original(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "baidu-crop.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    session = FakeSession(
        [
            FakeResponse({"access_token": "token-c", "expires_in": 3600}),
            FakeResponse(_ocr_payload(left=6, top=5)),
        ]
    )
    client = BaiduOCRClient(
        api_key="test-key-crop",
        secret_key="test-secret",
        session=session,
    )

    result = OCRWithPositionTool(
        backend="baidu",
        baidu_client=client,
    ).call(
        {
            "image_input": str(image_path),
            "bbox": [0.2, 0.2, 0.8, 0.8],
        }
    )

    assert result["status"] == "success"
    assert result["text_regions"][0]["bbox"] == [0.26, 0.25, 0.74, 0.4]
    assert result["requested_bbox"] == [0.2, 0.2, 0.8, 0.8]


def test_baidu_missing_credentials_does_not_fallback(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "baidu-missing.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    monkeypatch.delenv("BAIDU_OCR_API_KEY", raising=False)
    monkeypatch.delenv("BAIDU_OCR_SECRET_KEY", raising=False)
    monkeypatch.delenv("BAIDU_OCR_ACCESS_TOKEN", raising=False)

    result = OCRWithPositionTool(backend="baidu").call(
        {"image_input": str(image_path)}
    )

    assert result["status"] == "error"
    assert result["ocr_backend"] == "baidu"
    assert "BAIDU_OCR_API_KEY" in result["error"]
    assert result["subcalls"] == []
