import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.tools.vision_utils import (
    controlled_image_to_data_url,
    vision_tool_image_to_data_url,
)


def test_controlled_image_limits_resolution_and_records_sizes(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (2400, 1200), color=(30, 60, 90)).save(path)

    data_url, metadata = controlled_image_to_data_url(
        str(path),
        max_long_edge=640,
    )

    assert data_url.startswith("data:image/jpeg;base64,")
    assert metadata["original_size"] == [2400, 1200]
    assert metadata["sent_size"] == [640, 320]
    assert metadata["encoded_bytes"] > 0
    assert len(metadata["source_sha256"]) == 64
    assert len(metadata["sha256"]) == 64


def test_vision_tool_uses_light_compression(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (3000, 1500), color=(30, 60, 90)).save(path)

    data_url = vision_tool_image_to_data_url(str(path))

    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.split(",", 1)[1]

    with Image.open(BytesIO(base64.b64decode(encoded))) as sent:
        assert sent.size == (2048, 1024)
