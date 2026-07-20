from pathlib import Path

from PIL import Image

from src.tools.vision_utils import controlled_image_to_data_url


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
