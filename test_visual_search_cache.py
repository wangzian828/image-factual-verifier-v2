from __future__ import annotations

from pathlib import Path

from src.integrations.search.visual_search import ImageUploadClient


def test_image_upload_reuses_unexpired_url_for_same_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"stable-image-content")
    client = ImageUploadClient(
        provider="temp",
        temp_upload_url="https://upload.example.test",
        upload_cache_ttl_seconds=900,
    )
    calls: list[str] = []

    def fake_upload(path: Path, upload_url: str) -> str:
        calls.append(f"{path}:{upload_url}")
        return "https://cdn.example.test/stable-image"

    client._upload_via_http = fake_upload  # type: ignore[method-assign]

    first = client.upload(str(image_path))
    first_meta = dict(client.last_upload_meta or {})
    second = client.upload(str(image_path))
    second_meta = dict(client.last_upload_meta or {})

    assert first == second == "https://cdn.example.test/stable-image"
    assert len(calls) == 1
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert len(second_meta["image_sha256"]) == 64
