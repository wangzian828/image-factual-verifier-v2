from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from src.integrations.search.visual_search import ImageUploadClient


class _FakeRequestsSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeOssSession:
    def __init__(self) -> None:
        self.session = _FakeRequestsSession()


def _install_fake_oss(monkeypatch: Any, captured: dict[str, Any]) -> None:
    http_module = types.ModuleType("oss2.http")
    http_module.Session = _FakeOssSession

    class FakeAuth:
        def __init__(self, access_key_id: str, access_key_secret: str) -> None:
            self.access_key_id = access_key_id
            self.access_key_secret = access_key_secret

    class FakeBucket:
        def __init__(
            self,
            auth: Any,
            endpoint: str,
            bucket_name: str,
            *,
            session: Any,
        ) -> None:
            captured["auth"] = auth
            captured["endpoint"] = endpoint
            captured["bucket_name"] = bucket_name
            captured["session"] = session

        def put_object(self, object_name: str, handle: Any) -> None:
            captured["object_name"] = object_name
            captured["payload"] = handle.read()

        def sign_url(
            self,
            method: str,
            object_name: str,
            expiry: int,
            *,
            slash_safe: bool,
        ) -> str:
            return f"https://cdn.example.test/{object_name}"

    oss_module = types.ModuleType("oss2")
    oss_module.Auth = FakeAuth
    oss_module.Bucket = FakeBucket
    oss_module.http = http_module
    monkeypatch.setitem(sys.modules, "oss2", oss_module)
    monkeypatch.setitem(sys.modules, "oss2.http", http_module)


def test_oss_upload_uses_owned_session_and_closes_nested_requests_session(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "secret")
    monkeypatch.setenv("OSS_ENDPOINT", "https://oss.example.test")
    monkeypatch.setenv("OSS_BUCKET_NAME", "bucket")
    monkeypatch.setenv("OSS_USE_SIGNED_URL", "true")

    captured: dict[str, Any] = {}
    _install_fake_oss(monkeypatch, captured)
    client = ImageUploadClient(provider="oss")

    url = client._upload_to_oss(image_path)

    session = captured["session"]
    assert isinstance(session, _FakeOssSession)
    assert url.startswith("https://cdn.example.test/")
    assert session.session.closed is False

    client.close()

    assert session.session.closed is True


def test_oss_session_created_after_close_is_closed_immediately(
    monkeypatch: Any,
) -> None:
    _install_fake_oss(monkeypatch, {})
    client = ImageUploadClient(provider="oss")
    client.close()

    try:
        client._get_oss_session()
    except RuntimeError as exc:
        assert "already closed" in str(exc)
    else:
        raise AssertionError("closed upload client accepted a new OSS session")
