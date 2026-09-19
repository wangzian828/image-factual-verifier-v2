import json
from types import SimpleNamespace

import pytest

from scripts.collect_psd_rollouts_experimental import stat_image_verifier


def test_stat_image_verifier_does_not_require_payload_hashing(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"frozen-runtime-image")
    stat = image.stat()
    digest = "a" * 64
    identities = tmp_path / "asset-identities.jsonl"
    identities.write_text(json.dumps({
        "path": str(image.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
        "declared_image_sha256": digest,
    }) + "\n")
    verify = stat_image_verifier(identities)
    verify(SimpleNamespace(image_sha256=digest), str(image))
    image.write_bytes(b"changed-runtime-image")
    with pytest.raises(ValueError, match="stat identity changed"):
        verify(SimpleNamespace(image_sha256=digest), str(image))
