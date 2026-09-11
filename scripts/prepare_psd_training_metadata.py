"""Stream/hash the official train archive, storing metadata only (no image copies)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.psd_gemini_judge import _atomic_json

REPO = "jiashuhong/factcheck_train"
ARCHIVE = "factcheck_train-8490-20260907.tar.gz"
SHA256 = "2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448"
SIZE = 24127361006
MEMBERS = {"train-manifest.jsonl", "evaluator_private/private-gold-v1/train-private-gold.jsonl",
           "PACKAGE_MANIFEST.json"}


class HashingReader:
    def __init__(self, raw):
        self.raw, self.digest, self.count = raw, hashlib.sha256(), 0
        self.started, self.reported = time.monotonic(), 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        self.count += len(data)
        if self.count > SIZE:
            raise ValueError("official train archive exceeded pinned size")
        if self.count - self.reported >= 512 * 1024**2:
            self.reported = self.count
            print(json.dumps({"downloaded_bytes": self.count, "total_bytes": SIZE,
                              "seconds": round(time.monotonic() - self.started)}), flush=True)
        return data


def extract_metadata(response, output):
    """Consume the entire compressed stream before releasing verified metadata."""
    reader = HashingReader(response.raw)
    selected = {}
    with tarfile.open(fileobj=reader, mode="r|gz", bufsize=1024**2) as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe archive member")
            name = relative.as_posix()
            if name not in MEMBERS:
                continue
            if not member.isfile() or member.size > 128 * 1024**2 or name in selected:
                raise ValueError("unsafe/duplicate metadata member")
            with archive.extractfile(member) as handle:
                selected[name] = handle.read()
    # tar may stop before the final gzip trailer/padding; bind all source bytes.
    while reader.read(1024**2):
        pass
    if reader.count != SIZE or reader.digest.hexdigest() != SHA256:
        raise ValueError("official train archive hash/size mismatch")
    required = MEMBERS - {"PACKAGE_MANIFEST.json"}
    if not required.issubset(selected):
        raise ValueError("official training metadata missing")
    for name in required:
        rows = [json.loads(line) for line in selected[name].splitlines() if line.strip()]
        if len(rows) != 8490:
            raise ValueError("official training metadata row count mismatch")
    for name, data in selected.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    result = {"schema_version": "ifv-psd-training-metadata-v1", "archive_sha256": SHA256,
        "archive_bytes_verified": reader.count, "archive_stored": False, "images_stored": 0,
        "metadata_bytes": sum(map(len, selected.values())),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in selected.items()}}
    _atomic_json(output / "verified-metadata.json", result)
    return result


def main():
    from modelscope_hub.api import HubApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    api = HubApi()
    # Use the installed Hub transport's streaming endpoint, not its disk/cache
    # downloader, so a 24GB duplicate archive is never materialized.
    response = api.downloader._client.download_stream(repo_id=REPO, repo_type="dataset",
        file_path=ARCHIVE, revision="master", headers={"Accept-Encoding": "identity"})
    with response:
        response.raise_for_status()
        result = extract_metadata(response, args.output_dir)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
