from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=args.local_dir,
    )
    files = []
    for path in sorted(args.local_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(args.local_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": "ifv-model-asset-manifest-v1",
        "model": args.model,
        "revision": args.revision,
        "local_dir": str(args.local_dir.resolve()),
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
