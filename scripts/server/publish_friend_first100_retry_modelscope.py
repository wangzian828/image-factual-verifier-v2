"""Package and publish the six successful first-100 Pro retry results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile


ROOT = Path("/volume/ybo/wza/external-projects/wanglinhaotrain-20260915")
SOURCE = ROOT / "project/outputs/friend-pro-first100-failed6-retry-20260919"
PUBLISH = ROOT / "publish"
ARCHIVE_NAME = "friend-pro-first100-failed6-retry-20260919.zip"
ARCHIVE = PUBLISH / ARCHIVE_NAME
RECEIPT = ROOT / "state/modelscope-first100-retry-upload-20260919.json"
REPO_ID = "jiashuhong/wanglinhaotrain"
EXPECTED_IDS = (
    "sample-0005-6608a63f",
    "sample-0009-db2576f4",
    "sample-0023-e1cb3067",
    "sample-0027-56c24550",
    "sample-0029-f962a103",
    "sample-0031-8d90fe5e",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def locate(sample_id: str) -> tuple[Path, dict]:
    matches = list(SOURCE.glob(f"wave-*/{sample_id}/status.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one retry status for {sample_id}, found {len(matches)}")
    status_path = matches[0]
    status = json.loads(status_path.read_text())
    if status.get("state") != "succeeded" or status.get("attempts") != 1:
        raise RuntimeError(f"retry result is not one-attempt success: {sample_id}")
    output = status_path.parent / "output.pptx"
    if not output.is_file():
        raise RuntimeError(f"retry PPTX missing: {sample_id}")
    digest = sha256(output)
    if status.get("output_sha256") != digest:
        raise RuntimeError(f"retry PPTX digest differs from status: {sample_id}")
    checks = status.get("package_checks") or {}
    required = ("output_exists", "zip_readable", "exactly_one_slide",
        "native_size_matches")
    if not all(checks.get(key) is True for key in required):
        raise RuntimeError(f"retry PPTX structure checks incomplete: {sample_id}")
    return output, status


def package() -> None:
    if ARCHIVE.exists():
        raise RuntimeError("immutable publish archive already exists")
    summary = json.loads((SOURCE / "wave-00/summary.json").read_text())
    if summary.get("target_samples") != 6 or summary.get("states") != {"succeeded": 6}:
        raise RuntimeError("retry wave is not a complete 6/6 success")
    rows = []
    sources = []
    for index, sample_id in enumerate(EXPECTED_IDS, 1):
        output, status = locate(sample_id)
        archive_path = f"pro-retry/{index:03d}_{sample_id}.pptx"
        rows.append({
            "index": index,
            "sample_id": sample_id,
            "model": "gemini-3.1-pro-preview",
            "archive_path": archive_path,
            "sha256": status["output_sha256"],
            "bytes": output.stat().st_size,
            "wall_seconds": status.get("wall_seconds"),
            "output_tokens": (status.get("usage") or {}).get("output_tokens"),
            "thinking_tokens": (status.get("usage") or {}).get("thinking_tokens"),
            "slide_count": (status.get("package_checks") or {}).get("slide_count"),
            "native_size_matches": (status.get("package_checks") or {}).get("native_size_matches"),
        })
        sources.append((output, archive_path))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    readme = (
        "Friend campaign first-100 Pro retry supplement\n\n"
        "This is an additive supplement to "
        "friend-campaign-first100-corrected-20260918-111829.zip.\n"
        "It contains the six first-100 Pro cases that were retried on 2026-09-19; "
        "all six completed successfully on their first bounded retry.\n"
        "Only final PPTX files and compact audit metadata are included. Raw model "
        "transcripts, credentials, caches, generated intermediates and input images "
        "are excluded. Structural success is not a visual-quality score.\n"
    )
    metadata = {
        "schema_version": "friend-first100-pro-retry-supplement-v1",
        "base_archive": "friend-campaign-first100-corrected-20260918-111829.zip",
        "model": "gemini-3.1-pro-preview",
        "requested": 6,
        "succeeded": 6,
        "failed": 0,
        "selection": list(EXPECTED_IDS),
        "source_summary_sha256": sha256(SOURCE / "wave-00/summary.json"),
        "created_at_unix": time.time(),
    }
    PUBLISH.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        archive.writestr("README.txt", readme)
        archive.writestr("manifest.csv", buffer.getvalue())
        archive.writestr("summary.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        for source, target in sources:
            archive.write(source, target)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != 9:
            raise RuntimeError("publish archive verification failed")
    temporary.replace(ARCHIVE)
    save(PUBLISH / (ARCHIVE_NAME + ".receipt.json"), {
        "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE), "entries": 9,
        "pptx": 6, "raw_transcripts_included": False,
    })
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE), "pptx": 6}))


def api_from_stdin():
    from modelscope_hub.api import HubApi
    from modelscope_hub.config import HubConfig
    token = sys.stdin.read().strip()
    if len(token) < 16:
        raise RuntimeError("ModelScope session was not supplied")
    return HubApi(config=HubConfig(endpoint="https://modelscope.cn", token=token,
        config_dir=ROOT / "sdk-config", cache_dir=ROOT / "sdk-cache"))


def auth_check() -> None:
    identity = api_from_stdin().whoami()
    print(json.dumps({"authenticated": bool(identity)}))


def upload() -> None:
    if not ARCHIVE.is_file():
        raise RuntimeError("publish archive is missing")
    digest, size = sha256(ARCHIVE), ARCHIVE.stat().st_size
    api = api_from_stdin()
    before = {item.path: item for item in api.list_repo_files(REPO_ID, "dataset")}
    if ARCHIVE_NAME in before:
        remote = before[ARCHIVE_NAME]
        if remote.size != size or remote.sha256 != digest:
            raise RuntimeError("remote archive name exists with different content")
    else:
        api.upload_file(REPO_ID, "dataset", ARCHIVE, ARCHIVE_NAME,
            commit_message="Add six successful Pro retries for the first-100 results",
            commit_description=("Additive supplement to the existing first-100 "
                "Flash/Pro result archive; includes final PPTX and compact audit metadata."),
            disable_tqdm=True)
    after = {item.path: item for item in api.list_repo_files(REPO_ID, "dataset")}
    remote = after.get(ARCHIVE_NAME)
    if remote is None or remote.size != size or remote.sha256 != digest:
        raise RuntimeError("uploaded archive did not verify against ModelScope")
    save(RECEIPT, {"repo_id": REPO_ID, "path": ARCHIVE_NAME,
        "bytes": size, "sha256": digest, "verified": True, "time": time.time()})
    print(json.dumps({"uploaded": True, "repo_id": REPO_ID,
        "path": ARCHIVE_NAME, "bytes": size, "sha256": digest}))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("package", "auth-check", "upload"))
    {"package": package, "auth-check": auth_check, "upload": upload}[
        parser.parse_args().mode]()
