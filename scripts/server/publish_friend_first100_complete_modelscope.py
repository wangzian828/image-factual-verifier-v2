"""Package and publish all current successes for original samples 1-100."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile


ROOT = Path("/volume/ybo/wza/external-projects/wanglinhaotrain-20260915")
OUTPUTS = ROOT / "project/outputs"
FLASH_CAMPAIGN = OUTPUTS / "friend-flash-campaign-20260916"
PRO_CAMPAIGN = OUTPUTS / "friend-pro-campaign-20260916"
PRO_FIX19 = OUTPUTS / "friend-pro-first100-fix-20260918"
PRO_RETRY6 = OUTPUTS / "friend-pro-first100-failed6-retry-20260919"
PUBLISH = ROOT / "publish"
ARCHIVE_NAME = "friend-campaign-first100-complete-20260919.zip"
ARCHIVE = PUBLISH / ARCHIVE_NAME
RECEIPT = ROOT / "state/modelscope-first100-complete-upload-20260919.json"
REPO_ID = "jiashuhong/wanglinhaotrain"


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


def sample_index(sample_id: str) -> int:
    match = re.fullmatch(r"sample-(\d{4})-[0-9a-f]{8}", sample_id)
    if not match:
        raise RuntimeError("invalid sample id: " + sample_id)
    return int(match.group(1))


def verify_status(status_path: Path, expected_status_sha256: str | None = None) -> tuple[Path, dict]:
    if expected_status_sha256 and sha256(status_path) != expected_status_sha256:
        raise RuntimeError("selected status digest changed: " + str(status_path))
    status = json.loads(status_path.read_text())
    sample_id = status.get("sample_id")
    if status.get("state") != "succeeded":
        raise RuntimeError("selected status is not successful: " + str(status_path))
    output = status_path.parent / "output.pptx"
    if not output.is_file():
        raise RuntimeError("selected PPTX is missing: " + str(output))
    digest = sha256(output)
    if status.get("output_sha256") != digest:
        raise RuntimeError("selected PPTX digest differs from status: " + str(output))
    checks = status.get("package_checks") or {}
    required = ("output_exists", "zip_readable", "exactly_one_slide",
        "native_size_matches")
    if not all(checks.get(key) is True for key in required):
        raise RuntimeError("selected PPTX structure checks incomplete: " + str(output))
    if sample_id != status_path.parent.name:
        raise RuntimeError("selected status/sample directory mismatch")
    return output, status


def aggregate_successes(campaign: Path, limit: int = 100) -> dict[str, tuple[Path, dict, str]]:
    aggregate = json.loads((campaign / "aggregate.json").read_text())
    selected = {}
    for sample_id, record in aggregate["selected"].items():
        if not 1 <= sample_index(sample_id) <= limit:
            continue
        status_path = Path(record["status_path"])
        output, status = verify_status(status_path, record["status_sha256"])
        selected[sample_id] = (output, status, campaign.name)
    return selected


def direct_successes(run: Path, limit: int = 100) -> dict[str, tuple[Path, dict, str]]:
    selected = {}
    for status_path in sorted(run.glob("wave-*/sample-*/status.json")):
        sample_id = status_path.parent.name
        if not 1 <= sample_index(sample_id) <= limit:
            continue
        status = json.loads(status_path.read_text())
        if status.get("state") != "succeeded":
            continue
        output, status = verify_status(status_path)
        if sample_id in selected:
            raise RuntimeError("duplicate successful status in " + str(run))
        selected[sample_id] = (output, status, run.name)
    return selected


def current_first100() -> tuple[dict, dict, list[str], dict]:
    flash = aggregate_successes(FLASH_CAMPAIGN)
    if len(flash) != 100 or sorted(sample_index(x) for x in flash) != list(range(1, 101)):
        raise RuntimeError("Flash original first-100 is not complete")
    pro_main = aggregate_successes(PRO_CAMPAIGN)
    pro_fix = direct_successes(PRO_FIX19)
    pro_retry = direct_successes(PRO_RETRY6)
    pro = dict(pro_main)
    source_counts = {PRO_CAMPAIGN.name: len(pro_main), PRO_FIX19.name: 0, PRO_RETRY6.name: 0}
    for source_name, source in ((PRO_FIX19.name, pro_fix), (PRO_RETRY6.name, pro_retry)):
        for sample_id, result in source.items():
            if sample_id not in pro:
                pro[sample_id] = result
                source_counts[source_name] += 1
    ids = sorted(flash, key=sample_index)
    missing = [sample_id for sample_id in ids if sample_id not in pro]
    if missing or len(pro) != 100:
        raise RuntimeError("Pro original first-100 is incomplete: " + ",".join(missing))
    return flash, pro, ids, source_counts


def package() -> None:
    if ARCHIVE.exists():
        raise RuntimeError("immutable publish archive already exists")
    flash, pro, ids, pro_source_counts = current_first100()
    rows, sources = [], []
    for lane, model, selected in (
            ("flash", "gemini-3.6-flash", flash),
            ("pro", "gemini-3.1-pro-preview", pro)):
        for index, sample_id in enumerate(ids, 1):
            output, status, source_run = selected[sample_id]
            archive_path = f"{lane}/{index:03d}_{sample_id}.pptx"
            rows.append({
                "lane": lane, "index": index, "sample_id": sample_id,
                "model": model, "source_run": source_run,
                "archive_path": archive_path, "sha256": status["output_sha256"],
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
        "Friend campaign complete original first-100 results\n\n"
        "Contains the current successful final output.pptx for original samples "
        "0001 through 0100 in both lanes: 100 Gemini 3.6 Flash results and 100 "
        "Gemini 3.1 Pro Preview results. Pro results merge the main campaign, "
        "the bounded 19-case retry, and the final bounded 6-case retry.\n"
        "Only final PPTX files and compact audit metadata are included. Raw model "
        "transcripts, credentials, caches, generated intermediates and input images "
        "are excluded. Structural success is not a visual-quality score.\n"
    )
    metadata = {
        "schema_version": "friend-original-first100-complete-v1",
        "supersedes": "friend-campaign-first100-corrected-20260918-111829.zip",
        "selection": "original sample indexes 1-100",
        "included_per_model": {"flash": 100, "pro": 100},
        "included_total": 200,
        "pro_source_counts": pro_source_counts,
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
        if archive.testzip() is not None or len(archive.namelist()) != 203:
            raise RuntimeError("publish archive verification failed")
    temporary.replace(ARCHIVE)
    digest = sha256(ARCHIVE)
    save(PUBLISH / (ARCHIVE_NAME + ".receipt.json"), {
        "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": digest, "entries": 203, "pptx": 200,
        "raw_transcripts_included": False,
    })
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": digest, "flash": 100, "pro": 100,
        "pro_source_counts": pro_source_counts}))


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
            commit_message="Add complete original first-100 Flash and Pro results",
            commit_description=("Current structurally successful PPTX outputs for original "
                "samples 1-100, including bounded Pro retry recoveries."),
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
