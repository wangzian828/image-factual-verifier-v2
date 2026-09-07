#!/usr/bin/env python3
"""Build the code-only IFV Agent teacher-rollout handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "ifv-agent-teacher-handoff-v1"
SOURCE_REPOSITORY = "git@github.com:wangzian828/image-factual-verifier-v2.git"
SOURCE_BRANCH = "codex/gpu13-canary-20260804-plan-relaxation-01"
TRAIN_DATASET = "jiashuhong/factcheck_train"
TRAIN_ARCHIVE = "factcheck_train-8490-20260907.tar.gz"
TRAIN_ARCHIVE_SHA256 = (
    "2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*")) if path.is_file())


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_executable(path: Path, text: str) -> None:
    _write_text(path, text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_package(source_repo: Path, output_dir: Path) -> dict[str, object]:
    source_repo = source_repo.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    handoff = source_repo / "docs" / "teacher-rollout-api-handoff.md"
    env_template = source_repo / "configs" / "runtime.env.example"
    for source, destination in (
        (handoff, output_dir / "CODEX_HANDOFF.md"),
        (env_template, output_dir / "runtime.env.example"),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)

    _write_executable(
        output_dir / "bootstrap.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${{IFV_SOURCE_REPOSITORY:-{SOURCE_REPOSITORY}}}"
BRANCH="${{IFV_SOURCE_BRANCH:-{SOURCE_BRANCH}}}"
COMMIT="${{IFV_SOURCE_COMMIT:-{commit}}}"
REPO_ROOT="${{IFV_REPO_ROOT:-${{PWD}}/image-factual-verifier-v2}}"

if [[ ! -d "${{REPO_ROOT}}/.git" ]]; then
    git clone --branch "${{BRANCH}}" "${{REPOSITORY}}" "${{REPO_ROOT}}"
fi
if [[ -n "$(git -C "${{REPO_ROOT}}" status --porcelain)" ]]; then
    echo "Refusing to update a dirty checkout: ${{REPO_ROOT}}" >&2
    git -C "${{REPO_ROOT}}" status --short >&2
    exit 2
fi
git -C "${{REPO_ROOT}}" fetch origin "${{BRANCH}}"
git -C "${{REPO_ROOT}}" checkout "${{BRANCH}}"
git -C "${{REPO_ROOT}}" merge --ff-only "${{COMMIT}}"
test "$(git -C "${{REPO_ROOT}}" rev-parse HEAD)" = "${{COMMIT}}"

mkdir -p "${{HOME}}/.config/image-factual-verifier"
if [[ ! -f "${{HOME}}/.config/image-factual-verifier/runtime.env" ]]; then
    cp "${{REPO_ROOT}}/configs/runtime.env.example" \
        "${{HOME}}/.config/image-factual-verifier/runtime.env"
    chmod 600 "${{HOME}}/.config/image-factual-verifier/runtime.env"
fi

printf 'repo_root=%s\\ncommit=%s\\nruntime_env=%s\\n' \
    "${{REPO_ROOT}}" "${{COMMIT}}" \
    "${{HOME}}/.config/image-factual-verifier/runtime.env"
""",
    )
    _write_executable(
        output_dir / "run_smoke.sh",
        """#!/usr/bin/env bash
set -euo pipefail
: "${IFV_REPO_ROOT:?Run bootstrap.sh and export IFV_REPO_ROOT first.}"
exec "${IFV_REPO_ROOT}/scripts/server/start_teacher_rollout_portable.sh" \
    --limit 10 "$@"
""",
    )
    _write_executable(
        output_dir / "run_full.sh",
        """#!/usr/bin/env bash
set -euo pipefail
: "${IFV_REPO_ROOT:?Run bootstrap.sh and export IFV_REPO_ROOT first.}"
exec "${IFV_REPO_ROOT}/scripts/server/start_teacher_rollout_portable.sh" \
    --full "$@"
""",
    )
    _write_text(
        output_dir / "README.md",
        f"""# IFV Agent teacher-rollout handoff

This is a code-only control package. It contains no images, trajectories,
private gold, model weights, endpoint credentials, logs, or checkpoints.

1. Read `CODEX_HANDOFF.md`.
2. Run `bootstrap.sh`.
3. Configure the target server's large-Qwen teacher and judge API in the
   untracked runtime environment.
4. Export `IFV_REPO_ROOT`, then run `run_smoke.sh`.
5. Run `run_full.sh` only after the 10-case smoke passes every gate.

Canonical source:

- repository: `{SOURCE_REPOSITORY}`
- branch: `{SOURCE_BRANCH}`
- commit: `{commit}`

Canonical training data:

- dataset: `{TRAIN_DATASET}`
- archive: `{TRAIN_ARCHIVE}`
- SHA-256: `{TRAIN_ARCHIVE_SHA256}`
""",
    )

    rows = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _files(output_dir)
        if path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_date": "2026-09-07",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "branch": SOURCE_BRANCH,
            "commit": commit,
        },
        "training_data": {
            "dataset": TRAIN_DATASET,
            "archive": TRAIN_ARCHIVE,
            "archive_sha256": TRAIN_ARCHIVE_SHA256,
            "manifest_rows": 8490,
            "image_count": 8490,
        },
        "contains_credentials": False,
        "contains_dataset": False,
        "contains_model_weights": False,
        "files": rows,
    }
    _write_text(
        output_dir / "MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    checksum_rows = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in _files(output_dir)
        if path.name != "SHA256SUMS.txt"
    ]
    _write_text(
        output_dir / "SHA256SUMS.txt",
        "\n".join(checksum_rows) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(args.source_repo, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
