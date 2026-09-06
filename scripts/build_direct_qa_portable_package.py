#!/usr/bin/env python3
"""Build a self-contained Direct QA comparison package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


FILES = (
    "scripts/run_direct_qa_baseline.py",
    "scripts/audit_direct_qa_baseline.py",
    "scripts/prepare_direct_qa_inputs.py",
    "scripts/__init__.py",
    "src",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _prompt_from_runner(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "DEFAULT_PROMPT" not in names:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("DEFAULT_PROMPT must be a non-empty string")
        return value.strip()
    raise ValueError("DEFAULT_PROMPT was not found in direct-QA runner")


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".mypy_cache",
            ),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_package(
    *,
    source_repo: Path,
    test_package: Path,
    output_dir: Path,
) -> dict[str, object]:
    source_repo = source_repo.expanduser().resolve()
    test_package = test_package.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    if not source_repo.is_dir():
        raise FileNotFoundError(f"source repository does not exist: {source_repo}")
    if not test_package.is_dir():
        raise FileNotFoundError(f"test package does not exist: {test_package}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative in FILES:
        source = source_repo / relative
        if not source.exists():
            raise FileNotFoundError(f"required source is missing: {source}")
        _copy(source, output_dir / "runtime" / relative)

    runner = source_repo / "scripts" / "run_direct_qa_baseline.py"
    (output_dir / "prompt.txt").write_text(
        _prompt_from_runner(runner) + "\n",
        encoding="utf-8",
    )
    (output_dir / "requirements.txt").write_text(
        "httpx>=0.25.0\npython-dotenv>=1.0\npydantic>=2.0\nPillow>=10.0\n",
        encoding="utf-8",
    )

    test_destination = output_dir / "test-set"
    test_destination.mkdir(parents=True, exist_ok=True)
    copied_test_files: list[str] = []
    excluded_names = {
        "extracted",
        "dry-run-test-set-part-001",
        "dry-run-test-set-part-002",
        "verified-fixed",
    }
    for source in sorted(test_package.iterdir()):
        if source.name in excluded_names:
            continue
        if source.is_file() and (
            source.name.endswith(".tar.gz")
            or source.name in {"README.md", "SHA256SUMS.txt"}
        ):
            destination = test_destination / source.name
            shutil.copy2(source, destination)
            copied_test_files.append(destination.relative_to(output_dir).as_posix())

    (output_dir / "run_direct_qa.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$root = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "$forward = @('--prompt-file', (Join-Path $root 'prompt.txt')) + @args\n"
        "python $root/runtime/scripts/run_direct_qa_baseline.py @forward\n",
        encoding="utf-8",
    )
    (output_dir / "run_direct_qa.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ROOT=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "exec python \"$ROOT/runtime/scripts/run_direct_qa_baseline.py\" --prompt-file \"$ROOT/prompt.txt\" \"$@\"\n",
        encoding="utf-8",
    )
    (output_dir / "audit_direct_qa.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$root = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "python $root/runtime/scripts/audit_direct_qa_baseline.py @args\n",
        encoding="utf-8",
    )
    (output_dir / "audit_direct_qa.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ROOT=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "exec python \"$ROOT/runtime/scripts/audit_direct_qa_baseline.py\" \"$@\"\n",
        encoding="utf-8",
    )
    readme = f"""# IFV Direct QA comparison package

This package is evaluator-only and training-prohibited. It contains the exact
Direct QA runner and private post-hoc audit used by the project, plus the two
test-set shards and their image-generation scripts.

## Quick start

1. Extract both archives under `test-set/`.
2. Copy each shard's `config.example.json` to `config.json`, set the image API
   key named by `api_key_env`, and run its `generate.py` into a separate output
   directory.
3. Build the runnable manifest:

```bash
python runtime/scripts/prepare_direct_qa_inputs.py \\
  --package-root test-set/extracted \\
  --generated-root generated \\
  --output-dir direct-qa-inputs
```

4. Run Direct QA with only image plus the shared prompt:

```bash
python runtime/scripts/run_direct_qa_baseline.py \\
  --manifest direct-qa-inputs/manifest.jsonl \\
  --image-root direct-qa-inputs \\
  --prompt-file prompt.txt \\
  --output-dir direct-qa-run \\
  --model <model-id> \\
  --concurrency 4
```

5. Run the private post-hoc audit after setting the judge API key:

```bash
python runtime/scripts/audit_direct_qa_baseline.py \\
  --run-dir direct-qa-run \\
  --manifest direct-qa-inputs/manifest.jsonl \\
  --private-gold-sidecar direct-qa-inputs/private-gold.jsonl \\
  --output-dir direct-qa-run/private-gold-audit \\
  --judge-model <judge-model-id>
```

Only the runner request is model-visible. The manifest's factual metadata and
`private-gold.jsonl` are post-hoc evaluation inputs and are never sent to the
Direct QA model.

The package contains {len(copied_test_files)} copied test artifacts. Verify
`test-set/SHA256SUMS.txt` before extraction.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows: list[dict[str, str]] = []
    for path in _iter_files(output_dir):
        if path.name in {"MANIFEST.json", "SHA256SUMS.txt"}:
            continue
        manifest_rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "bytes": str(path.stat().st_size),
            }
        )
    manifest = {
        "schema_version": "ifv-direct-qa-portable-package-v1",
        "training_prohibited": True,
        "source_repo": str(source_repo),
        "source_commit": subprocess.run(
            ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown",
        "test_artifacts": copied_test_files,
        "files": manifest_rows,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--test-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(
        source_repo=args.source_repo,
        test_package=args.test_package,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
