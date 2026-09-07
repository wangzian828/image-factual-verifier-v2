#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"

usage() {
    cat >&2 <<'EOF'
usage: prepare_factcheck_dataset.sh --split train|test [OPTIONS]

Downloads one authoritative ModelScope archive, verifies its SHA-256, extracts
it atomically, and validates its manifest, private gold, and image references.

Options:
  --output-dir DIR     Extracted dataset destination.
  --archive-path FILE  Reuse an already downloaded archive.
  --force-download     Redownload the archive from ModelScope.
EOF
}

split=""
output_dir=""
archive_path=""
force_download="0"
while (($#)); do
    case "$1" in
        --split)
            (($# >= 2)) || { usage; exit 2; }
            split="$2"
            shift 2
            ;;
        --split=*)
            split="${1#*=}"
            shift
            ;;
        --output-dir)
            (($# >= 2)) || { usage; exit 2; }
            output_dir="$2"
            shift 2
            ;;
        --output-dir=*)
            output_dir="${1#*=}"
            shift
            ;;
        --archive-path)
            (($# >= 2)) || { usage; exit 2; }
            archive_path="$2"
            shift 2
            ;;
        --archive-path=*)
            archive_path="${1#*=}"
            shift
            ;;
        --force-download)
            force_download="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

case "${split}" in
    train)
        repo_id="jiashuhong/factcheck_train"
        archive_name="factcheck_train-8490-20260907.tar.gz"
        archive_sha256="2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448"
        expected_rows="8490"
        manifest_name="train-manifest.jsonl"
        gold_relative="evaluator_private/private-gold-v1/train-private-gold.jsonl"
        ;;
    test)
        repo_id="jiashuhong/factcheck_test"
        archive_name="factcheck_test-1682-20260907.tar.gz"
        archive_sha256="485dba3b8b3947913f372b57f85dbe30b456894022b3fa467c9460dcb8847ea5"
        expected_rows="1682"
        manifest_name="test-manifest.jsonl"
        gold_relative="evaluator_private/private-gold-v1/test-private-gold.jsonl"
        ;;
    *)
        echo "--split must be train or test." >&2
        usage
        exit 2
        ;;
esac

output_dir="${output_dir:-${IFV_DATA_ROOT}/datasets/factcheck_${split}-${expected_rows}-20260907}"
output_dir="$(realpath -m -- "${output_dir}")"
marker="${output_dir}/.complete"

validate_dataset() {
    python - "${1}" "${manifest_name}" "${gold_relative}" "${expected_rows}" "${split}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / sys.argv[2]
gold = root / sys.argv[3]
expected = int(sys.argv[4])
split = sys.argv[5]

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    return rows

rows = read_jsonl(manifest)
gold_rows = read_jsonl(gold)
if len(rows) != expected:
    raise ValueError(f"{manifest}: expected {expected} rows, got {len(rows)}")
if len(gold_rows) != expected:
    raise ValueError(f"{gold}: expected {expected} rows, got {len(gold_rows)}")

case_ids = {
    str(
        row.get("unified_case_id")
        or row.get("case_id")
        or row.get("record_id")
        or row.get("archive_source_version_id")
        or ""
    ).strip()
    for row in rows
}
if "" in case_ids or len(case_ids) != expected:
    raise ValueError(f"{manifest}: case IDs are missing or duplicated")
gold_case_ids = {
    str(
        row.get("case_id")
        or row.get("unified_case_id")
        or row.get("record_id")
        or row.get("archive_source_version_id")
        or ""
    ).strip()
    for row in gold_rows
}
if "" in gold_case_ids or len(gold_case_ids) != expected:
    raise ValueError(f"{gold}: case IDs are missing or duplicated")
if gold_case_ids != case_ids:
    raise ValueError(
        "manifest/private-gold case ID mismatch: "
        f"manifest_only={sorted(case_ids - gold_case_ids)[:3]}, "
        f"gold_only={sorted(gold_case_ids - case_ids)[:3]}"
    )

missing = []
resolved_images = set()
for row in rows:
    relative = str(
        row.get("unified_image_path")
        or row.get("local_image_path")
        or ""
    ).strip()
    if not relative:
        missing.append("<missing image path>")
        continue
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes dataset root: {relative}") from exc
    if not path.is_file():
        missing.append(relative)
    resolved_images.add(path)
if missing:
    raise FileNotFoundError(f"missing images: {missing[:3]}")
if len(resolved_images) != expected:
    raise ValueError(
        f"expected {expected} unique image references, got {len(resolved_images)}"
    )

package_manifest = root / "PACKAGE_MANIFEST.json"
if package_manifest.is_file():
    package = json.loads(package_manifest.read_text(encoding="utf-8"))
    if package.get("split") != split:
        raise ValueError(f"{package_manifest}: split mismatch")
    if int(package.get("manifest_rows", -1)) != expected:
        raise ValueError(f"{package_manifest}: row-count mismatch")

print(
    json.dumps(
        {
            "dataset_root": str(root),
            "split": split,
            "manifest_rows": len(rows),
            "gold_rows": len(gold_rows),
            "unique_images": len(resolved_images),
        },
        ensure_ascii=False,
    )
)
PY
}

if [[ -f "${marker}" ]] \
    && grep -Fxq "archive_sha256=${archive_sha256}" "${marker}"; then
    validate_dataset "${output_dir}"
    printf 'dataset_root=%s\n' "${output_dir}"
    exit 0
fi
if [[ -e "${output_dir}" ]]; then
    echo "dataset destination already exists without a matching .complete marker:" >&2
    echo "${output_dir}" >&2
    exit 2
fi

if [[ -z "${archive_path}" ]]; then
    download_dir="${IFV_DATA_ROOT}/downloads/modelscope/${repo_id#*/}"
    mkdir -p -- "${download_dir}"
    archive_path="${download_dir}/${archive_name}"
    if [[ "${force_download}" == "1" || ! -f "${archive_path}" ]]; then
        python - "${repo_id}" "${archive_name}" "${download_dir}" "${archive_sha256}" "${force_download}" <<'PY'
import os
import sys
from modelscope_hub.api import HubApi

repo_id, archive_name, download_dir, expected_sha256, force = sys.argv[1:]
token = (
    os.getenv("MODELSCOPE_API_TOKEN")
    or os.getenv("MODELSCOPE_TOKEN")
    or None
)
api = HubApi(token=token)
path = api.download_file(
    repo_id=repo_id,
    repo_type="dataset",
    file_path=archive_name,
    local_dir=download_dir,
    force=force == "1",
    expected_sha256=expected_sha256,
)
print(path)
PY
    fi
fi

archive_path="$(realpath -m -- "${archive_path}")"
if [[ ! -f "${archive_path}" ]]; then
    echo "dataset archive does not exist: ${archive_path}" >&2
    exit 2
fi
actual_sha256="$(sha256sum "${archive_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${archive_sha256}" ]]; then
    echo "dataset archive SHA-256 mismatch." >&2
    echo "expected=${archive_sha256}" >&2
    echo "actual=${actual_sha256}" >&2
    exit 1
fi

parent="$(dirname -- "${output_dir}")"
temporary="${output_dir}.extracting.$$"
mkdir -p -- "${parent}"
rm -rf -- "${temporary}"
trap 'rm -rf -- "${temporary}"' EXIT
mkdir -p -- "${temporary}"
tar -xzf "${archive_path}" -C "${temporary}"
validate_dataset "${temporary}"
mv -- "${temporary}" "${output_dir}"
trap - EXIT
cat > "${marker}" <<EOF
schema_version=ifv-factcheck-dataset-complete-v1
split=${split}
repo_id=${repo_id}
archive=${archive_name}
archive_sha256=${archive_sha256}
manifest_rows=${expected_rows}
EOF
printf 'dataset_root=%s\n' "${output_dir}"
