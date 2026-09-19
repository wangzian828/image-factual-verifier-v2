"""Freeze the next source-ordered 1,000 PSD cases without model-dependent selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/volume/ybo/wza")
POOL = ROOT / "data/psd-candidate-pool-4000-20260914-v3/train"
BENCHMARK = POOL / "runtime-release/runtime_input/cases.jsonl"
SPLITS = POOL / "evaluator_private/case_split.jsonl"
OLD_400 = ROOT / "runs/psd-production400x8-20260917-v6/binding.json"
OLD_1000 = ROOT / "runs/psd-production1000x4-20260918-v1/selection/case-list.txt"
OUTPUT = ROOT / "runs/psd-production1000x4-20260920-v2/selection"


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {OUTPUT}")
    old_400 = json.loads(OLD_400.read_text(encoding="utf-8"))["case_ids"]
    old_1000 = [line.strip() for line in OLD_1000.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(old_400) != 400 or len(set(old_400)) != 400:
        raise ValueError("old 400 binding is not exactly 400 unique cases")
    if len(old_1000) != 1000 or len(set(old_1000)) != 1000:
        raise ValueError("old 1000 binding is not exactly 1000 unique cases")
    if set(old_400) & set(old_1000):
        raise ValueError("old 400 and old 1000 overlap")
    excluded = set(old_400) | set(old_1000)

    split_by_id = {}
    for row in rows(SPLITS):
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in split_by_id:
            raise ValueError("split rows have a missing or duplicate case_id")
        split_by_id[case_id] = row
    source = list(rows(BENCHMARK))
    source_ids = [str(row.get("case_id") or "") for row in source]
    if len(source) != 4000 or len(set(source_ids)) != 4000 or not all(source_ids):
        raise ValueError("candidate pool is not exactly 4000 unique cases")
    selected = [row for row in source if row["case_id"] not in excluded][:1000]
    selected_ids = [row["case_id"] for row in selected]
    if len(selected_ids) != 1000 or set(selected_ids) & excluded:
        raise ValueError("next 1000 selection count/overlap gate failed")
    selected_splits = [split_by_id[case_id] for case_id in selected_ids]
    if any(row.get("split") != "train" for row in selected_splits):
        raise ValueError("next 1000 contains a non-train case")
    runtime_root = BENCHMARK.parent
    missing = [row["case_id"] for row in selected
               if not (runtime_root / row["image_path"]).is_file()]
    if missing:
        raise ValueError(f"next 1000 has missing image paths: {missing[:5]}")

    OUTPUT.mkdir(parents=True)
    case_list = OUTPUT / "case-list.txt"
    benchmark = OUTPUT / "benchmark.jsonl"
    train_cases = OUTPUT / "train-cases.jsonl"
    case_list.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    write_rows(benchmark, selected)
    write_rows(train_cases, selected_splits)
    report = {
        "schema_version": "ifv-psd-next1000-selection-v1",
        "status": "selected_not_started",
        "count": 1000,
        "rollouts_per_case": 4,
        "episodes_planned": 4000,
        "selection_by_answer_or_output": False,
        "selection_order": "source benchmark order after excluding frozen old 400 and old 1000",
        "source": {"benchmark": str(BENCHMARK), "benchmark_sha256": sha(BENCHMARK),
                   "case_split": str(SPLITS), "case_split_sha256": sha(SPLITS)},
        "exclusions": {"old_400": str(OLD_400), "old_400_count": 400,
                       "old_1000": str(OLD_1000), "old_1000_count": 1000,
                       "overlap": 0},
        "gates": {"source_cases": 4000, "selected_cases": 1000,
                  "selected_train_cases": 1000, "selected_existing_image_paths": 1000,
                  "image_content_rehash": False, "old_selection_overlap": 0},
        "artifacts": {"case_list": "case-list.txt", "case_list_sha256": sha(case_list),
                      "benchmark": "benchmark.jsonl", "benchmark_sha256": sha(benchmark),
                      "train_cases": "train-cases.jsonl", "train_cases_sha256": sha(train_cases)},
        "collection_started": False,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
