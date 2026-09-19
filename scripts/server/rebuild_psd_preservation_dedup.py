"""Rebuild frozen PSD targets with one preservation rollout per IFV case.

Consume the already-verified assembly instead of replaying the large repair
attempt ledger.  The latter is unnecessary after finalization and can consume
many gigabytes of memory.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-preserve-dedup-20260920-v76"
CODE = DEPLOY / "code"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
SEARCH = ROUND / "search-gemini37-flash-high"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
OUT = FINAL / "preservation-dedup-v74"
FROZEN_ASSEMBLED = SEARCH / "assembled"


def stream_first_preservation_per_case(source: Path, destination: Path) -> tuple[int, int]:
    from ifv_training.io import canonical_json, iter_jsonl

    seen: set[str] = set()
    source_rows = 0
    selected_rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(source):
            source_rows += 1
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"preservation row {source_rows - 1} has no case_id")
            if case_id in seen:
                continue
            seen.add(case_id)
            output.write(canonical_json(row) + "\n")
            selected_rows += 1
    return source_rows, selected_rows


def main() -> None:
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    spec = importlib.util.spec_from_file_location("owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    if owner.load(FINAL / "state.json")["phase"] != "requires_frozen_teacher_topk":
        raise ValueError("Frozen repair package is not ready for target rebuilding")
    from ifv_training.io import sha256_file, write_json
    from ifv_training.psd import build_psd_target_package

    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    assembled = OUT / "bank" / "assembled"
    assembled.mkdir(parents=True)
    shutil.copyfile(FROZEN_ASSEMBLED / "repairs.jsonl", assembled / "repairs.jsonl")
    source_rows, selected_rows = stream_first_preservation_per_case(
        FROZEN_ASSEMBLED / "preservation.jsonl", assembled / "preservation.jsonl")
    if source_rows != 971 or selected_rows != 246:
        raise RuntimeError("Frozen preservation audit changed")
    assembly = {
        "schema_version": "ifv-psd-preservation-dedup-v1",
        "source": {
            "frozen_repairs": str(FROZEN_ASSEMBLED / "repairs.jsonl"),
            "frozen_repairs_sha256": sha256_file(FROZEN_ASSEMBLED / "repairs.jsonl"),
            "frozen_preservation": str(FROZEN_ASSEMBLED / "preservation.jsonl"),
            "frozen_preservation_sha256": sha256_file(
                FROZEN_ASSEMBLED / "preservation.jsonl")},
        "counts": {"preservation_source_rows": source_rows,
            "preservation_unique_cases": selected_rows},
        "selection_policy": "first_verified_preservation_in_frozen_source_order",
        "status": "ready_for_target_build"}
    write_json(assembled / "manifest.json", assembly)
    targets = build_psd_target_package(
        repairs_path=assembled / "repairs.jsonl",
        preservation_path=assembled / "preservation.jsonl",
        output_dir=OUT / "bank" / "targets")
    result = {"assembly": assembly, "targets": targets,
        "status": "requires_frozen_teacher_topk"}
    if targets["status"] != "ready_for_topk_cache":
        raise RuntimeError("Deduplicated PSD bank did not reach teacher handoff")
    counts = targets["counts"]
    if (counts["repair_targets"] != 626
            or not 0 < counts["preservation_targets"] < 13026):
        raise RuntimeError("Deduplicated PSD bank counts violate the frozen audit")
    owner.save(OUT / "result.json", result)
    owner.save(OUT / "state.json", {"phase": "requires_frozen_teacher_topk",
        "training_started": False, "new_agent_or_provider_calls": False,
        "counts": counts})
    print(json.dumps({"status": result["status"], "counts": counts}))


if __name__ == "__main__":
    main()
