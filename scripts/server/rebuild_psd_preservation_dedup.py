"""Rebuild the frozen PSD bank with one preservation rollout per IFV case."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path("/volume/ybo/wza")
DEPLOY = ROOT / "training-artifacts/psd-preserve-dedup-20260920-v73"
CODE = DEPLOY / "code"
ROUND = ROOT / "runs/psd-production-round1-20260917-v1"
SEARCH = ROUND / "search-gemini37-flash-high"
FINAL = ROOT / "runs/psd-stopped-tail-finalization-20260919-v1"
OUT = FINAL / "preservation-dedup-v72"


def main() -> None:
    os.umask(0o077)
    sys.path[:0] = [str(CODE), str(CODE / "training")]
    spec = importlib.util.spec_from_file_location("owner",
        ROOT / "training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    if owner.load(FINAL / "state.json")["phase"] != "requires_frozen_teacher_topk":
        raise ValueError("Frozen repair package is not ready for target rebuilding")
    from ifv_training.psd_materialization import materialize_bank
    result = materialize_bank(output_dir=OUT / "bank",
        repair_candidates=SEARCH / "merged-stopped-tail-v1/repair_candidates.jsonl.gz",
        repair_attempts=SEARCH / "merged-stopped-tail-v1/repair_attempts.jsonl.gz",
        preservation_candidates=ROUND / "source-bank/candidates/preservation_candidates.jsonl",
        score_missing_topk=False)
    if result["status"] != "requires_frozen_teacher_topk":
        raise RuntimeError("Deduplicated PSD bank did not reach teacher handoff")
    counts = result["targets"]["counts"]
    if (result["assembly"]["counts"].get("preservation_unique_cases") != 246
            or counts["repair_targets"] != 626
            or not 0 < counts["preservation_targets"] < 13026):
        raise RuntimeError("Deduplicated PSD bank counts violate the frozen audit")
    owner.save(OUT / "result.json", result)
    owner.save(OUT / "state.json", {"phase": "requires_frozen_teacher_topk",
        "training_started": False, "new_agent_or_provider_calls": False,
        "counts": counts})
    print(json.dumps({"status": result["status"], "counts": counts}))


if __name__ == "__main__":
    main()
