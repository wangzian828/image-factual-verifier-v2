"""Wait for reviewed-only postprocess, then attest and build its PSD candidates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from ifv_training.io import load_json, write_json
from ifv_training.psd_candidates import build_psd_candidate_package_parallel
from ifv_training.psd_round import verify_psd_round_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-cases", type=Path, required=True)
    parser.add_argument("--serving-profile", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--postprocess-pid", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.candidate_workers <= 32:
        raise ValueError("candidate workers must be in [1, 32]")
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll seconds must be in [1, 60]")

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "psd-postprocess-progress.json"
    state_path = output / "state.json"
    while True:
        progress = load_json(progress_path) if progress_path.is_file() else {}
        if progress.get("status") == "completed":
            break
        try:
            os.kill(args.postprocess_pid, 0)
        except ProcessLookupError as exc:
            write_json(state_path, {"phase": "postprocess_exited_before_completion",
                "completed": progress.get("completed", 0), "total": progress.get("total")})
            raise RuntimeError("reviewed-only postprocess exited before completion") from exc
        write_json(state_path, {
            "phase": "waiting_reviewed_only_postprocess",
            "completed": progress.get("completed", 0),
            "total": progress.get("total"),
        })
        time.sleep(args.poll_seconds)

    postprocess = load_json(run_dir / "psd-postprocess.json")
    if postprocess.get("reviewed_only") is not True:
        raise ValueError("candidate handoff requires reviewed-only postprocess")
    if postprocess.get("episodes") != progress.get("total"):
        raise ValueError("postprocess completion count changed")
    write_json(state_path, {"phase": "attesting_rollout_subset"})
    gate_path = output / "rollout-gate.json"
    gate = verify_psd_round_rollout(
        round_index=1,
        run_dir=run_dir,
        train_cases_path=args.train_cases.resolve(),
        serving_profile_path=args.serving_profile.resolve(),
        round_start_checkpoint_manifest_path=args.checkpoint_manifest.resolve(),
        output=gate_path,
    )
    if gate.get("passed") is not True:
        write_json(state_path, {"phase": "rollout_gate_failed", "checks": gate.get("checks")})
        raise ValueError("reviewed-only rollout gate failed")

    write_json(state_path, {"phase": "building_candidates"})
    candidate_dir = output / "source-bank" / "candidates"
    manifest = build_psd_candidate_package_parallel(
        run_dir=run_dir,
        train_cases_path=args.train_cases.resolve(),
        rollout_gate_path=gate_path,
        output_dir=candidate_dir,
        workers=args.candidate_workers,
    )
    result = {
        "phase": "candidates_ready",
        "postprocess": postprocess,
        "counts": manifest["counts"],
        "repair_signals": manifest["repair_signals"],
        "training_started": False,
    }
    write_json(state_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
