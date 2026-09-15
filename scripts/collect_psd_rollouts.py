"""Training-only grouped collector; native Agent semantics with explicit T=0.7.

Run in a separate process. Never patch the main evaluation server/process.
The standard run_cases CLI follows --train-cases <private split manifest>.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from src.workflow import VerificationWorkflow
from ifv_training.io import load_jsonl, load_json, write_json, sha256_file


class PSDWorkflow(VerificationWorkflow):
    def _get_orchestrator(self, *, validate_startup=True):
        policy = super()._get_orchestrator(validate_startup=validate_startup)
        if not getattr(policy, "_psd_sampling_bound", False):
            original = policy._stage_generation_config
            def generation(stage):
                config = original(stage)
                if stage.upper() in {"UNIFIED_REACT", "UNIFIED_JUDGMENT"}:
                    config = {**config, "temperature": 0.7}
                return config
            policy._stage_generation_config = generation
            policy._psd_sampling_bound = True
        return policy

    def _new_batch_child(self, config):
        return PSDWorkflow(config)


def main():
    from src.eval import run_cases
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-cases", type=Path, required=True)
    private_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = run_cases._parse_args()
    if args.rollouts_per_case != 8:
        raise ValueError("published PSD collection requires eight rollouts per task")
    if not args.benchmark or not args.output_dir:
        raise ValueError("PSD collector requires explicit public benchmark and new output directory")
    rows = load_jsonl(private_args.train_cases)
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("PSD collection requires an explicitly training-only split")
    public = load_jsonl(Path(args.benchmark))
    if not {r["case_id"] for r in public} <= {r["case_id"] for r in rows}:
        raise ValueError("PSD public benchmark contains non-training cases")
    # Only the collector process gets this factory, including every isolated
    # batch child. The deployment's frozen VerificationWorkflow is unchanged.
    run_cases.VerificationWorkflow = PSDWorkflow
    summary = asyncio.run(run_cases._run_cases(args))
    manifest_path = Path(args.output_dir) / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["agent"]["psd_sampling"] = {"temperature": 0.7, "rollouts_per_case": 8,
        "collector_sha256": sha256_file(Path(__file__)), "stages": ["UNIFIED_REACT", "UNIFIED_JUDGMENT"]}
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
