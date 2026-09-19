"""Collect an isolated PSD source-sampling batch with a non-published group size.

This entry point deliberately keeps the native PSD capture path but does not
claim that a four-rollout group is the published eight-rollout PSD bank.
Artifacts from this command must remain in their own run directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from ifv_training.io import load_jsonl, load_json, sha256_file, write_json
from ifv_training.psd_capture_semantics import install_capture_semantics
from ifv_training.psd_collection import require_token_capture_environment
from scripts.collect_psd_rollouts import PSDWorkflow


def stat_image_verifier(path: Path):
    """Build a verifier that never reads frozen runtime image payloads."""
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[str(Path(row["path"]).resolve())] = row

    def verify(runtime_case, image_path):
        resolved = str(Path(image_path).resolve())
        expected = records.get(resolved)
        if expected is None or runtime_case.image_sha256 != expected["declared_image_sha256"]:
            raise ValueError("runtime image is absent from the frozen stat release")
        stat = Path(resolved).stat()
        actual = {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("frozen runtime image stat identity changed")

    return verify


def install_stat_image_verifier(path: Path) -> None:
    """Bind frozen runtime images by stat identity without reading their bytes."""
    verify = stat_image_verifier(path)
    import src.workflow as workflow_module
    import src.orchestrator.pipeline as pipeline_module
    from src.eval import run_eval as run_eval_module
    workflow_module.verify_case_image = verify
    pipeline_module.verify_case_image = verify
    run_eval_module.verify_case_image = verify


class CompactPSDWorkflow(PSDWorkflow):
    """Experimental collector with one authoritative successful trace."""

    async def run_single(self, image_path, image_id="", *, runtime_case=None):
        import copy
        import hashlib
        from ifv_training.psd_infrastructure_retry import (
            InfrastructureRetriesExhausted, PolicyInfrastructureFailure,
            retry_episode_compact,
        )
        from ifv_training.psd_source_completion import source_completion_failure
        from src.orchestrator.runtime_case import image_sha256
        from src.redaction import sanitize_for_persistence
        import src.workflow as workflow_module
        from src.workflow import VerificationWorkflow

        episode_id = image_id or (runtime_case.case_id if runtime_case else Path(image_path).name)
        trace_dir = Path(self.config.output_dir).resolve()
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in episode_id) + ".json"
        canonical = trace_dir / safe_name
        slot = trace_dir.parent / "psd-infrastructure-attempts" / hashlib.sha256(
            episode_id.encode()).hexdigest()
        image_hash = runtime_case.image_sha256 if runtime_case is not None else image_sha256(image_path)
        if runtime_case is not None:
            workflow_module.verify_case_image(runtime_case, image_path)
        identity = {"episode_id": episode_id,
            "case_id": runtime_case.case_id if runtime_case else episode_id,
            "image_sha256": image_hash, "sampling_seed": self.config.sampling_seed,
            "model": self.config.model_name, "provider": self.config.provider,
            "base_url": self.config.llm_base_url, "temperature": .7,
            "timeout": self.config.timeout}

        async def generate(directory):
            config = copy.copy(self.config)
            config.output_dir = str(directory / "traces")
            config.resume_from = None
            child = CompactPSDWorkflow(config)
            try:
                try:
                    result = await VerificationWorkflow.run_single(
                        child, image_path, episode_id, runtime_case=runtime_case)
                except PolicyInfrastructureFailure:
                    raise
                except Exception as error:
                    result = getattr(error, "_ifv_result", None)
                    if not isinstance(result, dict):
                        raise
                saved = Path(config.output_dir) / safe_name
                if not saved.is_file():
                    raise FileNotFoundError("native Agent did not persist its trace")
                return sanitize_for_persistence(result), saved
            finally:
                await child.aclose()

        try:
            return await retry_episode_compact(root=slot, identity=identity,
                canonical_path=canonical, generate=generate,
                validate_result=source_completion_failure)
        except InfrastructureRetriesExhausted as error:
            return {"image_id": episode_id, "image_path": str(image_path),
                "verdict": "error", "termination": "error", "error": str(error),
                "psd_infrastructure_pending": True}


def main() -> None:
    install_capture_semantics()
    require_token_capture_environment(os.environ)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-cases", type=Path, required=True)
    parser.add_argument("--runtime-stat-identities", type=Path, required=True)
    private_args, remaining = parser.parse_known_args()

    from src.eval import run_cases

    install_stat_image_verifier(private_args.runtime_stat_identities)

    sys.argv = [sys.argv[0], *remaining]
    args = run_cases._parse_args()
    if args.rollouts_per_case < 1:
        raise ValueError("experimental PSD group size must be positive")
    if not args.benchmark or not args.output_dir:
        raise ValueError("experimental PSD collection requires benchmark and output directory")

    train_rows = load_jsonl(private_args.train_cases)
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("experimental PSD collection requires a training-only split")
    public_rows = load_jsonl(Path(args.benchmark))
    public_ids = [row["case_id"] for row in public_rows]
    allowed_ids = {row["case_id"] for row in train_rows}
    if len(public_ids) != len(set(public_ids)) or not set(public_ids) <= allowed_ids:
        raise ValueError("experimental PSD benchmark is not covered by the training split")

    run_cases.VerificationWorkflow = CompactPSDWorkflow
    summary = asyncio.run(run_cases._run_cases(args))

    manifest_path = Path(args.output_dir) / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["agent"]["psd_sampling"] = {
        "temperature": 0.7,
        "rollouts_per_case": args.rollouts_per_case,
        "capture_policy_tokens": True,
        "policy_topk": 20,
        "collector_sha256": sha256_file(Path(__file__)),
        "stages": ["UNIFIED_REACT", "UNIFIED_JUDGMENT"],
        "formal_psd_bank": False,
        "experiment_label": "first_step_exploration_1000x4",
        "storage_policy": "single-canonical-trace-stat-v1",
        "runtime_image_validation": "frozen-stat-identity-v1",
    }
    manifest["experimental_scope"] = {
        "published_group_size": 8,
        "formal_training_allowed": False,
        "selection_by_answer": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
