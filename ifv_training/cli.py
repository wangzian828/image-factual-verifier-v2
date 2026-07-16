from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_derived_dataset
from .checkpoints import build_checkpoint_manifest, build_serving_profile
from .io import write_json
from .manifests import write_environment_manifest
from .perception import convert_perception_runs
from .policy import convert_policy_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ifv-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy = subparsers.add_parser("convert-policy")
    policy.add_argument("--input", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)

    perception = subparsers.add_parser("convert-perception")
    perception.add_argument("--run-dir", action="append", type=Path, required=True)
    perception.add_argument("--split-map", type=Path, required=True)
    perception.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--strict", action="store_true")

    environment = subparsers.add_parser("environment-manifest")
    environment.add_argument("--output", type=Path, required=True)
    environment.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    checkpoint = subparsers.add_parser("checkpoint-manifest")
    checkpoint.add_argument("--checkpoint-dir", type=Path, required=True)
    checkpoint.add_argument("--dataset-manifest", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--base-model-id", required=True)
    checkpoint.add_argument("--model-revision", default="")
    checkpoint.add_argument("--processor-revision", default="")
    checkpoint.add_argument(
        "--method",
        choices=("lora", "qlora", "full", "grpo-lora", "grpo-full"),
        required=True,
    )

    serving = subparsers.add_parser("serving-profile")
    serving.add_argument("--output", type=Path, required=True)
    serving.add_argument("--profile-id", required=True)
    serving.add_argument("--model-path", required=True)
    serving.add_argument("--engine", choices=("vllm", "lmdeploy"), required=True)
    serving.add_argument("--port", type=int, required=True)
    serving.add_argument("--tensor-parallel-size", type=int, required=True)
    serving.add_argument("--dtype", default="bfloat16")
    serving.add_argument("--context-length", type=int, required=True)
    serving.add_argument("--checkpoint-manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "convert-policy":
        result = convert_policy_dataset(args.input, args.output)
    elif args.command == "convert-perception":
        result = convert_perception_runs(
            args.run_dir,
            args.output,
            split_map_path=args.split_map,
        )
    elif args.command == "audit":
        result = audit_derived_dataset(args.input)
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "environment-manifest":
        result = write_environment_manifest(args.output, args.repo_root)
    elif args.command == "checkpoint-manifest":
        result = build_checkpoint_manifest(
            checkpoint_dir=args.checkpoint_dir,
            dataset_manifest_path=args.dataset_manifest,
            output_path=args.output,
            base_model_id=args.base_model_id,
            model_revision=args.model_revision,
            processor_revision=args.processor_revision,
            method=args.method,
        )
    elif args.command == "serving-profile":
        result = build_serving_profile(
            output_path=args.output,
            profile_id=args.profile_id,
            model_path=args.model_path,
            engine=args.engine,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            context_length=args.context_length,
            checkpoint_manifest_path=args.checkpoint_manifest,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
