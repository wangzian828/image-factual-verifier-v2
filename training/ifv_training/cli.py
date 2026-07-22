from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_derived_dataset
from .checkpoints import (
    audit_full_parameter_checkpoint,
    build_checkpoint_manifest,
    build_serving_profile,
)
from .io import load_json, load_jsonl, write_json, write_jsonl
from .manifests import write_environment_manifest
from .perception import convert_perception_runs
from .policy import convert_policy_dataset
from .rewards import (
    build_and_write_ledger,
    build_ledgers_from_run_artifacts,
    build_standard_grpo_groups,
    export_framework_reward,
    validate_reward_ledger,
    validate_semantic_reward_artifact,
)


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

    full_audit = subparsers.add_parser("audit-full-checkpoint")
    full_audit.add_argument("--checkpoint-dir", type=Path, required=True)
    full_audit.add_argument("--base-model-dir", type=Path, required=True)
    full_audit.add_argument("--output", type=Path, required=True)

    serving = subparsers.add_parser("serving-profile")
    serving.add_argument("--output", type=Path, required=True)
    serving.add_argument("--profile-id", required=True)
    serving.add_argument("--model-path", required=True)
    serving.add_argument("--engine", choices=("vllm", "lmdeploy"), required=True)
    serving.add_argument("--port", type=int, required=True)
    serving.add_argument("--tensor-parallel-size", type=int, required=True)
    serving.add_argument("--dtype", default="bfloat16")
    serving.add_argument("--context-length", type=int, required=True)
    serving.add_argument("--tool-call-parser", required=True)
    serving.add_argument("--reasoning-parser", required=True)
    serving.add_argument(
        "--thinking-enabled",
        choices=("true", "false"),
        required=True,
    )
    serving.add_argument("--checkpoint-manifest", type=Path)

    semantic_audit = subparsers.add_parser("audit-semantic-reward")
    semantic_audit.add_argument("--input", type=Path, required=True)
    semantic_audit.add_argument("--output", type=Path)
    semantic_audit.add_argument("--strict", action="store_true")

    reward_ledger = subparsers.add_parser("reward-ledger")
    reward_ledger.add_argument("--semantic-artifact", type=Path, required=True)
    reward_ledger.add_argument("--deterministic", type=Path)
    reward_ledger.add_argument("--profile", type=Path)
    reward_ledger.add_argument("--output", type=Path, required=True)

    ledger_audit = subparsers.add_parser("audit-reward-ledger")
    ledger_audit.add_argument("--input", type=Path, required=True)
    ledger_audit.add_argument("--output", type=Path)
    ledger_audit.add_argument("--strict", action="store_true")

    reward_export = subparsers.add_parser("export-reward")
    reward_export.add_argument("--ledger", type=Path, required=True)
    reward_export.add_argument(
        "--framework",
        choices=("rllm", "verl"),
        required=True,
    )
    reward_export.add_argument("--output", type=Path, required=True)

    grpo_groups = subparsers.add_parser("build-grpo-groups")
    grpo_groups.add_argument("--ledgers", type=Path, required=True)
    grpo_groups.add_argument("--rollout-members", type=Path, required=True)
    grpo_groups.add_argument("--output", type=Path, required=True)
    grpo_groups.add_argument("--minimum-valid-members", type=int, default=2)

    run_rewards = subparsers.add_parser("build-run-rewards")
    run_rewards.add_argument("--semantic-artifacts", type=Path, required=True)
    run_rewards.add_argument("--deterministic", type=Path, required=True)
    run_rewards.add_argument("--rollout-members", type=Path, required=True)
    run_rewards.add_argument("--profile", type=Path)
    run_rewards.add_argument("--ledger-output", type=Path, required=True)
    run_rewards.add_argument("--group-output", type=Path, required=True)
    run_rewards.add_argument("--minimum-valid-members", type=int, default=2)
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
    elif args.command == "audit-full-checkpoint":
        result = audit_full_parameter_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            base_model_dir=args.base_model_dir,
            output_path=args.output,
        )
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
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
            tool_call_parser=args.tool_call_parser,
            reasoning_parser=args.reasoning_parser,
            thinking_enabled=args.thinking_enabled == "true",
            checkpoint_manifest_path=args.checkpoint_manifest,
        )
    elif args.command == "audit-semantic-reward":
        result = validate_semantic_reward_artifact(load_json(args.input))
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "reward-ledger":
        result = build_and_write_ledger(
            semantic_artifact_path=args.semantic_artifact,
            deterministic_path=args.deterministic,
            profile_path=args.profile,
            output_path=args.output,
        )
    elif args.command == "audit-reward-ledger":
        result = validate_reward_ledger(load_json(args.input))
        if args.output:
            write_json(args.output, result)
        if args.strict and not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command == "export-reward":
        result = export_framework_reward(
            load_json(args.ledger),
            framework=args.framework,
        )
        write_json(args.output, result)
    elif args.command == "build-grpo-groups":
        result = build_standard_grpo_groups(
            load_jsonl(args.ledgers),
            load_jsonl(args.rollout_members),
            minimum_valid_members=args.minimum_valid_members,
        )
        write_jsonl(args.output, result)
    elif args.command == "build-run-rewards":
        semantic_paths = sorted(args.semantic_artifacts.glob("*.semantic_reward.json"))
        semantic_artifacts = [load_json(path) for path in semantic_paths]
        profile = load_reward_profile(args.profile)
        ledgers = build_ledgers_from_run_artifacts(
            semantic_artifacts=semantic_artifacts,
            deterministic_rows=load_jsonl(args.deterministic),
            profile=profile,
        )
        groups = build_standard_grpo_groups(
            ledgers,
            load_jsonl(args.rollout_members),
            minimum_valid_members=args.minimum_valid_members,
        )
        write_jsonl(args.ledger_output, ledgers)
        write_jsonl(args.group_output, groups)
        result = {
            "ledger_count": len(ledgers),
            "group_count": len(groups),
            "trainable_group_count": sum(
                1 for group in groups if group.get("trainable") is True
            ),
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
