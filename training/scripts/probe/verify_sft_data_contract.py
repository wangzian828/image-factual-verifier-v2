#!/usr/bin/env python3
"""Fail-closed binding between raw SFT data, its manifest, and processor audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ifv_training.audit import audit_derived_dataset


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _expected_template_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_length": args.max_context,
        "truncation_strategy": args.truncation_strategy,
        "max_pixels": args.max_pixels,
        "padding_free": args.padding_free,
        "sequence_parallel_size": args.sequence_parallel_size,
        "loss_scale": args.loss_scale,
        "enable_thinking": args.enable_thinking,
        "add_non_thinking_prefix": args.add_non_thinking_prefix,
        "image_max_token_num": args.image_max_token_num,
    }


def verify_sft_data_contract(
    *,
    train_jsonl: Path,
    validation_jsonl: Path,
    dataset_dir: Path,
    processor_report_path: Path,
    model: str,
    expected_template_contract: Mapping[str, Any],
    minimum_train_input_tokens: int | None = None,
    minimum_train_rows_at_or_above: int = 1,
) -> dict[str, Any]:
    errors: list[str] = []
    train_record = _file_record(train_jsonl)
    validation_record = _file_record(validation_jsonl)
    processor_report_path = processor_report_path.expanduser().resolve()
    processor_report = _load_object(processor_report_path)

    if (
        processor_report.get("schema_version")
        != "ifv-ms-swift-agent-processor-verification-v2"
    ):
        errors.append("processor report does not use the required v2 schema")
    if processor_report.get("passed") is not True:
        errors.append("processor report did not pass")

    expected_model = str(Path(model).expanduser().resolve())
    if processor_report.get("model") != expected_model:
        errors.append("processor report model does not match the training model")
    if processor_report.get("template_contract") != dict(
        expected_template_contract
    ):
        errors.append("processor template contract does not match the training profile")

    train_input_tokens_max: int | None = None
    train_rows_at_or_above: int | None = None
    if minimum_train_input_tokens is not None:
        exact_train_path = str(train_jsonl.expanduser().resolve())
        by_dataset = processor_report.get("input_tokens_by_dataset")
        by_dataset = by_dataset if isinstance(by_dataset, Mapping) else {}
        train_distribution = by_dataset.get(exact_train_path)
        if isinstance(train_distribution, Mapping) and isinstance(
            train_distribution.get("max"), int
        ):
            train_input_tokens_max = int(train_distribution["max"])
        if train_input_tokens_max is None:
            errors.append(
                "processor report does not expose the absolute-path train token distribution"
            )
        elif train_input_tokens_max < minimum_train_input_tokens:
            errors.append(
                "processor train rows do not reach the required long-context "
                f"boundary: required>={minimum_train_input_tokens}, "
                f"observed={train_input_tokens_max}"
            )
        boundary_map = processor_report.get("input_token_boundaries_by_dataset")
        boundary_map = boundary_map if isinstance(boundary_map, Mapping) else {}
        boundary_rows = boundary_map.get(exact_train_path)
        boundary_rows = boundary_rows if isinstance(boundary_rows, list) else []
        exact_boundary = next(
            (
                item
                for item in boundary_rows
                if isinstance(item, Mapping)
                and item.get("boundary_tokens") == minimum_train_input_tokens
            ),
            None,
        )
        if isinstance(exact_boundary, Mapping) and isinstance(
            exact_boundary.get("rows_at_or_above"), int
        ):
            train_rows_at_or_above = int(exact_boundary["rows_at_or_above"])
        if train_rows_at_or_above is None:
            errors.append(
                "processor report does not expose an exact train boundary count"
            )
        elif train_rows_at_or_above < minimum_train_rows_at_or_above:
            errors.append(
                "processor train boundary has too few real rows: "
                f"required>={minimum_train_rows_at_or_above}, "
                f"observed={train_rows_at_or_above}"
            )

    reported_files = processor_report.get("dataset_files")
    reported_files = reported_files if isinstance(reported_files, list) else []
    records_by_path = {
        str(item.get("path")): item
        for item in reported_files
        if isinstance(item, Mapping)
    }
    for label, current in (
        ("train", train_record),
        ("validation", validation_record),
    ):
        recorded = records_by_path.get(str(current["path"]))
        if not isinstance(recorded, Mapping):
            errors.append(f"processor report does not cover {label} JSONL")
            continue
        if recorded.get("size") != current["size"]:
            errors.append(f"processor report {label} size does not match")
        if recorded.get("sha256") != current["sha256"]:
            errors.append(f"processor report {label} SHA-256 does not match")

    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"dataset manifest is missing: {manifest_path}")
        dataset_audit: dict[str, Any] = {
            "passed": False,
            "error_count": 1,
            "errors": [str(errors[-1])],
        }
    else:
        dataset_audit = audit_derived_dataset(dataset_dir)
        if dataset_audit.get("passed") is not True:
            errors.append("strict dataset manifest audit did not pass")

        manifest = _load_object(manifest_path)
        artifact_paths = {
            str((dataset_dir / str(item.get("path", ""))).resolve())
            for item in (manifest.get("artifacts") or {}).values()
            if isinstance(item, Mapping)
        }
        for label, current in (
            ("train", train_record),
            ("validation", validation_record),
        ):
            if str(current["path"]) not in artifact_paths:
                errors.append(f"dataset manifest does not bind {label} JSONL")

    return {
        "schema_version": "ifv-sft-raw-data-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "model": expected_model,
        "template_contract": dict(expected_template_contract),
        "datasets": {
            "train": train_record,
            "validation": validation_record,
        },
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": (
            _file_record(manifest_path) if manifest_path.is_file() else None
        ),
        "dataset_audit": dataset_audit,
        "processor_report": {
            **_file_record(processor_report_path),
            "schema_version": processor_report.get("schema_version"),
            "passed": processor_report.get("passed"),
        },
        "long_context_boundary": {
            "required": minimum_train_input_tokens is not None,
            "minimum_train_input_tokens": minimum_train_input_tokens,
            "observed_train_input_tokens_max": train_input_tokens_max,
            "minimum_train_rows_at_or_above": minimum_train_rows_at_or_above,
            "observed_train_rows_at_or_above": train_rows_at_or_above,
            "passed": (
                train_input_tokens_max is not None
                and train_rows_at_or_above is not None
                and minimum_train_input_tokens is not None
                and train_input_tokens_max >= minimum_train_input_tokens
                and train_rows_at_or_above >= minimum_train_rows_at_or_above
                if minimum_train_input_tokens is not None
                else True
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--processor-report", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument(
        "--truncation-strategy",
        choices=("raise", "left", "right", "split"),
        default="raise",
    )
    parser.add_argument("--padding-free", type=_boolean, required=True)
    parser.add_argument("--sequence-parallel-size", type=int, required=True)
    parser.add_argument("--loss-scale", required=True)
    parser.add_argument("--enable-thinking", type=_boolean, required=True)
    parser.add_argument(
        "--add-non-thinking-prefix",
        type=_boolean,
        required=True,
    )
    parser.add_argument("--image-max-token-num", type=int, required=True)
    parser.add_argument("--minimum-train-input-tokens", type=int)
    parser.add_argument("--minimum-train-rows-at-or-above", type=int, default=1)
    args = parser.parse_args()
    if args.max_context < 1 or args.sequence_parallel_size < 1:
        parser.error("context and sequence parallel sizes must be positive")
    if (
        args.minimum_train_input_tokens is not None
        and args.minimum_train_input_tokens < 1
    ):
        parser.error("--minimum-train-input-tokens must be positive")
    if args.minimum_train_rows_at_or_above < 1:
        parser.error("--minimum-train-rows-at-or-above must be positive")

    result = verify_sft_data_contract(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        dataset_dir=args.dataset_dir,
        processor_report_path=args.processor_report,
        model=args.model,
        expected_template_contract=_expected_template_contract(args),
        minimum_train_input_tokens=args.minimum_train_input_tokens,
        minimum_train_rows_at_or_above=args.minimum_train_rows_at_or_above,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
