"""Bind a numerically audited merged LoRA export to its source checkpoint.

Some vLLM/model combinations cannot load multimodal LoRA adapters dynamically.
In that case the engine serves a standalone merged model, while PSD lineage must
still identify the immutable adapter checkpoint that defines the round policy.
This tool verifies both identities and emits an honest ``merged_lora`` serving
profile for the next-round rollout gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ifv_training.io import load_json, sha256_file, write_json


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolved(value: Any) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def _checkpoint_model_hashes(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in checkpoint.get("artifacts", []) or []:
        if not isinstance(item, Mapping) or item.get("scope") != "model":
            continue
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "").casefold()
        if path and digest:
            result[path] = digest
    return result


def attest(
    *,
    runtime_profile_path: Path,
    checkpoint_manifest_path: Path,
    merge_report_path: Path,
    numeric_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    runtime_profile_path = runtime_profile_path.expanduser().resolve()
    checkpoint_manifest_path = checkpoint_manifest_path.expanduser().resolve()
    merge_report_path = merge_report_path.expanduser().resolve()
    numeric_audit_path = numeric_audit_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    runtime = load_json(runtime_profile_path)
    checkpoint = load_json(checkpoint_manifest_path)
    merge = load_json(merge_report_path)
    numeric = load_json(numeric_audit_path)
    checkpoint_record = _mapping(checkpoint.get("checkpoint"))
    base_record = _mapping(checkpoint.get("base_model"))
    merge_source = _mapping(merge.get("source"))

    adapter = _resolved(checkpoint_record.get("path"))
    merged_model = _resolved(runtime.get("engine_model_path") or runtime.get("model_path"))
    base_model = _resolved(base_record.get("id_or_path"))
    checkpoint_hashes = _checkpoint_model_hashes(checkpoint)
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"

    merged_artifacts_valid = True
    merged_artifacts = _mapping(merge.get("model_artifacts"))
    if not merged_artifacts:
        merged_artifacts_valid = False
    for relative, record in merged_artifacts.items():
        artifact = (merged_model / str(relative)).resolve()
        if (
            not artifact.is_relative_to(merged_model)
            or not artifact.is_file()
            or sha256_file(artifact).casefold()
            != str(_mapping(record).get("sha256") or "").casefold()
        ):
            merged_artifacts_valid = False
            break

    checks = {
        "runtime_profile_schema": runtime.get("schema_version")
        == "ifv-qwen-serving-profile-v1",
        "runtime_serves_full_merged_model": runtime.get("deployment_mode")
        == "full_model"
        and _resolved(runtime.get("model_path")) == merged_model,
        "runtime_is_multimodal_128k": runtime.get("multimodal") is True
        and int(runtime.get("context_length") or 0) == 131072,
        "checkpoint_manifest_schema": checkpoint.get("schema_version")
        == "ifv-qwen-checkpoint-manifest-v1",
        "checkpoint_is_lora": checkpoint.get("training_method") == "lora",
        "source_paths_exist": adapter.is_dir()
        and merged_model.is_dir()
        and base_model.is_dir(),
        "adapter_files_present": adapter_config.is_file() and adapter_weights.is_file(),
        "adapter_config_bound": adapter_config.is_file()
        and checkpoint_hashes.get("adapter_config.json")
        == sha256_file(adapter_config).casefold()
        == str(merge_source.get("adapter_config_sha256") or "").casefold(),
        "adapter_weights_bound": adapter_weights.is_file()
        and checkpoint_hashes.get("adapter_model.safetensors")
        == sha256_file(adapter_weights).casefold()
        == str(merge_source.get("adapter_weights_sha256") or "").casefold(),
        "merge_report_schema": merge.get("schema_version")
        == "ifv-merged-lora-serving-export-v1",
        "merge_report_passed": merge.get("passed") is True,
        "merge_adapter_path": _resolved(merge_source.get("adapter")) == adapter,
        "merge_base_path": _resolved(merge_source.get("base_model")) == base_model,
        "merge_model_path": _resolved(merge.get("model_path")) == merged_model,
        "merged_artifacts_intact": merged_artifacts_valid,
        "numeric_audit_schema": numeric.get("schema_version")
        == "ifv-merged-lora-audit-v1",
        "numeric_audit_passed": numeric.get("passed") is True
        and bool(numeric.get("samples"))
        and all(
            _mapping(sample).get("passed") is True
            for sample in numeric.get("samples", [])
        ),
        "numeric_paths_bound": _resolved(numeric.get("adapter")) == adapter
        and _resolved(numeric.get("base_model")) == base_model
        and _resolved(numeric.get("merged_model")) == merged_model,
    }
    report = {
        "schema_version": "ifv-merged-lora-serving-attestation-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "runtime_profile": {
            "path": str(runtime_profile_path),
            "sha256": sha256_file(runtime_profile_path),
        },
        "checkpoint_manifest": {
            "path": str(checkpoint_manifest_path),
            "sha256": sha256_file(checkpoint_manifest_path),
        },
        "merge_report": {
            "path": str(merge_report_path),
            "sha256": sha256_file(merge_report_path),
        },
        "numeric_audit": {
            "path": str(numeric_audit_path),
            "sha256": sha256_file(numeric_audit_path),
        },
        "effective_adapter": str(adapter),
        "engine_model": str(merged_model),
        "base_model": str(base_model),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "merged-lora-attestation.json"
    write_json(report_path, report)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("merged LoRA serving attestation failed: " + ",".join(failed))

    effective_profile = dict(runtime)
    effective_profile.update(
        {
            "model_path": str(adapter),
            "engine_model_path": str(merged_model),
            "adapter_path": str(adapter),
            "deployment_mode": "merged_lora",
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "merged_lora_attestation": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
        }
    )
    write_json(output_dir / "serving-profile.json", effective_profile)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime-profile",
        "checkpoint-manifest",
        "merge-report",
        "numeric-audit",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = attest(
        runtime_profile_path=args.runtime_profile,
        checkpoint_manifest_path=args.checkpoint_manifest,
        merge_report_path=args.merge_report,
        numeric_audit_path=args.numeric_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
