from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.release_adapter import (
    image_only_case_from_runtime_row,
    load_runtime_release,
    resolve_runtime_image_path,
)


PUBLIC_ROW = {
    "case_id": "case_0123456789abcdef",
    "image_path": "assets/sha256/00/image.jpg",
    "image_sha256": "0" * 64,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release(tmp_path: Path, *, policy_active: bool = False) -> Path:
    root = tmp_path / "release"
    benchmark = root / "runtime_input" / "cases.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(json.dumps(PUBLIC_ROW) + "\n", encoding="utf-8")
    for relative, payload in (
        ("evaluator_private/gold.jsonl", ""),
        ("evaluation/classification_protocol.json", "{}"),
        ("evaluation/process_reference_protocol.json", "{}"),
        ("metadata/licenses.jsonl", ""),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    policy = {"active": False}
    if policy_active:
        policy_path = root / "evaluator_private" / "source_access_policy.json"
        _write_json(
            policy_path,
            {
                "schema_version": "source-access-policy-v1",
                "policy_id": "fixture",
                "excluded_domains": [],
                "excluded_urls": [],
            },
        )
        policy = {
            "active": True,
            "path": "evaluator_private/source_access_policy.json",
        }
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ifv-image-only-benchmark-release-v0.3",
            "release_id": "image-only-fixture",
            "release_stage": "development_subset",
            "runtime_contract_version": "ifv-image-only-runtime-v1",
            "input_mode": "image_only",
            "decision_policy_version": "reinspect-v2",
            "runtime_contract": {
                "allowed_keys": ["case_id", "image_path", "image_sha256"],
                "private_keys_absent": True,
            },
            "artifacts": {
                "agent_input": "runtime_input/cases.jsonl",
                "evaluation_gold": "evaluator_private/gold.jsonl",
                "classification_protocol": (
                    "evaluation/classification_protocol.json"
                ),
                "process_reference_protocol": (
                    "evaluation/process_reference_protocol.json"
                ),
                "licenses": "metadata/licenses.jsonl",
            },
            "source_access_policy": policy,
        },
    )
    return benchmark


def test_v03_manifest_and_three_field_case_are_runtime_owned(
    tmp_path: Path,
) -> None:
    benchmark = _release(tmp_path)
    release = load_runtime_release(benchmark)
    resolved = resolve_runtime_image_path(PUBLIC_ROW, benchmark)
    case = image_only_case_from_runtime_row(resolved)

    assert release is not None
    assert release.release_id == "image-only-fixture"
    assert release.input_mode == "image_only"
    assert release.decision_policy_version == "reinspect-v2"
    assert release.source_access_policy_active is False
    assert release.artifacts.source_access_policy is None
    assert case.case_id == PUBLIC_ROW["case_id"]
    assert Path(case.image_path) == (
        benchmark.parent / PUBLIC_ROW["image_path"]
    ).resolve()
    assert set(case.model_dump()) == {
        "case_id",
        "image_path",
        "image_sha256",
    }


def test_v03_rejects_private_or_claim_fields() -> None:
    for unexpected in (
        "claim_mode",
        "user_claim",
        "factual_status",
        "decisive_facts",
        "acceptable_evidence",
    ):
        with pytest.raises(ValueError, match=f"unexpected: {unexpected}"):
            image_only_case_from_runtime_row(
                {**PUBLIC_ROW, unexpected: None}
            )


def test_v03_manifest_fails_closed_on_contract_drift(tmp_path: Path) -> None:
    benchmark = _release(tmp_path)
    manifest_path = benchmark.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_policy_version"] = "reinspect-v1"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="expected 'reinspect-v2'"):
        load_runtime_release(benchmark)


def test_v03_paths_cannot_escape_release_boundaries(tmp_path: Path) -> None:
    benchmark = _release(tmp_path)

    with pytest.raises(ValueError, match="escapes runtime_input"):
        resolve_runtime_image_path(
            {**PUBLIC_ROW, "image_path": "../evaluator_private/gold.jsonl"},
            benchmark,
        )

    manifest_path = benchmark.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation_gold"] = "../gold.jsonl"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="escapes release root"):
        load_runtime_release(benchmark)


def test_v03_active_policy_is_explicit_and_required(tmp_path: Path) -> None:
    benchmark = _release(tmp_path, policy_active=True)
    release = load_runtime_release(benchmark)

    assert release is not None
    assert release.source_access_policy_active is True
    assert release.artifacts.source_access_policy == (
        benchmark.parent.parent
        / "evaluator_private"
        / "source_access_policy.json"
    )

    release.artifacts.source_access_policy.unlink()
    with pytest.raises(FileNotFoundError, match="source-access policy"):
        load_runtime_release(benchmark)
