from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.release_adapter import (
    release_companions,
    require_uniform_runtime_release,
    resolve_runtime_image_path,
    verification_case_from_runtime_row,
)
from src.orchestrator.source_access import SourceAccessPolicy


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "contracts" / "runtime-release" / "v0.2"
)
PUBLIC_KEYS = {
    "case_id",
    "image_path",
    "image_sha256",
    "claim_mode",
    "user_claim",
    "claim_surface",
    "claim_source_region",
    "claim_observed_at",
    "decision_policy_version",
}
FORBIDDEN_PUBLIC_KEYS = {
    "factual_status",
    "target_claim",
    "acceptable_evidence",
    "core_loop_gold",
    "unverifiable_reasons",
    "image_origin",
    "ground_truth",
    "bucket",
    "source_article_url",
    "world_id",
    "intervention",
    "visual_facts",
    "tasks",
    "findings",
    "reflection",
}


def _json(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_v02_public_fixtures_project_to_runtime_owned_cases(tmp_path: Path) -> None:
    benchmark = tmp_path / "release" / "runtime_input" / "cases.jsonl"
    for name, expected_mode in (
        ("external_claim.json", "external_claim"),
        ("embedded_claim.json", "embedded_claim"),
    ):
        public = _json(name)
        assert set(public) == PUBLIC_KEYS
        assert not FORBIDDEN_PUBLIC_KEYS.intersection(public)
        resolved = resolve_runtime_image_path(public, benchmark)
        case = verification_case_from_runtime_row(resolved)

        assert case is not None
        assert case.case_id == public["case_id"]
        assert case.claim_mode.value == expected_mode
        assert case.image_path == str(
            (benchmark.parent / public["image_path"]).resolve()
        )
        assert case.decision_policy_version == "reinspect-v1"


def test_v02_private_gold_and_policy_fixtures_are_joinable() -> None:
    public_ids = {
        _json("external_claim.json")["case_id"],
        _json("embedded_claim.json")["case_id"],
    }
    private = _json("evaluator_private.json")
    gold = _json("evaluation_gold.json")
    policy = SourceAccessPolicy.from_dict(_json("source_access_policy.json"))

    assert {row["sample_id"] for row in private} == public_ids
    assert {row["case_id"] for row in gold} == public_ids
    expected_verdict = {
        "supported": "real",
        "refuted": "fake",
        "unverifiable": "unverifiable",
    }
    private_by_id = {row["sample_id"]: row for row in private}
    for row in gold:
        assert private_by_id[row["case_id"]]["ground_truth"] == expected_verdict[
            row["factual_status"]
        ]
    assert not policy.allows("https://factcheck.example/cases/external-claim")
    assert policy.allows("https://official.example/events/ceremony")


def test_v02_manifest_pins_the_vendored_fixture_bytes() -> None:
    manifest = _json("manifest.json")

    assert manifest["producer_baseline"]["commit"] == "dc3c625"
    assert manifest["consumer_baseline"]["commit"] == "1b4b49c"
    for name, expected_sha256 in manifest["files"].items():
        actual = hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest()
        assert actual == expected_sha256


def test_standard_release_companions_are_inferred(tmp_path: Path) -> None:
    benchmark = tmp_path / "release" / "runtime_input" / "cases.jsonl"
    companions = release_companions(benchmark)

    assert companions is not None
    assert companions.evaluator_private == (
        tmp_path / "release" / "evaluator_private" / "run_eval.jsonl"
    )
    assert companions.evaluation_gold == (
        tmp_path / "release" / "evaluation_gold" / "gold.jsonl"
    )


def test_image_only_is_fail_closed_until_v03_is_active() -> None:
    row = _json("external_claim.json")
    with pytest.raises(ValueError, match="v0.3/reinspect-v2"):
        verification_case_from_runtime_row(
            {
                **row,
                "claim_mode": "image_only",
                "user_claim": None,
                "claim_observed_at": None,
                "decision_policy_version": "reinspect-v2",
            }
        )


def test_unactivated_decision_policy_is_rejected() -> None:
    row = _json("external_claim.json")
    with pytest.raises(ValueError, match="accepts decision_policy_version=reinspect-v1"):
        verification_case_from_runtime_row(
            {**row, "decision_policy_version": "reinspect-v2"}
        )


def test_release_row_requires_explicit_policy_version() -> None:
    row = _json("external_claim.json")
    del row["decision_policy_version"]

    with pytest.raises(ValueError, match="missing: decision_policy_version"):
        verification_case_from_runtime_row(row)

    with pytest.raises(ValueError, match="unexpected: factual_status"):
        verification_case_from_runtime_row(
            {**_json("external_claim.json"), "factual_status": "supported"}
        )


def test_mixed_release_and_legacy_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="mixes runtime release rows"):
        require_uniform_runtime_release(
            [
                _json("external_claim.json"),
                {"sample_id": "legacy", "image_path": "image.jpg"},
            ]
        )
